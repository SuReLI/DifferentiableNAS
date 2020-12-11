export PRIMITIVES, uniform_α, DARTSModel, Cell, MixedOp, DARTSEvalModel, α14, MaskedDARTSModel

using Flux
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
using CUDA
using JuliennedArrays
using Zygote: @adjoint, _zero
using SliceMap


ReLUConvBN(channels_in, channels_out, kernel_size, stride, pad) = Chain(
    x -> relu.(x),
    Conv(kernel_size, channels_in => channels_out),
    BatchNorm(channels_out),
)  |> gpu

function FactorizedReduce(channels_in, channels_out, stride)
    odd = Conv((1, 1), channels_in => channels_out ÷ stride, stride = (stride, stride)) |> gpu
    even = Conv((1, 1), channels_in => channels_out ÷ stride, stride = (stride, stride)) |> gpu

    Chain(
    x -> relu.(x),
    x -> cat(odd(x), even(x[2:end, 2:end, :, :]), dims = 3),
    BatchNorm(channels_out)
    )  |> gpu
end

SepConv(channels_in, channels_out, kernel_size, stride, pad) = Chain(
    x -> relu.(x),
    DepthwiseConv(kernel_size, channels_in => channels_in, stride = (stride, stride), pad = (pad, pad)),
    Conv((1, 1), channels_in => channels_in, stride = (1, 1), pad = (0, 0)),
    BatchNorm(channels_in),
    x -> relu.(x),
    DepthwiseConv(kernel_size, channels_in => channels_in, pad = (pad, pad), stride = (1, 1)),
    Conv((1, 1), channels_in => channels_out, stride = (1, 1), pad = (0, 0)),
    BatchNorm(channels_out),
) |> gpu

DilConv(channels_in, channels_out, kernel_size, stride, pad, dilation) = Chain(
    x -> relu.(x),
    DepthwiseConv(
        kernel_size,
        channels_in => channels_in,
        pad = (pad, pad),
        stride = (stride, stride),
        dilation = dilation,
    ),
    Conv(kernel_size, channels_in => channels_out, stride = (1, 1), pad = (pad, pad)),
    BatchNorm(channels_out),
) |> gpu

SepConv_v(channels_in, channels_out, kernel_size, stride, pad) = Chain(
    x -> relu.(x),
    Conv(kernel_size, channels_in => channels_in, stride = (stride, stride), pad = (pad, pad)),
    Conv((1, 1), channels_in => channels_in, stride = (1, 1), pad = (0, 0)),
    BatchNorm(channels_in),
    x -> relu.(x),
    Conv(kernel_size, channels_in => channels_in, pad = (pad, pad), stride = (1, 1)),
    Conv((1, 1), channels_in => channels_out, stride = (1, 1), pad = (0, 0)),
    BatchNorm(channels_out),
) |> gpu

DilConv_v(channels_in, channels_out, kernel_size, stride, pad, dilation) = Chain(
    x -> relu.(x),
    Conv(
        kernel_size,
        channels_in => channels_in,
        pad = (pad*dilation, pad*dilation),
        stride = (stride, stride),
        dilation = dilation,
    ),
    Conv(kernel_size, channels_in => channels_out, stride = (1, 1), pad = (pad, pad)),
    BatchNorm(channels_out),
) |> gpu


Identity(stride, pad) = x -> x[1:stride:end, 1:stride:end, :, :] |> gpu
Zero(stride, pad) = x -> x[1:stride:end, 1:stride:end, :, :] * 0 |> gpu

SkipConnect(channels_in, channels_out, stride, pad) = stride == 1 ? Identity(stride, pad) |> gpu : FactorizedReduce(channels_in, channels_out, stride) |> gpu

struct AdaptiveMeanPool{N}
    target_out::NTuple{N,Int}
end

AdaptiveMeanPool(target_out::NTuple{N,Integer}) where {N} = AdaptiveMeanPool(target_out)

function (m::AdaptiveMeanPool)(x)
    w = size(x, 1) - m.target_out[1] + 1
    h = size(x, 2) - m.target_out[2] + 1
    return meanpool(x, (w, h); pad = (0, 0), stride = (1, 1)) |> gpu
end

PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    #"sep_conv_7x7",
    "dil_conv_3x3",
    "dil_conv_5x5",
    #"conv_7x1_1x7"
]

#TODO: change to NamedTuple
OPS = Dict(
    "none" => (channels, stride, w) -> Chain(Zero(stride, 1))  |> gpu,
    "avg_pool_3x3" =>
        (channels, stride, w) ->
            Chain(MeanPool((3, 3), stride = stride, pad = 1), BatchNorm(channels)) |> gpu,
    "max_pool_3x3" =>
        (channels, stride, w) ->
            Chain(MaxPool((3, 3), stride = stride, pad = 1), BatchNorm(channels))  |> gpu,
    "skip_connect" =>
        (channels, stride, w) -> Chain(SkipConnect(channels, channels, stride, 1))  |> gpu,
    "sep_conv_3x3" => (channels, stride, w) -> SepConv_v(channels, channels, (3, 3), stride, 1) |> gpu,
    "sep_conv_5x5" => (channels, stride, w)-> SepConv_v(channels, channels, (5, 5), stride, 2) |> gpu,
    "sep_conv_7x7" => (channels, stride, w)-> SepConv_v(channels, channels, (7, 7), stride, 3)  |> gpu,
    "dil_conv_3x3" => (channels, stride, w) -> DilConv_v(channels, channels, (3, 3), stride, 1, 2)  |> gpu,
    "dil_conv_5x5" => (channels, stride, w)-> DilConv_v(channels, channels, (5, 5), stride, 2, 2)  |> gpu,
    "conv_7x1_1x7" =>
        (channels, stride, w) -> Chain(
            x -> relu.(x),
            Conv((1, 7), channels => channels, pad = (0, 3), stride = (1, stride)),
            Conv((7, 1), channels => channels, pad = (3, 0), stride = (stride, 1)),
            BatchNorm(channels)
        ) |> gpu,
)


struct ParOp
    ops::AbstractArray
end

ParOp(channels::Int64, stride::Int64) = ParOp([OPS[prim](channels, stride, 1) |> gpu for prim in PRIMITIVES]) |> gpu

function mask(row::Int64, shape::AbstractArray)
    m = zeros(Float32, size(shape)...)
    m[row,:] .= 1
    m
end

function (m::ParOp)(x, αs)
    opouts = [op(x) for op in m.ops]
    #return
    #mapreduce((op, α) -> (α/sum(αs))*op(x), +, m.ops, αs)
    #mapped = map(op -> op(x), m.ops)
    #sum((mask(m.location, αs) .* αs) * mapped)
end

Flux.@functor ParOp



struct MixedOp
    location::Int64
    ops::AbstractArray
end

MixedOp(channels::Int64, stride::Int64, location::Int64) = MixedOp(location, [OPS[prim](channels, stride, 1) |> gpu for prim in PRIMITIVES]) |> gpu

function mask(row::Int64, shape::AbstractArray)
    m = zeros(Float32, size(shape)...)
    m[row,:] .= 1
    m
end

function (m::MixedOp)(x, αs)
    #αs = αs
    #αs = softmax(αs)
    mapreduce((op, α) -> (α/sum(αs))*op(x), +, m.ops, αs)
    #mapped = map(op -> op(x), m.ops)
    #sum((mask(m.location, αs) .* αs) * mapped)
end

Flux.@functor MixedOp

my_zero(xs::AbstractArray) = fill!(similar(xs), zero(eltype(xs)))

collecteachrow(x) = collect(eachrow(x))

@adjoint function collecteachrow(x)
    collecteachrow(x), dy -> begin
        dx = my_zero(x) # _zero is not in ZygoteRules, TODO
        foreach(copyto!, collecteachrow(dx), dy)
        (dx,)
    end
end

"""
@adjoint function mapreduce(op, func, xs; kwargs...)
    opbacks = Any[]
    backs = Any[]
    ys = Any[]
    function nop(x)
        y, back = pullback(op,x)
        push!(opbacks, back)
        y
    end
    function nfunc(x, x2)
        y, back = pullback(func, x, x2)
        push!(backs, back)
        push!(ys, y)
        return y
    end
    mapreduce(nop, nfunc, xs; kwargs...),
    function (adjy)
        offset = haskey(kwargs, :init) ? 0 : 1
        res = Vector{Any}(undef, length(ys)+offset)
        for i=length(ys):-1:1
            opback, back, y = opbacks[i+offset], backs[i], ys[i]
            adjy, adjthis = back(adjy)
            res[i+offset], = opback(adjthis)
        end
        if offset==1
            res[1], = opbacks[1](adjy)
        end
        return (nothing, nothing, res)
    end
end
"""

struct Cell
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1::Chain
    prelayer2::Chain
    mixedops::AbstractArray
end

function Cell(channels_before_last, channels_last, channels, reduce, reduce_previous, steps, multiplier)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last,channels,2) |> gpu
    else
        prelayer1 = ReLUConvBN(channels_before_last,channels,(1,1),1,0) |> gpu
    end
    prelayer2 = ReLUConvBN(channels_last,channels,(1,1),1,0) |> gpu
    mixedops = []
    for i = 0:steps-1
       for j = 0:2+i-1
            reduce && j < 2 ? stride = 2 : stride = 1
            mixedop = MixedOp(channels, stride, length(mixedops)+1) |> gpu
            push!(mixedops, mixedop)
        end
    end
    Cell(steps, reduce, multiplier, prelayer1, prelayer2, mixedops) |> gpu
end

function (m::Cell)(x1, x2, αs)
    state1 = m.prelayer1(x1)
    state2 = m.prelayer2(x2)

    states = Zygote.Buffer([state1], m.steps+2)

    states[1] = state1
    states[2] = state2
    # states[3] = m.mixedops[1](states[1],αs.α1) + m.mixedops[2](states[2],αs.α2)
    # states[4] = m.mixedops[3](states[1],αs.α3) + m.mixedops[4](states[2],αs.α4) + m.mixedops[5](states[3],αs.α5)
    # states[5] = m.mixedops[6](states[1],αs.α6) + m.mixedops[7](states[2],αs.α7) + m.mixedops[8](states[3],αs.α8) + m.mixedops[9](states[4],αs.α9)
    # states[6] = m.mixedops[10](states[1],αs.α10) + m.mixedops[11](states[2],αs.α11) + m.mixedops[12](states[3],αs.α12) + m.mixedops[13](states[4],αs.α13) + m.mixedops[14](states[5],αs.α14)
    #
    offset = 0
    #mo_α = collect(zip(m.mixedops, collect(eachrow(αs))))
    for step in 1:m.steps
        #state = mapreduce((mixedop, α, state) -> mixedop(state, α), +, m.mixedops[offset+1:offset+step+1], collecteachrow(αs)[offset+1:offset+step+1], states)
        #state = mapreduce(((mixedop, α), state) -> mixedop(state, α), +, mo_α[offset+1:offset+step+1], states)
        state = mapreduce((mixedop, α, previous_state) -> mixedop(previous_state, α), +, m.mixedops[offset+1:offset+step+1], αs[offset+1:offset+step+1], states)
        # to_sum = Zygote.Buffer([state1], step+1)
        # for i in 1:step+1
        #     mo = m.mixedops[offset+i]
        #     α = αs[offset+i]
        #     #α = getfield(αs,offset+i)
        #     state = states[i]
        #     to_sum[i] = mo(state,α)
        # end
        offset += step + 1
        #states[step+2] = sum(to_sum)
        states[step+2] = state
    end
    states_ = copy(states)
    cat(states_[m.steps+2-m.multiplier+1:m.steps+2]..., dims = 3)
end

Flux.@functor Cell

struct α14
    α1::AbstractArray
    α2::AbstractArray
    α3::AbstractArray
    α4::AbstractArray
    α5::AbstractArray
    α6::AbstractArray
    α7::AbstractArray
    α8::AbstractArray
    α9::AbstractArray
    α10::AbstractArray
    α11::AbstractArray
    α12::AbstractArray
    α13::AbstractArray
    α14::AbstractArray
end

α14() = α14([2e-3*(rand(length(PRIMITIVES)).-0.5) |> f32 |> gpu  for _ in 1:14]...)

Flux.@functor α14

struct DARTSModel
    normal_αs::AbstractArray
    reduce_αs::AbstractArray
    stem::Chain
    cells::AbstractArray
    global_pooling::AdaptiveMeanPool
    classifier::Dense
end

function DARTSModel(; α_init = (num_ops -> 2e-3*(rand(num_ops).-0.5) |> f32), num_classes = 10, num_cells = 8, channels = 16, steps = 4, mult = 4, stem_mult = 3)
    channels_current = channels*stem_mult
    stem = Chain(
        Conv((3,3), 3=>channels_current, pad=(1,1)),
        BatchNorm(channels_current)) |> gpu
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    reduce_previous = false
    cells = []
    for i = 1:num_cells
        if i == num_cells÷3+1 || i == 2*num_cells÷3+1
            channels_current = channels_current*2
            reduce = true
        else
            reduce = false
        end
        cell = Cell(channels_before_last, channels_last, channels_current, reduce, reduce_previous, steps, mult) |> gpu
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult*channels_current
    end

    global_pooling = AdaptiveMeanPool((1,1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    k = floor(Int, steps^2/2+3*steps/2)
    #α_normal = α14()
    #α_reduce = α14()
    num_ops = length(PRIMITIVES)
    α_normal = [α_init(num_ops) for _ in 1:k]
    α_reduce = [α_init(num_ops) for _ in 1:k]
    #α_normal = rand(Float32, length(PRIMITIVES), k)
    #α_reduce = rand(Float32, length(PRIMITIVES), k)
    DARTSModel(α_normal, α_reduce, stem, cells, global_pooling, classifier)
end


function MaskedDARTSModel(m::DARTSModel; normal_αs = [], reduce_αs = [])
    if length(normal_αs) == 0
        normal_αs = m.normal_αs
    end
    if length(reduce_αs) == 0
        reduce_αs = m.reduce_αs
    end
    DARTSModel(normal_αs, reduce_αs, m.stem, m.cells, m.global_pooling, m.classifier)
end

function (m::DARTSModel)(x)
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        #cell.reduction ? αs = softmax(m.reduce_αs, dims = 1) : αs = softmax(m.normal_αs, dims = 1)
        cell.reduction ? αs = m.reduce_αs : αs = m.normal_αs
        new_state = cell(s1, s2, αs)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    m.classifier(squeeze(out))
end

function (m::DARTSModel)(x; normal_αs = [], reduce_αs = [])
    if length(normal_αs) == 0
        normal_αs = m.normal_αs
    end
    if length(reduce_αs) == 0
        reduce_αs = m.reduce_αs
    end
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        #cell.reduction ? αs = softmax(m.reduce_αs, dims = 1) : αs = softmax(m.normal_αs, dims = 1)
        cell.reduction ? αs = reduce_αs : αs = normal_αs
        new_state = cell(s1, s2, αs)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    m.classifier(squeeze(out))
end

Flux.@functor DARTSModel


struct EvalCell
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1::Chain
    prelayer2::Chain
    ops::AbstractArray
end

function EvalCell(channels_before_last, channels_last, channels, reduce, reduce_previous, steps, multiplier, all_αs)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last,channels,2)
    else
        prelayer1 = ReLUConvBN(channels_before_last,channels,(1,1),1,0)
    end
    prelayer2 = ReLUConvBN(channels_last,channels,(1,1),1,0)
    ops = []
    for i = 0:steps-1
       for j = 0:2+i-1
            reduce && j < 2 ? stride = 2 : stride = 1
            op = OPS[PRIMITIVES[argmax(all_αs[length(ops)+1,:])]](channels, stride, 1)
            push!(ops, op)
        end
    end
    EvalCell(steps, reduce, multiplier, prelayer1, prelayer2, ops)
end

function (m::EvalCell)(x1, x2)
    state1 = m.prelayer1(x1)
    state2 = m.prelayer2(x2)

    states = Zygote.Buffer([state1], m.steps+2)

    states[1] = state1
    states[2] = state2
    offset = 0
    for step in 1:m.steps
        state = mapreduce((op, state) -> op(state), +, m.ops[offset+1:offset+step+1], states)
        offset += step + 1
        states[step+2] = state
    end
    states_ = copy(states)
    cat(states_[m.steps+2-m.multiplier+1:m.steps+2]..., dims = 3)
end

Flux.@functor EvalCell

struct DARTSEvalModel
    normal_αs::AbstractArray
    reduce_αs::AbstractArray
    stem::Chain
    cells::AbstractArray
    global_pooling::AdaptiveMeanPool
    classifier::Dense
end

function DARTSEvalModel(searchmodel::DARTSModel; num_cells = 8, channels = 16, num_classes = 10, steps = 4, mult = 4 , stem_mult = 3)
    α_normal = searchmodel.normal_αs
    α_reduce = searchmodel.reduce_αs
    channels_current = channels*stem_mult
    stem = Chain(
        Conv((3,3), 3=>channels_current, pad=(1,1)),
        BatchNorm(channels_current))
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    reduce_previous = false
    cells = []
    for i = 1:num_cells
        if i == num_cells÷3+1 || i == 2*num_cells÷3+1
            channels_current = channels_current*2
            reduce = true
            all_αs = α_reduce
        else
            reduce = false
            all_αs = α_normal
        end
        cell = EvalCell(channels_before_last, channels_last, channels_current, reduce, reduce_previous, steps, mult, all_αs)
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult*channels_current
    end

    global_pooling = AdaptiveMeanPool((1,1))
    classifier = Dense(channels_last, num_classes)
    DARTSEvalModel(α_normal, α_reduce, stem, cells, global_pooling, classifier)
end

function (m::DARTSEvalModel)(x)
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        new_state = cell(s1, s2)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    m.classifier(squeeze(out))
end

Flux.@functor DARTSEvalModel
