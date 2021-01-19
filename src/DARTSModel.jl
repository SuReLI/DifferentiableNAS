export PRIMITIVES,
DARTSModel,
Cell,
MixedOp,
DARTSEvalModel,
EvalCell,
Activations,
discretize,
DARTSEvalAuxModel,
DARTSModelBN

using Flux
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
using CUDA
using Zygote: @adjoint, dropgrad
using Distributions

#trainable(bn::BatchNorm) = () #TODO move to script?

ReLUConvBN(channels_in, channels_out, kernel_size, stride, pad) =
    Chain(
        x -> relu.(x),
        Conv(kernel_size, channels_in => channels_out, bias = false),
        BatchNorm(channels_out),
    ) |> gpu

function FactorizedReduce(channels_in, channels_out, stride)
    odd =
        Conv(
            (1, 1),
            channels_in => channels_out ÷ stride,
            stride = (stride, stride),
            bias = false,
        ) |> gpu
    even =
        Conv(
            (1, 1),
            channels_in => channels_out ÷ stride,
            stride = (stride, stride),
            bias = false,
        ) |> gpu

    Chain(
        x -> relu.(x),
        x -> cat(odd(x), even(x[2:end, 2:end, :, :]), dims = 3), #TODO get rid of indexing
        BatchNorm(channels_out),
    ) |> gpu
end

SepConv(channels_in, channels_out, kernel_size, stride, pad) =
    Chain(
        x -> relu.(x),
        DepthwiseConv(
            kernel_size,
            channels_in => channels_in,
            stride = (stride, stride),
            pad = (pad, pad),
            bias = false,
        ),
        Conv(
            (1, 1),
            channels_in => channels_in,
            stride = (1, 1),
            pad = (0, 0),
            bias = false,
        ),
        BatchNorm(channels_in),
        x -> relu.(x),
        DepthwiseConv(
            kernel_size,
            channels_in => channels_in,
            pad = (pad, pad),
            stride = (1, 1),
            bias = false,
        ),
        Conv(
            (1, 1),
            channels_in => channels_out,
            stride = (1, 1),
            pad = (0, 0),
            bias = false,
        ),
        BatchNorm(channels_out),
    ) |> gpu

DilConv(channels_in, channels_out, kernel_size, stride, pad, dilation) =
    Chain(
        x -> relu.(x),
        DepthwiseConv(
            kernel_size,
            channels_in => channels_in,
            pad = (pad, pad),
            stride = (stride, stride),
            dilation = dilation,
            bias = false,
        ),
        Conv(
            kernel_size,
            channels_in => channels_out,
            stride = (1, 1),
            pad = (pad, pad),
            bias = false,
        ),
        BatchNorm(channels_out),
    ) |> gpu

SepConv_v(channels_in, channels_out, kernel_size, stride, pad) =
    Chain(
        x -> relu.(x),
        Conv(
            kernel_size,
            channels_in => channels_in,
            stride = (stride, stride),
            pad = (pad, pad),
            bias = false,
        ),
        Conv(
            (1, 1),
            channels_in => channels_in,
            stride = (1, 1),
            pad = (0, 0),
            bias = false,
        ),
        BatchNorm(channels_in),
        x -> relu.(x),
        Conv(
            kernel_size,
            channels_in => channels_in,
            pad = (pad, pad),
            stride = (1, 1),
            bias = false,
        ),
        Conv(
            (1, 1),
            channels_in => channels_out,
            stride = (1, 1),
            pad = (0, 0),
            bias = false,
        ),
        BatchNorm(channels_out),
    ) |> gpu

DilConv_v(channels_in, channels_out, kernel_size, stride, pad, dilation) =
    Chain(
        x -> relu.(x),
        Conv(
            kernel_size,
            channels_in => channels_in,
            pad = (pad * dilation, pad * dilation),
            stride = (stride, stride),
            dilation = dilation,
            bias = false,
        ),
        Conv(
            kernel_size,
            channels_in => channels_out,
            stride = (1, 1),
            pad = (pad, pad),
            bias = false,
        ),
        BatchNorm(channels_out),
    ) |> gpu


Identity(stride, pad) = x -> x |> gpu
#Zero(stride, pad) = x -> x[1:stride:end, 1:stride:end, :, :] * 0 |> gpu
Zero(stride) = Chain(MeanPool((1, 1), stride = stride), x -> 0*x)

SkipConnect(channels_in, channels_out, stride, pad) =
    stride == 1 ? Identity(stride, pad) |> gpu :
    FactorizedReduce(channels_in, channels_out, stride) |> gpu

struct AdaptiveMeanPool{N}
    target_out::NTuple{N,Int}
end

AdaptiveMeanPool(target_out::NTuple{N,Integer}) where {N} =
    AdaptiveMeanPool(target_out)

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
    "none" => (channels, stride, w) -> Chain(Zero(stride)) |> gpu,
    "avg_pool_3x3" =>
        (channels, stride, w) ->
            Chain(
                MeanPool((3, 3), stride = stride, pad = 1),
                BatchNorm(channels),
            ) |> gpu,
    "max_pool_3x3" =>
        (channels, stride, w) ->
            Chain(
                MaxPool((3, 3), stride = stride, pad = 1),
                BatchNorm(channels),
            ) |> gpu,
    "skip_connect" =>
        (channels, stride, w) ->
            Chain(SkipConnect(channels, channels, stride, 1)) |> gpu,
    "sep_conv_3x3" =>
        (channels, stride, w) ->
            SepConv_v(channels, channels, (3, 3), stride, 1) |> gpu,
    "sep_conv_5x5" =>
        (channels, stride, w) ->
            SepConv_v(channels, channels, (5, 5), stride, 2) |> gpu,
    "sep_conv_7x7" =>
        (channels, stride, w) ->
            SepConv_v(channels, channels, (7, 7), stride, 3) |> gpu,
    "dil_conv_3x3" =>
        (channels, stride, w) ->
            DilConv_v(channels, channels, (3, 3), stride, 1, 2) |> gpu,
    "dil_conv_5x5" =>
        (channels, stride, w) ->
            DilConv_v(channels, channels, (5, 5), stride, 2, 2) |> gpu,
    "conv_7x1_1x7" =>
        (channels, stride, w) ->
            Chain(
                x -> relu.(x),
                Conv(
                    (1, 7),
                    channels => channels,
                    pad = (0, 3),
                    stride = (1, stride),
                    bias = false,
                ),
                Conv(
                    (7, 1),
                    channels => channels,
                    pad = (3, 0),
                    stride = (stride, 1),
                    bias = false,
                ),
                BatchNorm(channels),
            ) |> gpu,
)

function my_softmax(xs::AbstractArray; dims = 1)
    softmax!(similar(xs), xs, dims=dims)
end

Zygote.@adjoint function my_softmax(xs::AbstractArray; dims = 1)
    softmax(xs, dims = dims), Δ -> begin
        (∇softmax(Δ, xs, dims = dims),)
    end
end


struct OpActs
    name::String
    op::Any
end

function showlayer(
    x::AbstractArray,
    layer,
    opname::String,
    outs::Array{Array{Float32,3}},
)
    out = layer(x)
    Zygote.ignore() do
        if !(typeof(layer) <: Flux.BatchNorm) && !occursin("none", opname)
            push!(outs, dropdims(mean(out, dims = (1, 2, 3)), dims = 3) |> cpu)
        end
    end
    out
end

function (opwrap::OpActs)(xin::AbstractArray, acts::Dict, cellid::Int64)
    outs = Array{Array{Float32,3}}(undef, 0)
    xout = foldl(
        (x, layer) -> showlayer(x, layer, opwrap.name, outs),
        opwrap.op,
        init = xin,
    )
    Zygote.ignore() do
        if length(outs) > 0
            if haskey(acts, opwrap.name)
                acts[opwrap.name] = vcat(acts[opwrap.name], hcat(outs...))
            else
                acts[opwrap.name] = hcat(outs...)
            end
        end #acts dims: cellid x oplayer x image
    end
    xout
end

Flux.@functor OpActs

struct Op
    name::String
    op::Any
end

function (opwrap::Op)(
    xin::AbstractArray,
    acts::Union{Dict,Nothing},
    cellid::Int64 = 0,
)
    opwrap.op(xin)
end

Flux.@functor Op

struct MixedOp
    cellid::Int64
    name::String
    ops::AbstractArray
end

function MixedOp(
    cellid::Int64,
    name::String,
    channels::Int64,
    stride::Int64,
    track_acts::Bool = false,
)
    if track_acts
        op = OpActs
    else
        op = Op
    end
    MixedOp(
        cellid,
        name,
        [
            op(string(name, "-", prim), OPS[prim](channels, stride, 1) |> gpu)
            for prim in PRIMITIVES
        ],
    ) |> gpu
end

function (m::MixedOp)(
    x::AbstractArray,
    αs::AbstractArray,
    acts::Union{Nothing,Dict} = nothing,
)
    as = my_softmax(copy(αs))
    sum(as[i] * m.ops[i](x, acts, m.cellid) for i = 1:length(as))
end

Flux.@functor MixedOp

struct Cell
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1::Chain
    prelayer2::Chain
    mixedops::AbstractArray
end

function Cell(
    channels_before_last::Int64,
    channels_last::Int64,
    channels::Int64,
    reduce::Bool,
    reduce_previous::Bool,
    steps::Int64,
    multiplier::Int64,
    cellid::Int64,
    track_acts::Bool = false,
)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last, channels, 2) |> gpu
    else
        prelayer1 =
            ReLUConvBN(channels_before_last, channels, (1, 1), 1, 0) |> gpu
    end
    prelayer2 = ReLUConvBN(channels_last, channels, (1, 1), 1, 0) |> gpu
    mixedops = []
    for i = 3:steps+2 #op output
        for j = 1:i-1 #op input
            reduce && j < 3 ? stride = 2 : stride = 1
            mixedop =
                MixedOp(
                    cellid,
                    string(j, "-", i),
                    channels,
                    stride,
                    track_acts,
                ) |> gpu
            push!(mixedops, mixedop)
        end
    end
    Cell(steps, reduce, multiplier, prelayer1, prelayer2, mixedops) |> gpu
end

function (m::Cell)(
    x1::AbstractArray,
    x2::AbstractArray,
    αs::AbstractArray,
    acts::Union{Nothing,Dict} = nothing,
)
    state1 = m.prelayer1(x1)
    state2 = m.prelayer2(x2)

    states = Zygote.Buffer([state1], m.steps + 2)

    states[1] = state1
    states[2] = state2
    offset = 0
    for step = 1:m.steps
        state = mapreduce(
            (mixedop, α, previous_state) ->
                mixedop(previous_state, α, acts),
            +,
            m.mixedops[offset+1:offset+step+1],
            αs[offset+1:offset+step+1],
            states,
        )
        offset += step + 1
        states[step+2] = state
    end
    states_ = copy(states)
    cat(states_[m.steps+2-m.multiplier+1:m.steps+2]..., dims = 3)
end

Flux.@functor Cell

mutable struct Activations
    currentacts::Dict{String,Array{Float32,3}}
end

struct DARTSModel
    normal_αs::AbstractArray
    reduce_αs::AbstractArray
    stem::Chain
    cells::AbstractArray
    global_pooling::AdaptiveMeanPool
    classifier::Dense
    activations::Activations
end

function DARTSModel(;
    α_init = (num_ops -> 2e-3 * (rand(num_ops) .- 0.5) |> f32),
    num_classes::Int64 = 10,
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
    track_acts::Bool = false,
)
    channels_current = channels * stem_mult
    stem =
        Chain(
            Conv((3, 3), 3 => channels_current, pad = (1, 1), bias = false),
            BatchNorm(channels_current),
        ) |> gpu
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    reduce_previous = false
    cells = []
    for i = 1:num_cells
        if i == num_cells ÷ 3 + 1 || i == 2 * num_cells ÷ 3 + 1
            channels_current = channels_current * 2
            reduce = true
        else
            reduce = false
        end
        cell =
            Cell(
                channels_before_last,
                channels_last,
                channels_current,
                reduce,
                reduce_previous,
                steps,
                mult,
                i,
                track_acts,
            ) |> gpu
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
    end
    global_pooling = AdaptiveMeanPool((1, 1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    k = floor(Int, steps^2 / 2 + 3 * steps / 2)
    num_ops = length(PRIMITIVES)
    α_normal = [α_init(num_ops) for _ = 1:k]
    α_reduce = [α_init(num_ops) for _ = 1:k]
    activations = Activations(Dict{String,Array{Float32,3}}())
    DARTSModel(
        α_normal,
        α_reduce,
        stem,
        cells,
        global_pooling,
        classifier,
        activations,
    )
end

function (m::DARTSModel)(x; αs::AbstractArray = [])
    if length(αs) > 0
        normal_αs = αs[1]
        reduce_αs = αs[2]
    else
        normal_αs = m.normal_αs
        reduce_αs = m.reduce_αs
    end
    s1 = m.stem(x)
    s2 = m.stem(x)
    m.activations.currentacts = Dict{String,Array{Float32,3}}()
    for (i, cell) in enumerate(m.cells)
        cell.reduction ? αs = reduce_αs : αs = normal_αs
        new_state = cell(s1, s2, αs, m.activations.currentacts)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    m.classifier(squeeze(out))
end

Flux.@functor DARTSModel



struct MixedOpBN
    cellid::Int64
    name::String
    ops::AbstractArray
    batchnorm::BatchNorm
end

function MixedOpBN(
    cellid::Int64,
    name::String,
    channels::Int64,
    stride::Int64,
    track_acts::Bool = false,
)
    if track_acts
        op = OpActs
    else
        op = Op
    end
    MixedOpBN(
        cellid,
        name,
        [
            op(string(name, "-", prim), OPS[prim](channels, stride, 1) |> gpu)
            for prim in PRIMITIVES
        ],
        BatchNorm(channels)
    ) |> gpu
end

function (m::MixedOpBN)(
    x::AbstractArray,
    αs::AbstractArray,
    acts::Union{Nothing,Dict} = nothing,
)
    m.batchnorm(sum(αs[i] * m.ops[i](x, acts, m.cellid) for i = 1:length(αs) if αs[i] > 0))
end

Flux.@functor MixedOpBN

struct CellBN
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1::Chain
    prelayer2::Chain
    mixedops::AbstractArray
end

function CellBN(
    channels_before_last::Int64,
    channels_last::Int64,
    channels::Int64,
    reduce::Bool,
    reduce_previous::Bool,
    steps::Int64,
    multiplier::Int64,
    cellid::Int64,
    track_acts::Bool = false,
)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last, channels, 2) |> gpu
    else
        prelayer1 =
            ReLUConvBN(channels_before_last, channels, (1, 1), 1, 0) |> gpu
    end
    prelayer2 = ReLUConvBN(channels_last, channels, (1, 1), 1, 0) |> gpu
    mixedops = []
    for i = 3:steps+2 #op output
        for j = 1:i-1 #op input
            reduce && j < 3 ? stride = 2 : stride = 1
            mixedop =
                MixedOpBN(
                    cellid,
                    string(j, "-", i),
                    channels,
                    stride,
                    track_acts,
                ) |> gpu
            push!(mixedops, mixedop)
        end
    end
    CellBN(steps, reduce, multiplier, prelayer1, prelayer2, mixedops) |> gpu
end

function (m::CellBN)(
    x1::AbstractArray,
    x2::AbstractArray,
    αs::AbstractArray,
    acts::Union{Nothing,Dict} = nothing,
)
    state1 = m.prelayer1(x1)
    state2 = m.prelayer2(x2)

    states = Zygote.Buffer([state1], m.steps + 2)

    states[1] = state1
    states[2] = state2
    offset = 0
    for step = 1:m.steps
        state = mapreduce(
            (mixedop, α, previous_state) ->
                mixedop(previous_state, α, acts),
            +,
            m.mixedops[offset+1:offset+step+1],
            αs[offset+1:offset+step+1],
            states,
        )
        offset += step + 1
        states[step+2] = state
    end
    states_ = copy(states)
    cat(states_[m.steps+2-m.multiplier+1:m.steps+2]..., dims = 3)
end

Flux.@functor CellBN


struct DARTSModelBN
    normal_αs::AbstractArray
    reduce_αs::AbstractArray
    stem::Chain
    cells::AbstractArray
    global_pooling::AdaptiveMeanPool
    classifier::Dense
    activations::Activations
end

function DARTSModelBN(;
    α_init = (num_ops -> ones(num_ops) ./ num_ops + 2e-3 * (rand(num_ops) .- 0.5) |> f32),
    num_classes::Int64 = 10,
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
    track_acts::Bool = false,
)
    channels_current = channels * stem_mult
    stem =
        Chain(
            Conv((3, 3), 3 => channels_current, pad = (1, 1), bias = false),
            BatchNorm(channels_current),
        ) |> gpu
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    reduce_previous = false
    cells = []
    for i = 1:num_cells
        if i == num_cells ÷ 3 + 1 || i == 2 * num_cells ÷ 3 + 1
            channels_current = channels_current * 2
            reduce = true
        else
            reduce = false
        end
        cell =
            CellBN(
                channels_before_last,
                channels_last,
                channels_current,
                reduce,
                reduce_previous,
                steps,
                mult,
                i,
                track_acts,
            ) |> gpu
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
    end
    global_pooling = AdaptiveMeanPool((1, 1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    k = floor(Int, steps^2 / 2 + 3 * steps / 2)
    num_ops = length(PRIMITIVES)
    α_normal = [α_init(num_ops) for _ = 1:k]
    α_reduce = [α_init(num_ops) for _ = 1:k]
    activations = Activations(Dict{String,Array{Float32,3}}())
    DARTSModelBN(
        α_normal,
        α_reduce,
        stem,
        cells,
        global_pooling,
        classifier,
        activations,
    )
end

function (m::DARTSModelBN)(x; αs::AbstractArray = [])
    if length(αs) > 0
        normal_αs = αs[1]
        reduce_αs = αs[2]
    else
        normal_αs = m.normal_αs
        reduce_αs = m.reduce_αs
    end
    s1 = m.stem(x)
    s2 = m.stem(x)
    m.activations.currentacts = Dict{String,Array{Float32,3}}()
    for (i, cell) in enumerate(m.cells)
        cell.reduction ? αs = reduce_αs : αs = normal_αs
        new_state = cell(s1, s2, αs, m.activations.currentacts)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    m.classifier(squeeze(out))
end

Flux.@functor DARTSModelBN





struct EvalCell
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1::Chain
    prelayer2::Chain
    ops::AbstractArray
    inputindices::AbstractArray
end

function maxk(a::AbstractArray, k::Int64)
    b = partialsortperm(a, 1:k, rev = true)
    return collect(b)
end

function discretize(
    αs::AbstractArray,
    channels::Int64,
    reduce::Bool,
    steps::Int64,
    disclude_1::Bool = true,
)
    prim = [
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

    if length(αs[1]) == 7
        prim = prim[2:8]
        disclude_1 = false
    end
    ops = []
    opnames = []
    inputindices = []
    rows = 0
    for i = 3:steps+2 #TODO test this loop, make sure it's not none
        if disclude_1
            for j = 1:i-1
                αs[rows+j][1] = -Inf32
            end
        end
        options = [findmax(αs[rows+j]) for j = 1:i-1]
        top2 = partialsortperm(options, 1:2, by = x -> x[1], rev = true)
        top2names = Tuple(prim[options[i][2]] for i in top2)
        top2ops = Tuple(
            OPS[prim[options[i][2]]](channels, reduce && i < 3 ? 2 : 1, 1)
            for i in top2
        )
        push!(inputindices, top2)
        push!(opnames, top2names)
        push!(ops, top2ops)
        rows += i - 1
    end
    (inputindices, ops, opnames)
end

function EvalCell(
    channels_before_last::Int64,
    channels_last::Int64,
    channels::Int64,
    reduce::Bool,
    reduce_previous::Bool,
    steps::Int64,
    multiplier::Int64,
    αs::AbstractArray,
)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last, channels, 2)
    else
        prelayer1 = ReLUConvBN(channels_before_last, channels, (1, 1), 1, 0)
    end
    prelayer2 = ReLUConvBN(channels_last, channels, (1, 1), 1, 0)
    inputindices, ops, _ = discretize(αs, channels, reduce, steps)
    EvalCell(steps, reduce, multiplier, prelayer1, prelayer2, ops, inputindices)
end

function droppath(x, drop_prob)
    if drop_prob > 0.0
        mask = rand(Bernoulli(1 - drop_prob), 1, 1, size(x, 3), 1) |> gpu
        x = x .* mask / (typeof(x[1])(1 - drop_prob))
    end
    x
end

function (m::EvalCell)(x1::AbstractArray, x2::AbstractArray, drop_prob::Float32)
    state1 = m.prelayer1(x1)
    state2 = m.prelayer2(x2)

    states = Zygote.Buffer([state1], m.steps + 2)

    states[1] = state1
    states[2] = state2
    for step = 1:m.steps
        in1 = m.ops[step][1](states[m.inputindices[step][1]])
        if in1 != states[m.inputindices[step][1]]
            in1 = droppath(in1, drop_prob)
        end
        in2 = m.ops[step][2](states[m.inputindices[step][2]])
        if in2 != states[m.inputindices[step][2]]
            in2 = droppath(in2, drop_prob)
        end
        states[step+2] = in1 + in2
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

function DARTSEvalModel(
    α_normal::AbstractArray,
    α_reduce::AbstractArray;
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    num_classes::Int64 = 10,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
)
    channels_current = channels * stem_mult
    stem = Chain(
        Conv((3, 3), 3 => channels_current, pad = (1, 1), bias = false),
        BatchNorm(channels_current),
    )
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    reduce_previous = false
    cells = []
    for i = 1:num_cells
        if i == num_cells ÷ 3 + 1 || i == 2 * num_cells ÷ 3 + 1
            channels_current = channels_current * 2
            reduce = true
            αs = α_reduce
        else
            reduce = false
            αs = α_normal
        end
        cell = EvalCell(
            channels_before_last,
            channels_last,
            channels_current,
            reduce,
            reduce_previous,
            steps,
            mult,
            αs,
        )
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
    end

    global_pooling = AdaptiveMeanPool((1, 1))
    classifier = Dense(channels_last, num_classes)
    DARTSEvalModel(α_normal, α_reduce, stem, cells, global_pooling, classifier)
end

function (m::DARTSEvalModel)(
    x::AbstractArray,
    drop_prob::Float32 = Float32(0.0),
)
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        new_state = cell(s1, s2, drop_prob)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    out = m.classifier(squeeze(out))
    out
end

Flux.@functor DARTSEvalModel


struct DARTSEvalAuxModel
    normal_αs::AbstractArray
    reduce_αs::AbstractArray
    stem::Chain
    cells::AbstractArray
    auxiliary::Chain
    global_pooling::AdaptiveMeanPool
    classifier::Dense
end

function DARTSEvalAuxModel(
    α_normal::AbstractArray,
    α_reduce::AbstractArray;
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    num_classes::Int64 = 10,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
)
    channels_current = channels * stem_mult
    stem = Chain(
        Conv((3, 3), 3 => channels_current, pad = (1, 1), bias = false),
        BatchNorm(channels_current),
    )
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    channels_aux = channels
    reduce_previous = false
    cells = []
    for i = 1:num_cells
        if i == num_cells ÷ 3 + 1 || i == 2 * num_cells ÷ 3 + 1
            channels_current = channels_current * 2
            reduce = true
            αs = α_reduce
        else
            reduce = false
            αs = α_normal
        end
        cell = EvalCell(
            channels_before_last,
            channels_last,
            channels_current,
            reduce,
            reduce_previous,
            steps,
            mult,
            αs,
        )
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
        if i == 2 * num_cells ÷ 3 + 1
            channels_aux = channels_last
        end
    end
    auxiliary = Chain(
        x -> relu.(x), #inplace?
        MeanPool((5, 5), pad = 0, stride = 3),
        Conv((1, 1), channels_aux => 128, bias = false),
        BatchNorm(128),
        x -> relu.(x),
        Conv((2, 2), 128 => 768, bias = false),
        BatchNorm(768),
        x -> relu.(x),
        x -> dropdims(reshape(x, size(x, 3), :), dims = 2),
        Dense(768, num_classes),
    )
    global_pooling = AdaptiveMeanPool((1, 1))
    classifier = Dense(channels_last, num_classes)
    DARTSEvalAuxModel(
        α_normal,
        α_reduce,
        stem,
        cells,
        auxiliary,
        global_pooling,
        classifier,
    )
end

function (m::DARTSEvalAuxModel)(
    x::AbstractArray,
    is_training::Bool = false,
    drop_prob::Float32 = Float32(0.0),
)
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        new_state = cell(s1, s2, drop_prob)
        s1 = s2
        s2 = new_state
        if i == 2 * length(m.cells) ÷ 3 + 1 && is_training
            out_aux = m.auxiliary(s2)
        end
    end
    out = m.global_pooling(s2)
    out = m.classifier(squeeze(out))
    if !is_training
        out_aux = out
    end
    out, out_aux
end

Flux.@functor DARTSEvalAuxModel
