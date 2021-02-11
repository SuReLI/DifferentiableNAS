export PRIMITIVES,
DARTSModel,
Cell,
MixedOp,
DARTSEvalModel,
EvalCell,
Activations,
discretize,
DARTSEvalAuxModel,
DARTSModelBN,
parse_genotype

using Flux
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
using CUDA
using Zygote: @adjoint, dropgrad
using Distributions

ReLUConvBN(channels_in, channels_out, kernel_size, stride, pad) =
    Chain(
        x -> relu.(x),
        Conv(kernel_size, channels_in => channels_out, bias = false),
        BatchNorm(channels_out),
    )

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
    )
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
    )

DilConv(channels_in, channels_out, kernel_size, stride, pad, dilation) =
    Chain(
        x -> relu.(x),
        DepthwiseConv(
            kernel_size,
            channels_in => channels_in,
            pad = (pad * dilation, pad * dilation),
            stride = (stride, stride),
            dilation = dilation,
            bias = false,
        ),
        Conv(
            (1,1),
            channels_in => channels_out,
            stride = (1, 1),
            pad = (0, 0),
            bias = false,
        ),
        BatchNorm(channels_out),
    )

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
    )

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
            (1,1),
            channels_in => channels_out,
            stride = (1, 1),
            pad = (0, 0),
            bias = false,
        ),
        BatchNorm(channels_out),
    )


Identity() = x -> x
Zero(stride) =
    stride == 2 ? Chain(MeanPool((1, 1), stride = stride), x -> 0*x) :
    x -> 0*x

SkipConnect(channels_in, channels_out, stride, pad) =
    stride == 1 ? Identity() :
    FactorizedReduce(channels_in, channels_out, stride)

struct AdaptiveMeanPool{N}
    target_out::NTuple{N,Int}
end

AdaptiveMeanPool(target_out::NTuple{N,Integer}) where {N} =
    AdaptiveMeanPool(target_out)

function (m::AdaptiveMeanPool)(x)
    w = size(x, 1) - m.target_out[1] + 1
    h = size(x, 2) - m.target_out[2] + 1
    return meanpool(x, (w, h); pad = (0, 0), stride = (1, 1))
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

OPS = Dict(
    "none" => (channels, stride, w) -> Chain(Zero(stride)),
    "avg_pool_3x3" =>
        (channels, stride, w) ->
            Chain(
                MeanPool((3, 3), stride = stride, pad = 1),
                BatchNorm(channels),
            ),
    "max_pool_3x3" =>
        (channels, stride, w) ->
            Chain(
                MaxPool((3, 3), stride = stride, pad = 1),
                BatchNorm(channels),
            ),
    "skip_connect" =>
        (channels, stride, w) ->
            Chain(SkipConnect(channels, channels, stride, 1)),
    "sep_conv_3x3" =>
        (channels, stride, w) ->
            SepConv(channels, channels, (3, 3), stride, 1),
    "sep_conv_5x5" =>
        (channels, stride, w) ->
            SepConv(channels, channels, (5, 5), stride, 2),
    "sep_conv_7x7" =>
        (channels, stride, w) ->
            SepConv(channels, channels, (7, 7), stride, 3),
    "dil_conv_3x3" =>
        (channels, stride, w) ->
            DilConv(channels, channels, (3, 3), stride, 1, 2),
    "dil_conv_5x5" =>
        (channels, stride, w) ->
            DilConv(channels, channels, (5, 5), stride, 2, 2),
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
            ),
)

function my_softmax(xs::AbstractArray; dims = 1)
    softmax!(similar(xs), xs, dims=dims)
end

Zygote.@adjoint function my_softmax(xs::AbstractArray; dims = 1)
    softmax(xs, dims = dims), Δ -> begin
        (∇softmax(Δ, xs, dims = dims),)
    end
end

abstract type AbstractOp end

struct OpActs <: AbstractOp
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

struct Op <: AbstractOp
    name::String
    op::Chain
end

function (opwrap::Op)(
    xin::AbstractArray,
    acts::Union{Dict,Nothing},
    cellid::Int64 = 0,
)
    opwrap.op(xin)
end

Flux.@functor Op

struct MixedOp{T<:AbstractOp}
    cellid::Int64
    name::String
    ops::Vector{T}
end

function MixedOp(
    cellid::Int64,
    name::String,
    channels::Int64,
    stride::Int64,
    operations::Vector{String} = PRIMITIVES,
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
            op(string(name, "-", operation), OPS[operation](channels, stride, 1) |> gpu)
            for operation in operations
        ],
    )
end

function (m::MixedOp)(
    x::CuArray{Float32, 4},
    #out::CuArray{Float32, 4},
    as::Vector{Float32} = [1f0],
    acts::Union{Nothing,Dict} = nothing,
)
    as = softmax(as)
    sum(as[i] * m.ops[i](x, acts, m.cellid) for i = 1:length(as))
end

"""
function (m::MixedOp)(
    x::CuArray{Float32, 4},
    precompute::AbstractArray,
    αs::Vector{Float32} = [1f0],
)
    as = my_softmax(copy(αs))
    if !precompute[1]
        precompute[2] = [m.ops[i](x, nothing, m.cellid) for i = 1:length(as)]
        precompute[1] = true
    end
    sum(as[i] * precompute[2][i] for i = 1:length(as))
end
"""

Flux.@functor MixedOp

struct Cell
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1::Chain
    prelayer2::Chain
    mixedops::Vector{MixedOp}
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
    operations::Vector{String} = PRIMITIVES,
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
                    operations,
                    track_acts,
                )
            push!(mixedops, mixedop)
        end
    end
    Cell(steps, reduce, multiplier, prelayer1, prelayer2, mixedops)
end

function (m::Cell)(
    x1::CuArray{Float32, 4},
    x2::CuArray{Float32, 4},
    αs::Vector{Vector{Float32}},
    acts::Union{Nothing,Dict} = nothing,
)
    state1 = m.prelayer1(x1)

    states = Zygote.Buffer([state1], m.steps + 2)

    states[1] = state1
    states[2] = m.prelayer2(x2)
    offset = 0
    #precompute = Array{Array{Float32,4}}(undef, m.steps + 1)
    #precompute = Zygote.Buffer([], m.steps + 1)
    #for i = 1:m.steps+1
        #precompute[i] = [false, []]
    #end
    #@display(typeof(precompute))
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
    reduce(ccat, copy(states[m.steps+2-m.multiplier+1:m.steps+2]))
end

Flux.@functor Cell

mutable struct Activations
    currentacts::Dict{String,Array{Float32,3}}
end

struct DARTSModel
    normal_αs::Vector{Vector{Float32}}
    reduce_αs::Vector{Vector{Float32}}
    stem::Chain
    cells::Array{Cell,1}
    global_pooling::AdaptiveMeanPool
    classifier::Dense
    operations::Vector{String}
    activations::Activations
end

function DARTSModel(;
    α_init = (num_ops -> 2f-3 * (rand(Float32, num_ops) .- 0.5f0)),
    num_classes::Int64 = 10,
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
    operations::Vector{String} = PRIMITIVES,
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
                operations,
                track_acts
            )
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
    end
    global_pooling = AdaptiveMeanPool((1, 1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    k = floor(Int, steps^2 / 2 + 3 * steps / 2)
    num_ops = length(operations)
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
        operations,
        activations
    )
end

function (m::DARTSModel)(x::CuArray{Float32, 4})
    normal_αs = m.normal_αs
    reduce_αs = m.reduce_αs
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

function (m::DARTSModel)(x::CuArray{Float32, 4}, αs::Vector{Vector{Vector{Float32}}})
    normal_αs = αs[1]
    reduce_αs = αs[2]
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



struct MixedOpBN{T<:AbstractOp}
    cellid::Int64
    name::String
    ops::Vector{T}
    batchnorm::BatchNorm
end

function MixedOpBN(
    cellid::Int64,
    name::String,
    channels::Int64,
    stride::Int64,
    operations::Vector{String} = PRIMITIVES[2:end],
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
            for prim in operations
        ],
        BatchNorm(channels) |> gpu
    )
end

function (m::MixedOpBN)(
    x::CuArray{Float32, 4},
    αs::Vector{Float32} = [1f0],
    acts::Union{Nothing,Dict} = nothing,
)
    if any(αs .> 0)
        m.batchnorm(sum(tanh(αs[i]) * m.ops[i](x, acts, m.cellid) for i = 1:length(αs) if αs[i] > 0))
    else
        0.0 * m.ops[1](x, acts, m.cellid)
    end
end

Flux.@functor MixedOpBN

struct CellBN
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1::Chain
    prelayer2::Chain
    mixedops::Vector{MixedOpBN}
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
    operations::Vector{String} = PRIMITIVES[2:end],
    track_acts::Bool = false,
)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last, channels, 2) |> gpu
    else
        prelayer1 =
            ReLUConvBN(channels_before_last, channels, (1, 1), 1, 0) |> gpu
    end
    prelayer2 = ReLUConvBN(channels_last, channels, (1, 1), 1, 0) |> gpu
    mixedops = Vector{MixedOpBN}()
    for i = 3:steps+2 #op output
        for j = 1:i-1 #op input
            reduce && j < 3 ? stride = 2 : stride = 1
            mixedop =
                MixedOpBN(
                    cellid,
                    string(j, "-", i),
                    channels,
                    stride,
                    operations,
                    track_acts,
                )
            push!(mixedops, mixedop)
        end
    end
    CellBN(steps, reduce, multiplier, prelayer1, prelayer2, mixedops)
end

function (m::CellBN)(
    x1::CuArray{Float32, 4},
    x2::CuArray{Float32, 4},
    αs::Vector{Vector{Float32}},
    acts::Union{Nothing,Dict} = nothing,
)
    state1 = m.prelayer1(x1)

    states = Zygote.Buffer([state1], m.steps + 2)

    states[1] = state1
    states[2] = m.prelayer2(x2)
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
    reduce(ccat, copy(states[m.steps+2-m.multiplier+1:m.steps+2]))
end

Flux.@functor CellBN


struct DARTSModelBN
    normal_αs::Vector{Vector{Float32}}
    reduce_αs::Vector{Vector{Float32}}
    stem::Chain
    cells::Vector{CellBN}
    global_pooling::AdaptiveMeanPool
    classifier::Dense
    operations::Vector{String}
    activations::Activations
end

function DARTSModelBN(;
    α_init = (num_ops -> atanh.(ones(Float32, num_ops) ./ num_ops + 2f-3 * (rand(Float32, num_ops) .- 0.5f0))),
    num_classes::Int64 = 10,
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
    operations::Vector{String} = PRIMITIVES[2:end],
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
    cells = Vector{CellBN}(undef, num_cells)
    for i = 1:num_cells
        if i == num_cells ÷ 3 + 1 || i == 2 * num_cells ÷ 3 + 1
            channels_current = channels_current * 2
            reduce = true
        else
            reduce = false
        end
        cells[i] =
            CellBN(
                channels_before_last,
                channels_last,
                channels_current,
                reduce,
                reduce_previous,
                steps,
                mult,
                i,
                operations,
                track_acts,
            )

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
    end
    global_pooling = AdaptiveMeanPool((1, 1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    k = floor(Int, steps^2 / 2 + 3 * steps / 2)
    num_ops = length(operations)
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
        operations,
        activations,
    )
end

function (m::DARTSModelBN)(x)
    s1 = m.stem(x)
    s2 = m.stem(x)
    m.activations.currentacts = Dict{String,Array{Float32,3}}()
    for (i, cell) in enumerate(m.cells)
        cell.reduction ? αs = m.reduce_αs : αs = m.normal_αs
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
    ops::Vector{Vector{Chain}}
    inputindices::Vector{Vector{Int}}
    names::Vector{Tuple{String,String}}
end


function maxk(a::Vector{Vector{Float32}}, k::Int64)
    b = partialsortperm(a, 1:k, rev = true)
    return collect(b)
end

function discretize(
    αs::Vector{Vector{Float32}},
    channels::Int64,
    reduce::Bool,
    steps::Int64,
    operations::Vector{String} = PRIMITIVES,
    plotting::Bool = false,
    disclude_1::Bool = true,
)
    if operations == PRIMITIVES && size(αs[1],1) == 7
        operations = operations[2:end]
    end
    if "none" in operations
        disclude_1 = true
    else
        disclude_1 = false
    end
    ops = Vector{Vector{Chain}}(undef,steps)
    opnames = Vector{Tuple{String,String}}(undef,steps)
    inputindices = Vector{Vector{Int}}(undef,steps)
    rows = 0
    for i = 3:steps+2
        if disclude_1
            for j = 1:i-1
                αs[rows+j][1] = -Inf32
            end
        end
        options = [findmax(αs[rows+j]) for j = 1:i-1]
        top2 = partialsortperm(options, 1:2, by = x -> x[1], rev = true)
        top2names = Tuple(operations[options[i][2]] for i in top2)
        if !plotting
            top2ops = [
                OPS[operations[options[i][2]]](channels, reduce && i < 3 ? 2 : 1, 1) |> gpu
                for i in top2
            ]
            ops[i-2] = top2ops
        end
        inputindices[i-2] = top2
        opnames[i-2] = top2names
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
    αs::Vector{Vector{Float32}},
    operations::Vector{String} = PRIMITIVES,
)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last, channels, 2) |> gpu
    else
        prelayer1 = ReLUConvBN(channels_before_last, channels, (1, 1), 1, 0) |> gpu
    end
    prelayer2 = ReLUConvBN(channels_last, channels, (1, 1), 1, 0) |> gpu
    inputindices, ops, names = discretize(αs, channels, reduce, steps, operations)
    EvalCell(steps, reduce, multiplier, prelayer1, prelayer2, ops, inputindices, names)
end


DARTS_V2 = (
    normal = [
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("skip_connect", 0),
        ("dil_conv_3x3", 2),
    ],
    reduce = [
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
    ],
)

function parse_genotype(;
    genotype::NamedTuple=DARTS_V2,
    channels::Int64=1,
    reduce::Bool=false,
    steps::Int64=4,
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

    ops = Vector{Vector{Chain}}(undef,steps)
    opnames = Vector{Tuple{String,String}}(undef,steps)
    inputindices = Vector{Vector{Int}}(undef,steps)
    if reduce
        genocell = genotype.reduce
    else
        genocell = genotype.normal
    end
    for i = 1:steps
        top2 = [genocell[2*i-1][2]+1, genocell[2*i][2]+1]
        top2names = (genocell[2*i-1][1], genocell[2*i][1])
        top2ops = [
            OPS[top2names[j]](channels, reduce && top2[j] < 3 ? 2 : 1, 1) |> gpu
            for j in 1:2
        ]
        inputindices[i] = top2
        opnames[i] = top2names
        ops[i] = top2ops
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
)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last, channels, 2) |> gpu
    else
        prelayer1 = ReLUConvBN(channels_before_last, channels, (1, 1), 1, 0)  |> gpu
    end
    prelayer2 = ReLUConvBN(channels_last, channels, (1, 1), 1, 0) |> gpu
    inputindices, ops, names = parse_genotype(genotype=DARTS_V2, channels=channels, reduce=reduce, steps=steps)
    display(collect(zip(inputindices,names)))
    EvalCell(steps, reduce, multiplier, prelayer1, prelayer2, ops, inputindices, names)
end

function droppath(x::CuArray{Float32, 4}, drop_prob::Float32)
    if drop_prob > 0.0
        mask = rand(Bernoulli(1 - drop_prob), 1, 1, size(x, 3), 1) |> gpu
        x = x .* mask / (typeof(x[1])(1 - drop_prob))
    end
    x
end

function (m::EvalCell)(x1::CuArray{Float32, 4}, x2::CuArray{Float32, 4}, drop_prob::Float32)
    state1 = m.prelayer1(x1)

    states = Zygote.Buffer([state1], m.steps + 2)

    states[1] = state1
    states[2] = m.prelayer2(x2)
    for step = 1:m.steps
        state = states[m.inputindices[step][1]]
        in1 = m.ops[step][1](states[m.inputindices[step][1]])
        if m.names[step][1] != "skip_connect"
            in1 = droppath(in1, drop_prob)
        end
        in2 = m.ops[step][2](states[m.inputindices[step][2]])
        if m.names[step][2] != "skip_connect"
            in2 = droppath(in2, drop_prob)
        end
        states[step+2] = in1 + in2
    end
    reduce(ccat, copy(states[m.steps+2-m.multiplier+1:m.steps+2]))
end

Flux.@functor EvalCell

struct DARTSEvalModel
    normal_αs::Vector{Vector{Float32}}
    reduce_αs::Vector{Vector{Float32}}
    stem::Chain
    cells::Vector{EvalCell}
    global_pooling::AdaptiveMeanPool
    classifier::Dense
    operations::Vector{String}
end

function DARTSEvalModel(
    α_normal::Vector{Vector{Float32}},
    α_reduce::Vector{Vector{Float32}};
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    num_classes::Int64 = 10,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
    operations::Vector{String} = PRIMITIVES,
)
    channels_current = channels * stem_mult
    stem = Chain(
        Conv((3, 3), 3 => channels_current, pad = (1, 1), bias = false),
        BatchNorm(channels_current),
    ) |> gpu
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    reduce_previous = false
    cells = Vector{EvalCell}(undef,num_cells)
    for i = 1:num_cells
        if i == num_cells ÷ 3 + 1 || i == 2 * num_cells ÷ 3 + 1
            channels_current = channels_current * 2
            reduce = true
            αs = α_reduce
        else
            reduce = false
            αs = α_normal
        end
        cells[i] = EvalCell(
            channels_before_last,
            channels_last,
            channels_current,
            reduce,
            reduce_previous,
            steps,
            mult,
            αs,
            operations,
        )

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
    end

    global_pooling = AdaptiveMeanPool((1, 1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    DARTSEvalModel(α_normal, α_reduce, stem, cells, global_pooling, classifier, operations)
end

function (m::DARTSEvalModel)(
    x::CuArray{Float32, 4},
    drop_prob::Float32 = 0f0,
)
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        new_state = cell(s1, s2, drop_prob)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    m.classifier(squeeze(out))
end

Flux.@functor DARTSEvalModel


struct DARTSEvalAuxModel
    normal_αs::Vector{Vector{Float32}}
    reduce_αs::Vector{Vector{Float32}}
    stem::Chain
    cells::Vector{EvalCell}
    auxiliary::Chain
    global_pooling::AdaptiveMeanPool
    classifier::Dense
    operations::Vector{String}
end

function DARTSEvalAuxModel(
    α_normal::Vector{Vector{Float32}},
    α_reduce::Vector{Vector{Float32}};
    num_cells::Int64 = 8,
    channels::Int64 = 16,
    num_classes::Int64 = 10,
    steps::Int64 = 4,
    mult::Int64 = 4,
    stem_mult::Int64 = 3,
    operations::Vector{String} = PRIMITIVES,
)
    channels_current = channels * stem_mult
    stem = Chain(
        Conv((3, 3), 3 => channels_current, pad = (1, 1), bias = false),
        BatchNorm(channels_current),
    ) |> gpu
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    channels_aux = channels
    reduce_previous = false
    cells = Vector{EvalCell}(undef,num_cells)
    for i = 1:num_cells
        if i == num_cells ÷ 3 + 1 || i == 2 * num_cells ÷ 3 + 1
            channels_current = channels_current * 2
            reduce = true
            αs = α_reduce
        else
            reduce = false
            αs = α_normal
        end
        cells[i] = EvalCell(
            channels_before_last,
            channels_last,
            channels_current,
            reduce,
            reduce_previous,
            steps,
            mult,
            αs,
            operations,
        )

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult * channels_current
        if i == 2 * num_cells ÷ 3 + 1
            channels_aux = channels_last
        end
    end
    auxiliary = Chain(
        x -> relu.(x),
        MeanPool((5, 5), pad = 0, stride = 3),
        Conv((1, 1), channels_aux => 128, bias = false),
        BatchNorm(128),
        x -> relu.(x),
        Conv((2, 2), 128 => 768, bias = false),
        BatchNorm(768),
        x -> relu.(x),
        x -> dropdims(x, dims = (1,2)),
        Dense(768, num_classes),
    ) |> gpu
    global_pooling = AdaptiveMeanPool((1, 1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    DARTSEvalAuxModel(
        α_normal,
        α_reduce,
        stem,
        cells,
        auxiliary,
        global_pooling,
        classifier,
        operations,
    )
end

function DARTSEvalAuxModel(
    ;
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
    ) |> gpu
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
        else
            reduce = false
        end
        cell = EvalCell(
            channels_before_last,
            channels_last,
            channels_current,
            reduce,
            reduce_previous,
            steps,
            mult,
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
        x -> dropdims(x, dims = (1,2)),
        Dense(768, num_classes),
    ) |> gpu
    global_pooling = AdaptiveMeanPool((1, 1)) |> gpu
    classifier = Dense(channels_last, num_classes) |> gpu
    DARTSEvalAuxModel(
        [],
        [],
        stem,
        cells,
        auxiliary,
        global_pooling,
        classifier,
        PRIMITIVES
    )
end

function (m::DARTSEvalAuxModel)(
    x::CuArray{Float32, 4},
    is_training::Bool = false,
    drop_prob::Float32 = 0f0,
)
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        new_state = cell(s1, s2, Float32(drop_prob))
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
