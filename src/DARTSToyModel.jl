export LayeredOp, DARTSModel
export PRIMITIVES, uniform_α, DARTSNetwork
export Cell, MixedOp

using Flux
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
# using CUDA

struct Op
    chain::Any
end

(m::Op)(x) = m.chain(x)
Flux.@functor Op

struct LayeredOp
    weights::Array{Float32,2}
    chains::Array{Op,2}
end

function LayeredOp()
    l1 = [
        Op(Chain(
            x -> repeat(x, outer = [1, 1, 12, 1]),
            x ->
                x[1:2:end, 1:2:end, :, :] +
                x[2:2:end, 2:2:end, :, :] +
                x[1:2:end, 2:2:end, :, :] +
                x[2:2:end, 1:2:end, :, :],
            x -> relu.(x),
        )),
        Op(Chain(Conv((1, 1), 3 => 36, stride = (2, 2), pad = (0, 0)), x -> relu.(x))),
        Op(Chain(Conv((3, 3), 3 => 36, stride = (2, 2), pad = (1, 1)), x -> relu.(x))),
        Op(Chain(Conv((5, 5), 3 => 36, stride = (2, 2), pad = (2, 2)), x -> relu.(x))),
        Op(Chain(
            Conv((3, 3), 3 => 18, pad = (1, 1)),
            x -> relu.(x),
            Conv((3, 3), 18 => 36, stride = (2, 2), pad = (1, 1)),
            x -> relu.(x),
        )),
    ]
    l2 = [
        Op(Chain(
            x ->
                x[1:2:end, 1:2:end, :, :] +
                x[2:2:end, 2:2:end, :, :] +
                x[1:2:end, 2:2:end, :, :] +
                x[2:2:end, 1:2:end, :, :], #change to maxpool?
            x ->
                x[1:2:end, 1:2:end, :, :] +
                x[2:2:end, 2:2:end, :, :] +
                x[1:2:end, 2:2:end, :, :] +
                x[2:2:end, 1:2:end, :, :], #change to maxpool?
            x -> relu.(x),
        )),
        Op(Chain(Conv((1, 1), 36 => 36, stride = (4, 4), pad = (0, 0)), x -> relu.(x))),
        Op(Chain(Conv((3, 3), 36 => 36, stride = (4, 4), pad = (1, 1)), x -> relu.(x))),
        Op(Chain(Conv((5, 5), 36 => 36, stride = (4, 4), pad = (2, 2)), x -> relu.(x))),
        Op(Chain(
            Conv((3, 3), 36 => 36, pad = (1, 1)),
            x -> relu.(x),
            Conv((3, 3), 36 => 36, stride = (4, 4), pad = (1, 1)),
            x -> relu.(x),
        )),
    ]
    l3 = [
        Op(Chain(
            x ->
                x[1:2:end, 1:2:end, 1:2:end, :] +
                x[2:2:end, 2:2:end, 2:2:end, :] +
                x[1:2:end, 2:2:end, 1:2:end, :] +
                x[2:2:end, 1:2:end, 2:2:end, :], #change to maxpool?
            x ->
                x[1:2:end, 1:2:end, 1:2:end, :] +
                x[2:2:end, 2:2:end, 2:2:end, :] +
                x[1:2:end, 2:2:end, 1:2:end, :] +
                x[2:2:end, 1:2:end, 2:2:end, :], #change to maxpool?
            x -> cat(x, sum(x, dims = 3), dims = 3),
            x -> relu.(x),
        )),
        Op(Chain(Conv((1, 1), 36 => 10, stride = (4, 4), pad = (0, 0)), x -> relu.(x))),
        Op(Chain(Conv((3, 3), 36 => 10, stride = (4, 4), pad = (1, 1)), x -> relu.(x))),
        Op(Chain(Conv((5, 5), 36 => 10, stride = (4, 4), pad = (2, 2)), x -> relu.(x))),
        Op(Chain(
            Conv((3, 3), 36 => 24, pad = (1, 1)),
            x -> relu.(x),
            Conv((3, 3), 24 => 10, stride = (4, 4), pad = (1, 1)),
            x -> relu.(x),
        )),
    ]
    chains = [l1 l2 l3]
    weights = softmax(ones(Float32, size(chains)))
    LayeredOp(weights, chains)
end

function squeeze(A::AbstractArray) #TODO: generalize this?
    #print(A, " ", size(A))
    if ndims(A) == 3
        if size(A, 3) > 1
            return dropdims(A; dims = (1))
        elseif size(A, 3) == 1
            return dropdims(A; dims = (1, 3))
        end
    elseif ndims(A) == 4
        if size(A, 4) > 1
            return dropdims(A; dims = (1, 2))
        elseif size(A, 4) == 1
            return dropdims(A; dims = (1, 2, 4))
        end
    end
    return A
end

function get_size(A::AbstractArray)
    print("\tsize:", size(A), "\t")
    return A
end

function (m::LayeredOp)(x)
    for i = 1:size(m.weights, 2)
        mpw = softmax(m.weights[:, i])
        x = sum([chain(x) for chain in m.chains[:, i]] .* mpw)
    end
    squeeze(x)
end

Flux.@functor LayeredOp

struct DARTSModel
    αs::Array{Float32,2}
    chains
end

function DARTSModel()
    lo = LayeredOp()
    αs = lo.weights
    chains = lo.chains
    DARTSModel(αs, chains)
end

function (m::DARTSModel)(x)
    for i = 1:size(m.αs, 2)
        mpw = softmax(m.αs[:, i])
        x = sum([chain(x) for chain in m.chains[:, i]] .* mpw)
    end
    squeeze(x)
end

Flux.@functor DARTSModel



ReLUConvBN(C_in, C_out, kernel_size, stride, pad) = Chain(
    x -> relu.(x),
    Conv(kernel_size, C_in => C_out),# ,pad=(pad,pad), stride=(stride,stride)),
    BatchNorm(C_out),
)

FactorizedReduce(C_in, C_out, stride) = Chain(
    x -> relu.(x),
    x -> cat(
        Conv((1, 1), C_in => C_out ÷ stride)(x),
        Conv((1, 1), C_in => C_out ÷ stride)(x[:, :, 2:end, 2:end]),
        dims = 1,
    ), #or dims=2?
    BatchNorm(C_out),
)

SepConv(C_in, C_out, kernel_size, stride, pad) = Chain(
    x -> relu.(x),
    DepthwiseConv(kernel_size, C_in => C_in, stride = (stride, stride), pad = (pad, pad)),
    Conv((1, 1), C_in => C_in, stride = (1, 1), pad = (0, 0)),
    BatchNorm(C_in),
    x -> relu.(x),
    DepthwiseConv(kernel_size, C_in => C_in, pad = (pad, pad), stride = (1, 1)),
    Conv((1, 1), C_in => C_out, stride = (1, 1), pad = (0, 0)),
    BatchNorm(C_out),
)

DilConv(C_in, C_out, kernel_size, stride, pad, dilation) = Chain(
    x -> relu.(x),
    DepthwiseConv(
        kernel_size,
        C_in => C_in,
        pad = (pad, pad),
        stride = (stride, stride),
        dilation = dilation,
    ),
    Conv(kernel_size, C_in -> C_out),
    BatchNorm(C_out),
)

Identity(stride, pad) = x -> x[1:stride:end, 1:stride:end, :, :] |> f32
Zero(stride, pad) = x -> x[1:stride:end, 1:stride:end, :, :] * 0 |> f32

struct AdaptiveMeanPool{N}
    target_out::NTuple{N,Int}
end

AdaptiveMeanPool(target_out::NTuple{N,Integer}) where {N} = AdaptiveMeanPool(target_out)

function (m::AdaptiveMeanPool)(x)
    w = size(x, 1) - m.target_out[1] + 1
    h = size(x, 2) - m.target_out[2] + 1
    return meanpool(x, (w, h); pad = (0, 0), stride = (1, 1))
end

PRIMITIVES = [
    "none",
    "identity",
    "max_pool_3x3",
    #"avg_pool_3x3",
    #"skip_connect",
    "sep_conv_3x3",
    #"sep_conv_5x5",
    #"dil_conv_3x3",
    #"dil_conv_5x5"
]



OPS = Dict(
    "none" => (C, stride, w) -> Chain(Zero(stride, 1), x -> w .* x),
    "identity" => (C, stride, w) -> Chain(Identity(stride, 1), x -> w .* x),
    "avg_pool_3x3" =>
        (C, stride, w) ->
            Chain(MeanPool((3, 3), stride = stride, pad = 1), BatchNorm(C), x -> w .* x) |> f32,
    "max_pool_3x3" =>
        (C, stride, w) ->
            Chain(MaxPool((3, 3), stride = stride, pad = 1), BatchNorm(C), x -> w .* x),
    #"skip_connect" => (C, stride, w) -> Chain(stride == 1 ? Identity() : FactorizedReduce(C, C), x->w.*x),
    "skip_connect" =>
        (C, stride, w) -> Chain(FactorizedReduce(C, C, stride), x -> w .* x),
    "sep_conv_3x3" => (C, stride, w) -> SepConv(C, C, (3, 3), stride, 1),
    #"sep_conv_5x5" => (C, stride, w)-> SepConv(C, C, 5, stride, 2),
    #"sep_conv_7x7" => (C, stride, w)-> SepConv(C, C, 7, stride, 3),
    "dil_conv_3x3" => (C, stride, w) -> DilConv(C, C, (3, 3), stride, 2, 2),
    #"dil_conv_5x5" => (C, stride, w)-> DilConv(C, C, 5, stride, 4, 2),
    "conv_7x1_1x7" =>
        (C, stride, w) -> Chain(
            x -> relu.(x),
            Conv((1, 7), C -> C, pad = (0, 3), stride = (1, stride)),
            Conv((7, 1), C -> C, pad = (3, 0), stride = (stride, 1)),
            BatchNorm(C_out),
            x -> w .* x,
        ),
)

struct MixedOp
    ops
end

MixedOp(C::Int64, stride::Int64) = MixedOp([OPS[prim](C, stride, 1) for prim in PRIMITIVES])

function (m::MixedOp)(x, αs)
    #m.identity(x) + m.none(x) + m.maxpool(x) + m.sepconv(x)
    sum([op(x) for op in m.ops] .* softmax(αs))
end

Flux.@functor MixedOp

struct Cell
    mixedops
    p0
    p1
    steps
    reduction
end

function Cell(C_pp, C_p, C, red, red_p, steps = 4, multiplier = 4)
    if red_p
        p0 = FactorizedReduce(C_pp,C)
    else
        p0 = ReLUConvBN(C_pp,C,(1,1),1,0)
    end
    p1 = ReLUConvBN(C_p,C,(1,1),1,0)
    mixedops = []
    for i = 1:steps
       for j = 1:2+i
            red && j < 3 ? stride = 2 : stride = 1
            mixedop = MixedOp(C, stride)
            push!(mixedops, mixedop)
        end
    end
    Cell(mixedops, p0, p1, steps, red)
end

function (m::Cell)(x1, x2, αs)
    states = [m.p0(x1), m.p1(x2)]
    offset = 0
    for i in range(m.steps)
        push!(states, sum([m.mixops(offset+j)(h, αs[offset+j] for (j, h) in enumerate(states))]))
        offset += len(states)
    end
end

Flux.@functor Cell


function DARTSNetwork(α_normal, α_reduce, num_classes = 10, layers = 8, C = 16, steps = 4, mult = 4 , stem_mult = 3)
    C_c = C*stem_mult
    stem = Chain(
        Conv((3,3), 3=>C_c, pad=(1,1)),
        BatchNorm(C_c))
    C_pp = C_c
    C_p = C_c
    C_c = C
    red_p = false
    cells = []
    for i = 1:layers
        if i == layers//3 || i == 2*layers//3
            C_c = C_c*2
            red = true
            weights = α_reduce
        else
            red = false
            weights = α_normal
        end
        cell = Cell(C_pp, C_p, C_c, red, red_p, steps, mult)
        red_p = red
        #model = cell(model)
        #k = floor(Int, steps^2/2+3*steps/2)
        num_ops = length(PRIMITIVES)
        C_pp = C_p
        C_p = 4*C_c #4 is length of DARTS_V1 concat
        if i == 2*layers//3
            C_aux = C_p
        end
        push!(cells, cell)
    end

    global_pooling = AdaptiveMeanPool(1)
    classifier = Dense(C_p, num_classes)

    DARTSModel(vcat(α_normal, α_reduce), [s0, s1, cells..., global_pooling, classifier])
end

#Flux.@functor DARTSNetwork

function (m::DARTSModel)(x)
    stemmed1 = DARTSModel.chains[1](x)
    stemmed2 = DARTSModel.chains[2](x)
    for (i, cell) in enumerate(DARTSModel.chains[3:-2])
        #cell.reduction ? αs = softmax(DARTSModel)
    end
end
