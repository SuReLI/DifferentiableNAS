export PRIMITIVES, uniform_α, DARTSNetwork
export Cell, MixedOp, squeeze, all_αs


using Flux
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
# using CUDA


ReLUConvBN(channels_in, channels_out, kernel_size, stride, pad) = Chain(
    x -> relu.(x),
    Conv(kernel_size, channels_in => channels_out),# ,pad=(pad,pad), stride=(stride,stride)),
    BatchNorm(channels_out),
)

FactorizedReduce(channels_in, channels_out, stride) = Chain(
    x -> relu.(x),
    x -> cat(
        Conv((1, 1), channels_in => channels_out ÷ stride, stride = (stride, stride))(x),
        Conv((1, 1), channels_in => channels_out ÷ stride, stride = (stride, stride))(x[2:end, 2:end, :, :]),
        dims = 3,
    ), #dims?
    BatchNorm(channels_out),
)

SepConv(channels_in, channels_out, kernel_size, stride, pad) = Chain(
    x -> relu.(x),
    DepthwiseConv(kernel_size, channels_in => channels_in, stride = (stride, stride), pad = (pad, pad)),
    Conv((1, 1), channels_in => channels_in, stride = (1, 1), pad = (0, 0)),
    BatchNorm(channels_in),
    x -> relu.(x),
    DepthwiseConv(kernel_size, channels_in => channels_in, pad = (pad, pad), stride = (1, 1)),
    Conv((1, 1), channels_in => channels_out, stride = (1, 1), pad = (0, 0)),
    BatchNorm(channels_out),
)

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
)

Identity(stride, pad) = x -> x[1:stride:end, 1:stride:end, :, :] |> f32
Zero(stride, pad) = x -> x[1:stride:end, 1:stride:end, :, :] * 0 |> f32

SkipConnect(channels_in, channels_out, stride, pad) = stride == 1 ? Identity(stride, pad) : FactorizedReduce(channels_in, channels_out, stride)

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
    #identity",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    #"dil_conv_3x3",
    #"dil_conv_5x5"
]

#TODO: change to NamedTuple
OPS = Dict(
    "none" => (channels, stride, w) -> Chain(Zero(stride, 1)),
    #"identity" => (channels, stride, w) -> Chain(Identity(stride, 1)),
    "avg_pool_3x3" =>
        (channels, stride, w) ->
            Chain(MeanPool((3, 3), stride = stride, pad = 1), BatchNorm(channels)) |> f32,
    "max_pool_3x3" =>
        (channels, stride, w) ->
            Chain(MaxPool((3, 3), stride = stride, pad = 1), BatchNorm(channels)),
    #"skip_connect" => (channels, stride, w) -> Chain(stride == 1 ? Identity() : FactorizedReduce(channels, C), x->w.*x),
    "skip_connect" =>
        (channels, stride, w) -> Chain(SkipConnect(channels, channels, stride, 1)),
    "sep_conv_3x3" => (channels, stride, w) -> SepConv(channels, channels, (3, 3), stride, 1),
    "sep_conv_5x5" => (channels, stride, w)-> SepConv(channels, channels, (5, 5), stride, 2),
    #"sep_conv_7x7" => (channels, stride, w)-> SepConv(channels, channels, 7, stride, 3),
    "dil_conv_3x3" => (channels, stride, w) -> DilConv(channels, channels, (3, 3), stride, 1, 2),
    "dil_conv_5x5" => (channels, stride, w)-> DilConv(channels, channels, (5, 5), stride, 2, 2),
    "conv_7x1_1x7" =>
        (channels, stride, w) -> Chain(
            x -> relu.(x),
            Conv((1, 7), C -> channels, pad = (0, 3), stride = (1, stride)),
            Conv((7, 1), C -> channels, pad = (3, 0), stride = (stride, 1)),
            BatchNorm(channels_out)
        ),
)

struct MixedOp
    ops
end

MixedOp(channels::Int64, stride::Int64) = MixedOp([OPS[prim](channels, stride, 1) for prim in PRIMITIVES])

function (m::MixedOp)(x, αs)
    #print(size(x), " -> ")
    for op in m.ops
        #print(size(op(x)), " ")
    end
    #println()
    sum([op(x) for op in m.ops] .* αs)
end

Flux.@functor MixedOp

struct Cell
    steps::Int64
    reduction::Bool
    multiplier::Int64
    prelayer1
    prelayer2
    mixedops
end

function Cell(channels_before_last, channels_last, channels, reduce, reduce_previous, steps, multiplier)
    if reduce_previous
        prelayer1 = FactorizedReduce(channels_before_last,channels,1)
    else
        prelayer1 = ReLUConvBN(channels_before_last,channels,(1,1),1,0)
    end
    prelayer2 = ReLUConvBN(channels_last,channels,(1,1),1,0)
    mixedops = []
    for i = 1:steps
       for j = 1:2+i
            reduce && j < 3 ? stride = 2 : stride = 1
            #println("i:", i, " j:", j, " s:", stride)
            mixedop = MixedOp(channels, stride)
            push!(mixedops, mixedop)
        end
    end
    Cell(steps, reduce, multiplier, prelayer1, prelayer2, mixedops)
end

function (m::Cell)(x1, x2, αs)
    println("x1:", size(x1), " x2:", size(x2))
    println("pl1:", size(m.prelayer1(x1)), " pl2:", size(m.prelayer2(x2)))
    states = [m.prelayer1(x1), m.prelayer2(x2)]
    offset = 0
    for i in 1:m.steps
        println(offset+1, ":", offset+size(states,1))
        for (j, h) in enumerate(states)
            println("to sum ", j, ": ", size(m.mixedops[offset+j](h, αs[offset+j,:])))
        end
        state = sum([m.mixedops[offset+j](earlier_state, αs[offset+j,:]) for (j, earlier_state) in enumerate(states)])
        offset += size(states,1)
        println("state ", i, ": ",size(state))
        push!(states, state)
    end
    println(size(cat(states[m.steps-m.multiplier+2:m.steps]..., dims = 3)))
    cat(states[m.steps-m.multiplier+1:m.steps]..., dims = 3)
end

Flux.@functor Cell

struct DARTSModel
    normal_αs::Array{Float32, 2}
    reduce_αs::Array{Float32, 2}
    stem
    cells
    global_pooling
    classifier
end

Flux.@functor DARTSModel

function DARTSNetwork(α_normal, α_reduce; num_classes = 10, layers = 8, channels = 16, steps = 4, mult = 4 , stem_mult = 3)
    channels_current = channels*stem_mult
    stem = Chain(
        Conv((3,3), 3=>channels_current, pad=(1,1)),
        BatchNorm(channels_current))
    channels_before_last = channels_current
    channels_last = channels_current
    channels_current = channels
    reduce_previous = false
    cells = []
    for i = 1:layers
        if i == layers÷3+1 || i == 2*layers÷3+1
            channels_current = channels_current*2
            reduce = true
        else
            reduce = false
        end
        println(i, ": ", channels_before_last, ", ", channels_last, ", ", channels_current, ", ", reduce, ", ", reduce_previous, ", ", steps, ", ", mult)
        cell = Cell(channels_before_last, channels_last, channels_current, reduce, reduce_previous, steps, mult)
        push!(cells, cell)

        reduce_previous = reduce
        channels_before_last = channels_last
        channels_last = mult*channels_current
    end

    global_pooling = AdaptiveMeanPool((1,1))
    classifier = Dense(channels_last, num_classes)
    DARTSModel(α_normal, α_reduce, stem, cells, global_pooling, classifier)
end

#Flux.@functor DARTSNetwork

function (m::DARTSModel)(x)
    s1 = m.stem(x)
    s2 = m.stem(x)
    for (i, cell) in enumerate(m.cells)
        cell.reduction ? αs = softmax(m.reduce_αs) : αs = softmax(m.normal_αs)
        new_state = cell(s1, s2, αs)
        s1 = s2
        s2 = new_state
    end
    out = m.global_pooling(s2)
    m.classifier(squeeze(out))
end
