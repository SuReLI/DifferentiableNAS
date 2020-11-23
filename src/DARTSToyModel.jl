export DARTSToyModel

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

struct DARTSToyModel
    normal_αs::Array{Float32,2}
    reduce_αs::Array{Float32,2}
    chains::Array{Op,2}
end

function DARTSToyModel()
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
    αs = softmax(ones(Float32, size(chains)))
    DARTSToyModel(αs, αs, chains)
end

function (m::DARTSToyModel)(x)
    for i = 1:size(m.normal_αs, 2)
        mpw = softmax(m.normal_αs[:, i])
        x = sum([chain(x) for chain in m.chains[:, i]] .* mpw)
    end
    squeeze(x)
end

Flux.@functor DARTSToyModel
