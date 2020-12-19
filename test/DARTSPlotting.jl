using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase: mean
using Parameters
using CUDA
using Distributions
using BSON
using Plots
include("CIFAR10.jl")

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = 64
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

Base.@kwdef mutable struct histories
    normal_αs::Vector{Vector{Array{Float32, 1}}}
    reduce_αs::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

function (hist::histories)()
    push!(hist.normal_αs, m.normal_αs |> cpu)
    push!(hist.reduce_αs, m.reduce_αs |> cpu)
    push!(hist.activations, m.activations |> cpu)
    push!(hist.accuracies, accuracy_batched(m, val |> gpu))
end
histepoch = histories([],[],[],[])
histbatch = histories([],[],[],[])

file_name = "test/models/pretrainedmaskprogress.bson"
BSON.@load file_name m histepoch histbatch

p = Vector(undef, 14)
for i = 1:14
    p[i] = plot(title = "Op $i, 1st order", legend = :outertopright)
    for j = 1:8
        plot!([softmax(a[i])[j] for a in histbatch.normal_αs], xlabel="Batch", ylabel="alpha", label=PRIMITIVES[j])#labels=["2nd:1dConv 1 layer" "2nd:1dConv 2 layer" "2nd:2dConv 1 layer" "2nd:2dConv 2 layer"], legend=:right)
    end
end
gui(plot(p..., layout = (7,2), size = (2000,2000)))
savefig("test/models/fig.png")
