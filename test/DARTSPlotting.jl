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
    activations::Vector{Any}
    accuracies::Vector{Float32}
end

file_name = "test/models/pretrainedmaskprogress2020-12-19T13:59:31.902.bson"
BSON.@load file_name m histepoch histbatch

connects = vcat([[(j,i) for j = 1:i-1] for i = 3:6]...)

n_y_min = minimum([softmax(a[i])[j] for a in histbatch.normal_αs for i in 1:14 for j in 1:8])
n_y_max = maximum([softmax(a[i])[j] for a in histbatch.normal_αs for i in 1:14 for j in 1:8])
p = Vector(undef, 14)
for i = 1:14
    p[i] = plot(title = string("Op ",connects[i][1],"->",connects[i][2]), legend = :outertopright)
    for j = 1:8
        plot!([softmax(a[i])[j] for a in histbatch.normal_αs], xlabel="Batch", ylabel="alpha", label=PRIMITIVES[j], ylim=(n_y_min,n_y_max))
    end
end
gui(plot(p..., layout = (7,2), size = (2000,2000)))
savefig("test/models/fig_n.png")

r_y_min = minimum([softmax(a[i])[j] for a in histbatch.reduce_αs for i in 1:14 for j in 1:8])
r_y_max = maximum([softmax(a[i])[j] for a in histbatch.reduce_αs for i in 1:14 for j in 1:8])
p = Vector(undef, 14)
for i = 1:14
    p[i] = plot(title = string("Op ",connects[i][1],"->",connects[i][2]), legend = :outertopright, xlabel="Batch", ylabel="alpha", ylim=(r_y_min,r_y_max))
    for j = 1:8
        plot!([softmax(a[i])[j] for a in histbatch.reduce_αs], label=PRIMITIVES[j])
    end
end
gui(plot(p..., layout = (7,2), size = (2000,2000)))
savefig("test/models/fig_r.png")