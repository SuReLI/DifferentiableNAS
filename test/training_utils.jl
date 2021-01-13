using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase: mean
using Parameters
using CUDA
using Distributions
using BSON
using Dates
using Zygote
include("CIFAR10.jl")
@nograd onehotbatch

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = 64
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

function loss(m, x, y)
    out = m(x)
    loss = logitcrossentropy(squeeze(out), y)
    return loss
end

acccb() = @show(accuracy_batched(m, val))
function accuracy(m, x, y; pert = [])
    out = mean(onecold(m(x, αs = pert), 1:10) .== onecold(y, 1:10))
end
function accuracy_batched(m, xy; pert = [])
    CUDA.reclaim()
    GC.gc()
    score = 0.0
    count = 0
    for batch in CuIterator(xy)
        @show acc = accuracy(m, batch..., pert = pert)
        score += acc*length(batch)
        count += length(batch)
        CUDA.reclaim()
        GC.gc()
    end
    display(score / count)
    score / count
end
function accuracy_unbatched(m, xy; pert = [])
    CUDA.reclaim()
    GC.gc()
    xy = xy | gpu
    acc = accuracy(m, xy..., pert = pert)
    foreach(CUDA.unsafe_free!, xy)
    CUDA.reclaim()
    GC.gc()
    acc
end
Base.@kwdef mutable struct histories
    normal_αs::Vector{Vector{Array{Float32, 1}}}
    reduce_αs::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

histories() = histories([],[],[],[])

function (hist::histories)()
    push!(hist.normal_αs, deepcopy(m.normal_αs) |> cpu)
    push!(hist.reduce_αs, deepcopy(m.reduce_αs) |> cpu)
    #push!(hist.activations, m.activations |> cpu)
    #push!(hist.accuracies, accuracy_batched(m, val |> gpu))
end


Base.@kwdef mutable struct historiessm
    normal_αs_sm::Vector{Vector{Array{Float32, 1}}}
    reduce_αs_sm::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

historiessm() = historiessm([],[],[],[])

function (hist::historiessm)()
    @show losses
    push!(hist.normal_αs_sm, softmax.(deepcopy(m.normal_αs)) |> cpu)
    push!(hist.reduce_αs_sm, softmax.(deepcopy(m.reduce_αs)) |> cpu)
    #push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    CUDA.reclaim()
    GC.gc()
end


Base.@kwdef mutable struct historiessml
    normal_αs_sm::Vector{Vector{Array{Float32, 1}}}
    reduce_αs_sm::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
    train_losses::Vector{Float32}
    val_losses::Vector{Float32}
end

historiessml() = historiessml([],[],[],[],[],[])

function (hist::historiessml)()
    @show m.normal_αs[1]
    @show losses
    push!(hist.normal_αs_sm, softmax.(deepcopy(m.normal_αs)) |> cpu)
    push!(hist.reduce_αs_sm, softmax.(deepcopy(m.reduce_αs)) |> cpu)
    #push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    push!(hist.train_losses, losses[1])
    push!(hist.val_losses, losses[2])
    @show m.normal_αs[1]
    CUDA.reclaim()
    GC.gc()
end


struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)

function save_progress()
    m_cpu = m |> cpu
    normal = m_cpu.normal_αs
    reduce = m_cpu.reduce_αs
    BSON.@save joinpath(base_folder, "model.bson") m_cpu argparams optimizer_α optimizer_w
    BSON.@save joinpath(base_folder, "alphas.bson") normal reduce argparams optimizer_α optimizer_w
    BSON.@save joinpath(base_folder, "histepoch.bson") histepoch
    BSON.@save joinpath(base_folder, "histbatch.bson") histbatch
end
