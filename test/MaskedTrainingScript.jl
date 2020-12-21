using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using Zygote: @nograd
using StatsBase: mean
using Parameters
using CUDA
using Distributions
using BSON
using Dates
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

argparams = trial_params(trainval_fraction = 0.005, val_split = 0.2, batchsize = 32)

m = DARTSModel(num_cells = 5) |> gpu

num_ops = length(PRIMITIVES)

losscb() = @show(loss(m, test[1] |> gpu, test[2] |> gpu))
throttled_losscb = throttle(losscb, argparams.throttle_)
function loss(m, x, y)
    #x_g = x |> gpu
    #y_g = y |> gpu
    @show logitcrossentropy(squeeze(m(x)), y)
end

acccb() = @show(accuracy_batched(m, val |> gpu))
function accuracy(m, x, y; pert = [])
    #x_g = x |> gpu
    #y_g = y |> gpu
    #display(typeof(x_g))
    #display(typeof(y_g))
    out = mean(onecold(m(x, normal_αs = pert), 1:10) .== onecold(y, 1:10))
    #CUDA.
end
function accuracy_batched(m, xy; pert = [])
    score = 0.0
    count = 0
    for batch in CuIterator(xy)
        acc = accuracy(m, batch..., pert = pert)
        println(acc)
        score += acc*length(batch)
        count += length(batch)
    end
    score / count
end

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

Base.@kwdef mutable struct histories
    normal_αs::Vector{Vector{Array{Float32, 1}}}
    reduce_αs::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

function (hist::histories)()
    push!(hist.normal_αs, m.normal_αs |> cpu)
    push!(hist.reduce_αs, m.reduce_αs |> cpu)
    #push!(hist.activations, m.activations.activations |> cpu)
    #push!(hist.accuracies, accuracy_batched(m, val |> gpu))
end
histepoch = histories([],[],[],[])
histbatch = histories([],[],[],[])

datesnow = Dates.now()
trial_file = string("test/models/masktrain", datesnow, ".bson")
save_progress() = BSON.@save trial_file m histepoch histbatch argparams optimizer_α optimizer_w

struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)
cbepoch = CbAll(histepoch, save_progress)
cbbatch = histbatch

#BSON.@load "test/models/pretrainedmaskprogress2020-12-19T13:59:31.902.bson" m histepoch histbatch
#m = m |> gpu
#Flux.@epochs 2 DARTStrain1st!(loss, m, train, val, optimizer_α, optimizer_w; cbepoch = cbepoch, cbbatch = cbbatch)

Flux.@epochs 10 Maskedtrain1st!(accuracy_batched, loss, m, train, val, optimizer_w; cbepoch = cbepoch, cbbatch = cbbatch)
