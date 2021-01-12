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
using Plots
include("../CIFAR10.jl")
include("../training_utils.jl")


argparams = trial_params(batchsize = 32, val_split = 0.0)

m = DARTSModel(num_cells = 4, channels = 4) |> gpu

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

histepoch = historiessm([],[],[],[])
histbatch = historiessm([],[],[],[])

datesnow = Dates.now()
base_folder = string("test/models/sadmm_", datesnow)
mkpath(base_folder)

function (hist::historiessm)()
    push!(hist.normal_αs_sm, softmax.(copy(m.normal_αs)) |> cpu)
    push!(hist.reduce_αs_sm, softmax.(copy(m.reduce_αs)) |> cpu)
    #push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    CUDA.reclaim()
    GC.gc()
end

cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

zs = 0*scalingparams(m) |> gpu
us = 0*scalingparams(m) |> gpu

@show typeof(zs)
Flux.@epochs 10 ScalingADMMtrain1st!(loss, m, train, optimizer_w, zs, us, 1e-3; cbepoch = cbepoch, cbbatch = cbbatch)
