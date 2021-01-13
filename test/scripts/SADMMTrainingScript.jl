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

m = DARTSModel() |> gpu

optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))
optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

datesnow = Dates.now()
base_folder = string("test/models/sadmm_", datesnow)
mkpath(base_folder)

cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

zs = 0*scalingparams(m) |> gpu
us = 0*scalingparams(m) |> gpu

@show typeof(zs)
Flux.@epochs 50 ScalingADMMtrain1st!(loss, m, train, optimiser_w, zs, us, 1e-3, losses; cbepoch = cbepoch, cbbatch = cbbatch)
