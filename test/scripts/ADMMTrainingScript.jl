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
include("../CIFAR10.jl")
include("../training_utils.jl")
@nograd onehotbatch


argparams = trial_params(batchsize=32)

num_ops = length(PRIMITIVES)

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
@show (length(train), length(val))
test = get_test_data(argparams.test_fraction)

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

datesnow = Dates.now()
base_folder = string("test/models/admm_", datesnow)
mkpath(base_folder)

cbepoch = CbAll(CUDA.reclaim, GC.gc, histepoch, save_progress, CUDA.reclaim, GC.gc)
cbbatch = CbAll(CUDA.reclaim, GC.gc, histbatch, CUDA.reclaim, GC.gc)

m = DARTSModel()
m = gpu(m)
zs = 0*vcat(m.normal_αs, m.reduce_αs)
us = 0*vcat(m.normal_αs, m.reduce_αs)
Flux.@epochs 50 ADMMtrain1st!(loss, m, train, val, optimizer_w, optimizer_α, zs, us, 1e-3, losses; cbepoch = cbepoch, cbbatch = cbbatch)
