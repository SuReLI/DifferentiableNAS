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

argparams = trial_params(val_split = 0.1)

num_ops = length(PRIMITIVES)

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

val_batchsize = 32
train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction, val_batchsize)
test = get_test_data(argparams.test_fraction)

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

datesnow = Dates.now()
base_folder = string("test/models/masked_", datesnow)
mkpath(base_folder)

cbepoch = CbAll(CUDA.reclaim, GC.gc, histepoch, save_progress, CUDA.reclaim, GC.gc)
cbbatch = CbAll(CUDA.reclaim, GC.gc, histbatch, CUDA.reclaim, GC.gc)

#BSON.@load "test/models/pretrainedmaskprogress2020-12-21T17:38:09.58.bson" m_cpu histepoch histbatch optimizer_w
#pars = Flux.params(cpu(m_cpu))
#m_cpu = nothing
m = DARTSModel()
#Flux.loadparams!(m, pars)
m = gpu(m)
CUDA.memory_status()
Flux.@epochs 10 Standardtrain1st!(accuracy_batched, loss, m, train, optimizer_w, losses; cbepoch = cbepoch, cbbatch = cbbatch)
Flux.@epochs 10 Maskedtrain1st!(accuracy_batched, loss, m, train, val, optimizer_w, losses; cbepoch = cbepoch, cbbatch = cbbatch)
