using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch, Optimiser
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

optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))
optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

base_folder = prepare_folder("masked")

cbepoch = CbAll(CUDA.reclaim, GC.gc, histepoch, save_progress, CUDA.reclaim, GC.gc)
cbbatch = CbAll(CUDA.reclaim, GC.gc, histbatch, CUDA.reclaim, GC.gc)


m = DARTSModel(num_cells = argparams.num_cells, channels = argparams.channels) |> gpu
CUDA.memory_status()
for epoch in 1:10
    @show epoch
    Standardtrain1st!(accuracy_batched, loss, m, train, optimiser_w, losses; cbepoch = cbepoch, cbbatch = cbbatch)
end
for epoch in 11:argparams.epochs
    @show epoch
    Maskedtrain1st!(accuracy_batched, loss, m, train, val, optimiser_w, losses; cbepoch = cbepoch, cbbatch = cbbatch)
end
