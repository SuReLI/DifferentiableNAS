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
using Plots
include("../CIFAR10.jl")
include("../training_utils.jl")


argparams = trial_params()

m = DARTSModel(num_cells = argparams.num_cells, channels = argparams.channels) |> gpu

optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))
optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

base_folder = prepare_folder("scaling")

cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

for epoch in 1:argparams.epochs
    @show epoch
    Scalingtrain1st!(loss, m, train, val, optimiser_α, optimiser_w, 0.1, losses)
end
