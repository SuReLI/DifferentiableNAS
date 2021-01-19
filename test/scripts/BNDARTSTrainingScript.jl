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

argparams = trial_params()

num_ops = length(PRIMITIVES)

m = DARTSModelBN(num_cells = argparams.num_cells, channels = argparams.channels) |> gpu

function (hist::historiessml)()
    @show losses
    push!(hist.normal_αs_sm, m.normal_αs |> cpu)
    push!(hist.reduce_αs_sm, m.reduce_αs |> cpu)
    #push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    push!(hist.train_losses, losses[1])
    push!(hist.val_losses, losses[2])
    CUDA.reclaim()
    GC.gc()
end

optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))
optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

base_folder = prepare_folder("bndarts")

cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

for epoch in 1:argparams.epochs
    @show epoch
    DARTStrain1st!(loss, m, train, val, optimiser_α, optimiser_w, losses; cbepoch = cbepoch, cbbatch = cbbatch)
end
