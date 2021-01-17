using DifferentiableNAS
using Flux
using Flux: logitcrossentropy, onecold, onehotbatch, Optimiser
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

m = DARTSModel(track_acts = true, num_cells = argparams.num_cells, channels = argparams.channels) |> gpu

optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))
optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

function (hist::historiessml)()
    push!(hist.normal_αs_sm, softmax.(copy(m.normal_αs)) |> cpu)
    push!(hist.reduce_αs_sm, softmax.(copy(m.reduce_αs)) |> cpu)
    push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    CUDA.reclaim()
    GC.gc()
end
histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

base_folder = prepare_folder("acts")

cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

acts = activationpre(loss, m, val)
for epoch in 1:argparams.epochs
    @show epoch
    Activationtrain1st!(loss, m, train, val, optimiser_α, optimiser_w, acts, losses)
end
