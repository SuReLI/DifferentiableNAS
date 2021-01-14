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
using Colors
using ColorBrewer
include("../CIFAR10.jl")
include("../training_utils.jl")

@with_kw struct eval_params
    epochs::Int = 600
    batchsize::Int = 196
    test_batchsize::Int = 196
    throttle_::Int = 20
    val_split::Float32 = 0.0
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

argparams = eval_params()

optimiser = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction, argparams.test_batchsize)


function (hist::historiessml)()
    push!(hist.accuracies, accuracy_batched(m_eval, val))
end
histepoch = historiessml()

function save_progress()
    m_cpu = m |> cpu
    normal_αs = m_cpu.normal_αs
    reduce_αs = m_cpu.reduce_αs
    BSON.@save joinpath(base_folder, "model.bson") m_cpu argparams optimiser
    BSON.@save joinpath(base_folder, "histepoch.bson") histepoch
    BSON.@save joinpath(base_folder, "histbatch.bson") histbatch
end


datesnow = Dates.now()
base_folder = string("test/models/eval_", datesnow)
mkpath(base_folder)

trial_name = "test/models/alphas09.58.bson"

BSON.@load trial_name normal_ reduce_

cbepoch = CbAll(histepoch, save_progress)

m = DARTSEvalAuxModel(normal_, reduce_, num_cells=20, channels=36) |> gpu
Flux.@epochs 10 DARTSevaltrain1st!(loss, m, train, optimiser; cbepoch = cbepoch)
