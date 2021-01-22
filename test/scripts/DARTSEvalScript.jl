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

@with_kw struct eval_params
    epochs::Int = 600
    batchsize::Int = 196
    test_batchsize::Int = 196
    throttle_::Int = 20
    val_split::Float32 = 0.0
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

argparams = eval_params(batchsize::Int = 64)

optimiser = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction, argparams.test_batchsize)

losses = [0.0, 0.0]

function (hist::historiessml)()
    push!(hist.train_losses, losses[1])
end

function save_progress()
    m_cpu = m |> cpu
    normal_αs = m_cpu.normal_αs
    reduce_αs = m_cpu.reduce_αs
    BSON.@save joinpath(base_folder, "model.bson") m_cpu argparams optimiser
    BSON.@save joinpath(base_folder, "histeval.bson") histeval
end

trial_folder = "test/models/osirim/bnadmm_6642126"

function loss(m, x, y)
    out, aux = m(x)
    loss = logitcrossentropy(squeeze(out), y) + 0.4*logitcrossentropy(squeeze(aux), y)
    return loss
end


if "SLURM_JOB_ID" in keys(ENV)
    uniqueid = ENV["SLURM_JOB_ID"]
else
    uniqueid = Dates.now()
end
base_folder = string(trial_folder, "/eval_", uniqueid)
mkpath(base_folder)

BSON.@load string(trial_folder, "/histepoch.bson") histepoch
normal_ = histepoch.normal_αs_sm
reduce_ = histepoch.reduce_αs_sm

histeval = historiessml()
cbepoch = CbAll(histeval, save_progress)

m = DARTSEvalAuxModel(normal_[length(normal_)], reduce_[length(reduce_)], num_cells=20, channels=36) |> gpu
for epoch in 1:argparams.epochs
    @show epoch
    DARTSevaltrain1st!(loss, m, train, optimiser, losses; cbepoch = cbepoch)
    if epoch % 10 == 0
        accuracy_batched(m, val)
    end
end
