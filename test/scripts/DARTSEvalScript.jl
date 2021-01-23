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

optimiser = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(args["val_split"], args["batchsize"], args["trainval_fraction"])
test = get_test_data(args["test_fraction"], args["test_batchsize"])

losses = [0.0, 0.0]

function (hist::historiessml)()
    push!(hist.train_losses, losses[1])
end

function save_progress()
    m_cpu = m |> cpu
    normal_αs = m_cpu.normal_αs
    reduce_αs = m_cpu.reduce_αs
    BSON.@save joinpath(base_folder, "model.bson") m_cpu optimiser
    BSON.@save joinpath(base_folder, "histeval.bson") histeval
end

trial_folder = "test/models/bnadmm_6642126"

function loss(m, x, y)
    out, aux = m(x, true, args["droppath"])
    showmx = m(x)[1] |>cpu
    showy = y|>cpu
    for i in 1:size(showmx,2)
        @show (softmax(showmx[:,i]), showy[:,i])
    end
    loss = logitcrossentropy(squeeze(out), y) + args["aux"]*logitcrossentropy(squeeze(aux), y)
    return loss
end
function accuracy(m, x, y)
    mx = m(x)
    showmx = m(x)[1] |>cpu
    showy = y|>cpu
    for i in 1:size(showmx,2)
        @show (softmax(showmx[:,i]), showy[:,i])
    end
    mean(onecold(mx[1], 1:10)|>cpu .== onecold(y|>cpu, 1:10))
end
function accuracy_batched(m, xy)
    CUDA.reclaim()
    GC.gc()
    score = 0.0
    count = 0
    for batch in TestCuIterator(xy)
        acc = accuracy(m, batch...)
        score += acc*length(batch)
        count += length(batch)
        CUDA.reclaim()
        GC.gc()
    end
    @show ("accuracy ", score / count)
    score / count
end

if "SLURM_JOB_ID" in keys(ENV)
    uniqueid = ENV["SLURM_JOB_ID"]
else
    uniqueid = Dates.now()
end
base_folder = string(trial_folder, "/eval_", uniqueid)
mkpath(base_folder)
BSON.@save joinpath(base_folder, "args.bson") args

BSON.@load string(trial_folder, "/histepoch.bson") histepoch
normal_ = histepoch.normal_αs_sm
reduce_ = histepoch.reduce_αs_sm

histeval = historiessml()
cbepoch = CbAll(histeval, save_progress)

m = DARTSEvalAuxModel(normal_[length(normal_)], reduce_[length(reduce_)], num_cells=20, channels=36) |> gpu
for epoch in 1:args["epochs"]
    @show epoch
    DARTSevaltrain1st!(loss, m, train, optimiser, losses; cbepoch = cbepoch)
    if epoch % 1 == 0
        accuracy_batched(m, test)
    end
end
