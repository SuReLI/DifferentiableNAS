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

@show beginscript = now()

@show args = parse_commandline_eval()

if args["random_seed"] > -1
    Random.seed!(args["random_seed"])
end

optimiser = Optimiser(WeightDecay(3f-4),CosineAnnealing(args["epochs"]),Momentum(0.025f0, 0.9f0))

train, val = get_processed_data(args["val_split"], args["batchsize"], args["trainval_fraction"], args["random_seed"])
test = get_test_data(args["test_fraction"], args["random_seed"])

losses = [0f0, 0f0]

function (hist::historiessml)()
    push!(hist.train_losses, losses[1])
    push!(hist.accuracies, losses[2])
end

function save_progress()
    m_cpu = m |> cpu
    normal_αs = m_cpu.normal_αs
    reduce_αs = m_cpu.reduce_αs
    #BSON.@save joinpath(base_folder, "model.bson") m_cpu optimiser
    BSON.@save joinpath(base_folder, "histeval.bson") histeval
end

#trial_folder = "test/models/bnadmm_6642126"
trial_folder = "test/models/evaldarts/"

function loss(m, x, y)
    out, aux = m(x, true, args["droppath"])
    loss = logitcrossentropy(squeeze(out), y) + args["aux"]*logitcrossentropy(squeeze(aux), y)
    return loss
end
function accuracy(m, x, y)
    mx = m(x)
    showmx = mx[1] |>cpu
    showy = y|>cpu
    for i in 1:size(showmx,2)
        @show (softmax(showmx[:,i]), showy[:,i])
    end
    mean(onecold(mx[1], 1:10)|>cpu .== onecold(y|>cpu, 1:10))
end
function accuracy_batched(m, xy)
    CUDA.reclaim()
    GC.gc()
    score = 0f0
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

#BSON.@load string(trial_folder, "/histepoch.bson") histepoch
#normal_ = histepoch.normal_αs_sm
#reduce_ = histepoch.reduce_αs_sm

histeval = historiessml()
cbepoch = CbAll(histeval, save_progress)

#m = DARTSEvalAuxModel(normal_[length(normal_)], reduce_[length(reduce_)], num_cells=20, channels=36) |> gpu
m = DARTSEvalAuxModel(num_cells = args["num_cells"], channels = args["channels"]) |> gpu
for epoch in 1:args["epochs"]
    @show epoch
    display(Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS"))
    @time DARTSevaltrain1st!(loss, m, train, optimiser, losses, epoch; cbepoch = cbepoch)
    if epoch % 10 == 0
        @time losses[2] = accuracy_batched(m, test)
    end
end
display(("done", Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS")))
