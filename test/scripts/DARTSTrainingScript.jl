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

@show beginscript = now()

@show args = parse_commandline()
#argparams = trial_params()

num_ops = length(PRIMITIVES)

m = DARTSModel(num_cells = args["num_cells"], channels = args["channels"]) |> gpu

optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))
optimiser_w = Optimiser(WeightDecay(3e-4),CosineAnnealing(args["epochs"]),Momentum(0.025, 0.9))

train, val = get_processed_data(args["val_split"], args["batchsize"], args["trainval_fraction"])
test = get_test_data(args["test_fraction"])

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

base_folder = prepare_folder("darts", args)

cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

for epoch in 1:args["epochs"]
    @show epoch
    @show Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS")
    DARTStrain1st!(loss, m, train, val, optimiser_α, optimiser_w, losses, epoch; cbepoch = cbepoch, cbbatch = cbbatch)
end
@show "done", Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS")
