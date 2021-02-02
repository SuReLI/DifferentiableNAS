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

if args["random_seed"] > -1
    Random.seed!(args["random_seed"])
end

m = DARTSModel(num_cells = args["num_cells"], channels = args["channels"]) |> gpu

optimiser_α = Optimiser(WeightDecay(1f-3),ADAM(3f-4,(0.5f0,0.999f0)))
optimiser_w = Optimiser(WeightDecay(3f-4),CosineAnnealing(args["epochs"]),Momentum(0.025f0, 0.9f0))

train, val = get_processed_data(args["val_split"], args["batchsize"], args["trainval_fraction"], args["random_seed"])
test = get_test_data(args["test_fraction"], args["random_seed"])

histepoch = historiessml()
histbatch = historiessml()
losses = [0f0, 0f0]

base_folder = prepare_folder("darts", args)

cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

for epoch in 1:args["epochs"]
    @show epoch
    display(Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS"))
    @time DARTStrain1st!(loss, m, train, val, optimiser_α, optimiser_w, losses, epoch; cbepoch = cbepoch, cbbatch = cbbatch)
end
display(("done", Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS")))
