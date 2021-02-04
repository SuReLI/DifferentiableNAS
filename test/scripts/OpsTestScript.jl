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

@show args = parse_commandline()

train, val = get_processed_data(args["val_split"], args["batchsize"], 0.02f0, args["random_seed"])
@show size(train[1][1])

if args["random_seed"] > -1
    Random.seed!(args["random_seed"])
end
base_folder = prepare_folder("optest", args)

for _ in 1:2
    for prim in PRIMITIVES
        @show beginscript = now()
        global args
        global base_folder
        global cbepoch
        global cbbatch

        m = DARTSModel(num_cells = args["num_cells"], channels = args["channels"], operations = [prim])

        optimiser_α = Optimiser(WeightDecay(1f-3),ADAM(3f-4,(0.5f0,0.999f0)))
        optimiser_w = Optimiser(WeightDecay(3f-4),CosineAnnealing(args["epochs"]),Momentum(0.025f0, 0.9f0))

        histepoch = historiessml()
        histbatch = historiessml()
        cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
        cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)
        losses = [0f0, 0f0]

        for epoch in 1:1
            @show epoch
            display(Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS"))
            @show prim
            @time DARTStrain1st!(loss, m, train, val, optimiser_α, optimiser_w, losses, epoch; cbepoch = cbepoch, cbbatch = cbbatch)
        end
        display(("done", Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS")))
    end
