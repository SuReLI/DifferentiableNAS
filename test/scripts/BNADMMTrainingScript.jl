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
num_ops = length(PRIMITIVES)

optimiser_α = Optimiser(WeightDecay(1f-3),ADAM(3f-4,(0.5f0,0.999f0)))
optimiser_w = Optimiser(WeightDecay(3f-4),CosineAnnealing(args["epochs"]),Momentum(0.025f0, 0.9f0))
#optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(args["val_split"], args["batchsize"], args["trainval_fraction"], args["random_seed"])
test = get_test_data(args["test_fraction"], args["random_seed"])

histepoch = historiessml()
histbatch = historiessml()
losses = [0.0, 0.0]

base_folder = prepare_folder("bnadmm", args)

cbepoch = CbAll(CUDA.reclaim, GC.gc, histepoch, save_progress, CUDA.reclaim, GC.gc)
cbbatch = CbAll(CUDA.reclaim, GC.gc, histbatch, CUDA.reclaim, GC.gc)

function (hist::historiessml)()
    @show losses
    push!(hist.normal_αs_sm, [tanh.(relu.(a)) for a in m.normal_αs |> cpu])
    push!(hist.reduce_αs_sm, [tanh.(relu.(a)) for a in m.reduce_αs |> cpu])
    #push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    push!(hist.train_losses, losses[1])
    push!(hist.val_losses, losses[2])
    CUDA.reclaim()
    GC.gc()
end

m = DARTSModelBN(num_cells = args["num_cells"], channels = args["channels"])
zu = ADMMaux(0f0*vcat(m.normal_αs, m.reduce_αs), 0f0*vcat(m.normal_αs, m.reduce_αs))
local epoch
for epoch in 1:args["epochs"]
    @show epoch
    display(Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS"))
    @time ADMMtrain1st!(loss, m, train, val, optimiser_w, optimiser_α, zu, args["rho"], losses, epoch; cbepoch = cbepoch, cbbatch = cbbatch)
end
display(("done", Dates.format(convert(DateTime,now()-beginscript), "HH:MM:SS")))
