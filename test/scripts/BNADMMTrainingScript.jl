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
#argparams = trial_params()

num_ops = length(PRIMITIVES)

optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))
optimiser_w = Optimiser(WeightDecay(3e-4),CosineAnnealing(args["epochs"]),Momentum(0.025, 0.9))
#optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

train, val = get_processed_data(args["val_split"], args["batchsize"], args["trainval_fraction"])
test = get_test_data(args["test_fraction"])

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

m = DARTSModelBN(num_cells = args["num_cells"], channels = args["channels"]) |> gpu
zu = ADMMaux(0*vcat(m.normal_αs, m.reduce_αs), 0*vcat(m.normal_αs, m.reduce_αs))
for epoch in 1:args["epochs"]
    @show epoch
    ADMMtrain1st!(loss, m, train, val, optimiser_w, optimiser_α, zu, args["rho"], losses, epoch, args["epochs"]; cbepoch = cbepoch, cbbatch = cbbatch)
end
