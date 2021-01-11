using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
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

argparams = eval_params(batchsize = 16, test_batchsize = 16)

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction, argparams.test_batchsize)


function (hist::historiessm)()
    push!(hist.accuracies, accuracy_batched(m_eval, val))
end
histepoch = historiessm([],[],[],[])

datesnow = Dates.now()
base_folder = string("test/models/eval_", datesnow)
mkpath(base_folder)

trial_name = "test/models/alphas09.58.bson"

BSON.@load trial_name normal_ reduce_

cbepoch = CbAll(acccb, histepoch, save_progress)

m = DARTSEvalModel(normal_, reduce_, num_cells=20, channels=36) |> gpu
optimizer = Nesterov(3e-4,0.9)
Flux.@epochs 10 DARTSevaltrain1st!(loss, m, train, optimizer; cbepoch = cbepoch)
