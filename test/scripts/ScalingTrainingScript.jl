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
include("../CIFAR10.jl")
include("../training_utils.jl")


argparams = trial_params(batchsize = 32, trainval_fraction = 0.01)

m = DARTSModel(num_cells = 4, channels = 4) |> gpu

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

histepoch = historiessm([],[],[],[])
histbatch = historiessm([],[],[],[])

datesnow = Dates.now()
base_folder = string("test/models/darts_", datesnow)
mkpath(base_folder)
function save_progress()
    m_cpu = m |> cpu
    normal = m_cpu.normal_αs
    reduce = m_cpu.reduce_αs
    BSON.@save joinpath(base_folder, "model.bson") m_cpu argparams optimizer_α optimizer_w
    BSON.@save joinpath(base_folder, "alphas.bson") normal reduce argparams optimizer_α optimizer_w
    BSON.@save joinpath(base_folder, "histepoch.bson") histepoch
    BSON.@save joinpath(base_folder, "histbatch.bson") histbatch
end


struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)
cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

Flux.@epochs 10 Scalingtrain1st!(loss, m, train, val, optimizer_α, optimizer_w, 0.1)
