using DifferentiableNAS
using Flux
using Parameters
using CUDA
using Distributions
using BSON
include("CIFAR10.jl")

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = 64
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

argparams = trial_params(batchsize = 32, trainval_fraction = 0.005)

function loss(m, x, y)
    @show Flux.logitcrossentropy(squeeze(m(x)), y)
end

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9)

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)

trial_file = string("test/models/saveload.bson")



m = DARTSModel(num_cells = 3, channels = 4) |> gpu

Flux.@epochs 1 DARTStrain1st!(loss, m, train, val, optimizer_α, optimizer_w)
@show m_cpu = m |> cpu
BSON.@save trial_file m_cpu
m_cpu = nothing
BSON.@load trial_file m_cpu
m_loaded = m_cpu |> gpu
Flux.@epochs 1 DARTStrain1st!(loss, m_loaded, train, val, optimizer_α, optimizer_w)

