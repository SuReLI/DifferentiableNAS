using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using Zygote: @nograd
using StatsBase: mean
using CUDA
include(".../CIFAR10.jl")
@nograd onehotbatch
@nograd softmax

@testset "DARTS training test" begin
    sing DifferentiableNAS
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

    argparams = trial_params(batchsize = 32)

    num_ops = length(PRIMITIVES)

    m = DARTSModel(num_cells = 4, channels = 4) |> gpu

    optimiser_α = Optimiser(WeightDecay(1e-3),ADAM(3e-4,(0.5,0.999)))
    optimiser_w = Optimiser(WeightDecay(3e-4),Momentum(0.025, 0.9))

    train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
    test = get_test_data(argparams.test_fraction)

    histepoch = historiessml()
    histbatch = historiessml()
    losses = [0.0, 0.0]

    datesnow = Dates.now()
    base_folder = string("test/models/darts_", datesnow)
    mkpath(base_folder)

    cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
    cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

    DARTStrain1st!(loss, m, train, val, optimiser_α, optimiser_w, losses; cbepoch = cbepoch, cbbatch = cbbatch)

    loss1 = loss(m, train[1][1] |> gpu, train[1][2] |> gpu)
    acc1 = accuracy(m, test...)

    DARTStrain1st!(loss, m, train, val, optimiser; cb = evalcb)

    loss2 = loss(m, train[1][1] |> gpu, train[1][2] |> gpu)
    acc2 = accuracy(m, test...)

    @test loss2 < loss1
    @test acc2 > acc1
end
