using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using Zygote: @nograd
using StatsBase: mean
using CUDA
include("CIFAR10.jl")
@nograd onehotbatch
@nograd softmax

@testset "DARTS training test" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)

    m = DARTSModel() |> gpu
    batchsize = 64
    throttle_ = 2
    splitr = 0.5
    evalcb = throttle(() -> @show(loss(m, test[1] |> gpu, test[2] |> gpu)), throttle_)
    function loss(m, x, y)
        logitcrossentropy(squeeze(m(x)), y)
    end
    function accuracy(m, x, y)
        x_g = x |> gpu
        y_g = y |> gpu
        mean(onecold(m(x_g), 1:10) .== onecold(y_g, 1:10))
    end
    optimizer = ADAM()
    train, val = get_processed_data(splitr, batchsize)
    test = get_test_data(0.01)

    loss1 = loss(m, train[1][1] |> gpu, train[1][2] |> gpu)
    acc1 = accuracy(m, test...)

    DARTStrain1st!(loss, m, train, val, optimizer; cb = evalcb)

    loss2 = loss(m, train[1][1] |> gpu, train[1][2] |> gpu)
    acc2 = accuracy(m, test...)

    @test loss2 < loss1
    @test acc2 > acc1
end
