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
    singlify(x::Float64) = Float32(x)
    singlify(x::Complex{Float64}) = Complex{Float32}(x)
    random_α(dim1::Int64, dim2::Int64) = singlify.(2e-3*(rand(Float32, dim1, dim2) .- 0.5))
    softmaxrandom_α(dim1::Int64,dim2::Int64) = singlify.(softmax(random_α(dim1, dim2), dims = 2))
    uniform_α(dim1::Int64, dim2::Int64) = singlify.(softmax(ones((dim1, dim2)), dims = 2))
    α_normal = uniform_α(k, num_ops)  |> gpu
    α_rand = softmax(random_α(k, num_ops), dims = 2)  |> gpu
    α_reduce = uniform_α(k, num_ops)  |> gpu
    a_n = cu.([2e-3*(rand(Float32, num_ops).-0.5) for _ in 1:k])
    a_r = cu.([2e-3*(rand(Float32, num_ops).-0.5) for _ in 1:k])

    m = DARTSModel(a_n, a_r) |> gpu
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
