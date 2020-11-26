using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold
using StatsBase: mean
include("CIFAR10.jl")

@testset "DARTS training test" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    singlify(x::Float64) = Float32(x)
    singlify(x::Complex{Float64}) = Complex{Float32}(x)
    random_α(dim1::Int64, dim2::Int64) = singlify.(2e-3*(rand(Float32, dim1, dim2) .- 0.5))
    softmaxrandom_α(dim1::Int64,dim2::Int64) = singlify.(softmax(random_α(dim1, dim2), dims = 2))
    uniform_α(dim1::Int64, dim2::Int64) = singlify.(softmax(ones((dim1, dim2)), dims = 2))
    α_normal = uniform_α(k, num_ops)
    α_rand = softmax(random_α(k, num_ops), dims = 2)
    α_reduce = uniform_α(k, num_ops)

    m = DARTSNetwork(α_normal, α_reduce)
    batchsize = 64
    throttle_ = 2
    splitr = 0.5
    evalcb = throttle(() -> @show(loss(m, test...)), throttle_)
    loss(m, x, y) = logitcrossentropy(squeeze(m(x)), y)
    function accuracy(m, x, y)
        mean(onecold(m(x), 1:10) .== onecold(y, 1:10))
    end
    optimizer = ADAM()
    train, val = get_processed_data(splitr, batchsize)
    test = get_test_data(0.01)

    loss(m, train[1]...)
    accuracy(m, test...)

    DARTStrain1st!(loss, m, train[1:5], val[1:5], optimizer; cb = evalcb)

    loss(m, train[1]...)
    accuracy(m, test...)

end
