using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold
using StatsBase: mean

@testset "DARTS Model" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    random_α(dim1, dim2) = 2e-3*(rand(Float32, dim1, dim2) .- 0.5)
    uniform_α(dim1, dim2) = softmax(ones(Float32, (dim1, dim2)))
    α_normal = uniform_α(k, num_ops)
    @test all(y->y==α_normal[1], α_normal)
    α_reduce = uniform_α(k, num_ops)
    m = DARTSNetwork(α_normal, α_reduce)
    params(m)
    params(m.normal_αs)
    all_αs(m)
    params(m.cells)

    co2 = MixedOp(4,1)
    params(co2)
    size(co2(rand(Float32,8,8,4,2), rand(Float32,num_ops)))
    α = rand(num_ops)
    gradient(x -> sum(co2(x, α)), rand(8,8,4,2))

    cell = Cell(4, 4, 1, false, false, 4, 4)
    params(cell)
    αs = rand(k, num_ops)
    cell(rand(Float32,8,8,4,2), rand(Float32,8,8,4,2), αs)
    gradient((x1, x2) -> sum(cell(x1, x2, αs)), rand(8,8,4,2), rand(8,8,4,2))


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
    test = get_test_data()

    m = DARTSNetwork(α_normal, α_reduce, stem_mult = 3)

    loss(m, train[1]...)
    accuracy(m, test...)

    m
    DARTStrain1st!(loss, m, train[1:5], val[1:5], optimizer; cb = evalcb)


    loss(m, train[1]...)
    accuracy(m, test...)


end
