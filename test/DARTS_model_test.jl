using DifferentiableNAS
using Flux
using CUDA
include("CIFAR10.jl")

@testset "DARTS MixedOp" begin
    num_ops = length(PRIMITIVES)
    mo = MixedOp(4,1)  |> gpu
    @test length(params(mo).order |> cpu) > 0
    data = rand(Float32,8,8,4,2)  |> gpu
    α = rand(Float32, num_ops)  |> gpu
    println(typeof(mo.ops[1]),typeof(mo.ops))
    @test size(data) == size(mo(data, α))
    @test size(data) == size(gradient(x -> sum(mo(x, α)), data)[1])
end

@testset "DARTS Cell" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    data = rand(Float32,8,8,4,2) |> gpu
    cell = Cell(4, 4, 1, false, false, 4, 4) |> gpu
    @test length(params(cell).order) > 0
    as = cu.([2e-3*rand(Float32, num_ops).-0.5 for _ in 1:k])
    @test size(data) == size(cell(data, data, as))
    grad = gradient((x1, x2, αs) -> sum(cell(x1, x2, αs)), data, data, as)
    @test size(data) == size(grad[1])

end

@testset "DARTS Model" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    @test k == (steps+1)*(steps+2)/2-1
    num_ops = length(PRIMITIVES)
    singlify(x::Float64) = Float32(x)
    singlify(x::Complex{Float64}) = Complex{Float32}(x)
    random_α(dim1::Int64, dim2::Int64) = singlify.(2e-3*(rand(Float32, dim1, dim2) .- 0.5))
    softmaxrandom_α(dim1::Int64,dim2::Int64) = singlify.(softmax(random_α(dim1, dim2), dims = 2))
    uniform_α(dim1::Int64, dim2::Int64) = singlify.(softmax(ones((dim1, dim2)), dims = 2))
    α_normal = uniform_α(k, num_ops) |> gpu
    α_rand = softmax(random_α(k, num_ops), dims = 2) |> gpu
    α_reduce = uniform_α(k, num_ops) |> gpu
    a_n = cu.([2e-3*rand(Float32, num_ops).-0.5 for _ in 1:k])
    a_r = cu.([2e-3*rand(Float32, num_ops).-0.5 for _ in 1:k])

    m = DARTSModel(a_n, a_r) |> gpu
    @test length(params(m).order) > 1
    @test length(all_αs(m).order) == 2*k
    @test length(all_αs(m).order) + length(all_ws(m).order) == length(params(m).order)
    @test length(params(m.cells).order) > 1
    test_image = rand(Float32, 32, 32, 3, 1) |> gpu
    grad = gradient(x->sum(m(x)), test_image)
    @test size(test_image) ==  size(grad)
end
