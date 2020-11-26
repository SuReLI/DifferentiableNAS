using DifferentiableNAS
using Flux
include("CIFAR10.jl")

@testset "DARTS MixedOp" begin
    num_ops = length(PRIMITIVES)
    mo = MixedOp(4,1)
    @test length(params(mo).order) > 0
    input = rand(Float32,8,8,4,2)
    @test size(input) == size(mo(input, rand(Float32,num_ops)))
    α = rand(Float32, num_ops)
    @test size(input) == size(gradient(x -> sum(mo(x, α)), input)[1])
end

@testset "DARTS Cell" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    input = rand(Float32,8,8,4,2)
    cell = Cell(4, 4, 1, false, false, 4, 4)
    @test length(params(cell).order) > 0
    αs = rand(Float32, k, num_ops)
    @test size(input) == size(cell(input, input.*2, αs))
    grad = gradient((x1, x2) -> sum(cell(x1, x2, αs)), input, input.*2)
    @test size(input) == size(grad[1])

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
    α_normal = uniform_α(k, num_ops)
    α_rand = softmax(random_α(k, num_ops), dims = 2)
    @test all(y->y==α_normal[1], α_normal)
    α_reduce = uniform_α(k, num_ops)
    @test all(y->y==α_normal[1], α_normal)

    m = DARTSNetwork(α_normal, α_reduce)
    @test length(params(m).order) > 1
    @test length(all_αs(m).order) == 2
    @test length(all_αs(m).order) + length(all_ws(m).order) == length(params(m).order)
    @test length(params(m.cells).order) > 1
end
