using DifferentiableNAS
using Flux
using CUDA
using SliceMap
using Zygote: @showgrad
using Zygote
include("CIFAR10.jl")

@testset "DARTS MixedOp" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    mo = MixedOp("1-2",4,1)  |> gpu
    @test length(Flux.params(mo).order |> cpu) > 0
    data = rand(Float32,8,8,4,2)  |> gpu
    α = rand(Float32, num_ops)  |> gpu
    @test size(data) == size(mo(data, α))
    g = gradient((x,α) -> sum(mo(x, α)), data, α)
    @test size(data) == size(g[1])
    @test size(α) == size(g[2])
end

@testset "DARTS Cell" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    data = rand(Float32,8,8,4,2) |> gpu
    cell = Cell(4, 4, 1, false, false, 4, 4) |> gpu
    @test length(Flux.params(cell).order) > 0
    as = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    @test size(data) == size(cell(data, data, as))
    grad = gradient((x1, x2, αs) -> sum(cell(x1, x2, αs)), data, data, as)
    @test size(data) == size(grad[1])
end

@testset "DARTS Model" begin
    m = DARTSModel() |> gpu
    @test length(Flux.params(m).order) > 1
    @test length(all_αs(m).order) + length(all_ws(m).order) == length(Flux.params(m).order)
    @test length(Flux.params(m.cells).order) > 1
    test_image = rand(Float32, 32, 32, 3, 1) |> gpu
    grad = gradient(x->sum(m(x)), test_image)
    @test size(test_image) ==  size(grad[1])
    loss(m, x) = sum(m(x))
    gws = gradient(Flux.params(m.cells)) do
        sum(m(test_image))
    end
    @test typeof(gws[Flux.params(m.cells)[1]]) != Nothing
    gαs = gradient(Flux.params(m.normal_αs)) do
        sum(m(test_image))
    end
    @test typeof(gαs[Flux.params(m.normal_αs)[1]]) != Nothing
    @test length(m.activations.activations) > 0
end

@testset "DARTS Eval Cell" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    data = rand(Float32,8,8,4,2) |> gpu
    αs = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    cell = EvalCell(4, 4, 1, false, false, 4, 4, αs) |> gpu
    @test length(Flux.params(cell).order) > 0
    as = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    @test size(data) == size(cell(data, data, 0.4))
    grad = gradient((x1, x2, αs) -> sum(cell(x1, x2, αs, 0.4)), data, data, as)
    @test size(data) == size(grad[1])
end

@testset "DARTS Eval Model" begin
    normal = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    reduce = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    m = DARTSEvalModel(normal, reduce) |> gpu
    @test length(Flux.params(m).order) > 1
    @test length(Flux.params(m.cells).order) > 1
    test_image = rand(Float32, 32, 32, 3, 1) |> gpu
    grad = gradient(x->sum(m(x)), test_image)
    @test size(test_image) ==  size(grad[1])
    loss(m, x) = sum(m(x, 0.4))
    gws = gradient(Flux.params(m.cells)) do
        sum(m(test_image))
    end
    @test typeof(gws[Flux.params(m.cells)[1]]) != Nothing
end
