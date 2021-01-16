using DifferentiableNAS
using Flux
using CUDA
using Distributions: Bernoulli
include("../CIFAR10.jl")

@testset "Softmax" begin

end

@testset "DARTS MixedOp" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    mo = MixedOp(1,"1-2",4,1)  |> gpu
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
    cell = Cell(4, 4, 1, false, false, 4, 4, 1) |> gpu
    @test length(Flux.params(cell).order) > 0
    as = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    @test size(data) == size(cell(data, data, as))
    grad = gradient((x1, x2, αs) -> sum(cell(x1, x2, αs)), data, data, as)
    @test size(data) == size(grad[1])
end

@testset "DARTS Model" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    data = rand(Float32,8,8,4,2) |> gpu
    masked_αs = [2e-3*(rand(num_ops).-0.5).*rand(Bernoulli(),num_ops) |> f32 |> gpu  for _ in 1:k]
    m = DARTSModel(track_acts = true) |> gpu
    @test length(Flux.params(m).order) > 1
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
    @test length(m.activations.currentacts) > 0
    key = collect(keys(m.activations.currentacts))[1]
    sample1 = m.activations.currentacts[key]
    out = m(test_image, αs=[masked_αs, masked_αs])
    sample2 = m.activations.currentacts[key]
    @test sample1 != sample2
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
    @test size(data) == size(cell(data, data, Float32(0.4)))
    grad = gradient((x1, x2) -> sum(cell(x1, x2, Float32(0.4))), data, data)
    @test size(data) == size(grad[1])
end

@testset "DARTS Eval Model" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    normal = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    reduce = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    m = DARTSEvalModel(normal, reduce) |> gpu
    @test length(Flux.params(m).order) > 1
    @test length(Flux.params(m.cells).order) > 1
    test_image = rand(Float32, 32, 32, 3, 1) |> gpu
    grad = gradient(x->sum(m(x)), test_image)
    @test size(test_image) ==  size(grad[1])
    loss(m, x) = sum(m(x, Float32(0.4)))
    gws = gradient(Flux.params(m.cells)) do
        sum(m(test_image))
    end
    @test typeof(gws[Flux.params(m.cells)[1]]) != Nothing
end


@testset "DARTS Eval Aux Model" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)
    normal = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    reduce = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    m = DARTSEvalAuxModel(normal, reduce) |> gpu
    @test length(Flux.params(m).order) > 1
    @test length(Flux.params(m.cells).order) > 1
    test_image = rand(Float32, 32, 32, 3, 1) |> gpu
    out, out_aux = m(test_image, true)
    @test size(out) == size(out_aux)
    grad = gradient(x->sum(m(x, true)[1]), test_image)
    @test size(test_image) ==  size(grad[1])
    loss(m, x) = sum(m(x, true, Float32(0.4)))
    gws = gradient(Flux.params(m.cells)) do
        sum(sum(m(test_image, true)))
    end
    @test typeof(gws[Flux.params(m.cells)[1]]) != Nothing
end
