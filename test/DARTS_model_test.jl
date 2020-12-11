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
    mo = MixedOp(4,1,1)  |> gpu
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
    #as = α14()
    as = [2e-3*(rand(num_ops).-0.5) |> f32 |> gpu  for _ in 1:k]
    #as = rand(Float32, k, num_ops) #|> gpu
    @test size(data) == size(cell(data, data, as))
    grad = gradient((x1, x2, αs) -> sum(cell(x1, x2, αs)), data, data, as)
    @test size(data) == size(grad[1])
    display(typeof(grad[3]))
end

@testset "DARTS Model" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    @test k == (steps+1)*(steps+2)/2-1
    num_ops = length(PRIMITIVES)
    m = DARTSModel() |> gpu
    @test length(Flux.params(m).order) > 1
    @test length(all_αs(m).order) == 2*k
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
    for k in keys(gαs.grads)
        if isa(gαs.grads[k],NamedTuple)
            display(gαs.grads[k].:normal_αs)
        end
    end
    for α in Flux.params(m.normal_αs).order
        for v in values(gαs.grads)
            if isa(v,AbstractArray) && size(α) == size(v)
                display(v)
            end
        end
    end
    @test typeof(gαs[Flux.params(m.normal_αs)[1]]) != Nothing
end
