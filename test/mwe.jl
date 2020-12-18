using Flux
using Zygote

function my_softmax(xs; dims = 1)
    @show typeof(xs) xs
    softmax(xs, dims = dims)
end

Zygote.@adjoint function my_softmax(xs; dims = 1)
    softmax(xs, dims = dims), Δ -> begin
        @show typeof(Δ) Δ typeof(xs) xs ∇softmax(Δ, xs, dims = dims)
        (∇softmax(Δ, xs, dims = dims),)
    end
end

ReLUConv(channels_in, channels_out, kernel_size, pad) =
    Chain(x -> relu.(x), Conv(kernel_size, channels_in => channels_out, pad = pad))


struct MixedOperation
    operations::AbstractArray
end

MixedOperation(channels::Int64, kernel_options::AbstractArray) =
    MixedOperation([ReLUConv(channels, channels, (i, i), i ÷ 2) for i in kernel_options])

function (m::MixedOperation)(x::AbstractArray, αs::AbstractArray)
    #αs = softmax(αs)
    αs = my_softmax(αs)
    #mapreduce((op, α) -> α * op(x), +, m.operations, αs) #errors out in gradient of above line
    #sum(αs[i]*m.operations[i](x) for i in 1:length(αs))
    sum(αs .* map(op -> op(x), m.operations))
end

Flux.@functor MixedOperation

using Test
using CUDA

m = MixedOperation(3, [1, 3, 5, 7]) |> gpu
αs = rand(Float32, 4) |> gpu
test_image = rand(Float32, 16, 16, 3, 1) |> gpu
@test sum(m(test_image, αs)) != 0
grad = gradient((x,αs) -> sum(m(x,αs)), test_image, αs)

gαs = gradient(Flux.params(αs)) do
    sum(m(test_image, αs))
end
for a in Flux.params(αs)
    @show gαs[a]
    @test isa(gαs[a], CuArray)
end
