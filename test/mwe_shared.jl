using Flux

struct SharedDense{F,T<:AbstractVector}
  b::T
  σ::F
end

SharedDense(b) = SharedDense(b, identity)

function SharedDense(in::Integer, out::Integer, σ = identity)
  return SharedDense(rand(Float32, out), σ)
end

Flux.@functor SharedDense

function (a::SharedDense)(x::AbstractArray, W::AbstractArray)
  #b, σ = a.b, a.σ
  a.σ.(W*x .+ a.b)
end


struct SharedWeights
    W1::AbstractArray
    W2::AbstractArray
end

Flux.@functor SharedWeights


struct SharedModel
    SW::SharedWeights
    layers::Array{SharedDense,1}
end

function SharedModel()
    layers = [SharedDense(8,8) for _ in 1:3]
    SW = SharedWeights(rand(Float32,8,8),rand(Float32,8,8))
    SharedModel(SW, layers)
end

function (m::SharedModel)(x::AbstractArray)
    state = x
    for layer in m.layers
        state = layer(state, m.SW.W1)
        state = layer(state, m.SW.W2)
    end
    state
end

Flux.@functor SharedModel


using Test
using CUDA

m = SharedModel() |> gpu
data = rand(Float32, 8) |> gpu
@test sum(m(data)) != 0
grad = gradient(x -> sum(m(x)), data)

loss(m, x) = sum(m(x))
gws = gradient(params(m)) do
    sum(m(data))
end
for ws in params(m)
    @test gws[ws] != Nothing
end
