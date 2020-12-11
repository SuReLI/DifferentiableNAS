using Flux

struct SoftmaxWeightedSum
    fields
end

function (a::SharedDense)(x::AbstractArray, W::AbstractArray)
  #b, σ = a.b, a.σ
  a.σ.(W*x .+ a.b)
end
