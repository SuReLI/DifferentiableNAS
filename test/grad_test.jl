using Flux
using LinearAlgebra

p = Tuple([rand(5) for i=1:3])
p1 = p[1]
p2 = p[2]
function myloss()
    norm(p1) + norm(p2)
end
grad = gradient(Flux.Zygote.Params(p)) do
  myloss()
end
display(grad[p[1]])
display(grad.grads)

function myloss2()
    norm(p[1]) + norm(p[2])
end
grad2 = gradient(Flux.params(p)) do
  myloss2()
end
display(grad2[p[1]])
display(grad2.grads)

p = rand(5,3)
p1 = p[:,1]
p2 = p[:,2]
function myloss()
    norm(p1) + norm(p2)
end
grad = gradient(Flux.Zygote.Params(p)) do
  myloss()
end
display(grad.grads)

function myloss2()
    norm(p[:,1]) + norm(p[:,2])
end
grad2 = gradient(Flux.params(p...)) do
  myloss2()
end
display(grad2.grads)
for v in values(grad.grads)
  if isa(v,AbstractArray) && size(p1) == size(v)
    display(v)
  end
end

using SliceMap
using JuliennedArrays
using DifferentiableNAS

A = rand(Float32, num_ops, k) |> gpu
B = JuliennedArrays.Slices(A,1) |> gpu
C = JuliennedArrays.Align(B,1) |> gpu
D = reduce(hcat,B)

steps = 4
k = floor(Int, steps^2/2+3*steps/2)
num_ops = length(PRIMITIVES)
singlify(x::Float64) = Float32(x)
singlify(x::Complex{Float64}) = Complex{Float32}(x)
random_α(dim1::Int64, dim2::Int64) = singlify.(2e-3*(rand(Float32, dim1, dim2) .- 0.5))
softmaxrandom_α(dim1::Int64,dim2::Int64) = singlify.(softmax(random_α(dim1, dim2), dims = 2))
uniform_α(dim1::Int64, dim2::Int64) = singlify.(softmax(ones((dim1, dim2)), dims = 2))
α_normal = random_α(k, num_ops) |> gpu
α_rand = softmax(random_α(k, num_ops), dims = 2) |> gpu
α_reduce = random_α(k, num_ops) |> gpu

m = DARTSModel(num_cells = 3, channels = 4) |> gpu

struct Αs
  αs
end

Αs(α...) = Αs(α)

a = Αs([rand(Float32, num_ops)|>gpu for _ in 1:k]...)

typeof(a.αs)

reduce(hcat, a.αs)