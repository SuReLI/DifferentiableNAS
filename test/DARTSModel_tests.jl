using DifferentiableNAS
using Flux


steps = 4
k = floor(Int, steps^2/2+3*steps/2)
k = 5
num_ops = length(PRIMITIVES)
random_α(dim1, dim2) = 2e-3*(rand(dim1, dim2) .- 0.5)
uniform_α(dim1, dim2) = softmax(ones(Float32, (dim1, dim2)))
α_normal = uniform_α(k, num_ops)
α_reduce = uniform_α(k, num_ops)
m = DARTSNetwork(α_normal, α_reduce)
params(m)
params(m.αs)
params(m.chains)

co2 = MixedOp(64,1)
params(co2)

cell = Cell(64, 64, 64, false, false)
params(cell)

DARTStrain!(loss, m, train[1:5], val[1:5], optimizer, "second"; cb = evalcb)
