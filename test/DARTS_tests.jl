using DifferentiableNAS
using Parameters: @with_kw
using Flux: logitcrossentropy, ADAM, params, gradient, @epochs, onecold
using StatsBase

"""
using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end
"""

@with_kw mutable struct Args
    batchsize::Int = 1
    throttle::Int = 10
    lr::Float64 = 3e-4
    epochs::Int = 50
    splitr_::Float64 = 0.5
end

args = Args()
train, val = get_processed_data(args)


optimizer = ADAM()
loss(m, x, y) = logitcrossentropy(squeeze(m(x)), y)

model = DARTSModel(4,2)

loss(model, train[1]...)
train!(loss, model, train, val, optimizer)
loss(model, train[1]...)

model

test = get_test_data()
accuracy(m, x, y) = mean(onecold(dropdims(m(x);dims=(1,2)), 1:10) .== onecold(y, 1:10))
accuracy(model, test...)
