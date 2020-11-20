using DifferentiableNAS
using Parameters: @with_kw
using Flux: logitcrossentropy, ADAM, params, gradient, @epochs, onecold, throttle
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
    batchsize::Int = 64
    throttle::Int = 10
    lr::Float64 = 3e-4
    epochs::Int = 50
    splitr_::Float64 = 0.5
end

args = Args()
train, val = get_processed_data(args)


optimizer = ADAM()
loss(m, x, y) = logitcrossentropy(squeeze(m(x)), y)

model = DARTSModel()

test = get_test_data()
accuracy(m, x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))
accuracy(model, test...)
evalcb = throttle(() -> @show(loss(model, val...)), args.throttle)

DARTStrain!(loss, model, train, val, optimizer, "second"; cb = evalcb)
loss(model, train[1]...)
accuracy(model, test...)
model

DARTStrain!(loss, model, train, val, optimizer, "second"; cb = evalcb)
loss(model, train[1]...)
accuracy(model, test...)
model

DARTStrain!(loss, model, train, val, optimizer, "second"; cb = evalcb)
loss(model, train[1]...)
accuracy(model, test...)
model
