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

loss(m, x, y) = logitcrossentropy(squeeze(m(x)), y)
accuracy(m, x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))


@with_kw mutable struct Args
    batchsize::Int = 64
    throttle::Int = 2
    lr::Float64 = 3e-4
    epochs::Int = 50
    splitr_::Float64 = 0.5
end

args = Args()

evalcb = throttle(() -> @show(loss(model, test...)), args.throttle)
#evalcb = throttle(() -> @show(mean([loss(model, val_batch...) for val_batch in val])), args.throttle)

train, val = get_processed_data(args)

optimizer = ADAM()
model = DARTSModel()
test = get_test_data()

loss(model, train[1]...)
accuracy(model, test...)
model

DARTStrain!(loss, model, train, val, optimizer, "second"; cb = evalcb)

loss(model, train[1]...)
accuracy(model, test...)
model

DARTStrain!(loss, model, train[1:5], val[1:5], optimizer, "second"; cb = evalcb)
