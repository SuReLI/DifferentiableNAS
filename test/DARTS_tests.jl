using DifferentiableNAS
using Flux #: logitcrossentropy, ADAM, params, gradient, @epochs, onecold, throttle
using Flux: throttle, logitcrossentropy, ADAM, params, gradient, @epochs, onecold
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
function accuracy(m, x, y)
    mean(onecold(m(x), 1:10) .== onecold(y, 1:10))
end

batchsize = 64
throttle_ = 2
epochs = 50
splitr = 0.5

evalcb = throttle(() -> @show(loss(model, test...)), throttle_)
#evalcb = throttle(() -> @show(mean([loss(model, val_batch...) for val_batch in val])), args.throttle)

train, val = get_processed_data(splitr, batchsize)

optimizer = ADAM()
model = DARTSToyModel()
test = get_test_data()

loss(model, train[1]...)
accuracy(model, test...)
model

DARTStrain!(loss, model, train[1:5], val[1:5], optimizer, "second"; cb = evalcb)

loss(model, train[1]...)
accuracy(model, test...)
model

DARTStrain!(loss, model, train[5:15], val[5:15], optimizer, "second"; cb = evalcb)
