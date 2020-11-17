using DifferentiableNAS

using Parameters: @with_kw
@with_kw mutable struct Args
    batchsize::Int = 64
    throttle::Int = 10
    lr::Float64 = 3e-4
    epochs::Int = 50
    splitr_::Float64 = 0.5
end

args = Args()
#train, valX, valY = get_CIFAR10_train(64, 0.5)
train, val = get_processed_data(args)
