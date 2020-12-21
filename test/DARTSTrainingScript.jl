using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using Zygote: @nograd
using StatsBase: mean
using Parameters
using CUDA
using Distributions
using BSON
using Dates
include("CIFAR10.jl")
@nograd onehotbatch

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = 64
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

argparams = trial_params(batchsize = 32)

num_ops = length(PRIMITIVES)

m = DARTSModel(num_cells = 5) |> gpu

losscb() = @show(loss(m, test[1] |> gpu, test[2] |> gpu))
throttled_losscb = throttle(losscb, argparams.throttle_)
function loss(m, x, y)
    @show logitcrossentropy(squeeze(m(x)), y)
end

acccb() = @show(accuracy_batched(m, val))
function accuracy(m, x, y; pert = [])
    mean(onecold(m(x, normal_αs = pert), 1:10) .== onecold(y, 1:10))
end
function accuracy_batched(m, xy; pert = [])
    CUDA.reclaim()
    score = 0.0
    count = 0
    for batch in CuIterator(xy)
        acc = accuracy(m, batch..., pert = pert)
        score += acc*length(batch)
        count += length(batch)
        CUDA.reclaim()
    end
    @show score / count
end

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

function (hist::histories)()
    display("begin histories")
    push!(hist.normal_αs, m.normal_αs |> cpu)
    push!(hist.reduce_αs, m.reduce_αs |> cpu)
    display("alphas recorded")
    push!(hist.activations, m.activations.activations |> cpu)
    display("activations recorded")
    #push!(hist.accuracies, accuracy_batched(m, val))
end
histepoch = histories([],[],[],[])
histbatch = histories([],[],[],[])

datesnow = Dates.now()
trial_file = string("test/models/pretrainedmaskprogress", datesnow, ".bson")

function save_progress()
    m_cpu = m |> cpu
    BSON.@save trial_file m_cpu histepoch histbatch argparams optimizer_α optimizer_w
end


struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)
cbepoch = CbAll(CUDA.reclaim, histepoch, save_progress, CUDA.reclaim)
cbbatch = CbAll(CUDA.reclaim, histbatch, CUDA.reclaim)

Flux.@epochs 10 DARTStrain1st!(loss, m, train, val, optimizer_α, optimizer_w; cbepoch = cbepoch, cbbatch = cbbatch)
