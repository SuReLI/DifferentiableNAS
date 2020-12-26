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

argparams = trial_params(val_split = 0.1)

num_ops = length(PRIMITIVES)

losscb() = @show(loss(m, test[1] |> gpu, test[2] |> gpu))
throttled_losscb = throttle(losscb, argparams.throttle_)
function loss(m, x, y)
    #x_g = x |> gpu
    #y_g = y |> gpu
    @show logitcrossentropy(squeeze(m(x)), y)
end

acccb() = @show(accuracy_batched(m, val))
function accuracy(m, x, y; pert = [])
    out = mean(onecold(m(x, αs = pert), 1:10) .== onecold(y, 1:10))
end
function accuracy_batched(m, xy; pert = [])
    CUDA.reclaim()
    GC.gc()
    score = 0.0
    count = 0
    for batch in CuIterator(xy)
        @show acc = accuracy(m, batch..., pert = pert)
        score += acc*length(batch)
        count += length(batch)
        CUDA.reclaim()
        GC.gc()
    end
    display(score / count)
    score / count
end
function accuracy_unbatched(m, xy; pert = [])
    CUDA.reclaim()
    GC.gc()
    xy = xy | gpu
    acc = accuracy(m, xy..., pert = pert)
    foreach(CUDA.unsafe_free!, xy)
    CUDA.reclaim()
    GC.gc()
    acc
end

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

val_batchsize = 32
train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction, val_batchsize)
test = get_test_data(argparams.test_fraction)

Base.@kwdef mutable struct histories
    normal_αs::Vector{Vector{Array{Float32, 1}}}
    reduce_αs::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

function (hist::histories)()#accuracies = false)
    push!(hist.normal_αs, m.normal_αs |> cpu)
    push!(hist.reduce_αs, m.reduce_αs |> cpu)
    push!(hist.activations, m.activations.activations |> cpu)
    #if accuracies
    #	CUDA.reclaim()
    #	push!(hist.accuracies, accuracy_batched(m, val))
    #end
    CUDA.reclaim()
    GC.gc()
end
histepoch = histories([],[],[],[])
histbatch = histories([],[],[],[])

datesnow = Dates.now()
base_file = string("test/models/masked_", datesnow)

function save_progress()
    m_cpu = m |> cpu
    BSON.@save string(base_file, "model.bson") m_cpu argparams optimizer_α optimizer_w
    BSON.@save string(base_file, "histepoch.bson") histepoch
    BSON.@save string(base_file, "histbatch.bson") histbatch
end
struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)
cbepoch = CbAll(CUDA.reclaim, GC.gc, histepoch, save_progress, CUDA.reclaim, GC.gc)
cbbatch = CbAll(CUDA.reclaim, GC.gc, histbatch, CUDA.reclaim, GC.gc)

BSON.@load "test/models/pretrainedmaskprogress2020-12-21T17:38:09.58.bson" m_cpu histepoch histbatch optimizer_w
pars = Flux.params(cpu(m_cpu))
m = DARTSModel()
Flux.loadparams!(m, pars)
m = gpu(m)
CUDA.memory_status()

Flux.@epochs 16 Maskedtrain1st!(accuracy_batched, loss, m, train, val, optimizer_w; cbepoch = cbepoch, cbbatch = cbbatch)
