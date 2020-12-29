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
using Plots
using Colors
using ColorBrewer
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
@with_kw struct eval_params
    epochs::Int = 600
    batchsize::Int = 196
    test_batchsize::Int = 196
    throttle_::Int = 20
    val_split::Float32 = 0.0
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

argparams = eval_params(batchsize = 16, test_batchsize = 16)

num_ops = length(PRIMITIVES)

#m = DARTSModel(α_init = (num_ops -> ones(num_ops) |> f32), num_cells = 3, channels = 4) |> gpu

losscb() = @show(loss(m, test[1] |> gpu, test[2] |> gpu))
throttled_losscb = throttle(losscb, argparams.throttle_)
function loss(m, x, y)
    #x_g = x |> gpu
    #y_g = y |> gpu
    mx = m(x)
    @show(logitcrossentropy(squeeze(mx), y))
end

acccb() = @show(accuracy_batched(m_eval, test))
function accuracy(m, x, y)
    mx = m(x)
    mean(onecold(mx, 1:10) .== onecold(y, 1:10))
end
function accuracy_batched(m, xy)
    @show typeof(xy)
    score = 0.0
    count = 0
    for batch in CuIterator(xy)
        acc = accuracy(m, batch...)
        println(acc)
        score += acc*length(batch)
        count += length(batch)
        CUDA.reclaim()
        GC.gc()
    end
    score / count
end

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction, argparams.test_batchsize)

Base.@kwdef mutable struct histories
    normal_αs::Vector{Vector{Array{Float32, 1}}}
    reduce_αs::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

function (hist::histories)()
    #push!(hist.normal_αs, m.normal_αs |> cpu)
    #push!(hist.reduce_αs, m.reduce_αs |> cpu)
    #push!(hist.activations, m.activations |> cpu)
    push!(hist.accuracies, accuracy_batched(m_eval, val))
end
histepoch = histories([],[],[],[])

datesnow = Dates.now()
base_folder = string("test/models/eval_", datesnow)
mkpath(base_folder)
function save_progress()
    m_cpu = m_eval |> cpu
    normal = m_cpu.normal_αs
    reduce = m_cpu.reduce_αs
    BSON.@save joinpath(base_folder, "model.bson") m_cpu argparams optimizer_α optimizer_w
    BSON.@save joinpath(base_folder, "alphas.bson") normal reduce argparams optimizer_α optimizer_w
    BSON.@save joinpath(base_folder, "histepoch.bson") histepoch
    BSON.@save joinpath(base_folder, "histbatch.bson") histbatch
end


trial_name = "test/models/alphas09.58.bson"

BSON.@load trial_name normal_ reduce_


struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)
cbepoch = CbAll(acccb, histepoch, save_progress)

m_eval = DARTSEvalModel(normal_, reduce_, num_cells=20, channels=36) |> gpu
optimizer = Nesterov(3e-4,0.9)
Flux.@epochs 10 DARTSevaltrain1st!(loss, m_eval, train, optimizer; cbepoch = cbepoch)

