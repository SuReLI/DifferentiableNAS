using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
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

argparams = trial_params(trainval_fraction = 0.01)

num_ops = length(PRIMITIVES)

m = DARTSModel(α_init = (num_ops -> ones(num_ops) |> f32), num_cells = 3, channels = 4) |> gpu


losscb() = @show(loss(m, test[1] |> gpu, test[2] |> gpu))
throttled_losscb = throttle(losscb, argparams.throttle_)
function loss(m, x, y)
    #x_g = x |> gpu
    #y_g = y |> gpu
    logitcrossentropy(squeeze(m(x)), y)
end

acccb() = @show(accuracy_batched(m, val |> gpu))
function accuracy(m, x, y; pert = [])
    x_g = x |> gpu
    y_g = y |> gpu
    mean(onecold(m(x_g, normal_αs = pert), 1:10) .== onecold(y_g, 1:10))
end
function accuracy_batched(m, xy; pert = [])
    score = 0.0
    count = 0
    for batch in xy
        acc = accuracy(m, batch..., pert = pert)
        println(acc)
        score += acc*length(batch)
        count += length(batch)
    end
    score / count
end

optimizer_α = ADAM(3e-4,(0.9,0.999))
optimizer_w = Nesterov(0.025,0.9) #change?

train, val = get_processed_data(argparams.val_split, argparams.batchsize, argparams.trainval_fraction)
test = get_test_data(argparams.test_fraction)

Base.@kwdef mutable struct histories
    normal_αs::Vector{Vector{Array{Float32, 1}}}
    reduce_αs::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end


function (hist::histories)()
    push!(hist.normal_αs, m.normal_αs |> cpu)
    push!(hist.reduce_αs, m.reduce_αs |> cpu)
    push!(hist.activations, m.activations |> cpu)
    push!(hist.accuracies, accuracy_batched(m, val |> gpu))
end
histepoch = histories([],[],[],[])
histbatch = histories([],[],[],[])

datesnow = Dates.now()
trial_file = string("test/models/pretrainedmaskprogress", datesnow, ".bson")
save_progress() = BSON.@save trial_file m histepoch histbatch

struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)
cbepoch = CbAll(acccb, histepoch, save_progress)
cbbatch = CbAll(throttled_losscb, histbatch)

Flux.@epochs 1 DARTStrain1st!(loss, m, train, val, optimizer_α, optimizer_w; cbepoch = cbepoch, cbbatch = cbbatch)


using Plots
using Colors
using ColorBrewer
cur_colors = get_color_palette(:auto, plot_color(:white), 17)
p = Vector(undef, 8)
for j = 1:8
    p[j] = plot(title = "Op $j, 1st order", legend = :outertopright)
    for i = 1:14
        plot!([a[i][j] for a in hist.normal_αs], xlabel="Batch", ylabel="alpha")#,label=labels[i])#labels=["2nd:1dConv 1 layer" "2nd:1dConv 2 layer" "2nd:2dConv 1 layer" "2nd:2dConv 2 layer"], legend=:right)
    end
end
plot(p..., layout = (4,2), size = (600,1100))

#need to clear GPU first
m_eval = DARTSEvalModel(m, num_cells=20, channels=36) |> gpu
optimizer = Nesterov(3e-4,0.9)
batchsize = 96
train, _ = get_processed_data(0.0, batchsize)
epochs = 600
Flux.@epochs 1 DARTSevaltrain1st!(loss, m_eval, train, optimizer; cb = CbAll(losscb))
