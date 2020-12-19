using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase: mean
using CUDA
using Distributions
using BSON
using Dates
include("CIFAR10.jl")
@nograd onehotbatch

num_ops = length(PRIMITIVES)

m = DARTSModel(α_init = (num_ops -> ones(num_ops) |> f32), num_cells = 3, channels = 4) |> gpu
epochs = 50
batchsize = 64
#batchsize = 32
throttle_ = 20
#splitr = 0.5
splitr = 0.2

Flux.@epochs 1 Standardtrain1st!(accuracy_batched, loss, m, train, val, optimizer_w; cb = cbs)
BSON.@save string("test/models/pretrainedmasktest", datesnow, ".bson") m histepoch histbatch

#BSON.@load "test/models/pretrainedmasktest_.bson" m
Flux.@epochs 10 Maskedtrain1st!(accuracy_batched, loss, m, train, val, optimizer_w; cb = cbs)

"""
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
"""


#need to clear GPU first
m_eval = DARTSEvalModel(m, num_cells=20, channels=36) |> gpu
optimizer = Nesterov(3e-4,0.9)
batchsize = 96
train, _ = get_processed_data(0.0, batchsize)
epochs = 600
Flux.@epochs 1 DARTSevaltrain1st!(loss, m_eval, train, optimizer; cb = CbAll(losscb))
