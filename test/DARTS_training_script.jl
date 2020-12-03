using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using Zygote: @nograd
using StatsBase: mean
using CUDA
include("CIFAR10.jl")
@nograd onehotbatch

steps = 4
k = floor(Int, steps^2/2+3*steps/2)
num_ops = length(PRIMITIVES)

a_n = cu.([2e-3*(rand(Float32, num_ops).-0.5) for _ in 1:k])
a_r = cu.([2e-3*(rand(Float32, num_ops).-0.5) for _ in 1:k])

m = DARTSModel(a_n, a_r, layers = 3, channels = 4) |> gpu
epochs = 50
batchsize = 64
throttle_ = 20
splitr = 0.5
losscb() = @show(loss(m, test[1] |> gpu, test[2] |> gpu))
throttled_cb = throttle(losscb, throttle_)
function loss(m, x, y)
    #x_g = x |> gpu
    #y_g = y |> gpu
    logitcrossentropy(squeeze(m(x)), y)
end
function accuracy(m, x, y)
    x_g = x |> gpu
    y_g = y |> gpu
    mean(onecold(m(x_g), 1:10) .== onecold(y_g, 1:10))
end
function accuracy_batched(m, xy)
    score = 0.0
    count = 0
    for batch in xy
        acc = accuracy(m, batch...)
        println(acc)
        score += acc*length(batch)
        count += batch
    end
    score / batch
end
optimizer = ADAM(3e-4,(0.5,0.999))

train, val = get_processed_data(splitr, batchsize)
test = get_test_data(0.01)

Base.@kwdef mutable struct α_histories
    normal_αs::Vector{Array{Array{Float32, 1}, 1}}
    reduce_αs::Vector{Array{Array{Float32, 1}, 1}}
end

function (hist::α_histories)()
    push!(hist.normal_αs, m.normal_αs |> cpu)
    push!(hist.reduce_αs, m.reduce_αs |> cpu)
end
hist = α_histories([],[])

struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)
cbs = CbAll(losscb, hist)

Flux.@epochs 10 DARTStrain1st!(loss, m, train, val, optimizer; cb = cbs)


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
