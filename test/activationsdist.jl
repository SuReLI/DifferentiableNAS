ENV["GKSwstype"]="100"


using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase: mean, median
using Parameters
using CUDA
using Distributions
using BSON
using Plots
using Plots.PlotMeasures
include("CIFAR10.jl")

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = 64
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

folder_name = "test/models/osirim/darts_2021-01-04T16:25:28.826"
BSON.@load joinpath(folder_name,"histbatch.bson") histbatch
BSON.@load joinpath(folder_name,"histepoch.bson") histepoch

datadict = Dict{String, Array{Float32,4}}()
for (k,v) in histbatch.activations[1]
    datadict[k] = Array{Float32,4}(undef, length(histbatch.activations), size(v)...)
end
for (i, batch) in enumerate(histbatch.activations)
    for (k,v) in batch
        datadict[k][i,:,:,:] .= v
    end
end
layerdict = Dict(
    "max_pool_3x3" => [""],
    "avg_pool_3x3" => [""],
    "skip_connect" => [""],
    "sep_conv_3x3" => ["relu", "3x3conv", "1x1conv", "relu", "3x3conv", "1x1conv"],
    "sep_conv_5x5" => ["relu", "5x5conv", "1x1conv", "relu", "5x5conv", "1x1conv"],
    "dil_conv_3x3" => ["relu", "3x3dilconv", "3x3conv"],
    "dil_conv_5x5" => ["relu", "5x5dilconv", "5x5conv"]
)
ops = sort(collect(keys(datadict)), by = x -> (x[5:6], x[1], x[3]))
connects = vcat([[(j,i) for j = 1:i-1] for i = 3:6]...)
a_conn = [collect(zip(connects, a)) for a in histbatch.normal_αs]
sortin = sort(connects)
sorta = [[s[2] for s in sort(a)] for a in a_conn]
for (j, op) in enumerate(PRIMITIVES[2:length(PRIMITIVES)])
    layers = size(datadict[string("2-3-", op)],2)
    p = Vector(undef, layers+1)
    p[1] = plot(title = string(op, " alphas"), legend = :outertopright, left_margin = 10mm)
    for n = 1:14
        plot!([softmax(a[n])[j+1] for a in sorta], label=string(sortin[n][1],"->",sortin[n][2]))
    end
    for i = 1:layers
        p[i+1] = plot(title = string(op, " ",layerdict[op][i]))
        for (n1, n2) in sortin
            cellop = string(n1,"-",n2,"-",op)
            plot!(dropdims(median(datadict[cellop][:,i,:,:], dims = (2,3)), dims=(2,3)), label=string(n1,"->",n2), legend = :outerright, left_margin = 10mm)
        end
    end
    plot(p..., layout = (layers+1,1), size = (1200, 400*(layers+1)))
    savefig(joinpath(folder_name,string("fig_med_", op, ".png")));
end



"""
n_y_min = minimum([softmax(a[i])[j] for a in histbatch.normal_αs for i in 1:14 for j in 1:8])
n_y_max = maximum([softmax(a[i])[j] for a in histbatch.normal_αs for i in 1:14 for j in 1:8])
p = Vector(undef, 14)
for i = 1:14
    p[i] = plot(title = string("Op ",connects[i][1],"->",connects[i][2]), ylim=(n_y_min,n_y_max), legend=false)
    for j = 1:8
        plot!([softmax(a[i])[j] for a in histbatch.normal_αs], label=PRIMITIVES[j])
    end
end
plot(p..., layout = (2,7), size = (2200,600));
savefig(joinpath(folder_name,"fig_n.png"))

r_y_min = minimum([softmax(a[i])[j] for a in histbatch.reduce_αs for i in 1:14 for j in 1:8])
r_y_max = maximum([softmax(a[i])[j] for a in histbatch.reduce_αs for i in 1:14 for j in 1:8])
p = Vector(undef, 14)
for i = 1:14
    p[i] = plot(title = string("Op ",connects[i][1],"->",connects[i][2]), ylim=(r_y_min,r_y_max), legend=false)
    for j = 1:8
        plot!([softmax(a[i])[j] for a in histbatch.reduce_αs], label=PRIMITIVES[j])
    end
end
plot(p..., layout = (2,7), size = (2200,600));
savefig(joinpath(folder_name,"fig_r.png"))

#normal_ = m_cpu.normal_αs
#reduce_ = m_cpu.reduce_αs
#BSON.@save "test/models/alphas09.58.bson" normal_ reduce_ histepoch histbatch argparams
"""