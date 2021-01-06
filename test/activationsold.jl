ENV["GKSwstype"]="100"


using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase: mean
using Parameters
using CUDA
using Distributions
using BSON
using Plots
include("CIFAR10.jl")

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = 64
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

folder_name = "test/models/osirim/darts_2021-01-04T11:54:07.793"
BSON.@load joinpath(folder_name,"histbatch.bson") histbatch
BSON.@load joinpath(folder_name,"histepoch.bson") histepoch

datadict = Dict{String, Array{Float32,2}}()
for (k,v) in histbatch.activations[1]
    datadict[k] = Matrix{Float32}(undef, length(histbatch.activations), length(v))
end
for (i, batch) in enumerate(histbatch.activations)
    for (k,v) in batch
        datadict[k][i,:] .= v
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
sortin = sort(connects)
for op in PRIMITIVES[2:length(PRIMITIVES)]
    layers = size(datadict[string("2-3-", op)],2)
    p = Vector(undef, layers)
    for i = 1:layers
        p[i] = plot(title = string(op, " ",layerdict[op][i]))
        for (n1, n2) in sortin
            cellop = string(n1,"-",n2,"-",op)
            plot!(datadict[cellop][:,i], label=cellop, legend = :outerright)
        end
    end
    plot(p..., layout = (layers,1), size = (1200, 400*layers))
    savefig(joinpath(folder_name,string("fig_", op, ".png")));
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