ENV["GKSwstype"]="100"


using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase
using Parameters
using CUDA
using Distributions
using BSON
using Plots
using Plots.PlotMeasures
include("CIFAR10.jl")

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = 32
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = 1.0
    test_fraction::Float32 = 1.0
end

C(g::ColorGradient) = RGB[g[z] for z=LinRange(0,1,30)]
g = :inferno
color = cgrad(g) |> C


#acts dims: cellid x oplayer x image
folder_name = "test/models/osirim/darts_2021-01-04T22:44:32.328"
BSON.@load joinpath(folder_name,"histbatch.bson") histbatch
BSON.@load joinpath(folder_name,"histepoch.bson") histepoch

datadict = Dict{String, Array{Float32,4}}()
datadictepoch = Dict{String, Array{Float32,4}}()

epochs = length(histepoch.activations)
images = sum([size(d["3-6-sep_conv_3x3"],3) for d in histbatch.activations]) ÷ epochs

for (k,v) in histbatch.activations[1]
    datadict[k] = Array{Float32,4}(undef, length(histbatch.activations), size(v)...)
    datadictepoch[k] = Array{Float32,4}(undef, length(histepoch.activations), size(v,1), size(v,2), images)
end

e = 1
ep = 1
for (i, batch) in enumerate(histbatch.activations)
    for (k,v) in batch
        a = 1
        while a < 32
            datadict[k][i,:,:,a:size(v,3)+a-1] .= v #batch x cellid x oplayer x image
            a += size(v,3)
        end
        datadictepoch[k][ep,:,:,e:size(v,3)+e-1] .= v #epoch x cellid x oplayer x image
    end
    e += size(batch["3-6-sep_conv_3x3"],3)
    if e > images
        e = 1
        @show ep += 1
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
    layers = size(datadict[string("2-3-", op)],3)
    p = Vector(undef, layers+1)
    p[1] = plot(title = string(op, " alphas"), legend = :outertopright, left_margin = 10mm)
    for n = 1:14
        plot!([softmax(a[n])[j+1] for a in sorta], label=string(sortin[n][1],"->",sortin[n][2]))
    end
    for i = 1:layers
        p[i+1] = plot(title = string(op, " ",layerdict[op][i]))
        for (n1, n2) in sortin
            #cellop = string(n1,"-",n2,"-",op)
            #plot!(datadictepoch[cellop][1,:,i,:], label=string(n1,"->",n2), legend = :outerright, left_margin = 10mm)
            hdata = dropdims(mean(datadictepoch["4-6-dil_conv_3x3"][:,:,2,:], dims = 2), dims = 2)
            bins = minimum(hdata):0.01:maximum(hdata)
            h = hcat([fit(Histogram, hdata[i,:], bins).weights for i in 1:epochs]...)
            ticks = [(bins[i]+bins[i+1])/2 for i in 1:length(bins)-1]
            heatmap(1:epochs, ticks, h, xlabel="epoch", ylabel="spatial+channel+cell meaned activation", title="4-6-dil_conv_3x3, 3x3conv layer")
        end
    end
    plot(p..., layout = (layers+1,1), size = (1200, 400*(layers+1)))
    savefig(joinpath(folder_name,string("fig_", op, ".png")));
end


#heatmap(datadictepoch["3-6-dil_conv_3x3"][:,4,2,:]', xlabel="epoch", ylabel="image", title="3-6-dil_conv_3x3, cell 3, dilconv layer")
hdata = dropdims(mean(datadictepoch["4-6-sep_conv_3x3"][:,:,2,:], dims = 2), dims = 2)
bins = minimum(hdata):0.001:maximum(hdata)
h = hcat([fit(Histogram, hdata[i,:], bins).weights for i in 1:epochs]...)
ticks = [(bins[i]+bins[i+1])/2 for i in 1:length(bins)-1]
heatmap(1:epochs, ticks, h, xlabel="epoch", ylabel="spatial+channel+cell meaned activation", title="4-6-dil_conv_3x3, dilconv layer")

hdata = datadict["4-6-sep_conv_3x3"][:,4,5,:]
bins = minimum(hdata):0.01:maximum(hdata)
h = hcat([fit(Histogram, hdata[i,:], bins).weights for i in 1:size(datadict["4-6-sep_conv_3x3"][:,4,2,:],1)]...)
ticks = [(bins[i]+bins[i+1])/2 for i in 1:length(bins)-1]
heatmap(1:size(datadict["4-6-sep_conv_3x3"][:,4,2,:],1), ticks, h, xlabel="batch", ylabel="spatial+channel meaned activation", title="4-6-sep_conv_3x3, sepconv layer")



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
