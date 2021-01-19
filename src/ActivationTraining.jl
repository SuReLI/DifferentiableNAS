export Activationtrain1st!, activationupdate, collectweights, activationpre

using Flux
using Flux: onehotbatch
using Juno
using Base.Iterators
using StatsBase
using Zygote
using Zygote: @nograd
using LinearAlgebra
using CUDA
using Plots
#include("utils.jl")
#include("DARTSModel.jl")

function collectweights(model)
    for (i,cell) in enumerate(model.cells)
        #cell.reduction
        for (j,mixedop) in enumerate(cell.mixedops)
            for (k,op) in enumerate(mixedop.ops)
                for (l,layer) in enumerate(op.op)
                    if typeof(layer) <: Flux.Conv
                        @show (i,j,k,l,norm(layer.weight),size(layer.weight))
                    end
                end
            end
        end
    end
end

connectstrings = vcat([[string(j,"-",i) for j = 1:i-1] for i = 3:6]...)
axlab = vcat([string("n",e) for e in connectstrings],[string("r",e) for e in connectstrings])

edgedict = Dict{String, Int64}()
for (i,strconn) in enumerate(connectstrings)
    edgedict[strconn] = i
end
primdict = Dict{String, Int64}()
for (i,prim) in enumerate(PRIMITIVES)
    primdict[prim] = i
end
function activationupdatema(model)
    update = all_αs(model).*0
    reducecells = Array{Int64, 1}()
    normalcells = Array{Int64, 1}()
    for (i,cell) in enumerate(model.cells)
        if cell.reduction
            push!(reducecells, i)
        else
            push!(normalcells, i)
        end
    end
    for key in keys(model.activations.currentacts)
        acts = model.activations.currentacts[key]
        if size(acts,2) == 3
            acts = acts[:,2:3,:]
        elseif size(acts,2) == 6
            acts = acts[:,[2,3,5,6],:]
        end
        reduce = acts[reducecells,:,:]
        normal = acts[normalcells,:,:]
        statreduce = x -> mean(abs.(x))
        #statreduce = x -> std(x)
        update[edgedict[key[1:3]]][primdict[key[5:length(key)]]] = statreduce(normal)
        update[edgedict[key[1:3]]+14][primdict[key[5:length(key)]]] = statreduce(reduce)
    end
    update
end
function activationupdatesd(model)
    update = all_αs(model).*0
    reducecells = Array{Int64, 1}()
    normalcells = Array{Int64, 1}()
    for (i,cell) in enumerate(model.cells)
        if cell.reduction
            push!(reducecells, i)
        else
            push!(normalcells, i)
        end
    end
    for key in keys(model.activations.currentacts)
        acts = model.activations.currentacts[key]
        if size(acts,2) == 3
            acts = acts[:,2:3,:]
        elseif size(acts,2) == 6
            acts = acts[:,[2,3,5,6],:]
        end
        reduce = acts[reducecells,:,:]
        normal = acts[normalcells,:,:]
        #statreduce = x -> mean(abs.(x))
        statreduce = x -> std(x)
        update[edgedict[key[1:3]]][primdict[key[5:length(key)]]] = statreduce(normal)
        update[edgedict[key[1:3]]+14][primdict[key[5:length(key)]]] = statreduce(reduce)
    end
    update
end


all_αs(model::DARTSModel) = Flux.params([model.normal_αs, model.reduce_αs])
all_ws(model::DARTSModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

i = 1

function activationpre(loss, model, val)
    acts = []
    for val_batch in CuIterator(val)
        val_loss = loss(model, val_batch...)
        foreach(CUDA.unsafe_free!, val_batch)
        CUDA.reclaim()
        GC.gc()
        ac1 = activationupdatesd(model)
        push!(acts, ac1)
        CUDA.reclaim()
    end
    acts
end

function Activationtrain1st!(loss, model, train, val, opt_α, opt_w, baseacts, losses=[0.0,0.0]; cbepoch = () -> (), cbbatch = () -> ())
    local train_loss
    local val_loss
    local gsα
    w = all_ws(model)
    α = all_αs(model)
    fakeg = all_αs(model).*0
    fakea = all_αs(model).*0
    global i
    for (train_batch, val_batch, base_act) in zip(CuIterator(train), CuIterator(val), acts)
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...)
            return train_loss
        end
        losses[1] = train_loss
        foreach(CUDA.unsafe_free!, train_batch)
        Flux.Optimise.update!(opt_w, w, gsw)
        CUDA.reclaim()

        gsα = gradient(α) do
            val_loss = loss(model, val_batch...)
            return val_loss
        end
        losses[2] = val_loss
        foreach(CUDA.unsafe_free!, val_batch)
        CUDA.reclaim()
        GC.gc()
        gs = [gsα[x] for x in α]
        for (f,g) in zip(fakeg, gs)
            f .-= g
        end
        CUDA.reclaim()
        GC.gc()
        curracts = activationupdatesd(model)
        actupdate = [log.(curr./base) for (curr, base) in zip(curracts, baseacts)]
        for (f,a) in zip(fakea, actupdate)
            f .+= a
        end
        CUDA.reclaim()
        cbbatch()
    end

    #ac1 = activationupdatesd(model)
    p1 = heatmap(axlab, PRIMITIVES, hcat((fakea |> cpu)...), xrotation = 90, title = "Activation Standard Deviation")
    #ac2 = activationupdatema(model)
    #p2 = heatmap(axlab, PRIMITIVES, hcat((ac2 |> cpu)...), xrotation = 90, title = "Activation Mean of Absolute Value")
    #gs = [gsα[x] for x in α]
    p3 = heatmap(axlab, PRIMITIVES, hcat((fakeg |> cpu)...), xrotation = 90 , title = "Alpha Gradient Update")
    plot(p1,p3,layout = (2,1),size = (1200, 800))
    savefig(string("test/models/fig_gsvsacts", i, ".png"));

    i += 1
    cbepoch()
end
