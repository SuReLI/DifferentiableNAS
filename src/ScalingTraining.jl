export Scalingtrain1st!, scalingupdate, collectscales

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
include("utils.jl")
include("DARTSModel.jl")

function collectscales(model)
    for (i,cell) in enumerate(m.cells)
        #cell.reduction
        for (j,mixedop) in enumerate(cell.mixedops)
            for (k,op) in enumerate(mixedop.ops)
                for (l,layer) in enumerate(op.op)
                    if typeof(layer) <: Flux.BatchNorm
                        @show (i,j,k,l,layer.λ)
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

function scalingupdate(model)
    reducecells = Array{Int64, 1}()
    normalcells = Array{Int64, 1}()
    scales = Array{Float32, 3}(undef, length(model.cells), length(model.cells[1].mixedops), length(model.cells[1].mixedops[1].ops)) |> gpu#cell id x mixop id x op id
    for (i,cell) in enumerate(model.cells)
        if cell.reduction
            push!(reducecells, i)
        else
            push!(normalcells, i)
        end
        for (j,mixedop) in enumerate(cell.mixedops)
            for (k,op) in enumerate(mixedop.ops)
                scales[i,j,k] = 0.0
                for (l,layer) in enumerate(op.op)
                    if typeof(layer) <: Flux.BatchNorm
                        #@show(k, mean(layer.γ .- 1.0))
                        scales[i,j,k] = mean(layer.γ .- 1.0)  #automatically gets only last BatchNorm in op
                    end
                end
            end
        end
    end
    reduce = scales[reducecells,:,:]
    normal = scales[normalcells,:,:]
    display(vcat(dropdims(mean(normal,dims=1),dims=1),dropdims(mean(reduce,dims=1),dims=1)))
    vcat(dropdims(mean(normal,dims=1),dims=1),dropdims(mean(reduce,dims=1),dims=1))'
end

function scalingreg(model)
    reducecells = Array{Int64, 1}()
    normalcells = Array{Int64, 1}()
    collectscales = 0.0
    for (i,cell) in enumerate(model.cells)
        for (j,mixedop) in enumerate(cell.mixedops)
            for (k,op) in enumerate(mixedop.ops)
                for (l,layer) in enumerate(op.op)
                    if typeof(layer) <: Flux.BatchNorm
                        collectscales += abs(mean(layer.γ))
                    end
                end
            end
        end
    end
    collectscales
end

all_ws(model::DARTSModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

i = 1
function Scalingtrain1st!(loss, model, train, val, opt_α, opt_w, λ=0.0001, losses=[0.0,0.0]; cbepoch = () -> (), cbbatch = () -> ())
    local train_loss
    local val_loss
    local gsα
    w = all_ws(model)
    α = all_αs(model)
    fakeg = all_αs(model).*0
    fakea = all_αs(model).*0
    global i
    for (train_batch, val_batch) in zip(CuIterator(train), CuIterator(val))
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...) + λ*scalingreg(model)#sum([sum(abs, ss) for ss in scalingupdate(model)])
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
        cbbatch()
    end
    p1 = heatmap(axlab, PRIMITIVES, scalingupdate(model)|>cpu, xrotation = 90, title = "Mean Scaling Delta")
    #ac2 = activationupdatema(model)
    #p2 = heatmap(axlab, PRIMITIVES, hcat((ac2 |> cpu)...), xrotation = 90, title = "Activation Mean of Absolute Value")
    #gs = [gsα[x] for x in α]
    p3 = heatmap(axlab, PRIMITIVES, hcat((fakeg |> cpu)...), xrotation = 90 , title = "Alpha Gradient Update")
    plot(p1,p3,layout = (2,1),size = (1200, 800))
    savefig(string("test/models/fig_scale", i, ".png"));

    i += 1
    cbepoch()
end
