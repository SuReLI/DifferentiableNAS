export ScalingADMMtrain1st!

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

function scalingparams(model)
    reducecells = Array{Bool, 1}(undef, length(model.cells))
    #scales = ones(Float32, length(model.cells), length(model.cells[1].mixedops), length(model.cells[1].mixedops[1].ops)) |> gpu#cell id x mixop id x op id
    rscales = Zygote.Buffer([], Float32, 2, length(model.cells[1].mixedops), length(model.cells[1].mixedops[1].ops))
    nscales = Zygote.Buffer([], Float32, length(model.cells)-2, length(model.cells[1].mixedops), length(model.cells[1].mixedops[1].ops))
    n = 1
    r = 1
    for (i,cell) in enumerate(model.cells)
        for (j,mixedop) in enumerate(cell.mixedops)
            for (k,op) in enumerate(mixedop.ops)
                if cell.reduction
                    rscales[r,j,k] = 1
                else
                    nscales[n,j,k] = 1
                end
                for (l,layer) in enumerate(op.op)
                    if typeof(layer) <: Flux.BatchNorm
                        if cell.reduction
                            rscales[r,j,k] = mean(layer.γ)  #automatically gets only last BatchNorm in op
                        else
                            nscales[n,j,k] = mean(layer.γ)
                        end
                    end
                end
            end
        end
        if cell.reduction
            r += 1
        else
            n += 1
        end
    end
    reduce = copy(nscales) |> gpu
    normal = copy(rscales) |> gpu
    vcat(dropdims(mean(normal,dims=1),dims=1),dropdims(mean(reduce,dims=1),dims=1))
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

collecteachrow(x) = collect(eachrow(x))

@adjoint function collecteachrow(x)
    collecteachrow(x), dy -> begin
        dx = 0*similar(x) # _zero is not in ZygoteRules, TODO
        foreach(copyto!, collecteachrow(dx), dy)
        (dx,)
    end
end

function euclidmaps(aus, cardinality)
    for i in 1:size(aus,1)
        sp = copy(aus[i,:])
        to_mask = sortperm(sp)[1:size(aus,2)-cardinality]
        for j in to_mask
            aus[i,j] = 0
        end
    end
    aus
end

function regterms(m, zs, us)
    gs = scalingparams(m)
    out = 0.0
    for (g, z, u) in zip(collecteachrow(gs), collecteachrow(zs), collecteachrow(us))
        out += sum(abs2,g - z + u)
    end
    out
end

trainable(bn::BatchNorm) = (bn.β, bn.γ)
all_ws(model::DARTSModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

function ScalingADMMtrain1st!(loss, model, train, opt_w, zs, us, ρ=1e-3, losses=[0.0,0.0]; cbepoch = () -> (), cbbatch = () -> ())
    w = all_ws(model)
    for (i, train_batch) in zip(1:length(train), TrainCuIterator(train))
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...) + ρ/2*regterms(model, zs, us)
            return train_loss
        end
        losses[1] = train_loss
        foreach(CUDA.unsafe_free!, train_batch)
        Flux.Optimise.update!(opt_w, w, gsw)
        CUDA.reclaim()
        GC.gc()
        if i%10 == 0
            gs = scalingparams(model)
            display(gs)
            zs = euclidmap(gs+us, 1)
            display(zs)
            us += gs - zs
            display(us)
        end
        cbbatch()
    end
    cbepoch()
end
