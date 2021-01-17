export ADMMtrain1st!, euclidmap, regterm, ADMMaux

using Flux
using Flux: onehotbatch
using Juno
using Base.Iterators
using StatsBase: mean
using Zygote
using Zygote: @nograd
using LinearAlgebra
using CUDA
include("utils.jl")
include("DARTSModel.jl")

function euclidmap(aus, cardinality)
    if cardinality == -1 #full DARTS discretization
        for i in 1:size(aus,1)
            #aus[i][1] = 0 #ensure we don't choose none
            to_mask = sortperm(aus[i])[1:length(aus[i])-1]
            aus[i][to_mask] .= 0
        end
        i = 1
        for r in 1:4
            maxes = maximum(aus[i:i+r][:], dims = 2)
            keep_rows = sortperm(maxes, rev=true)[1:2]
            for k in 1:r+1
                if !(k in keep_rows)
                    aus[i+k-1][:] .= 0
                end
            end
            i += r+1
        end
    else
        for i in 1:size(aus,1)
            to_mask = sortperm(aus[i])[1:length(aus[i])-cardinality]
            aus[i][to_mask] .= 0
        end
    end
    aus
end

function collect_αs(model)
    vcat([exp.(n) for n in model.normal_αs], [exp.(n) for n in model.reduce_αs])
end
function regterm(m::DARTSModel, zs, us)
    as = collect_αs(m)
    out = 0.0
    for (a, z, u) in zip(as, zs, us)
        out += sum(abs2,a - z + u)
    end
    out
end

mutable struct ADMMaux
    zs::AbstractArray
    us::AbstractArray
end

function ADMMtrain1st!(loss, model, train, val, opt_w, opt_α, zu, ρ=1e-3, losses=[0.0,0.0], epoch = 1; cbepoch = () -> (), cbbatch = () -> ())
    zs = zu.zs
    us = zu.us
    w = all_ws_sansbn(model)
    α = all_αs(model)
    local train_loss
    local val_loss
    @show admmupdate = length(train)÷epoch
    @show disc = length(zs[1])-epoch÷3-1 #hyperparam
    if disc < 1
        disc = -1
    end
    for (i, train_batch, val_batch) in zip(1:length(train), CuIterator(train), CuIterator(val))
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...)
            return train_loss
        end
        losses[1] = train_loss
        foreach(CUDA.unsafe_free!, train_batch)
        Flux.Optimise.update!(opt_w, w, gsw)
        CUDA.reclaim()
        gsα = gradient(α) do
            val_loss = loss(model, val_batch...) + ρ/2*regterm(model, zs, us)
            return val_loss
        end
        losses[2] = val_loss
        foreach(CUDA.unsafe_free!, val_batch)
        Flux.Optimise.update!(opt_α, α, gsα)
        CUDA.reclaim()
        if i%admmupdate == 0
            as = collect_αs(model)
            display(as)
            zs = euclidmap(as+us, disc)
            display(zs)
            us += as - zs
            display(us)
        end
        cbbatch()
    end
    cbepoch()
    zu.zs = zs
    zu.us = us
end
