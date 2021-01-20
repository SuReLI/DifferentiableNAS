export Maskedtrain1st!, Standardtrain1st!

using Flux
using Flux: onehotbatch
using Juno
using Base.Iterators
using StatsBase: mean
using Zygote
using Zygote: @nograd
using LinearAlgebra
using CUDA
#include("utils.jl")
#include("DARTSModel.jl")
@nograd onehotbatch
@nograd softmax

function perturb(αs::AbstractArray)
    counter = [ones(length(a)) for a in αs]
    while sum(sum(counter)) > 0
        rn = rand([i for i in 1:length(counter) if sum(counter[i]) > 0])
        row = rand([i for i in 1:length(counter[rn]) if counter[rn][i] == 1])
        inds = findall(softmax(αs[rn][row]) .> 0)
        if length(inds) <= 1
            counter[rn][row] = 0
        else
            perturbs = [copy(αs) for i in inds]
            for i in 1:length(inds)
                perturbs[i] = deepcopy(αs)
                perturbs[i][rn][row][inds[i]] = -Inf32
            end
            return (rn, row, inds, perturbs)
        end
    end
    return (-1, -1, [], [])
end

function Standardtrain1st!(accuracy, loss, model, train, opt, losses=[0.0,0.0]; cbepoch = () -> (), cbbatch = () -> ())
    local train_loss
    w = all_ws_sansbn(model)
    for train_batch in TrainCuIterator(train)
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...)
            return train_loss
        end
        losses[1] = train_loss
        CUDA.reclaim()
        GC.gc()
        Flux.Optimise.update!(opt, w, gsw)
        cbbatch()
    end
    cbepoch()
end

function Maskedtrain1st!(accuracy, loss, model, train, val, opt, losses=[0.0,0.0]; cbepoch = () -> (), cbbatch = () -> ())
    w = all_ws_sansbn(model)
    for _ in 1:1
        rn, row, inds, perturbs = perturb([model.normal_αs, model.reduce_αs])
        if rn == -1
            continue
        end
        vals = [accuracy(model, val, pert = pert) for pert in perturbs]
        to_remove = sortperm(vals)[1:length(vals)-1]
        if rn == 1
            model.normal_αs[row][to_remove] .= -Inf32
            display((rn, row, softmax(model.normal_αs[row])))
        else
            model.reduce_αs[row][to_remove] .= -Inf32
            display((rn. row, softmax(model.reduce_αs[row])))
        end
    end
    cbbatch()
    for train_batch in CuIterator(train)
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...)
            return train_loss
        end
        losses[1] = train_loss
        CUDA.reclaim()
        GC.gc()
        Flux.Optimise.update!(opt, w, gsw)
        cbbatch()
    end

    cbepoch()
end
