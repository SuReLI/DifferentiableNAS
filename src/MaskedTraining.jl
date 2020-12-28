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
include("utils.jl")
include("DARTSModel.jl")
@nograd onehotbatch
@nograd softmax

function flatten_grads(grads)
    xs = Zygote.Buffer([])
    fmap(grads) do x
        x isa AbstractArray && push!(xs, x)
        return x
    end
    flat_gs = vcat(vec.(copy(xs))...)
end

function update_all!(opt, ps::Params, gs)
    for (p, g) in zip(ps, gs)
        g == nothing && continue
        Flux.Optimise.update!(opt, p, g)
    end
end

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

all_ws(model::DARTSModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

function perturb(αs::AbstractArray)
    rn = rand(1:2)
    row = rand(1:length(αs[rn]))
    inds = findall(softmax(αs[rn][row]) .> 0)
    perturbs = [copy(αs) for i in inds]
    for i in 1:length(inds)
        perturbs[i] = deepcopy(αs)
        perturbs[i][rn][row][inds[i]] = -Inf32
    end
    (rn, row, inds, perturbs)
end

function Standardtrain1st!(accuracy, loss, model, train, val, opt; cbepoch = () -> (), cbbatch = () -> ())
    function grad_loss(model, ps, batch, verbose = false)
        gs = gradient(ps) do
            loss(model, batch...)
        end
    end

    w = all_ws(model)

    for train_batch in CuIterator(train)
        gsw = grad_loss(model, w, train_batch)
        CUDA.reclaim()
        GC.gc()
        Flux.Optimise.update!(opt, w, gsw)
        cbbatch()
    end
    cbepoch()
end

function Maskedtrain1st!(accuracy, loss, model, train, val, opt; cbepoch = () -> (), cbbatch = () -> ())
    function grad_loss(model, ps, batch, verbose = false)
        gs = gradient(ps) do
            loss(model, batch...)
        end
    end

    w = all_ws(model)

    for _ in 1:14
        rn, row, inds, perturbs = perturb([model.normal_αs, model.reduce_αs])
        vals = [accuracy(model, val, pert = pert) for pert in perturbs]
        if rn == 1
            model.normal_αs[row][findmax(vals)[2]] = -Inf32
            display((row, softmax(model.normal_αs[row])))
        else
            model.reduce_αs[row][findmax(vals)[2]] = -Inf32
            display((row, softmax(model.reduce_αs[row])))
        end
    end
    cbbatch()
    for train_batch in CuIterator(train)
        gsw = grad_loss(model, w, train_batch)
        CUDA.reclaim()
        GC.gc()
        Flux.Optimise.update!(opt, w, gsw)
        cbbatch()
    end

    cbepoch()
end
