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
include("DARTS_model.jl")
@nograd onehotbatch
@nograd softmax

function flatten_grads(grads)
    xs = Zygote.Buffer([])
    fmap(grads) do x
        x isa AbstractArray && push!(xs, x)
        #println("x ",x)
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
    row = rand(1:length(αs))
    inds = findall(αs[row] .> 0)
    perturbs = [copy(αs) for i in inds]
    for i in 1:length(inds)
        perturbs[i] = deepcopy(αs)
        perturbs[i][row][inds[i]] = 0.0
    end
    #display(inds)
    (row, inds, perturbs)
end

function Standardtrain1st!(accuracy, loss, model, train, val, opt; cb = () -> ())
    function grad_loss(model, ps, batch, verbose = false)
        gs = gradient(ps) do
            loss(model, batch...)
        end
    end

    w = all_ws(model)

    for train_batch in train
        t_gpu = train_batch |> gpu
        gsw = grad_loss(model, w, t_gpu)
        Flux.Optimise.update!(opt, w, gsw)
        cb()
    end
    cb()
end


function Maskedtrain1st!(accuracy, loss, model, train, val, opt; cb = () -> ())
    function grad_loss(model, ps, batch, verbose = false)
        gs = gradient(ps) do
            loss(model, batch...)
        end
    end

    w = all_ws(model)

    for train_batch in train
        t_gpu = train_batch |> gpu
        gsw = grad_loss(model, w, t_gpu)
        Flux.Optimise.update!(opt, w, gsw)
        cb()
    end

    row, inds, perturbs = perturb(model.normal_αs)
    vals = [accuracy(model, val, pert = pert) for pert in perturbs]
    model.normal_αs[row][findmax(vals)[2]] = 0
    display(row, model.normal_αs[row])

    cb()
end


function MaskedEval(model, test, accuracy; cb = () -> ())
    for batch in test
        cb()
    end
end