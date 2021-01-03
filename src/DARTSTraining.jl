export DARTStrain1st!, DARTStrain2nd!, DARTSevaltrain1st!, all_ws, all_αs

using Flux
using Flux: onehotbatch
using Juno
using Base.Iterators
using StatsBase: mean
using Zygote
using Zygote: @nograd
using LinearAlgebra
using CUDA
#using TensorBoardLogger
using Logging
include("utils.jl")
include("DARTSModel.jl")

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

all_αs(model::DARTSModel) = Flux.params([model.normal_αs, model.reduce_αs])
all_ws(model::DARTSModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

function DARTStrain1st!(loss, model, train, val, opt_α, opt_w; cbepoch = () -> (), cbbatch = () -> ())
    local train_loss
    local val_loss
    w = all_ws(model)
    α = all_αs(model)
    for (train_batch, val_batch) in zip(CuIterator(train), CuIterator(val))
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...)
            return train_loss
        end
        foreach(CUDA.unsafe_free!, train_batch)
        Flux.Optimise.update!(opt_w, w, gsw)
        CUDA.reclaim()
        if length(model.activations.currentacts) > 0
            @show model.activations.currentacts["5-6-dil_conv_5x5"]
        end
        gsα = gradient(α) do
            val_loss = loss(model, val_batch...)
            return val_loss
        end
        if length(model.activations.currentacts) > 0
            @show model.activations.currentacts["5-6-dil_conv_5x5"]
        end
        foreach(CUDA.unsafe_free!, val_batch)
        Flux.Optimise.update!(opt_α, α, gsα)
        CUDA.reclaim()
        cbbatch()
    end
    cbepoch()
end


function DARTStrain2nd!(loss, model, train, val, opt; cb = () -> ())
    function grad_loss(model, ps, batch, verbose = false)
        gs = gradient(ps) do
            loss(model, batch...)
        end
    end

    ξ = opt.eta

    w = all_ws(model)
    α = all_αs(model)

    for (train_batch, val_batch) in zip(CuIterator(train), CuIterator(val))
        gsα = grad_loss(model, α, val_batch)
        model_prime = deepcopy(model)
        model_prime_α = deepcopy(model_prime)
        w_prime = all_ws(model_prime_α)

        gsw_prime = grad_loss(model_prime, w_prime, val_batch)

        Flux.Optimise.update!(Descent(ξ), w_prime, gsw_prime)

        model_prime_minus = deepcopy(model_prime)

        w_prime_minus = all_ws(model_prime_minus)

        gsw_prime_minus = grad_loss(model_prime_minus, w_prime_minus, val_batch)

        model_minus = deepcopy(model)
        model_plus = deepcopy(model)

        w_minus = all_ws(model_minus)
        w_plus = all_ws(model_plus)

        epsilon = 0.01 ./ norm([gsw_prime_minus[w_] for w_ in w_prime_minus.order])
        update_all!(
            Descent(epsilon),
            w_minus,
            [gsw_prime_minus[w_] for w_ in w_prime_minus.order],
        )
        update_all!(
            Descent(-1 * epsilon),
            w_plus,
            [gsw_prime_minus[w_] for w_ in w_prime_minus.order],
        )

        gsα_prime = grad_loss(model_prime_α, all_αs(model_prime_α), val_batch)

        gsα_plus = grad_loss(model_plus, all_αs(model_plus), train_batch)
        gsα_minus = grad_loss(model_minus, all_αs(model_minus), train_batch)

        update_all!(opt, α, [gsα_prime[a] for a in all_αs(model_prime_α).order])
        update_all!(
            Descent(-1 * ξ^2 / (2 * epsilon)),
            α,
            [gsα_plus[a] for a in all_αs(model_plus).order],
        )
        update_all!(
            Descent(ξ^2 / (2 * epsilon)),
            α,
            [gsα_minus[a] for a in all_αs(model_minus).order],
        )

        gsw = grad_loss(model, w, train_batch)
        Flux.Optimise.update!(opt, w, gsw)

        cb()
    end
end

all_ws(model::DARTSEvalModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

function DARTSevaltrain1st!(loss, model, train, opt_w; cbepoch = () -> ())
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
        Flux.Optimise.update!(opt_w, w, gsw)
        CUDA.reclaim()
        GC.gc()
    end
    CUDA.reclaim()
    GC.gc()
    cbepoch()
    CUDA.reclaim()
    GC.gc()
end
