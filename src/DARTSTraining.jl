export DARTStrain!, squeeze

using Flux
using Juno
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
# using CUDA

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

function all_params(chains)
    ps = Params()
    for chain in chains
        Flux.params!(ps, chain)
    end
    return ps
end

all_αs(model) = all_params([model.normal_αs, model.reduce_αs])

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

function DARTStrain!(loss, model, train, val, opt, order; cb = () -> ())
    function grad_loss(model, ps, batch, verbose = false)
        gs = gradient(ps) do
            loss(model, batch...)
        end
    end

    if order == "first"
        ξ = 0
    else
        ξ = opt.eta
    end
    w = all_params(model.chains)
    α = all_αs(model)

    cb = runall(cb)
    #@progress
    for (train_batch, val_batch) in zip(train, val)
        gsα = grad_loss(model, α, val_batch)
        if ξ != 0
            model_prime = deepcopy(model)
            model_prime_α = deepcopy(model_prime)
            w_prime = all_params(model_prime_α.chains)

            gsw_prime = grad_loss(model_prime, w_prime, val_batch)

            Flux.Optimise.update!(Descent(ξ), w_prime, gsw_prime)

            model_prime_minus = deepcopy(model_prime)

            w_prime_minus = all_params(model_prime_minus.chains)

            gsw_prime_minus = grad_loss(model_prime_minus, w_prime_minus, val_batch)

            model_minus = deepcopy(model)
            model_plus = deepcopy(model)

            w_minus = all_params(model_minus.chains)
            w_plus = all_params(model_plus.chains)

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
        else
            Flux.Optimise.update!(opt, α, gsα)
        end
        gsw = grad_loss(model, w, train_batch)
        Flux.Optimise.update!(opt, w, gsw)
        cb()
    end
end
