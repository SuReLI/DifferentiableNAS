using Metalhead, Images
using Images.ImageCore
using Flux
using Flux.Optimise
using Base.Iterators
using StatsBase:mean
using Zygote
using LinearAlgebra
# using CUDA

struct Op

end

struct DARTSModel
  αs::Array{Float32, 2}
  #chains::Array{Op,2}
end

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
    Flux.params!(ps,chain)
  end
  return ps
end


function Flux.Optimise.train!(loss, model::DARTSModel, data, opt; cb = () -> ())
  function grad_loss(model, ps, verbose = false) #do I need batch here?
    gs = gradient(ps) do
      loss(model, batch...) #does model work here?
      return gs
    end

    w = all_params(model.chains)
    α = params(model.αs)
    cb = runall(cb)
    #@progress
    for batch in data
      gsα = grad_loss(model, α)
      if ξ != 0
        model_prime = deepcopy(model)
        model_prime_α = deepcopy(model_prime)
        w_prime = all_params(model_prime_α.chains)

        gsw_prime = grad_loss(model_prime, w_prime, loss)
        for a in all_params(model_prime.chains).order
          #println(" gsw_prime ", norm(gsw_prime[a]))
        end

        Flux.Optimise.update!(Descent(ξ), w_prime, gsw_prime)

        model_prime_minus = deepcopy(model_prime)

        w_prime_minus = all_params(model_prime_minus.model.chains)

        gsw_prime_minus = grad_loss(model_prime_minus, w_prime_minus)
        for a in params(model_prime_minus.model.chains).order
          #println(" gsw_prime_minus ", norm(gsw_prime_minus[a]))
        end

        model_minus = deepcopy(model)
        model_plus = deepcopy(model)

        w_minus = all_params(model_minus.chains)
        w_plus = all_params(model_plus.chains)

        epsilon = 0.01 ./ norm([gsw_prime_minus[w] for w in w_prime_minus.order if gsw_prime_minus[w] != nothing])
        update_all!(Descent(epsilon), w_minus, [gsw_prime_minus[w] for w in w_prime_minus.order])
        update_all!(Descent(-1*epsilon), w_plus, [gsw_prime_minus[w] for w in w_prime_minus.order])

        gsα_prime = grad_loss(model_prime_α, params(model_prime_α.weights))

        gsα_plus = grad_loss(model_plus, params(Q_plus.weights))
        gsα_minus = grad_loss(model_minus, params(Q_minus.weights))

        for a in params(model_prime_α.weights).order
          #println(a, " gsα_prime ", gsα_prime[a])
        end

        for a in params(model_plus.weights).order
          #println(gsα_plus[a])
        end

        update_all!(opt, α, [gsα_prime[a] for a in params(model_prime_α.weights).order])

        update_all!(Descent(-1*ξ^2/(2*epsilon)), α, [gsα_plus[a] for a in params(model_plus.weights).order])
        update_all!(Descent(ξ^2/(2*epsilon)), α, [gsα_minus[a] for a in params(model_minus.weights).order])
      else
        Flux.Optimise.update!(opt, α, gsα)
      end
      gsw = grad_loss(Q, w)
      Flux.Optimise.update!(opt, w, gsw)
      cb()
    end
  end
end
