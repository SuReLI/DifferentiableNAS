export LayeredOp, DARTSModel, DARTStrain!, squeeze

using Metalhead, Images
using Images.ImageCore
using Flux
using Flux.Optimise
using Base.Iterators
using StatsBase:mean
using Zygote
using LinearAlgebra
using Juno
# using CUDA

struct Op
  chain
end

(m::Op)(x) = m.chain(x)
Flux.@functor Op

struct LayeredOp
    weights::Array{Float32, 2}
    chains::Array{Op,2}
end

#add in skip connections?

function LayeredOp()
    l1 = [
        Op(Chain(
            x -> repeat(x, outer = [1, 1, 12, 1]),
            x ->
                x[1:2:end, 1:2:end, :, :] +
                x[2:2:end, 2:2:end, :, :] +
                x[1:2:end, 2:2:end, :, :] +
                x[2:2:end, 1:2:end, :, :],
            x -> relu.(x),
        )),
        Op(Chain(Conv((1, 1), 3 => 36, stride = (2, 2), pad = (0, 0)), x -> relu.(x))),
        Op(Chain(Conv((3, 3), 3 => 36, stride = (2, 2), pad = (1, 1)), x -> relu.(x))),
        Op(Chain(Conv((5, 5), 3 => 36, stride = (2, 2), pad = (2, 2)), x -> relu.(x))),
        Op(Chain(
            Conv((3, 3), 3 => 18, pad = (1, 1)),
            x -> relu.(x),
            Conv((3, 3), 18 => 36, stride = (2, 2), pad = (1, 1)),
            x -> relu.(x),
        )),
    ]
    l2 = [
        Op(Chain(
            x ->
                x[1:2:end, 1:2:end, :, :] +
                x[2:2:end, 2:2:end, :, :] +
                x[1:2:end, 2:2:end, :, :] +
                x[2:2:end, 1:2:end, :, :], #change to maxpool?
            x ->
                x[1:2:end, 1:2:end, :, :] +
                x[2:2:end, 2:2:end, :, :] +
                x[1:2:end, 2:2:end, :, :] +
                x[2:2:end, 1:2:end, :, :], #change to maxpool?
            x -> relu.(x),
        )),
        Op(Chain(Conv((1, 1), 36 => 36, stride = (4, 4), pad = (0, 0)), x -> relu.(x))),
        Op(Chain(Conv((3, 3), 36 => 36, stride = (4, 4), pad = (1, 1)), x -> relu.(x))),
        Op(Chain(Conv((5, 5), 36 => 36, stride = (4, 4), pad = (2, 2)), x -> relu.(x))),
        Op(Chain(
            Conv((3, 3), 36 => 36, pad = (1, 1)),
            x -> relu.(x),
            Conv((3, 3), 36 => 36, stride = (4, 4), pad = (1, 1)),
            x -> relu.(x),
        )),
    ]
    l3 = [
        Op(Chain(
            x ->
                x[1:2:end, 1:2:end, 1:2:end, :] +
                x[2:2:end, 2:2:end, 2:2:end, :] +
                x[1:2:end, 2:2:end, 1:2:end, :] +
                x[2:2:end, 1:2:end, 2:2:end, :], #change to maxpool?
            x ->
                x[1:2:end, 1:2:end, 1:2:end, :] +
                x[2:2:end, 2:2:end, 2:2:end, :] +
                x[1:2:end, 2:2:end, 1:2:end, :] +
                x[2:2:end, 1:2:end, 2:2:end, :], #change to maxpool?
            x -> cat(x, sum(x, dims = 3), dims = 3),
            x -> relu.(x),
        )),
        Op(Chain(Conv((1, 1), 36 => 10, stride = (4, 4), pad = (0, 0)), x -> relu.(x))),
        Op(Chain(Conv((3, 3), 36 => 10, stride = (4, 4), pad = (1, 1)), x -> relu.(x))),
        Op(Chain(Conv((5, 5), 36 => 10, stride = (4, 4), pad = (2, 2)), x -> relu.(x))),
        Op(Chain(
            Conv((3, 3), 36 => 24, pad = (1, 1)),
            x -> relu.(x),
            Conv((3, 3), 24 => 10, stride = (4, 4), pad = (1, 1)),
            x -> relu.(x),
        )),
    ]
    chains = [l1 l2 l3]
    weights = softmax(ones(Float32, size(chains)))
    LayeredOp(weights, chains)
end

function squeeze(A::AbstractArray)
    #print(A, " ", size(A))
    if ndims(A) == 3
        if size(A, 3) > 1
            return dropdims(A; dims = (1))
        elseif size(A, 3) == 1
            return dropdims(A; dims = (1,3))
        end
    elseif ndims(A) == 4
        if size(A, 4) > 1
            return dropdims(A; dims = (1,2))
        elseif size(A, 4) == 1
            return dropdims(A; dims = (1,2,4))
        end
    end
    return A
end

function get_size(A::AbstractArray)
    print("\tsize:", size(A), "\t")
    return A
end

function (m::LayeredOp)(x)
    for i = 1:size(m.weights,2)
        mpw = softmax(m.weights[:,i])
        x = sum([chain(x) for chain in m.chains[:,i]].*mpw)
    end
    squeeze(x)
end

Flux.@functor LayeredOp

struct DARTSModel
  αs::Array{Float32, 2}
  chains::Array{Op,2}
end

function DARTSModel()
  lo = LayeredOp()
  αs = lo.weights
  chains = lo.chains
  DARTSModel(αs, chains)
end

function (m::DARTSModel)(x)
  for i = 1:size(m.αs,2)
      mpw = softmax(m.αs[:,i])
      x = sum([chain(x) for chain in m.chains[:,i]].*mpw)
  end
  squeeze(x)
end

Flux.@functor DARTSModel

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

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

function DARTStrain!(loss, model, train, val, opt, order; cb = () -> ())
  function grad_loss(model, ps, batch, verbose = false) #do I need batch here?
    #println(loss(model, batch...))
    gs = gradient(ps) do
      loss(model, batch...) #does model work here?
    end
  end

  if order == "first"
      ξ = 0
  else
      ξ = opt.eta
  end
  w = all_params(model.chains)
  α = params(model.αs)
  #α = params(model.weights)
  cb = runall(cb)
  @progress for (train_batch, val_batch) in zip(train, val)
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
      update_all!(Descent(epsilon), w_minus, [gsw_prime_minus[w_] for w_ in w_prime_minus.order])
      update_all!(Descent(-1*epsilon), w_plus, [gsw_prime_minus[w_] for w_ in w_prime_minus.order])

      gsα_prime = grad_loss(model_prime_α, params(model_prime_α.αs), val_batch)

      gsα_plus = grad_loss(model_plus, params(model_plus.αs), train_batch)
      gsα_minus = grad_loss(model_minus, params(model_minus.αs), train_batch)

      update_all!(opt, α, [gsα_prime[a] for a in params(model_prime_α.αs).order])
      update_all!(Descent(-1*ξ^2/(2*epsilon)), α, [gsα_plus[a] for a in params(model_plus.αs).order])
      update_all!(Descent(ξ^2/(2*epsilon)), α, [gsα_minus[a] for a in params(model_minus.αs).order])
    else
      Flux.Optimise.update!(opt, α, gsα)
    end
    gsw = grad_loss(model, w, train_batch)
    Flux.Optimise.update!(opt, w, gsw)
    #cb()
  end
end
