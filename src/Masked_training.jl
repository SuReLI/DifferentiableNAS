export DARTStrain1st!, DARTStrain2nd!, all_ws, all_αs

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

all_αs(model::DARTSModel) = params([model.normal_αs, model.reduce_αs])
all_ws(model::DARTSModel) = params([model.stem, model.cells..., model.global_pooling, model.classifier])

function Maskedtrain1st!(loss, model, train, val, opt; cb = () -> ())
    function grad_loss(model, ps, batch, verbose = false)
        gs = gradient(ps) do
            loss(model, batch...)
        end
    end

    w = all_ws(model)
    α = all_αs(model)

    #cb = runall(cb)
    #@progress
    for (train_batch, val_batch) in zip(train, val)
        v_gpu = val_batch |> gpu
        gsα = grad_loss(model, α, v_gpu)
        Flux.Optimise.update!(opt, α, gsα)
        #CUDA.unsafe_free!(v_gpu[1])

        t_gpu = train_batch |> gpu
        gsw = grad_loss(model, w, t_gpu)
        Flux.Optimise.update!(opt, w, gsw)
        #CUDA.unsafe_free!(t_gpu[1])
        cb()
    end
    cb()
end


function MaskedEval(model, test, accuracy; cb = () -> ())
    for batch in test:
        cb()
    end
end
