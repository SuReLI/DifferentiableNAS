export DARTStrain1st!, DARTStrain2nd!, DARTSevaltrain1st!, all_ws_sansbn, all_αs

using Flux
using Flux: onehotbatch, onecold
using Juno
using Base.Iterators
using StatsBase: mean
using Zygote
using Zygote: @nograd
using LinearAlgebra
using CUDA


function accuracy(m, x, y)
    mx = m(x)
    showmx = mx |>cpu
    showy = y|>cpu
    for i in 1:size(showmx,2)
        @show collect(zip(softmax(showmx[:,i]), showy[:,i]))
    end
    mean(onecold(mx, 1:10)|>cpu .== onecold(y|>cpu, 1:10))
end



function DARTStrain1st!(loss, model, train, val, opt_α, opt_w, losses=[0f0,0f0], epoch = 1; cbepoch = () -> (), cbbatch = () -> ())
    local train_loss
    local val_loss
    w = all_ws_sansbn(model)
    α = all_αs(model)
    opt_w.os[2].t = epoch - 1
    for (train_batch, val_batch) in zip(TrainCuIterator(train), TrainCuIterator(val))
        if epoch == 29 || epoch == 30
            @show accuracy(model, train_batch...)
            @show accuracy(model, val_batch...)
        end
        display("grad weights")
        @time begin
            gsw = gradient(w) do
                train_loss = loss(model, train_batch...)
                return train_loss
            end
            losses[1] = train_loss
        end
        display("UnGPU val")
        @time foreach(CUDA.unsafe_free!, train_batch)
        display("Update weights")
        @time Flux.Optimise.update!(opt_w, w, gsw)
        display("Reclaim")
        @time CUDA.reclaim()
        display("grad alpha")
        @time begin
            gsα = gradient(α) do
                val_loss = loss(model, val_batch...)
                return val_loss
            end
            losses[2] = val_loss
            if epoch == 29 || epoch == 30
                @show losses
            end
        end
        display("UnGPU val")
        @time foreach(CUDA.unsafe_free!, val_batch)
        display("Update alpha")
        @time Flux.Optimise.update!(opt_α, α, gsα)
        display("Reclaim")
        @time CUDA.reclaim()
        display("cbbatch")
        @time cbbatch()
    end
    cbepoch()
end

all_ws(model::DARTSEvalModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

function DARTSevaltrain1st!(loss, model, train, opt_w, losses=[0f0,0f0], epoch = 1; cbepoch = () -> (), cbbatch = () -> ())
    w = all_ws(model)
    local train_loss
    opt_w.os[2].t = epoch - 1
    for train_batch in EvalCuIterator(train)
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...)
            return train_loss
        end
        losses[1] = train_loss
        CUDA.reclaim()
        GC.gc()
        Flux.Optimise.update!(opt_w, w, gsw)
        cbbatch()
    end
    cbepoch()
end

function update_all!(opt, ps::Params, gs)
    for (p, g) in zip(ps, gs)
        g == nothing && continue
        Flux.Optimise.update!(opt, p, g)
    end
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
