export squeeze, histories, all_αs, all_ws_sansbn

function squeeze(A::AbstractArray) #generalize this?
    if ndims(A) == 3
        if size(A, 3) > 1
            return dropdims(A; dims = (1))
        elseif size(A, 3) == 1
            return dropdims(A; dims = (1, 3))
        end
    elseif ndims(A) == 4
        if size(A, 4) > 1
            return dropdims(A; dims = (1, 2))
        elseif size(A, 4) == 1
            return dropdims(A; dims = (1, 2, 4))
        end
    end
    return A
end

function all_params(submodels)
    ps = Params()
    for submodel in submodels
        Flux.params!(ps, submodel)
    end
    return ps
end

all_αs(model::DARTSModel) = Flux.params([model.normal_αs, model.reduce_αs])

function all_ws_sansbn(model::DARTSModel) #without batchnorm params
    all_w = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])
    for (i,cell) in enumerate(model.cells)
        for (j,mixedop) in enumerate(cell.mixedops)
            for (k,op) in enumerate(mixedop.ops)
                for (l,layer) in enumerate(op.op)
                    if typeof(layer) <: Flux.BatchNorm
                        delete!(all_w, layer.γ)
                        delete!(all_w, layer.β)
                    end
                end
            end
        end
    end
    for (l,layer) in enumerate(model.stem.layers)
        if typeof(layer) <: Flux.BatchNorm
            delete!(all_w, layer.γ)
            delete!(all_w, layer.β)
        end
    end
    all_w
end
