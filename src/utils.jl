export squeeze, all_ws, all_αs

function squeeze(A::AbstractArray) #generalize this? move to utils?
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
