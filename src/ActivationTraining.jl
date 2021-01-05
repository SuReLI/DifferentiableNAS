export Activationtrain1st!, activationupdate

using Flux
using Flux: onehotbatch
using Juno
using Base.Iterators
using StatsBase
using Zygote
using Zygote: @nograd
using LinearAlgebra
using CUDA
include("utils.jl")
include("DARTSModel.jl")

function collectweights(model)
    for (i,cell) in enumerate(model.cells)
        #cell.reduction
        for (j,mixedop) in enumerate(cell.mixedops)
            for (k,op) in enumerate(mixedop.ops)
                for l in op.op
                    if typeof(l) <: Flux.Conv
                        #@show typeof(l.weight)
                    end
                end
                #@show [typeof(p) for p in Flux.params(op.op).params]
            end
        end
    end
end


connectstrings = vcat([[string(j,"-",i) for j = 1:i-1] for i = 3:6]...)
axlab = vcat([string("n",e) for e in connectstrings],[string("r",e) for e in connectstrings])

edgedict = Dict{String, Int64}()
for (i,strconn) in enumerate(connectstrings)
    edgedict[strconn] = i
end
primdict = Dict{String, Int64}()
for (i,prim) in enumerate(PRIMITIVES)
    primdict[prim] = i
end
function activationupdatema(model)
    update = all_αs(model).*0
    reducecells = Array{Int64, 1}()
    normalcells = Array{Int64, 1}()
    for (i,cell) in enumerate(model.cells)
        if cell.reduction
            push!(reducecells, i)
        else
            push!(normalcells, i)
        end
    end
    for key in keys(model.activations.currentacts)
        acts = model.activations.currentacts[key]
        if size(acts,2) == 3
            acts = acts[:,2:3,:]
        elseif size(acts,2) == 6
            acts = acts[:,[2,3,5,6],:]
        end
        reduce = acts[reducecells,:,:]
        normal = acts[normalcells,:,:]
        statreduce = x -> mean(abs.(x))
        #statreduce = x -> std(x)
        update[edgedict[key[1:3]]][primdict[key[5:length(key)]]] = statreduce(normal)
        update[edgedict[key[1:3]]+14][primdict[key[5:length(key)]]] = statreduce(reduce)
    end
    update
end
function activationupdatesd(model)
    update = all_αs(model).*0
    reducecells = Array{Int64, 1}()
    normalcells = Array{Int64, 1}()
    for (i,cell) in enumerate(model.cells)
        if cell.reduction
            push!(reducecells, i)
        else
            push!(normalcells, i)
        end
    end
    for key in keys(model.activations.currentacts)
        acts = model.activations.currentacts[key]
        if size(acts,2) == 3
            acts = acts[:,2:3,:]
        elseif size(acts,2) == 6
            acts = acts[:,[2,3,5,6],:]
        end
        reduce = acts[reducecells,:,:]
        normal = acts[normalcells,:,:]
        #statreduce = x -> mean(abs.(x))
        statreduce = x -> std(x)
        update[edgedict[key[1:3]]][primdict[key[5:length(key)]]] = statreduce(normal)
        update[edgedict[key[1:3]]+14][primdict[key[5:length(key)]]] = statreduce(reduce)
    end
    update
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

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

all_αs(model::DARTSModel) = Flux.params([model.normal_αs, model.reduce_αs])
all_ws(model::DARTSModel) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

i = 1

function Activationtrain1st!(loss, model, train, val, opt_α, opt_w; cbepoch = () -> (), cbbatch = () -> ())
    local train_loss
    local val_loss
    local gsα
    w = all_ws(model)
    α = all_αs(model)
    fake = all_αs(model).*0
    global i
    for (train_batch, val_batch) in zip(CuIterator(train), CuIterator(val))
        gsw = gradient(w) do
            train_loss = loss(model, train_batch...)
            return train_loss
        end
        foreach(CUDA.unsafe_free!, train_batch)
        Flux.Optimise.update!(opt_w, w, gsw)
        CUDA.reclaim()

        gsα = gradient(α) do
            val_loss = loss(model, val_batch...)
            return val_loss
        end
        foreach(CUDA.unsafe_free!, val_batch)
        gs = [gsα[x] for x in α]
        for (f,g) in zip(fake, gs)
            Flux.Optimise.update!(opt_α, f, g)
        end
        @show norm(fake), norm(α)
        CUDA.reclaim()
        cbbatch()
    end
    """
    val_batch = val[1] |> gpu
    gsα = gradient(α) do
        val_loss = loss(model, val_batch...)
        return val_loss
    end
    foreach(CUDA.unsafe_free!, val_batch)
    """
    ac1 = activationupdatesd(model)
    p1 = heatmap(axlab, PRIMITIVES, hcat((ac1 |> cpu)...), xrotation = 90, title = "Activation Standard Deviation")
    ac2 = activationupdatema(model)
    p2 = heatmap(axlab, PRIMITIVES, hcat((ac2 |> cpu)...), xrotation = 90, title = "Activation Mean of Absolute Value")
    #gs = [gsα[x] for x in α]
    p3 = heatmap(axlab, PRIMITIVES, hcat((fake |> cpu)...), xrotation = 90 , title = "Alpha Gradient Update")
    plot(p1,p2,p3,layout = (3,1),size = (1300, 1500))
    @show string(i, "fig_gsvsacts.png")
    savefig(string(i, "fig_gsvsacts.png"));
    i += 1
    cbepoch()
end
