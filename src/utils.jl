export squeeze, histories, all_αs, all_ws_sansbn, process_batch!, TrainCuIterator, EvalCuIterator
using Adapt
using CUDA

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

all_αs(model) = Flux.params([model.normal_αs, model.reduce_αs])
all_ws(model) = Flux.params([model.stem, model.cells..., model.global_pooling, model.classifier])

function all_ws_sansbn(model) #without batchnorm params
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

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] |> f32
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] |> f32

function process_batch!(batch::Array{Float32,4}, cutout::Int = -1)
	mean_im = repeat(reshape(CIFAR_MEAN, (1,1,3)), outer = [32,32,1])
	std_im = repeat(reshape(CIFAR_STD, (1,1,3)), outer = [32,32,1])
	for image in 1:size(batch,4)
		orig = copy(batch[:,:,:,image])
		flip = rand(Bool)
		shiftx = rand(-4:4)
		shifty = rand(-4:4)
		for x in 1:size(batch,1)
			for y in 1:size(batch,2)
				if minimum([x,y,x+shiftx,y+shifty]) >= 1 && maximum([x,y,x+shiftx,y+shifty]) <= size(batch,1)
					batch[x,y,:,image] .= orig[x+shiftx,y+shifty,:]
				end
			end
		end
		if shiftx > 0
			for x in size(batch,1)-shiftx+1:size(batch,1)
				batch[x,:,:,image] .= 0
			end
		elseif shiftx < 0
			for x in 1:-shiftx
				batch[x,:,:,image] .= 0
			end
		end
		if shifty > 0
			for y in size(batch,2)-shifty+1:size(batch,2)
				batch[:,y,:,image] .= 0
			end
		elseif shifty < 0
			for y in 1:-shifty
				batch[:,y,:,image] .= 0
			end
		end
		if flip
			mask = reverse(batch[:,:,:,image], dims=2)
			batch[:,:,:,image] .= mask
		end
		if cutout > 0
			cutx = rand(1:size(batch,1))
			cuty = rand(1:size(batch,2))
			minx = maximum([cutx-cutout÷2,1])
			maxx = minimum([cutx+cutout÷2,size(batch,1)])
			miny = maximum([cuty-cutout÷2,1])
			maxy = minimum([cuty+cutout÷2,size(batch,2)])
			batch[minx:maxx,miny:maxy,:,image] .= 0
		end
		batch[:,:,:,image] = (batch[:,:,:,image].-mean_im)./std_im
	end
end


mutable struct TrainCuIterator{B}
    batches::B
    previous::Any
    TrainCuIterator(batches) = new{typeof(batches)}(batches)
end
function Base.iterate(c::TrainCuIterator, state...)
    item = iterate(c.batches, state...)
    isdefined(c, :previous) && foreach(CUDA.unsafe_free!, c.previous)
    item === nothing && return nothing
    batch, next_state = item
	process_batch!(batch[1], -1)
    cubatch = map(x -> adapt(CuArray, x), batch)
    c.previous = cubatch
    return cubatch, next_state
end

mutable struct EvalCuIterator{B}
    batches::B
    previous::Any
    EvalCuIterator(batches) = new{typeof(batches)}(batches)
end
function Base.iterate(c::EvalCuIterator, state...)
    item = iterate(c.batches, state...)
    isdefined(c, :previous) && foreach(CUDA.unsafe_free!, c.previous)
    item === nothing && return nothing
    batch, next_state = item
	process_batch!(batch[1], 16)
    cubatch = map(x -> adapt(CuArray, x), batch)
    c.previous = cubatch
    return cubatch, next_state
end
