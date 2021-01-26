export squeeze,
    histories,
    all_αs,
    all_ws_sansbn,
    flip_batch!,
    shift_batch!,
	cutout_batch!,
	norm_batch!,
    TrainCuIterator,
    EvalCuIterator,
    TestCuIterator,
    CosineAnnealing,
	apply!
using Adapt
using CUDA
using Flux
import Flux.Optimise.apply!

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

function flip_batch!(batch::Array{Float32,4})
	flips = falses(size(batch,4))
	for image in 1:size(batch,4)
		orig = copy(batch[:,:,:,image])
		flip = rand(Bool)
		if flip
			flipped = reverse(orig, dims=2)
			batch[:,:,:,image] = flipped
		end
		flips[image] = flip
	end
	flips
end

function shift_batch!(batch::Array{Float32,4})
	shifts = Array{Int64}(undef,size(batch,4),2)
	for image in 1:size(batch,4)
		orig = copy(batch[:,:,:,image])
		shiftx = rand(-4:4)
		shifty = rand(-4:4)
		if shiftx > 0
			batch[1:size(batch,1)-shiftx,:,:,image] = orig[shiftx+1:size(batch,1),:,:]
			batch[size(batch,1)-shiftx+1:size(batch,1),:,:,image] .= 0
		elseif shiftx < 0
			batch[1:-shiftx,:,:,image] .= 0
			batch[1-shiftx:size(batch,1),:,:,image] = orig[1:size(batch,1)+shiftx,:,:]
		end
		orig = copy(batch[:,:,:,image])
		if shifty > 0
			batch[:,1:size(batch,2)-shifty,:,image] = orig[:,shifty+1:size(batch,2),:]
			batch[:,size(batch,2)-shifty+1:size(batch,2),:,image] .= 0
		elseif shifty < 0
			batch[:,1:-shifty,:,image] .= 0
			batch[:,1-shifty:size(batch,2),:,image] = orig[:,1:size(batch,2)+shifty,:]
		end
		shifts[image,:] = [shiftx;shifty]
	end
	shifts
end

function cutout_batch!(batch::Array{Float32,4}, cutout::Int = -1)
	cutouts = Array{Int64}(undef,size(batch,4),2)
	if cutout > 0
		for image in 1:size(batch,4)
			cutx = rand(1:size(batch,1))
			cuty = rand(1:size(batch,2))
			minx = maximum([cutx-cutout÷2,1])
			maxx = minimum([cutx+cutout÷2-1,size(batch,1)])
			miny = maximum([cuty-cutout÷2,1])
			maxy = minimum([cuty+cutout÷2-1,size(batch,2)])
			batch[minx:maxx,miny:maxy,:,image] .= 0
			cutouts[image,:] = [cutx;cuty]
		end
	end
end

function norm_batch!(batch::Array{Float32,4})
	mean_im = repeat(reshape(CIFAR_MEAN, (1,1,3,1)), outer = [32,32,1,size(batch,4)])
	std_im = repeat(reshape(CIFAR_STD, (1,1,3,1)), outer = [32,32,1,size(batch,4)])
	batch = (batch.-mean_im)./std_im
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
	flip_batch!(batch[1])
	shift_batch!(batch[1])
	norm_batch!(batch[1])
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
	flip_batch!(batch[1])
	shift_batch!(batch[1])
	norm_batch!(batch[1])
	cutout_batch!(batch[1], 16)
    cubatch = map(x -> adapt(CuArray, x), batch)
    c.previous = cubatch
    return cubatch, next_state
end

mutable struct TestCuIterator{B}
    batches::B
    previous::Any
    TestCuIterator(batches) = new{typeof(batches)}(batches)
end
function Base.iterate(c::TestCuIterator, state...)
    item = iterate(c.batches, state...)
    isdefined(c, :previous) && foreach(CUDA.unsafe_free!, c.previous)
    item === nothing && return nothing
    batch, next_state = item
	process_test_batch!(batch[1])
	norm_batch!(batch[1])
    cubatch = map(x -> adapt(CuArray, x), batch)
    c.previous = cubatch
    return cubatch, next_state
end


mutable struct CosineAnnealing
  tmax::Int64
  t::Int64
end

CosineAnnealing(tmax::Int64 = 1) = CosineAnnealing(tmax, 0)

function Flux.Optimise.apply!(o::CosineAnnealing, x, Δ)
  tmax = o.tmax
  t = o.t
  Δ .*= (1 + cos(t/tmax*pi))/2
  return Δ
end
