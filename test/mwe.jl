using Flux
using Zygote


ReLUConv(channels_in, channels_out, kernel_size, pad) =
    Chain(x -> relu.(x), Conv(kernel_size, channels_in => channels_out, pad = pad))


struct MixedOperation
    operations::AbstractArray
end

MixedOperation(channels::Int64, kernel_options::AbstractArray) =
    MixedOperation([ReLUConv(channels, channels, (i, i), i ÷ 2) for i in kernel_options])

function (m::MixedOperation)(x::AbstractArray, αs::AbstractArray)
    mapreduce((op, α) -> α * op(x), +, m.operations, αs)
end

Flux.@functor MixedOperation


struct MWECell
    steps::Int64
    outstates::Int64
    mixedops::Array{MixedOperation,1}
end

function MWECell(
    steps::Int64,
    outstates::Int64,
    channels::Int64,
    kernel_options::AbstractArray,
)
    mixedops = [MixedOperation(channels, kernel_options) for _ = 1:steps]
    MWECell(steps, outstates, mixedops)
end

function (m::MWECell)(x::AbstractArray, all_αs::AbstractArray)
    state1 = m.mixedops[1](x, all_αs[1, :])
    states = Zygote.Buffer([state1], m.steps)
    states[1] = state1
    for step = 2:m.steps
        states[step] = m.mixedops[step](states[step-1], all_αs[step,:]) #GPU version errors out here
    end
    states_ = copy(states)
    out = cat(states_[m.steps-m.outstates+1:m.steps]..., dims = 3)
    out
end

Flux.@functor MWECell


struct MWEModel
    all_αs::AbstractArray
    cells::Array{MWECell,1}
end

function MWEModel(
    kernel_options::AbstractArray;
    num_cells = 3,
    channels = 3,
    steps = 4,
    outstates = 2,
)
    cells = [
        MWECell(steps, outstates, channels * outstates^index, kernel_options)
        for index = 0:num_cells-1
    ]
    all_αs = rand(Float32, steps, length(kernel_options)) #offending data structure
    MWEModel(all_αs, cells)
end

function (m::MWEModel)(x::AbstractArray)
    state = x
    αs = softmax(m.all_αs, dims = 2)
    for cell in m.cells
        state = cell(state, αs)
    end
    state
end

Flux.@functor MWEModel


using Test
using CUDA

m = MWEModel([1, 3, 5]) |> gpu
test_image = rand(Float32, 32, 32, 3, 1) |> gpu
@test sum(m(test_image)) != 0
grad = gradient(x -> sum(m(x)), test_image)

loss(m, x) = sum(m(x))
gαs = gradient(params(m.all_αs)) do
    sum(m(test_image))
end
for αs in params(m.all_αs)
    @test gαs[αs] != Nothing
end
gws = gradient(params(m.cells)) do
    sum(m(test_image))
end
for ws in params(m.cells)
    @test gws[ws] != Nothing
end
