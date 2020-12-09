using Flux

struct InterconnectedModel
    out_states::Int64
    steps::Int64
    prelayer
    nodes
end

Flux.@functor InterconnectedModel

ConvBNReLU(channels_in, channels_out) = Chain(
    Conv((3,3), channels_in => channels_out, pad = (1,1)),
    BatchNorm(channels_out),
    x -> relu.(x),
)

function InterconnectedModel(steps::Int64, channels_in::Int64, channels_rest::Int64, out_states::Int64)
    prelayer = ConvBNReLU(channels_in, channels_rest)
    nodes =[ConvBNReLU(channels_rest, channels_rest) for step in 1:(steps*(steps+1))÷2]
    InterconnectedModel(out_states, steps, prelayer, nodes)
end


function (model::InterconnectedModel)(x)
    #go through the default first layer
    state = model.prelayer(x)
    states_size = (size(state,1), size(state,2), (model.steps+1)*size(state,3), size(state,4))
    states = Array{Float32,4}(undef,states_size...)
    states_indices = CartesianIndices((1:size(state,1),1:size(state,2),1:size(state,3),1:size(state,4)))
    copyto!(states,states_indices,state,CartesianIndices(size(state)))
    chunk = size(state,3)

    #for the remaining steps, apply the respective layer to each prior state and sum to get current state
    offset = 1
    for step in 1:model.steps
        #outs = [model.nodes[offset+i](states[:,:,i*chunk+1:(i+1)*chunk,:]) for i in 0:step-1]
        #state = sum(outs)
        #states_indices = CartesianIndices((1:size(state,1),1:size(state,2),step*chunk+1:(step+1)*chunk,1:size(state,4)))
        #copyto!(states, states_indices, state, CartesianIndices(size(state))) #This line causes "Mutating arrays is not supported" error
        states[:,:,step*chunk+1:(step+1)*chunk,:] = sum([model.nodes[offset+i](states[:,:,i*chunk+1:(i+1)*chunk,:]) for i in 0:step-1])
        offset += step
    end

    #concatenate the states we want and return
    states_indices = CartesianIndices((1:size(states,1),1:size(states,2),(size(states,3)-model.out_states*chunk)+1:size(states,3),1:size(states,4)))
    states[states_indices]
end

model = InterconnectedModel(5,3,8,3)
model(rand(Float32,8,8,3,2))
gradient(x -> sum(model(x)), rand(Float32,8,8,3,2))
