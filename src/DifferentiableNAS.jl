module DifferentiableNAS

using Flux
using Juno
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
# using CUDA

include("DARTSToyModel.jl")
include("DARTSModel.jl")
include("DARTSTraining.jl")
include("utils.jl")

end # module
