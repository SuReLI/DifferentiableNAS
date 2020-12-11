module DifferentiableNAS

using Flux
using Juno
using Base.Iterators
using StatsBase: mean
using Zygote
using LinearAlgebra
# using CUDA

include("DARTS_model.jl")
include("DARTS_training.jl")
include("Masked_training.jl")
include("utils.jl")

end # module
