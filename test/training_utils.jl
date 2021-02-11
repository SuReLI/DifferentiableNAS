export apply!

using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase: mean
using Parameters
using CUDA
using Distributions
using BSON
using Dates
using Zygote
using ArgParse

function parse_commandline()
    gpumem = CUDA.totalmem(collect(CUDA.devices())[1])/(1024^3)
    if gpumem < 2.0
        batchsize_ = 4
        num_cells_ = 4
        channels_ = 4
        trainval_fraction_ = 2f-3
    elseif gpumem < 12.0
        batchsize_ = 32
        num_cells_ = 8
        channels_ = 16
        trainval_fraction_ = 1f0
    elseif gpumem < 16.0
        batchsize_ = 64
        num_cells_ = 8
        channels_ = 16
        trainval_fraction_ = 1f0
    else
        batchsize_ = 128
        num_cells_ = 8
        channels_ = 16
        trainval_fraction_ = 1f0
    end
    global batchsize_
    global num_cells_
    global channels_
    global trainval_fraction_

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--epochs"
            help = "number of epochs"
            arg_type = Int
            default = 50
        "--random_seed"
            help = "random seed (-1 for no random seed)"
            arg_type = Int
            default = 32
        "--checkpoint"
            help = "how often to save checkpoints (-1 for no checkpointing)"
            arg_type = Int
            default = -1
        "--batchsize"
            help = "batchsize"
            arg_type = Int
            default = batchsize_
        "--val_split"
            help = "fraction of train/val set to use as val set"
            arg_type = Float32
            default = 5f-1
        "--trainval_fraction"
            help = "total fraction of train/val set to use"
            arg_type = Float32
            default = trainval_fraction_
        "--test_fraction"
            help = "fraction of test set to use"
            arg_type = Float32
            default = 1f0
        "--num_cells"
            help = "number of cells in supernet"
            arg_type = Int
            default = num_cells_
        "--channels"
            help = "number of initial in supernet"
            arg_type = Int
            default = channels_
        "--rho"
            help = "admm parameter"
            arg_type = Float32
            default = 1f-3
    end

    return parse_args(s)
end


function parse_commandline_eval()
    gpumem = CUDA.totalmem(collect(CUDA.devices())[1])/(1024^3)
    if gpumem < 2.0
        batchsize_ = 1
        num_cells_ = 2
        channels_ = 18
        trainval_fraction_ = 4f-4
    elseif gpumem < 12.0
        batchsize_ = 96
        num_cells_ = 20
        channels_ = 36
        trainval_fraction_ = 1f0
    elseif gpumem < 16.0
        batchsize_ = 96
        num_cells_ = 20
        channels_ = 36
        trainval_fraction_ = 1f0
    else
        batchsize_ = 128
        num_cells_ = 20
        channels_ = 36
        trainval_fraction_ = 1f0
    end
    global batchsize_
    global num_cells_
    global channels_
    global trainval_fraction_

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--epochs"
            help = "number of epochs"
            arg_type = Int
            default = 600
        "--random_seed"
            help = "random seed (-1 for no random seed)"
            arg_type = Int
            default = 32
        "--checkpoint"
            help = "how often to save checkpoints (-1 for no checkpointing)"
            arg_type = Int
            default = -1
        "--batchsize"
            help = "batchsize"
            arg_type = Int
            default = batchsize_
        "--test_batchsize"
            help = "testset batchsize"
            arg_type = Int
            default = 248
        "--val_split"
            help = "fraction of train/val set to use as val set"
            arg_type = Float32
            default = 0f0
        "--trainval_fraction"
            help = "total fraction of train/val set to use"
            arg_type = Float32
            default = trainval_fraction_
        "--test_fraction"
            help = "fraction of test set to use"
            arg_type = Float32
            default = 1f0
        "--num_cells"
            help = "number of cells in supernet"
            arg_type = Int
            default = num_cells_
        "--channels"
            help = "number of initial in supernet"
            arg_type = Int
            default = channels_
        "--droppath"
            help = "droppath probability"
            arg_type = Float32
            default = 2f-1
        "--aux"
            help = "weight of auxiliary loss"
            arg_type = Float32
            default = 4f-1
    end

    return parse_args(s)
end

@with_kw struct trial_params
    epochs::Int = 50
    batchsize::Int = batchsize_
    throttle_::Int = 20
    val_split::Float32 = 0.5
    trainval_fraction::Float32 = trainval_fraction_
    test_fraction::Float32 = 1.0
    num_cells::Int = num_cells_
    channels::Int = channels_
end

function loss(m, x, y)
    out = m(x)
    loss = logitcrossentropy(squeeze(out), y)
    return loss
end

acccb() = @show(accuracy_batched(m, val))
function accuracy(m, x, y)
    out = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))
end
function accuracy(m, x, y, pert)
    out = mean(onecold(m(x, pert), 1:10) .== onecold(y, 1:10))
end
function accuracy_batched(m, xy)
    CUDA.reclaim()
    GC.gc()
    score = 0f0
    count = 0
    for batch in CuIterator(xy)
        @show acc = accuracy(m, batch...)
        score += acc*length(batch)
        count += length(batch)
        CUDA.reclaim()
        GC.gc()
    end
    display(score / count)
    score / count
end
function accuracy_batched(m, xy, pert)
    CUDA.reclaim()
    GC.gc()
    score = 0f0
    count = 0
    for batch in CuIterator(xy)
        @show acc = accuracy(m, batch..., pert)
        score += acc*length(batch)
        count += length(batch)
        CUDA.reclaim()
        GC.gc()
    end
    display(score / count)
    score / count
end
function accuracy_unbatched(m, xy; pert = [])
    CUDA.reclaim()
    GC.gc()
    xy = xy | gpu
    acc = accuracy(m, xy..., pert = pert)
    foreach(CUDA.unsafe_free!, xy)
    CUDA.reclaim()
    GC.gc()
    acc
end
Base.@kwdef mutable struct histories
    normal_αs::Vector{Vector{Array{Float32, 1}}}
    reduce_αs::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

histories() = histories([],[],[],[])

function (hist::histories)()
    push!(hist.normal_αs, deepcopy(m.normal_αs) |> cpu)
    push!(hist.reduce_αs, deepcopy(m.reduce_αs) |> cpu)
    #push!(hist.activations, m.activations |> cpu)
    #push!(hist.accuracies, accuracy_batched(m, val |> gpu))
end


Base.@kwdef mutable struct historiessm
    normal_αs_sm::Vector{Vector{Array{Float32, 1}}}
    reduce_αs_sm::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
end

historiessm() = historiessm([],[],[],[])

function (hist::historiessm)()
    #@show losses
    push!(hist.normal_αs_sm, softmax.(deepcopy(m.normal_αs)) |> cpu)
    push!(hist.reduce_αs_sm, softmax.(deepcopy(m.reduce_αs)) |> cpu)
    #push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    CUDA.reclaim()
    GC.gc()
end


Base.@kwdef mutable struct historiessml
    normal_αs_sm::Vector{Vector{Array{Float32, 1}}}
    reduce_αs_sm::Vector{Vector{Array{Float32, 1}}}
    activations::Vector{Dict}
    accuracies::Vector{Float32}
    train_losses::Vector{Float32}
    val_losses::Vector{Float32}
end

historiessml() = historiessml([],[],[],[],[],[])

function (hist::historiessml)()
    @show losses
    push!(hist.normal_αs_sm, softmax.(deepcopy(m.normal_αs)) |> cpu)
    push!(hist.reduce_αs_sm, softmax.(deepcopy(m.reduce_αs)) |> cpu)
    #push!(hist.activations, copy(m.activations.currentacts) |> cpu)
    push!(hist.train_losses, losses[1])
    push!(hist.val_losses, losses[2])
    CUDA.reclaim()
    GC.gc()
end


struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

function timefunc(func)
    @show func
    @time out = func()
    out
end


(cba::CbAll)() = foreach(cb -> cb(), cba.cbs)

function prepare_folder(algo::String, args::Dict)
    if "SLURM_JOB_ID" in keys(ENV)
        uniqueid = ENV["SLURM_JOB_ID"]
    else
        uniqueid = Dates.now()
    end
    if ispath("/gpfs/work/p21001/maile/dnas/models/")
        model_dir = "/gpfs/work/p21001/maile/dnas/models/"
    elseif ispath("/gpfs/work/p21001/maile/dnas/models/")
        model_dir = "/projets/reva/kmaile/dnas/models/"
    else
        model_dir = "test/models/"
    end
    base_folder = string(model_dir, algo, "_", uniqueid)
    mkpath(base_folder)
    BSON.@save joinpath(base_folder, "args.bson") args
    base_folder
end

function save_progress()
    m_cpu = m |> cpu
    normal_αs = m_cpu.normal_αs
    reduce_αs = m_cpu.reduce_αs
    #BSON.@save joinpath(base_folder, "model.bson") m_cpu args
    BSON.@save joinpath(base_folder, "alphas.bson") normal_αs reduce_αs
    BSON.@save joinpath(base_folder, "histepoch.bson") histepoch
    BSON.@save joinpath(base_folder, "histbatch.bson") histbatch
    if args["checkpoint"] > 0 && length(histepoch.train_losses) % args["checkpoint"] == 0
        BSON.@save joinpath(base_folder, string("model", length(histepoch.train_losses), ".bson")) m_cpu args
    end
end
