ENV["GKSwstype"]="100"


using DifferentiableNAS
using Flux
using Flux: throttle, logitcrossentropy, onecold, onehotbatch
using StatsBase: mean
using Parameters
using CUDA
using Distributions
using BSON
using Plots
gr()
include("CIFAR10.jl")
include("training_utils.jl")
include("visualize.jl")

#folder_name = "test/models/osirim/darts_2021-01-13T14:17:49.647"
#folder_name = "test/models/osirim/darts_2021-01-13T14:24:11.711"
folder_name = "test/models/osirim/admm_2021-01-13T17:43:21.59"

trials = Dict("test/models/osirim/darts_2021-01-13T14:17:49.647" => "DARTS_none",
                "test/models/osirim/darts_2021-01-13T14:24:11.711"=> "DARTS_sans_none",
                "test/models/osirim/admm_2021-01-13T17:43:21.59"=> "ADMM_sans_none",
                "test/models/osirim/admm_2021-01-14T16:55:27.481"=> "ADMM_none",
                "test/models/osirim/admm_2021-01-14T17:16:51.535"=> "ADMM_128_RTX",
                "test/models/osirim/admm_2021-01-15T12:06:38.741"=> "ADMM_128_RTX_sans_none_disc",)


for folder_name in keys(trials)

    file_name = string(folder_name, "/histbatch.bson")
    BSON.@load file_name histbatch

    prim = [
        "none",
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        #"sep_conv_7x7",
        "dil_conv_3x3",
        "dil_conv_5x5",
        #"conv_7x1_1x7"
    ]

    nodelabels = ["c_{k-2}", "c_{k-1}", "1", "2", "3", "4", "c_{k}"]

    connects = vcat([[(j,i) for j = 1:i-1] for i = 3:6]...)
    if typeof(histbatch)<:histories
        normal_ = histbatch.normal_αs
        reduce_ = histbatch.reduce_αs

        if length(normal_[1][1]) == 7
            prim = prim[2:8]
        end
        n_y_min = minimum([softmax(a[i])[j] for a in histbatch.normal_αs for i in 1:14 for j in 1:length(prim)])
        n_y_max = maximum([softmax(a[i])[j] for a in histbatch.normal_αs for i in 1:14 for j in 1:length(prim)])
        p = Vector(undef, 14)
        for i = 1:14
            p[i] = plot(title = string("Op ",nodelabels[connects[i][1]],"->",nodelabels[connects[i][2]]), ylim=(n_y_min,n_y_max), legend=false)
            for j = 1:length(prim)
                plot!([softmax(a[i])[j] for a in histbatch.normal_αs], label=prim[j])
            end
        end
        plot(p..., layout = (2,7), size = (2200,600));
        savefig(string(folder_name, "/normal_plots.png"))

        r_y_min = minimum([softmax(a[i])[j] for a in histbatch.reduce_αs for i in 1:14 for j in 1:length(prim)])
        r_y_max = maximum([softmax(a[i])[j] for a in histbatch.reduce_αs for i in 1:14 for j in 1:length(prim)])
        p = Vector(undef, 14)
        for i = 1:14
            p[i] = plot(title = string("Op ",nodelabels[connects[i][1]],"->",nodelabels[connects[i][2]]), ylim=(r_y_min,r_y_max), legend=false)
            for j = 1:length(prim)
                plot!([softmax(a[i])[j] for a in histbatch.reduce_αs], label=prim[j])
            end
        end
        plot(p..., layout = (2,7), size = (2200,600));
        savefig(string(folder_name, "/reduce_plots.png"))
    else
        normal_ = histbatch.normal_αs_sm
        reduce_ = histbatch.reduce_αs_sm
        if length(normal_[1][1]) == 7
            prim = prim[2:8]
        else
            n_y_min = minimum([a[i][j] for a in normal_ for i in 1:14 for j in 2:length(prim)])
            n_y_max = maximum([a[i][j] for a in normal_ for i in 1:14 for j in 2:length(prim)])
            p = Vector(undef, 15)
            for i = 1:14
                p[i] = plot(title = string("Op ",nodelabels[connects[i][1]],"->",nodelabels[connects[i][2]]), ylim=(n_y_min,n_y_max), legend=false)
                for j = length(prim):-1:2
                    plot!([a[i][j] for a in normal_], label=prim[j])
                end
            end
            p[15] = plot(showaxis=false, grid = false, xlims=(-1,-1))
            for j = length(prim):-1:1
                plot!([a[1][j] for a in normal_], label=prim[j])
            end
            plot(p..., layout = (3,5), size = (1200,800));
            savefig(string(folder_name, "/", trials[folder_name], "-normal_plots_sans_none.png"))

            r_y_min = minimum([a[i][j] for a in reduce_ for i in 1:14 for j in 1:length(prim)])
            r_y_max = maximum([a[i][j] for a in reduce_ for i in 1:14 for j in 1:length(prim)])
            p = Vector(undef, 15)
            for i = 1:14
                p[i] = plot(title = string("Op ",nodelabels[connects[i][1]],"->",nodelabels[connects[i][2]]), ylim=(r_y_min,r_y_max), legend=false)
                for j = length(prim):-1:2
                    plot!([a[i][j] for a in reduce_], label=prim[j])
                end
            end
            p[15] = plot(showaxis=false, grid = false, xlims=(-1,-1))
            for j = length(prim):-1:2
                plot!([a[1][j] for a in reduce_], label=prim[j])
            end
            plot(p..., layout = (3,5), size = (1200,800));
            savefig(string(folder_name, "/", trials[folder_name], "-reduce_plots_sans_none.png"))
        end
        n_y_min = minimum([a[i][j] for a in normal_ for i in 1:14 for j in 1:length(prim)])
        n_y_max = maximum([a[i][j] for a in normal_ for i in 1:14 for j in 1:length(prim)])
        p = Vector(undef, 15)
        for i = 1:14
            p[i] = plot(title = string("Op ",nodelabels[connects[i][1]],"->",nodelabels[connects[i][2]]), ylim=(n_y_min,n_y_max), legend=false)
            for j = length(prim):-1:1
                plot!([a[i][j] for a in normal_], label=prim[j])
            end
        end
        p[15] = plot(showaxis=false, grid = false, xlims=(-1,-1))
        for j = length(prim):-1:1
            plot!([a[1][j] for a in normal_], label=prim[j])
        end
        plot(p..., layout = (3,5), size = (1200,800));
        savefig(string(folder_name, "/", trials[folder_name], "-normal_plots.png"))

        r_y_min = minimum([a[i][j] for a in reduce_ for i in 1:14 for j in 1:length(prim)])
        r_y_max = maximum([a[i][j] for a in reduce_ for i in 1:14 for j in 1:length(prim)])
        p = Vector(undef, 15)
        for i = 1:14
            p[i] = plot(title = string("Op ",nodelabels[connects[i][1]],"->",nodelabels[connects[i][2]]), ylim=(r_y_min,r_y_max), legend=false)
            for j = length(prim):-1:1
                plot!([a[i][j] for a in reduce_], label=prim[j])
            end
        end
        p[15] = plot(showaxis=false, grid = false, xlims=(-1,-1))
        for j = length(prim):-1:1
            plot!([a[1][j] for a in reduce_], label=prim[j])
        end
        plot(p..., layout = (3,5), size = (1200,800));
        savefig(string(folder_name, "/", trials[folder_name], "-reduce_plots.png"))
        if typeof(histbatch)<:historiessml
            p = Vector(undef, 2)
            p = plot(title = "Losses")
            plot!(histbatch.train_losses, label = "training loss")
            plot!(histbatch.val_losses, label = "validation loss")
            """
            p = Vector(undef, 2)
            p[1] = plot(histbatch.train_losses, label = "training loss")
            p[2] = plot(histbatch.val_losses, label = "validation loss", seriescolor = :red)
            plot(p..., layout = (2,1), size = (600,800), title = "Losses")
            """
            savefig(string(folder_name, "/", trials[folder_name], "-loss.png"))
        end
    end

    visualize(normal_[length(normal_)], string(folder_name, "/", trials[folder_name], "-normal_graph.png"))
    visualize(reduce_[length(reduce_)], string(folder_name, "/", trials[folder_name], "-reduce_graph.png"))
end
