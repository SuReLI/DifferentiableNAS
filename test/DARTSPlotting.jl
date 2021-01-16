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
using PyPlot
gr()
include("CIFAR10.jl")
include("training_utils.jl")
include("visualize.jl")


trials = Dict("test/models/osirim/darts_2021-01-13T14:17:49.647" => "DARTS_none",
                "test/models/osirim/darts_2021-01-13T14:24:11.711"=> "DARTS_sans_none",
                "test/models/osirim/admm_2021-01-13T17:43:21.59"=> "ADMM_sans_none",
                "test/models/osirim/admm_2021-01-14T16:55:27.481"=> "ADMM_none",
                "test/models/osirim/admm_2021-01-14T17:16:51.535"=> "ADMM_128_RTX",
                "test/models/osirim/admm_2021-01-15T12:06:38.741"=> "ADMM_128_RTX_sans_none_disc",)


for folder_name in keys(trials)

    file_name = string(folder_name, "/histbatch.bson")
    BSON.@load file_name histbatch

    BSON.@load joinpath(folder_name, "model.bson") argparams

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
        batches = 1:length(normal_)
        epochs = batches ./ (1+(25000 ÷ argparams.batchsize))
        if length(normal_[1][1]) == 7
            prim = prim[2:8]
        else
            fig, axs = plt.subplots(4, 5, sharex="col", sharey="row")
            for (i,con) in enumerate(connects)
                if con[2] == 6
                    axs[con[2]-2,con[1]].set_xlabel(latexstring(nodelabels[con[1]]))
                end
                if con[1] == 1
                    axs[con[2]-2,con[1]].set_ylabel(string(nodelabels[con[2]]))
                end
                for j = length(prim):-1:2
                    axs[con[2]-2,con[1]].plot(epochs, [softmax(a[i])[j] for a in histbatch.normal_αs_sm], label = prim[j])
                end
            end
            for i in 1:3
                for j in 2+i:5
                    axs[i,j].axis("off")
                end
            end
            handles, labels = axs[1,1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right")
            PyPlot.savefig(string(folder_name, "/", trials[folder_name], "-normal_plots_sans_none.png"))

            fig, axs = plt.subplots(4, 5, sharex="col", sharey="row")
            for (i,con) in enumerate(connects)
                if con[2] == 6
                    axs[con[2]-2,con[1]].set_xlabel(latexstring(nodelabels[con[1]]))
                end
                if con[1] == 1
                    axs[con[2]-2,con[1]].set_ylabel(string(nodelabels[con[2]]))
                end
                for j = length(prim):-1:2
                    axs[con[2]-2,con[1]].plot(epochs, [softmax(a[i])[j] for a in histbatch.reduce_αs_sm], label = prim[j])
                end
            end
            for i in 1:3
                for j in 2+i:5
                    axs[i,j].axis("off")
                end
            end
            handles, labels = axs[1,1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right")
            PyPlot.savefig(string(folder_name, "/", trials[folder_name], "-reduce_plots_sans_none.png"))
        end

        fig, axs = plt.subplots(4, 5, sharex="col", sharey="row")
        for (i,con) in enumerate(connects)
            if con[2] == 6
                axs[con[2]-2,con[1]].set_xlabel(latexstring(nodelabels[con[1]]))
            end
            if con[1] == 1
                axs[con[2]-2,con[1]].set_ylabel(string(nodelabels[con[2]]))
            end
            for j = length(prim):-1:1
                axs[con[2]-2,con[1]].plot(epochs, [softmax(a[i])[j] for a in histbatch.normal_αs_sm], label = prim[j])
            end
        end
        for i in 1:3
            for j in 2+i:5
                axs[i,j].axis("off")
            end
        end
        handles, labels = axs[1,1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        PyPlot.savefig(string(folder_name, "/", trials[folder_name], "-normal_plots.png"))

        fig, axs = plt.subplots(4, 5, sharex="col", sharey="row")
        for (i,con) in enumerate(connects)
            if con[2] == 6
                axs[con[2]-2,con[1]].set_xlabel(latexstring(nodelabels[con[1]]))
            end
            if con[1] == 1
                axs[con[2]-2,con[1]].set_ylabel(string(nodelabels[con[2]]))
            end
            for j = length(prim):-1:1
                axs[con[2]-2,con[1]].plot(epochs, [softmax(a[i])[j] for a in histbatch.reduce_αs_sm], label = prim[j])
            end
        end
        for i in 1:3
            for j in 2+i:5
                axs[i,j].axis("off")
            end
        end
        handles, labels = axs[1,1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        PyPlot.savefig(string(folder_name, "/", trials[folder_name], "-reduce_plots.png"))
        if typeof(histbatch)<:historiessml
            fig, ax = plt.subplots(1)
            ax.set_title("Losses")
            ax.plot(epochs, histbatch.train_losses, label = "training loss")
            ax.plot(epochs, histbatch.val_losses, label = "validation loss")
            ax.legend()
            PyPlot.savefig(string(folder_name, "/", trials[folder_name], "-loss.png"))
        end
    end

    visualize(normal_[length(normal_)], string(folder_name, "/", trials[folder_name], "-normal_graphviz"))
    visualize(reduce_[length(reduce_)], string(folder_name, "/", trials[folder_name], "-reduce_graphviz"))
    PyPlot.close("all")
end
