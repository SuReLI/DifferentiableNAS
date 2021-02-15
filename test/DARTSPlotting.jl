ENV["GKSwstype"]="100"


using DifferentiableNAS
using Flux
using Flux: logitcrossentropy, onecold, onehotbatch
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


trials = Dict(  #"test/models/osirim/darts_2021-01-13T14:17:49.647" => "DARTS_none",
                #"test/models/osirim/darts_2021-01-13T14:24:11.711"=> "DARTS_sans_none",
                #"test/models/osirim/admm_2021-01-13T17:43:21.59"=> "ADMM_sans_none",
                #"test/models/osirim/admm_2021-01-14T16:55:27.481"=> "ADMM_none",
                #"test/models/osirim/admm_2021-01-14T17:16:51.535"=> "ADMM_128_RTX",
                #"test/models/osirim/admm_2021-01-15T12:06:38.741"=> "ADMM_128_RTX_sans_none_disc",
                #"test/models/osirim/admm_6641599" => "ADMM_128_sliding_freq_sliding_disc_1",
                #"test/models/osirim/admm_6641601" => "ADMM_128_sliding_freq",
                #"test/models/osirim/admm_6641606" => "ADMM_sliding_freq_sliding_disc",
                #"test/models/osirim/darts_6641607" => "BNDARTS",
                #"test/models/osirim/bnadmm_6641778" => "BNADMM_sliding_freq_sliding_disc",
                #"test/models/osirim/bnadmm_6641847" => "BNADMM_none_fixed",
                #"test/models/osirim/bnadmm_6641882" => "BNADMM_progressive_1e-2",
                #"test/models/osirim/sigadmm_6642019" => "SIGADMM",
                #"test/models/osirim/bndarts_6642020" => "BNDARTS",
                #"test/models/osirim/bnadmm_6642126" => "BNADMM_prog_1e-1",
                #"test/models/osirim/bnadmm_6642153" => "BNADMM_prog_relutanh",
                #"test/models/osirim/bnadmm_6642226" => "BNADMM_rho_disc_1e-3",
                #"test/models/osirim/bnadmm_6642233" => "BNADMM_rho_disc_5e-3",
                #"test/models/osirim/bnadmm_6642280" => "BNADMM_eval",
                #"test/models/osirim/bnadmm_6642317" => "BNADMM_rho_disc_1e-3_2",
                #"test/models/osirim/bnadmm_6642319" => "BNADMM_rho_disc_alr9e-4",
                #"test/models/osirim/bnadmm_6642365" => "BNADMM_fix_data",
                #"test/models/osirim/bnadmm_6642368" => "BNADMM_eval",
                #"test/models/osirim/darts_6642375" => "DARTS",
                #"test/models/osirim/bnadmm_6642376" => "BNADMM_rho_1e-3",
                #"test/models/osirim/bnadmm_6642377" => "BNADMM_rho_5e-3",
                #"test/models/osirim/bnadmm_6642384" => "BNADMM_rho_1e-2",
                #"test/models/osirim/bnadmm_6642392" => "BNADMM_rho_1e-3",
                #"test/models/osirim/bnadmm_6642393" => "BNADMM_eval",
                #"test/models/osirim/darts_6642395" => "DARTS_none",
                #"test/models/osirim/bnadmm_6642397" => "BNADMM_alr_3e-3",
                #"test/models/osirim/bnadmm_6642406" => "BNADMM_eval",
                #"test/models/osirim/bnadmm_6642409" => "DARTS_eval",
                #"test/models/osirim/bnadmm_6642571" => "BNADMM_singleupdate",
                #"test/models/osirim/darts_6642654" => "DARTS_cosanneal",
                #"test/models/osirim/darts_6642774" => "DARTS_test30epochs_frac",
                #"test/models/osirim/darts_6642805" => "DARTS_test30epochs_full",
                #"test/models/osirim/darts_6642941" => "DARTS_test30check_small",
                #"test/models/osirim/darts_6642942" => "DARTS_test30check_big",
                #"test/models/osirim/bnadmm_6644989" => "BNADMM_speedup",
                #"test/models/osirim/admm_6645078" => "BNADMM_scheduled",
                #"test/models/osirim/darts_6645238" => "DARTS_speedup",
                #"test/models/osirim/darts_6645482" => "DARTS_32_dep",
                #"test/models/osirim/darts_6645483" => "DARTS_32_oldv",
                #"test/models/osirim/darts_6645488" => "DARTS_64_dep",
                #"test/models/osirim/darts_6645542" => "DARTS_64_dep_cuda11",
                "test/models/osirim/darts_6645746" => "DARTS64",
                "test/models/osirim/admm_6645848" => "BNADMM64scheduled",
                "test/models/osirim/bnadmm_6645849" => "BNADMM64",
                "test/models/osirim/evaldarts/eval_6645902" => "eval",
                #"test/models/olympe/bnadmm_528425" => "BNADMM_o1",
                #"test/models/olympe/bnadmm_528440" => "BNADMM_o2",
                #"test/models/olympe/bnadmm_528445" => "BNADMM_o3",
                #"test/models/olympe/bnadmm_528487" => "BNADMM_o4",
                #"test/models/olympe/bnadmm_529404" => "BNADMM_o5",
                #"test/models/olympe/bnadmm_529405" => "BNADMM_o6",
                #"test/models/olympe/bnadmm_529406" => "BNADMM_o7",
                #"test/models/olympe/bnadmm_530364" => "BNADMM_o8",
                #"test/models/olympe/bnadmm_530427" => "BNADMM_o9",
                #"test/models/olympe/bnadmm_532776" => "BNADMM_agpu",
                #"test/models/olympe/bnadmm_532778" => "BNADMM_acpu",
                "test/models/olympe/evaldarts/eval_540757" => "eval",
                "test/models/olympe/bnadmm_540761" => "BNADMM",
                "test/models/olympe/admm_540763" => "BNADMMscheduled",
                "test/models/olympe/darts_540764" => "DARTS",
                )

for folder_name in keys(trials)

    if ispath(joinpath(folder_name, "histbatch.bson"))
        @show file_name = string(folder_name, "/histbatch.bson")
        BSON.@load file_name histbatch

        if ispath(joinpath(folder_name, "args.bson"))
            BSON.@load joinpath(folder_name, "args.bson") args
            batchsize = args["batchsize"]
            trainval_fraction = args["trainval_fraction"]
        else
            BSON.@load joinpath(folder_name, "model.bson") argparams
            batchsize = argparams.batchsize
            trainval_fraction = 1.0
        end
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

        nodelabels = ["c_{k-2}", "c_{k-1}", "0", "1", "2", "3", "c_{k}"]

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
            epochs = batches ./ (1+(25000 ÷ batchsize)*trainval_fraction)
            if length(normal_[1][1]) == 7
                if occursin("DC",trials[folder_name])
                    prim = prim[1:7]
                else
                    prim = prim[2:8]
                end
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
                        axs[con[2]-2,con[1]].plot(epochs, [a[i][j] for a in histbatch.normal_αs_sm], label = prim[j])
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
                        axs[con[2]-2,con[1]].set_xlabel(string(nodelabels[con[1]]))
                    end
                    if con[1] == 1
                        axs[con[2]-2,con[1]].set_ylabel(string(nodelabels[con[2]]))
                    end
                    for j = length(prim):-1:2
                        axs[con[2]-2,con[1]].plot(epochs, [a[i][j] for a in histbatch.reduce_αs_sm], label = prim[j])
                    end
                    if con[1] in [1,2]
                        axs[con[2]-2,con[1]].plot(epochs, [a[i][findall(x->x=="skip_connect", prim)[1]] for a in histbatch.reduce_αs_sm], label = "fact_red", color = "tab:olive")
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
                    axs[con[2]-2,con[1]].plot(epochs, [a[i][j] for a in histbatch.normal_αs_sm], label = prim[j])
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
                    axs[con[2]-2,con[1]].plot(epochs, [a[i][j] for a in histbatch.reduce_αs_sm], label = prim[j])
                end
                if con[1] in [1,2]
                    axs[con[2]-2,con[1]].plot(epochs, [a[i][findall(x->x=="skip_connect", prim)[1]] for a in histbatch.reduce_αs_sm], label = "fact_red", color = "tab:olive")
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
                fig, axs = plt.subplots(2)
                axs[1].plot(epochs, histbatch.train_losses, label = "training loss")
                axs[2].plot(epochs, histbatch.val_losses, label = "validation loss", color = "tab:orange")
                axs[1].legend()
                axs[2].legend()
                PyPlot.savefig(string(folder_name, "/", trials[folder_name], "-loss_subplots.png"))
            end
        end

        visualize(normal_[length(normal_)], string(folder_name, "/", trials[folder_name], "-normal_graphviz"))
        visualize(reduce_[length(reduce_)], string(folder_name, "/", trials[folder_name], "-reduce_graphviz"))
        PyPlot.close("all")
    else
        @show file_name = string(folder_name, "/histeval.bson")
        BSON.@load file_name histeval
        fig, axs = plt.subplots(2)
        axs[1].plot(histeval.train_losses, label = "training loss")
        axs[2].plot(histeval.accuracies, label = "accuracy", color = "tab:green")
        axs[1].legend()
        axs[2].legend()
        PyPlot.savefig(string(folder_name, "/", trials[folder_name], "-plots.png"))
    end
end
