export visualize

using DifferentiableNAS
using Flux
using CUDA
using LightGraphs
using GraphPlot
using Cairo
using Compose
using BSON
using LaTeXStrings
using PyCall
graphviz = pyimport("graphviz")
include("training_utils.jl")


function visualize(αs, filename::String)
    nodelabels = ["c_{k-2}", "c_{k-1}", "0", "1", "2", "3", "c_{k}"]
    inputindices, _, opnames = discretize(αs, 1, false, 4; plotting = true)
    g = graphviz.Digraph(
          format="png",
          edge_attr=Dict("fontsize"=>"20", "fontname"=>"times"),
          node_attr=Dict("style"=>"filled", "shape"=>"rect", "align"=>"center", "fontsize"=>"20", "height"=>"0.5", "width"=>"0.5", "penwidth"=>"2", "fontname"=>"times"),
          engine="dot")
    g.body = ["rankdir=LR"]

    g.node(nodelabels[1], fillcolor="darkseagreen2")
    g.node(nodelabels[2], fillcolor="darkseagreen2")

    for i in 3:6
      g.node(nodelabels[i], fillcolor="lightblue")
    end

    for i in 1:length(inputindices)
        g.edge(nodelabels[inputindices[i][1]], nodelabels[i+2], label=opnames[i][1], fillcolor="gray")
        g.edge(nodelabels[inputindices[i][2]], nodelabels[i+2], label=opnames[i][2], fillcolor="gray")
    end

    g.node(nodelabels[7], fillcolor="palegoldenrod")
    for i in 3:6
      g.edge(nodelabels[i], nodelabels[7], fillcolor="gray")
    end
    g.render(filename, view=false)
end

function visualize(filename::String)
    nodelabels = ["c_{k-2}", "c_{k-1}", "0", "1", "2", "3", "c_{k}"]
    inputindices, _, opnames = parse_genotype(channels=1, reduce=false, steps=4)
    g = graphviz.Digraph(
          format="png",
          edge_attr=Dict("fontsize"=>"20", "fontname"=>"times"),
          node_attr=Dict("style"=>"filled", "shape"=>"rect", "align"=>"center", "fontsize"=>"20", "height"=>"0.5", "width"=>"0.5", "penwidth"=>"2", "fontname"=>"times"),
          engine="dot")
    g.body = ["rankdir=LR"]

    g.node(nodelabels[1], fillcolor="darkseagreen2")
    g.node(nodelabels[2], fillcolor="darkseagreen2")

    for i in 3:6
      g.node(nodelabels[i], fillcolor="lightblue")
    end

    for i in 1:length(inputindices)
        g.edge(nodelabels[inputindices[i][1]], nodelabels[i+2], label=opnames[i][1], fillcolor="gray")
        g.edge(nodelabels[inputindices[i][2]], nodelabels[i+2], label=opnames[i][2], fillcolor="gray")
    end

    g.node(nodelabels[7], fillcolor="palegoldenrod")
    for i in 3:6
      g.edge(nodelabels[i], nodelabels[7], fillcolor="gray")
    end
    g.render(filename, view=false)
end


function visualize_supercell(filename::String)
    nodelabels = ["c_{k-2}", "c_{k-1}", "0", "1", "2", "3", "c_{k}"]
    g = graphviz.Digraph(
          format="png",
          edge_attr=Dict("fontsize"=>"20", "fontname"=>"times"),
          node_attr=Dict("style"=>"filled", "shape"=>"rect", "align"=>"center", "fontsize"=>"20", "height"=>"0.5", "width"=>"0.5", "penwidth"=>"2", "fontname"=>"times"),
          engine="dot")
    g.body = ["rankdir=LR"]

    g.node(nodelabels[1], fillcolor="darkseagreen2")
    g.node(nodelabels[2], fillcolor="darkseagreen2")

    for i in 3:6
      g.node(nodelabels[i], fillcolor="lightblue")
    end

    connects = vcat([[(j,i) for j = 1:i-1] for i = 3:6]...)

    for connect in connects
        g.edge(nodelabels[connect[1]], nodelabels[connect[2]], fillcolor="black", penwidth="3")
    end

    g.node(nodelabels[7], fillcolor="palegoldenrod")
    for i in 3:6
      g.edge(nodelabels[i], nodelabels[7], fillcolor="gray")
    end
    g.render(filename, view=false)
end
