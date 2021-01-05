using LightGraphs
using GraphPlot
using Cairo
using Compose

function visualize(αs, filename)
    inputindices, _, opnames = discretize(αs, 1, false, 4)
    g = SimpleDiGraph(7)
    nodelabels = ["c_{k-2}", "c_{k-1}", "1", "2", "3", "4", "c_{k}"]
    locs_x = [-1.5,-1.5,1,2,3,4,6]
    locs_y = [4.5,3,3.5,2,4.5,3,2]
    nodefillcs = ["lightgreen", "lightgreen", "lightblue", "lightblue", "lightblue", "lightblue", "lightyellow"]
    edgelabels = []
    for i in 1:length(inputindices)
        add_edge!(g, inputindices[i][1], i+2)
        push!(edgelabels, (opnames[i][1], inputindices[i][1], i+2))
        add_edge!(g, inputindices[i][2], i+2)
        push!(edgelabels, (opnames[i][2], inputindices[i][2], i+2))
    end
    for i in 3:6
        add_edge!(g, i, 7)
        push!(edgelabels,  ("", i, 7))
    end
    sort!(edgelabels, by = x -> (x[2],x[3]))
    edgelabels = [e[1] for e in edgelabels]
    @show collect(zip(edgelabels, edges(g)))
    gp = gplot(g, locs_x, locs_y, edgelabel=edgelabels, nodelabel=nodelabels, nodefillc=nodefillcs, NODELABELSIZE=3.0, edgelabeldisty = -0.5)
    draw(PNG(filename), gp)
end

using BSON
BSON.@load "test/models/alphas_only.bson" normal_ reduce_
visualize(normal_, "test/models/normal.png")
visualize(reduce_, "test/models/reduce.png")
