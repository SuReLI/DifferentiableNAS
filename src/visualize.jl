using LightGraphs

function visualize(αs)
    inputindices, _, opnames = discretize(αs, -1, -1)
    g = SimpleDiGraph(7)
    nodelabels = ["c_{k-2}", "c_{k-1}", "1", "2", "3", "4", "c_{k}"]
    edgelabels = []
    for i in 1:length(inputindices)
        add_edge!(g, inputindices[i][1], i+2)
        push!(edgelabels, opnames[i][1])
        add_edge!(g, inputindices[i][2], i+2)
        push!(edgelabels, opnames[i][2])
    end
    for i in 3:6
        add_edge!(g, i, 7)
        push!(edgelabels,  "")
    end
    gplot(g, edgelabel=edgelabels, nodelabel=nodelabels, edgelabeldistx=2, edgelabeldisty=2)
end

using BSON
BSON.@load "test/models/alphas_final.bson" normal_
visualize(normal_)
