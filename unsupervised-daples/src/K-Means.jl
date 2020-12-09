using Pkg

Pkg.add("StatsBase")
using StatsBase

include("Tools.jl");

# Cost Function
#   U -> (k, n) membership matrix
#   dists -> (k, n) Distance matrix
function cost_fun(U, dists)
    return sum(U .* dists)
end

# Algorithm
#   data -> (n, p) dataset matrix
#   k -> Number of clusters
#   dist -> Similarity function
#   γ -> Tolerance for stop criterion
#   arg -> Additional parameter for similarity function
#   norm -> Whether the data will be normalized or not
function k_means(data, k, dist; γ=0.001, arg=nothing, norm=true)
    # Normalize data
    data = norm ? normalize(data) : data
    # Initialize prototypes
    n = size(data, 1)
    indexes = collect(1:n)
    selection = sample(indexes, k, replace=false)
    protos = data[selection, :]
    protos_iter = [protos]
    # Iterate k-means
    improv = Inf
    cost = 0
    improvs = []
    while improv >= γ
        # New groups and prototypes
        U, dists = find_membership(data, dist, protos, arg)
        protos = (U * data) ./ sum(U, dims=2)
        push!(protos_iter, protos)
        # Evaluate improvement
        prev_cost = cost
        cost = cost_fun(U, dists)
        improv = abs(prev_cost - cost)
        push!(improvs, improv)
    end
    U, _ = find_membership(data, dist, protos, arg)
    return data, protos, U, improvs, protos_iter
end
