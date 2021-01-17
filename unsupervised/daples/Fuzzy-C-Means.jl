include("Tools.jl");

# Initialize membership matrix with random vector
#   c -> Number of clusters
#   n -> Number of data points
function initialize_membership(c, n)
    U = zeros(c, n)
    for j in 1:n
        r = rand(c, 1)
        U[:, j] = r / sum(r)
    end
    return U
end

# Cost Function
#   Uᵐ -> (c, n) fuzzy membership matrix
#   dists -> (c, n) Distance matrix
function cost_fun(Uᵐ, dists)
    return sum(Uᵐ .* dists.^2)
end

# Algorithm
#   data -> (n, p) dataset matrix
#   c -> Number of clusters
#   dist -> Similarity function
#   m -> Weighting exponent for fuzzy membership
#   γ -> Tolerance for stop criterion
#   arg -> Additional parameter for similarity function
#   norm -> Whether the data will be normalized or not
function fuzzy_c_means(data, c, dist, m; γ=0.001, arg=nothing, norm=true)
    # Normalize data
    data = norm ? normalize(data) : data
    n, p = size(data)
    # Initialize
    U = initialize_membership(c, n)
    Uᵐ = U.^m
    # Iterate
    improv = Inf
    dists = zeros(c, n)
    cost = 0
    improvs = []
    protos_iter = []
    while improv >= γ
        # Find current clusters
        global protos = Uᵐ * data ./ sum(Uᵐ, dims=2)
        push!(protos_iter, protos)
        for j in 1:c
            dists[j, :] = dist(data, reshape(protos[j, :], (1, p)), arg)
        end
        # Evaluate improvement
        prev_cost = cost
        cost = cost_fun(Uᵐ, dists)
        improv = abs(prev_cost - cost)
        push!(improvs, improv)
        # New memberships
        for i in 1:c
            for j in 1:n
                U[i, j] = 1 / sum((dists[i, j] ./ dists[:, j]) .^ (2 / (m - 1)))
            end
        end
        Uᵐ = U.^m
    end
    return data, protos, U, improvs, protos_iter
end
