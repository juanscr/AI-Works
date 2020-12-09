include("K-Means.jl")
include("PlotTools.jl")

# Graph Laplacian
#   data -> (n, p) dataset matrix
#   dist -> Similarity function
#   m -> Minimum number of nearest neighbors
#   arg -> Additional parameter for similarity function
function laplacian_matrix(data, dist; m=6, arg=nothing)
    n, p = size(data)
    dists = zeros(n, n)
    adj_matrix = zeros(n, n)
    for j in 1:n
        dists[j, :] = dist(data, reshape(data[j, :], (1, p)), arg)
    end
    for i in 1:n
        for j in 1:m
            ind_min = argmin(dists[i, :])
            adj_matrix[i, ind_min] = 1
            adj_matrix[ind_min, i] = 1
            dists[i, ind_min] = Inf
        end
    end

    degree_matrix = diagm(vec(sum(adj_matrix, dims=2)))
    L = degree_matrix - adj_matrix
    return L
end

# Algorithm
#   data -> (n, p) dataset matrix
#   k -> Number of clusters
#   dist -> Similarity function
#   m -> Minimum number of nearest neighbors
#   arg -> Additional parameter for similarity function
function spectral_cluster(data, k, dist; m=6, arg=nothing)
    L = laplacian_matrix(data, dist, arg=arg, m=m)
    eigvals, eigvecs = eigen(L)
    eigvecs = eigvecs[:, sortperm(eigvals)]
    W = eigvecs[:, 1:k]
    _, _, U, _, _ = k_means(W, k, dist)
    return U
end

# Test spectral clustering with circular data
function test_spectral()
    r1 = 5
    r2 = 1
    xs1 = collect(-r1:0.05:r1)
    xs2 = collect(-r2:0.05:r2)
    data1 = []
    data2 = []
    f(x, r) = sqrt(r^2 - x^2) + rand(Normal(0, 0.5), 1)[1]

    data1 = zeros(2*size(xs1, 1), 2)
    data1[:, 1] = vcat(xs1, xs1)

    data2 = zeros(2*size(xs2, 1), 2)
    data2[:, 1] = vcat(xs2, xs2)

    for i in 1:size(xs1, 1)
        data1[i, 2] = f(xs1[i], r1)
        data1[i+size(xs1, 1), 2] = -f(xs1[i], r1)
    end

    for i in 1:size(xs2, 1)
        data2[i, 2] = f(xs2[i], r2)
        data2[i+size(xs2, 1), 2] = -f(xs2[i], r2)
    end

    dir = "MvN-Mixture"
    circles = vcat(data1, data2)
    U = spectral_cluster(circles, 2, euclidean, m=10)
    plot_clustered_data(2, U, circles, "test_spectral.pdf"; dir=dir)
end
