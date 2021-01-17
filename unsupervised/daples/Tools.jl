using Pkg
Pkg.add("Distributions")
Pkg.add("LinearAlgebra")
Pkg.add("TSne")

using Distributions
using LinearAlgebra
using TSne

## Similarities
function pnorm(data, y, arg)
    return (sum(abs.(data .- y).^arg, dims=2)).^(1/arg)
end

function euclidean(data, y, arg)
    return pnorm(data, y, 2)
end

function manhattan(data, y, arg)
    return pnorm(data, y, 1)
end

function mahal(data, y, IV)
    diff = data .- y
    return sqrt.(sum(diff * IV .* diff, dims=2))
end

## Misc
# Find memebership matrix of data to given cluster centers
function find_membership(data, dist, protos, arg)
    n = size(data, 1)
    k, p = size(protos)
    U = zeros(k, n)
    dists = zeros(k, n)
    for j in 1:k
        dists[j, :] = dist(data, reshape(protos[j, :], (1, p)), arg)
    end
    indexes_min = argmin(dists, dims=1)
    U[indexes_min] .= 1
    return U, dists
end

# Randomizes parameters for a Multivariate Normal Distributions for gen([a,b])
function rand_params_MvN(a, b, p)
    u = Uniform(a, b)
    μ = rand(u, p)
    Σ = rand(p, p)
    Σ = 0.5 * (Σ + Σ') + p*I
    return μ, Σ
end

# Generates data from a Multivariate Normal Distribution
function generate_MvN_Mixture(p, k, n; b = 25)
    data = rand(MvNormal(rand_params_MvN(0, b, p)...), n)';
    for j = 1:k-1
        data = vcat(data, rand(MvNormal(rand_params_MvN(0, b, p)...), n)')
    end
    return data
end

# Normalize data with max|x_i| -> [-1,1] hypercube.
function normalize(data)
    return data ./ maximum(abs.(data), dims=1)
end

# Read Iris dataset
function read_iris(iris_file)
    iris_file = open(iris_file, "r") do io
        read(io, String)
    end
    lines = split(iris_file, "\n")[1:end-1]
    lines = replace.(lines, "\r" => "")
    data = split.(lines, ",")
    tags = map(x -> x[end], data)
    data = map(x -> x[1:end-1], data)
    n = size(data)[1]
    p = size(data[1])[1]
    data = [parse.(Float64, point) for point in data]
    aux = reshape(data[1], (1, p))
    for j in 2:n
        aux = vcat(aux, reshape(data[j], (1, p)))
    end
    return aux, tags
end

# Read Counties dataset
function read_countries(country_file)
    country_file = open(country_file, "r") do io
        read(io, String)
    end
    lines = split(country_file, "\n")
    lines = lines[1:end-1]
    data = split.(lines, ",")
    header = split(data[1], ",")[1]
    data = data[2:end]
    countries = map(x -> x[1], data)
    data = map(x -> x[2:end], data)
    n = size(data)[1]
    p = size(data[1])[1]
    data = [parse.(Float64, point) for point in data]
    aux = reshape(data[1], (1, p))
    for j in 2:n
        aux = vcat(aux, reshape(data[j], (1, p)))
    end
    return aux, countries, header
end

# Print 2D matrix
function print_2Dmatrix(mat)
    m, n = size(mat)
    print("[\n")
    for i in 1:m
        for j in 1:n-1
            print(mat[i, j], "  ")
        end
        print(mat[i, n], ";\n")
    end
    print("]\n")
end

## Data Analysis
# Removes collinear dimensions and returns new data
function remove_collinear(data; γ=0.85)
    n, p = size(data)
    indexes = collect(1:p)
    rms = []
    new_data = data
    og_R = nothing
    R = nothing
    k = 1
    while true
        Σ = cov(new_data)
        d1 = diag(Σ)
        d2 = diag(inv(Σ))
        R =  1 .- 1 ./ (d1.*d2)
        if k == 1
            og_R = R
        end
        k += 1
        M, ind_max = findmax(R)
        if M <= γ
            break
        end
        append!(rms, splice!(indexes, indexes[ind_max]))
        new_data = data[:, indexes]
    end
    return new_data, rms, R, og_R
end

# Calculate the screes (sorted eigenvalues) of data
function screes(data; corr=false)
    Σ = ~corr ? cov(data) : cor(data)
    return collect(1:size(data, 2)), sort(eigvals(Σ), rev=true)
end

# Estimates Tukey's statistical depth
function tukey(point, data; n=500)
    r, c = size(data)
    u = rand(Normal(0, 1), (c, n))
    scalar1 = data * u
    scalar2 = point * u
    replic = ones(r, 1) * scalar2
    diff = scalar1 - replic
    diff_indicator = convert.(Int, diff .> 0)
    return minimum(mean(diff_indicator, dims=1))
end

# Calculates the crossed statistical depths between two samples
function depths(S₁, S₂)
    sample = vcat(S₁, S₂)
    n, p = size(sample)
    Z₁ = zeros(n)
    Z₂ = zeros(n)
    for i in 1:n
        point = reshape(sample[i, :], (1, p))
        Z₁[i] = tukey(point, S₁)
        Z₂[i] = tukey(point, S₂)
    end
    return Z₁, Z₂
end
