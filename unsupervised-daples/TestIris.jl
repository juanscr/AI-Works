include("Fuzzy-C-Means.jl");
include("K-Means.jl");
include("Mountain.jl");
include("Subtractive.jl");

include("PlotTools.jl");
include("Tools.jl");

## Read Iris Dataset
data, tags = read_iris("data/iris.data")
dist = euclidean

## Mountain
n_grid = 10
σ = 0.1
β = 0.15
data, protos_mount, U_mount, grid, aux, evals_mount = mountain_cluster(
    data, dist, n_grid, σ, β; n_it = 10, γ=0.4, arg=nothing, norm=true
)

## Subtractive
rₐ = 0.5
_, protos_sub, U_sub, evals_sub = subtractive_cluster(
    data, dist, rₐ; ϵ_up = 0.5, ϵ_down = 0.15, n_it=10, arg=nothing, norm=true
)
k = size(protos_sub, 1)

## K-Means
_, protos_kmeans, U_kmeans, improvs_kmeans = k_means(
    data, k, dist; γ=0.001, arg=nothing, norm=true
)

## Fuzzy C-Means
m = 2
_, protos_cmeans, U_cmeans, improvs_cmeans = fuzzy_c_means(
    data, k, dist, m; γ=0.001, arg=nothing, norm=true
)

## Output Plots
# Generate 3D proyections
plot_nDimGroups(protos_mount, U_mount, data, "mountain-Iris", dir="Iris-Mount")
plot_nDimGroups(protos_sub, U_sub, data, "subtractive-Iris", dir="Iris-Sub")
plot_nDimGroups(protos_kmeans, U_kmeans, data, "KM-Iris", dir="Iris-KM")
plot_nDimGroups(protos_cmeans, U_cmeans, data, "FCM-Iris", dir="Iris-FCM")

# Generate GIFs for densities
generate_density_gif(aux, evals_mount, "iris.gif", dir="IrisDensityGifs")
