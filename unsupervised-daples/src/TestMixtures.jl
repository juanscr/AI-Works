include("Fuzzy-C-Means.jl");
include("K-Means.jl");
include("Mountain.jl");
include("Subtractive.jl");

include("Tools.jl");
include("PlotTools.jl");

# Create dataset from a Mixture of Multivariate Normal Distributions
#   k -> Number of different distributions (clusters)
#   p -> Number of variables (set as 2 for standard plotting)
#   n -> Size of EACH of the k samples
#   data -> (k*n, p) data matrix
p = 2;
k = 3;
n = 500;
data = generate_MvN_Mixture(p, k, n)

# Define the similarity measure in the procedure
#   dist -> Function. It can be set to 'euclidean', 'manhattan', 'mahal' or
#           'pnorm'. More similarities will be implemented.
#   arg -> Extra parameter of the 'dist' function. It affects only the 'mahal'
#          and the 'pnorm' similarity measures, as the INVERSE of the weighting
#          matrix and the value of p, respectively.
dist = euclidean
arg = nothing

# Directory inside 'figs' where the following results will be stored
dir = "MvN-Mixture"

## Testing
# Based on the dataset created, each clustering algorithm is applied and the
# results are saved on a 2D plot, which is saved inside 'dir', within 'figs'.
# The 'plot_k_clusters' method can only plot 2D data (see 'Tools.jl').

# K-Means
data, protos, U, improvs, protos_iter = k_means(data, k, dist, arg=arg)
plot_k_clusters(protos, U, data, "mixture_k_means.pdf", dir=dir;
    protos_iter = protos_iter
)

# Mountain
data, protos, U, _, _, _ = mountain_cluster(data, dist, 25, 0.1, 0.2, arg=arg)
plot_k_clusters(protos, U, data, "mixture_mountain.pdf", dir=dir)

# Fuzzy C Means
data, protos, U, improvs, protos_iter = fuzzy_c_means(data, k, dist, 2, arg=arg)
plot_k_clusters(protos, U, data, "mixture_fuzzy_c_means.pdf", dir=dir;
    protos_iter = protos_iter
)

# Subtractive
data, protos, U, evals = subtractive_cluster(data, dist, 0.5, arg=arg)
plot_k_clusters(protos, U, data, "mixture_subtractive.pdf", dir=dir)

# Test GIF generation
p = 2;
k = 10;
n = 250;
data = generate_MvN_Mixture(p, k, n; b=100)
data, protos, U, grid, aux, evals = mountain_cluster(data, dist,
    25, 0.1, 0.2, arg=arg
)
generate_density_gif(aux, evals, "GIF_density_MvNMixture.gif", dir=dir)

# Test spectral with simple circle-based data (plots in same folder)
test_spectral()
