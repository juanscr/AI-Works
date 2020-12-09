include("Fuzzy-C-Means.jl");
include("K-Means.jl");
include("Mountain.jl");
include("Subtractive.jl");
include("Spectral.jl")

include("Tools.jl");
include("PlotTools.jl");

# Read data
data, countries, header = read_countries("data/country-data.csv");

## Statistical Analysis (over original data)
dig = 2
dir = "Data-Analysis"

# Mean Vector
μ = mean(data, dims = 1)
println(h, "Mean Vector:\n", round.(μ; digits=dig))

# Covariance Matrix
Σ = cov(data)
println("Covariance Matrix:")
print_2Dmatrix(round.(Σ, digits=dig))

# Screes (PCA) - Covariance
js, screes_cov = screes(data)
fig = plotPL(js, screes_cov, xlabel=L"j", ylabel=L"$\lambda_j$")
save(fig, "screes_cov.pdf", dir=dir)

# Screes (PCA) - Correlation
js, screes_cor = screes(data, corr=true)
fig = plotPL(js, screes_cor, xlabel=L"j", ylabel=L"$\lambda_j$")
save(fig, "screes_cor.pdf", dir=dir)

# Homogeneity test
function homogeneity(data, part, filename; legend=:bottomright)
    # Depth scores
    S₁ = data[1:part, :]
    S₂ = data[part+1:end, :]
    Z₁, Z₂ = depths(S₁, S₂)
    fig = ddplot(Z₁, Z₂, "ddplot_"*filename*".pdf", dir=dir, legend=legend)
    return Z₁, Z₂, fig
end

# Multicollinearity
cΣ = cond(Σ);
println(h, "Condition number of Covariance Matrix: ", cΣ);
new_data, rms, R, og_R = remove_collinear(data)

# Homogeneity
res = homogeneity(data, 84, "og")
res = homogeneity(new_data, 84, "new_dims")

## # Unsupervised Learning
# Normalize data
norm_data = normalize(data)

# Dimension Reduction
red_data = normalize(tsne(data, 3))

## Explore space
# Apply Mountain and Subtractive to explore the space with some parameters
function explore_space(data, dataset, dir; n_grid=5)
    # Explore space by Mountain clustering
    σs = [0.1, 0.25, 0.5]
    βs = [0.15, 0.375, 0.75]
    γs = [0.1, 0.3, 0.5]
    nf = true
    for γ in γs
        explore_mountain(data, euclidean, n_grid, σs, βs, γ,
            newfile = nf, dataset=dataset
        )
        nf = false
    end

    # Explore space by Subtractive Clustering
    rₐs = collect(0.1:0.01:0.9)
    ks = explore_subtractive(data, euclidean, rₐs, dataset=dataset)
    fig = plotPL(rₐs, ks, xlabel=L"r_a", ylabel=L"k")
    save(fig, "K_"*dataset*"_subtractive.pdf", dir=dir)
end

# Set to 'true' to run the combinations
explore = false

# Explore space using normalized high-dimensions data
dataset_og = "og_country"
dir_og = "OG-Country-Analysis"
if explore
    explore_space(norm_data, dataset_og, dir_og)
end

# Explore space using normalized embedded-dimensions data
dataset_red = "red_country"
dir_red = "Red-Country-Analysis"
if explore
    explore_space(red_data, dataset_red, dir_red)
end

# Improve clustering with K-Means and Fuzzy-C-Means
function clusterize(data, k, dataset, dir)
    _, protos_KM, U_KM, _, _ = k_means(data, k, euclidean, norm=true)
    _, protos_FCM, U_FCM, _, _ = fuzzy_c_means(data, k, euclidean, 2, norm=true)
    U_spec = spectral_cluster(data, k, euclidean)

    if size(data, 2) > 3
        emb_data = tsne(data, 3)
    else
        emb_data = data
    end

    plot_3DClusters(protos_KM, U_KM, emb_data, "KM_"*dataset*".pdf",
        ("", "", ""), dir=dir, plot_protos = false
    )
    plot_3DClusters(protos_FCM, U_FCM, emb_data, "FCM_"*dataset*".pdf",
        ("", "", ""), dir=dir, plot_protos = false
    )
    plot_3DClusters(nothing, U_spec, emb_data, "Spec_"*dataset*".pdf",
        ("", "", ""), dir=dir, plot_protos = false, k = 3
    )
    return [protos_KM, protos_FCM], [U_KM, U_FCM]
end

# Improve clustering
clusterize(norm_data, 3, dataset_og, dir_og)
clusterize(red_data, 3, dataset_red, dir_red)
