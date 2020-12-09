include("CountryAnalysis.jl")
include("ClusterValidation.jl")

data, tags = read_iris("data/iris.data")

## Statistical Analysis (over original data)
dig = 2
dir = "Data-Analysis-Iris"

# Mean Vector
μ = mean(data, dims = 1)
println("Mean Vector:\n", round.(μ; digits=dig))

# Covariance Matrix
Σ = cov(data)
println("Covariance Matrix:")
print_2Dmatrix(round.(Σ, digits=dig))

# Screes (PCA) - Covariance
js, screes_cov = screes(data)
fig = plotPL(js, screes_cov, xlabel=L"j", ylabel=L"$\lambda_j$")
save(fig, "screes_cov_iris.pdf", dir=dir)

# Screes (PCA) - Correlation
js, screes_cor = screes(data, corr=true)
fig = plotPL(js, screes_cor, xlabel=L"j", ylabel=L"$\lambda_j$")
save(fig, "screes_cor_iris.pdf", dir=dir)

# Multicollinearity
cΣ = cond(Σ)
println("Condition number of Covariance Matrix: ", cΣ)
new_data, rms, R, og_R = remove_collinear(data)

# Homogeneity
res = homogeneity(data, 75, "iris", legend=:topright)
res = homogeneity(new_data, 75, "new_dims_iris", legend=:topright)

## # Unsupervised Learning
# Normalize data
norm_data = normalize(data)

# Dimension Reduction
red_data = normalize(tsne(data, 3))

## Explore space
# Set to 'true' to run the combinations
explore = false
# Using normalized high-dimensions data
dataset_og = "og_iris"
dir_og = "OG-Iris-Analysis"
if explore
    explore_space(norm_data, dataset_og, dir_og)
end

# Using normalized embedded-dimensions data
dataset_red = "red_iris"
dir_red = "Red-Iris-Analysis"
if explore
    explore_space(red_data, dataset_red, dir_red)
end

## Improve clusters
# Original data
Ps_og, Us_og = clusterize(norm_data, 3, dataset_og, dir_og)
# Embedded data
Ps_red, Us_red = clusterize(red_data, 3, dataset_red, dir_red)

## Internal clustering validation
# Original data
print_DB(norm_data, Ps_og, Us_og, euclidean)
print_CH(norm_data, Ps_og, Us_og, euclidean)
# Embedded data
print_DB(red_data, Ps_red, Us_red, euclidean)
print_CH(red_data, Ps_red, Us_red, euclidean)
