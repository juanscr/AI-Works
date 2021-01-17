# Returns a list with matrices corresponding to the data in the i-th cluster
function get_data_clusters(data, U)
    k, n = size(U)
    clusters = []
    for j in 1:k
        aux = argmax(U, dims=1)[[x[1] for x in argmax(U, dims=1)] .== j]
        data_cluster = data[[x[2] for x in aux], :]
        push!(clusters, data_cluster)
    end
    return clusters
end

# Calculates de Davies-Bouldin intra-cluster index
function DB(data, protos, U, dist; arg=nothing)
    k, n = size(U)
    _, p = size(data)
    clusters = get_data_clusters(data, U)
    μs = zeros(k)
    for i in 1:k
        Cᵢ = clusters[i]
        cᵢ = reshape(protos[i, :], (1, p))
        μs[i] = sum(dist(Cᵢ, cᵢ, arg)) / size(Cᵢ, 1)
    end

    dbs = zeros(k)
    for i in 1:k
        aux = zeros(k)
        cᵢ = reshape(protos[i, :], (1, p))
        for j in 1:k
            cⱼ = reshape(protos[j, :], (1, p))
            if i == j
                continue
            end
            dist_protos = dist(cᵢ, cⱼ, arg)[1]
            aux[j] = (μs[i] + μs[j]) / dist_protos
        end
        dbs[i] = maximum(aux)
    end
    return sum(dbs) / k
end

# Prints the above method for the K-Means and Fuzzy-C-Means results
function print_DB(data, Ps, Us, dist; arg=nothing)
    print("")
    tags = ["K-Means", "Fuzzy-C-Means"]
    for i in 1:size(Ps, 1)
        s = DB(data, Ps[i], Us[i], dist, arg=arg)
        println("Davies-Bouldin index for "*tags[i]*": "*string(s))
    end
end

# Calculates de Calinski-Harabasz
function CH(data, protos, U, dist; arg=nothing)
    k, n = size(U)
    _, p = size(data)
    clusters = get_data_clusters(data, U)
    c = mean(data, dims=1)

    dists = vec(dist(protos, c, arg)).^2
    ns = [size(clusters[i], 1) for i in 1:k]
    num = sum(ns .* dists)

    denom = 0
    for i in 1:k
        cᵢ = reshape(protos[i, :], (1, p))
        denom += sum(dist(clusters[i], cᵢ, arg).^2)
    end
    ind = num / denom
    return ind * (n - k)/(k - 1)
end

# Prints the above method for the K-Means and Fuzzy-C-Means results
function print_CH(data, Ps, Us, dist; arg=nothing)
    print("")
    tags = ["K-Means", "Fuzzy-C-Means"]
    for i in 1:size(Ps, 1)
        s = CH(data, Ps[i], Us[i], dist, arg=arg)
        println("Calinski-Harabasz index for "*tags[i]*": "*string(s))
    end
end
