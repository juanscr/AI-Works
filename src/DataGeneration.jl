using Clustering
using CSV
using Plots
using Random
using Tables
include("modules/Tools.jl")
using TSne

# Fixed seed
Random.seed!(1234)

# ============ Transform data ============ #
function get_es(es_string)
    r = Random.rand(Float64)
    if es_string == "No decirle a nadie."
        return -255 + r*55
    elseif es_string == "Pido consejo a alguien cercano sobre cómo solucionarlo."
        return 100 + r*100
    elseif es_string == "Hablo lo que siento con la persona que tengo rabia."
        return 200 + r*55
    else
        return -200 + r*100
    end
end;

function get_output(output)
    r = Random.rand(Float64)
    if output == "Mucho."
        return 75 + r*25
    elseif output == "Muy poco."
        return r*25
    else
        return 40 + r*20
    end
end;

# ============ Read data ============ #
rows = CSV.Rows("data/results.csv")
data = []
for row in rows
    row_data = zeros(6)
    row_data[1] = parse(Float64, row[2])
    row_data[2] = parse(Float64, row[3])
    row_data[3] = get_es(row[4])
    row_data[4] = get_output(row[5])
    row_data[5] = get_output(row[6])
    row_data[6] = get_output(row[7])
    push!(data, row_data)
end;
data = collect(transpose(hcat(data...)))

# ============ Embedded Data ============ #
data_emb = normalize(tsne(normalize(data)))
CSV.write("data/embedded-data.csv", Tables.table(data_emb))

# ============ Write new data ============ #
function find_membership(data, dist, prototypes, args)
    k = size(prototypes)[1]
    u = zeros(Float64, k, size(data)[1])
    distances = zeros(Float64, k, size(data)[1])
    for j in 1:k
        distances[j, :] = dist(data, prototypes[j, :], args...)
    end
    indexes_min = argmin(distances, dims = 1)
    u[indexes_min] .= 1

    return u, distances
end

function euclidean(data, center; norm = false)
    center_aux = reshape(center, 1, length(center))
    diff = data .- center_aux
    value = sum(diff .^ 2, dims = 2)
    if norm
        return value
    end

    return sqrt.(value)
end;

# Clustering
protos = transpose(Clustering.fuzzy_cmeans(transpose(data), 4, 2).centers)
u, _ = find_membership(data, euclidean, protos, [])

# CSV creation
final_data = [data reshape(u[1, :], size(data, 1), 1)]
CSV.write("data/num-data.csv", Tables.table(final_data))


# ============ Separate data ============ #
indexes = 1:size(data, 1)

# Training
n_training = trunc(Int, 0.6 * size(data)[1])
indexes_tr = sort(sample(collect(indexes), n_training, replace = false))

# Testing
n_testing = trunc(Int, 0.2 * size(data)[1])
indexes_avail = []
for i in indexes
    if i ∉ indexes_tr
        push!(indexes_avail, i)
    end
end
indexes_te = sort(sample(indexes_avail, n_testing, replace = false))

# Validation
indexes_val = []
for i in indexes
    if i ∉ indexes_tr && i ∉ indexes_te
        push!(indexes_val, i)
    end
end
indexes_val = sort(indexes_val)

# Fill holes
indexes_te = vcat(indexes_te, -ones(n_training - n_testing))
indexes_val = vcat(indexes_val, -ones(n_training - length(indexes_val)))
indexes_mat = [indexes_tr indexes_te indexes_val]

# File of indexes
CSV.write("data/indexes.csv", Tables.table(indexes_mat))
