using Clustering
using CSV
using Plots
using Random
using Tables
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
function normalize(A)
    return A ./ maximum(abs.(A), dims = 1)
end

function plot_data(data)
    scatter(data[:, 1], data[:, 2],
            markersize = 5,
            markerstrokewidth = 0,
            label = false,
            color = "black",
            grid = false)
end;

data_emb = tsne(normalize(data))
plot_data(data_emb)
savefig("figs/embedded-data.pdf")

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
