# Normalize data
function normalize(data)
    return data ./ maximum(data, dims = 1)
end

# Clean indexes
function purge(indexes :: Vector{Int64})
    index_neg = findfirst(x -> x < 0, indexes)
    return indexes[1:(index_neg - 1)]
end
