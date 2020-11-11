function normalize(data)
    return data ./ maximum(data, dims = 1)
end
