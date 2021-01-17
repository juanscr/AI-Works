using Distributions

# Normalize data with max|x_i| -> [-1,1] hypercube.
function normalize(data)
    return data ./ maximum(abs.(data), dims=1)
end

# Initialize a nxm matrix with [-1,1] uniformly distributed random numbers.
function init_weights(n, m)
    return rand(Uniform(-1, 1), (n, m))
end

function read_data(file)
    data_file = open(file, "r") do io
        read(io, String)
    end
    lines = split(data_file, "\n")
    lines = lines[1:end-1]
    data = split.(lines, ",")
    header = split(data[1], ",")[1]
    data = data[2:end]
    n = size(data)[1]
    p = size(data[1])[1]
    data = [parse.(Float64, point) for point in data]
    aux = reshape(data[1], (1, p))
    for j in 2:n
        aux = vcat(aux, reshape(data[j], (1, p)))
    end
    return aux, header
end
