using CSV

# Normalize data
function normalize(data)
    return data ./ maximum(data, dims = 1)
end

# Clean indexes
function purge(indexes :: Vector{Int64})
    index_neg = findfirst(x -> x < 0, indexes)
    return indexes[1:(index_neg - 1)]
end

# Get data
function create_data(name_file :: String, sep = true)
    # Data reading
    data_csv = CSV.File("data/num-data.csv")
    data = zeros(length(data_csv), 7)
    k = 1
    for row in data_csv
        data[k, 1] = row.Column1
        data[k, 2] = row.Column2
        data[k, 3] = row.Column3
        data[k, 4] = row.Column4
        data[k, 5] = row.Column5
        data[k, 6] = row.Column6
        data[k, 7] = row.Column7
        k += 1
    end
    data = normalize(data)

    if !sep
        return data
    end

    # Separate data
    indexes_csv = CSV.File("data/indexes.csv")
    indexes_tr = trunc.(Int, indexes_csv.columns[1])
    indexes_te = purge(trunc.(Int, indexes_csv.columns[2]))
    indexes_val = purge(trunc.(Int, indexes_csv.columns[3]))

    # Create data
    train_datax = data[indexes_tr, 1:6]
    train_datay = data[indexes_tr, 7:end]

    test_datax = data[indexes_te, 1:6]
    test_datay = data[indexes_te, 7:end]

    val_datax = data[indexes_val, 1:6]
    val_datay = data[indexes_val, 7:end]

    return (train_datax, train_datay), (test_datax, test_datay),
           (val_datax, val_datay)
end
