using CSV
using Ipopt
using JuMP

# Normalize data
function normalize(data)
    return data ./ maximum(abs.(data), dims = 1)
end

# Clean indexes
function purge(indexes :: Vector{Int64})
    index_neg = findfirst(x -> x < 0, indexes)
    return indexes[1:(index_neg - 1)]
end

# Get data
function create_data(name_file :: String; sep = true, norm = true,
                     indexes_file="data/indexes.csv")
    # Data reading
    data_col = CSV.File(name_file).columns
    data = zeros(length(data_col[1]), length(data_col))
    for i in 1:length(data_col)
        data[:, i] = data_col[i]
    end
    if norm
        data = normalize(data)
    end

    if !sep
        return data[:, 1:(end - 1)], data[:, end:end]
    end

    # Separate data
    indexes_csv = CSV.File(indexes_file)
    indexes_tr = trunc.(Int, indexes_csv.columns[1])
    indexes_te = purge(trunc.(Int, indexes_csv.columns[2]))
    indexes_val = purge(trunc.(Int, indexes_csv.columns[3]))

    # Create data
    train_datax = data[indexes_tr, 1:(end - 1)]
    train_datay = data[indexes_tr, end:end]

    test_datax = data[indexes_te, 1:(end - 1)]
    test_datay = data[indexes_te, end:end]

    val_datax = data[indexes_val, 1:(end - 1)]
    val_datay = data[indexes_val, end:end]

    return (train_datax, train_datay), (test_datax, test_datay),
           (val_datax, val_datay)
end

# Calculate ε-dimension
function calculate_min_error(m :: Int64, δ :: Float64, VC :: Float64)
    model = Model(with_optimizer(Ipopt.Optimizer, print_level = 0))

    # Variables
    @variable(model, 0 <= ε)

    # Constraints
    @constraint(model, 4 * log(2 / δ) / m <= ε)
    @NLconstraint(model, 8 * VC * log(13 / ε) / ε <= m)

    # Objective function
    @objective(model, Min, ε^2)

    # Optimization
    JuMP.optimize!(model)
    return JuMP.value(ε)
end
