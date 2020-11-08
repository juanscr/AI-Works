# ====== Global Behavior ====== #
using CSV
using Flux
using Plots

# ====== Transform data ====== #
function get_training_data(train_datax :: Matrix{Float64},
                           train_datay :: Matrix{Float64})
    data = []
    for i in 1:size(train_datax, 1)
        push!(data, (train_datax[i, :], train_datay[i, :]))
    end
    return data
end

# ====== NN ====== #
function nn(ls :: Vector{Int64}, η :: Float64, train_data, ϕs)
    # Layers Construction
    layers = [Dense(length(train_data[1][1]), ls[1], ϕs[1])]
    for i in 2:(length(ls) - 1)
        push!(layers, Dense(ls[i - 1], ls[i], ϕs[i]))
    end
    push!(layers, Dense(ls[end], length(train_data[1][2]), ϕs[end]))

    # NN construction
    chain = Chain(layers...)

    # Initialization for training
    params = Flux.params(chain)
    opt = Flux.Optimise.Descent(η)
    loss(x, y) = Flux.Losses.mse(chain(x), y)
    # Training
    for i in 1:50
        Flux.train!(loss, params, train_data, opt)
    end
end

# ====== Main ====== #
# Data reading
data_csv = CSV.File("data/num-data.csv")
data = zeros(length(data_csv), 7)
i = 1
for row in data_csv
    data[i, 1] = row.Column1
    data[i, 2] = row.Column2
    data[i, 3] = row.Column3
    data[i, 4] = row.Column4
    data[i, 5] = row.Column5
    data[i, 6] = row.Column6
    data[i, 7] = row.Column7
end

# Testing
nn([1], 0.2, get_training_data(data[1:6, :], data[7:end, :]), [σ])
