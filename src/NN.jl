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

    Δs = zeros(50, trunc(Int, length(params) / 2))

    # Training
    for i in 1:100
        Flux.train!(loss, params, train_data, opt)

        # Gradients
        gs = gradient(params) do
            loss(train_data[end]...)
        end

        for j in 1:2:length(params)
            Δs[i, trunc(Int, (j + 1)/2)] = sum(gs[params[j]])
        end
        break
    end

    # Gradients
    return chain, Δs
end

# ====== Main ====== #
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
    global k += 1
end

# Data testing
dat0 = [([0, 0], [0]),
        ([1, 0], [1]),
        ([0, 1], [1]),
        ([1, 1], [1])]

# Testing
chain, Δs = nn([2], 0.5, dat0, [σ, σ])
plot(Δs)
savefig("testing.pdf")

for dat1 in dat0
    println(chain(dat1[1]), " == ", dat1[2])
end
