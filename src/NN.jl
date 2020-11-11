# ====== Global Behavior ====== #
include("modules/ActFunctions.jl")
using CSV
include("modules/Brain.jl")
using Plots
using Random

Random.seed!(1234)

# ====== Transform data ====== #
function get_training_data(train_datax :: Matrix{Float64},
                           train_datay :: Matrix{Float64})
    data = []
    for i in 1:size(train_datax, 1)
        push!(data, (train_datax[i, :], train_datay[i, :]))
    end
    return data
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
data = data ./ maximum(data, dims = 1)

# Activation Functions
sig = Sigmoid()
relu = ReLu()
tant = Tanh()
h = Heaviside()

# Brain
brain = Brain(6, 1, [2], [sig, sig])
grads, avg_err = brain.learn_data(data[:, 1:6], data[:, 7:end],
                                            epocs=1000, η=0.9, α=0.1)
plot(grads)
savefig("testing.pdf")
plot(sum(avg_err, dims = 2))
savefig("testing2.pdf")
