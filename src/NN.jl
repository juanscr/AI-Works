# ====== Global Behavior ====== #
include("modules/ActFunctions.jl")
include("modules/Brain.jl")
using Random
include("modules/Tools.jl")

# Random seed
Random.seed!(1234)

# ====== Functions ====== #
function get_neuron_combinations(L :: Int64, ls :: Vector{Int64})
    prod = Iterators.product([ls for i in 1:L]...)

    # Transformation to matrix
    grid = reshape(collect(prod), 1, :)
    array = vcat(map(x -> transpose(collect(x)), grid)...)

    return array
end

function write_info_run(grads :: Matrix{Float64}, avg_err :: Vector{Float64},
                        name_file :: String, η :: Float64, ls :: Vector{Int64})
    open(name_file, "a") do io
        # Write run information
        write(io, string("eta,", η, "\n"))
        write(io, "ls,")
        for l in ls
            write(io, string(l, ","))
        end
        write(io, "\n")

        # Write run results
        write(io, "Gradients\n")
        for i in 1:size(grads, 1)
            for j in 1:size(grads, 2)
                write(io, string(grads[i, j], ","))
            end
            write(io, "\n")
        end

        write(io, "Average Error\n")
        for i in 1:length(avg_err)
            write(io, string(avg_err[i], "\n"))
        end
    end
end

function run_brain(η :: Float64, ls :: Vector{Int64}, tr_datax :: Matrix{Float64},
                   tr_datay :: Matrix{Float64}, name_file :: String)
    # Activation function
    sig = Sigmoid()

    # Brain
    brain = Brain(size(tr_datax, 2), size(tr_datay, 2), ls,
                  [sig for i in 1:(length(ls) + 1)])
    grads, avg_err = brain.learn_data(tr_datax, tr_datay, η=η, α=0.1, epocs=50)
    avg_err = reshape(sum(avg_err, dims = 2), :)

    # File writing
    write_info_run(grads, avg_err, name_file, η, ls)
end

# ====== Main ====== #
train_data, _, _ = create_data("data/num-data.csv")
train_datax = train_data[1]
train_datay = train_data[2]

# Parameters for running
ηs = [0.2, 0.5, 0.9]
ls = [1, 2, 3]
neurons = [1, 2, 3]

# Run all combinations
for η in ηs
    for l in ls
        possible_neurons = get_neuron_combinations(l, neurons)
        for i in 1:size(possible_neurons, 1)
            run_brain(η, possible_neurons[i, :], train_datax, train_datay,
                      "../results/nn-results.csv")
        end
    end
end
