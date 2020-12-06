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
                        name_file :: String, η :: Float64, ls :: Vector{Int64},
                        Z :: Vector{Float64})
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

        write(io, "Output\n")
        for i in 1:length(Z)
            write(io, string(Z[i], "\n"))
        end
    end
end

function write_brain(brain :: Brain, grads :: Matrix{Float64},
                     avg_err :: Matrix{Float64}, η :: Float64,
                     name_file :: String, file_data :: String)
    # Number of neurons
    ls = Vector{Int64}([])
    for ω in brain.ω[1:(end-1)]
        push!(ls, size(ω, 2))
    end

    # Output of NN
    datax, _ = create_data(file_data, sep = false)
    Z = zeros(size(datax, 1))
    for i in 1:size(datax, 1)
        Z[i] = brain.propagate(datax[i, :])[1, 1]
    end

    # Write data
    write_info_run(grads, reshape(avg_err, :), name_file, η, ls, Z)
end

function run_combinations(ηs :: Vector{Float64}, ls :: Vector{Int64},
                          neurons :: Vector{Int64}, tr_datax :: Matrix{Float64},
                          tr_datay :: Matrix{Float64}, selection_criterion,
                          number_of_brains :: Int64)
    # Activation functions
    sig = Sigmoid()

    # Run all combinations
    selected_brains = []
    for η in ηs
        for l in ls
            possible_neurons = get_neuron_combinations(l, neurons)
            for i in 1:size(possible_neurons, 1)
                ls_aux = possible_neurons[i, :]

                # Run brain
                brain = Brain(size(tr_datax, 2), size(tr_datay, 2), ls_aux,
                            [sig for i in 1:(length(ls_aux) + 1)])
                grads, avg_err = brain.learn_data(tr_datax, tr_datay, η=η,
                                                α=0.1, epocs=50)
                avg_err = sum(avg_err / size(avg_err, 2), dims = 2)

                # Fill list
                if length(selected_brains) < number_of_brains
                    push!(selected_brains, (brain, grads, avg_err, η))
                    sort!(selected_brains, by = selection_criterion)
                    continue
                end

                # Replace list
                err = selection_criterion((brain, grads, avg_err, η))

                # Best ones
                for i in 1:(number_of_brains - 1)
                    if selection_criterion(selected_brains[i]) > err
                        selected_brains[(i+1):(end - 1)] = selected_brains[i:(end - 2)]
                        selected_brains[i] = (brain, grads, avg_err, η)
                        break
                    end
                end

                # Worst one
                if selection_criterion(selected_brains[end]) < err
                    selected_brains[end] = (brain, grads, avg_err, η)
                end
            end
        end
    end
    return selected_brains
end

# ====== Main ====== #
train_data, _, _ = create_data("data/num-data.csv")
train_datax = train_data[1]
train_datay = train_data[2]

train_data_emb, _, _ = create_data("data/embedded-data.csv")
train_dataxe = train_data_emb[1]
train_dataye = train_data_emb[2]

# Parameters for running
ηs = [0.2, 0.5, 0.9]
ls = [1, 2, 3]
neurons = [1, 2, 3]

# Selection
number_of_brains = 3
selection_criterion = x -> x[3][end]

# Run
selected_brains = run_combinations(ηs, ls, neurons, train_datax, train_datay,
                                   selection_criterion, number_of_brains)
selected_brainse = run_combinations(ηs, ls, neurons, train_dataxe, train_dataye,
                                    selection_criterion, number_of_brains)

# Write files
for selected_brain in selected_brains
    write_brain(selected_brain..., "../results/nn-results.csv",
                "data/num-data.csv")
end

for selected_brain in selected_brainse
    write_brain(selected_brain..., "../results/nn-results-emb.csv",
                "data/embedded-data.csv")
end
