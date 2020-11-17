# ==== Global behavior ==== #
using CSV
using PyCall
using ScikitLearn
using Tables
include("modules/Tools.jl")

@pyimport joblib
@sk_import svm: SVC

# ==== Functions ==== #
function export_error(ms :: Vector{Int64}, δ :: Float64, VC :: Float64,
                      name_file :: String)
    aux_func(m) = calculate_min_error(m, δ, VC)
    εs = map(aux_func, ms)

    # Output
    out = [εs ms]
    CSV.write(name_file, Tables.table(out))
end

function run_data(prefix :: String, train_datax :: Matrix{Float64},
                  train_datay :: Vector{Float64}, ms :: Vector{Int64},
                  δ :: Float64)
    # Possible kernles
    kernels = ["linear", "rbf", "poly"]
    n = size(train_datax, 2)
    VCs = [n + 1, nothing, (n + 3) * (n + 2) * (n + 1) / 6]

    for i in 1:length(kernels)
        svm = SVC(kernel = kernels[i])
        fit!(svm, train_datax, reshape(train_datay, :))
        joblib.dump(svm, string(prefix, kernels[i], ".joblib"))

        if !isnothing(VCs[i])
            export_error(ms, δ, VCs[i], string("../results/", prefix,
                                               kernels[i], "-eps.csv"))
        end
    end
end

# ==== Main ==== #
### High dimensional Data
data_name = "data/num-data.csv"
train_data, _, _ = create_data(data_name)
train_datax = train_data[1]
train_datay = reshape(train_data[2], :)

# Output
prefix = "../results/svm-"

# Error aproximation data
m = size(train_datax, 1)
ms = trunc.(Int, floor(0.8*m):1:floor(8*m))
δ = 0.3

run_data(prefix, train_datax, train_datay, ms, δ)


### Embedded data
data_name = "data/embedded-data.csv"
train_datae, _, _ = create_data(data_name)
train_dataxe = train_datae[1]
train_dataye = reshape(train_datae[2], :)

# Output
prefix = "../results/svm-emb-"

run_data(prefix, train_dataxe, train_dataye, ms, δ)
