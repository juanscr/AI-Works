# ==== Global behavior ==== #
using CSV
using PyCall
using ScikitLearn
include("modules/Tools.jl")

@pyimport joblib
@sk_import tree: DecisionTreeClassifier

# ==== Function ==== #
function read_meshgrid(name_file :: String)
    rows = CSV.Rows(name_file)

    # Mesh
    mesh = []
    for row in rows
        row_data = zeros(6)
        for i in 1:6
            row_data[i] = parse(Float64, row[i])
        end
        push!(mesh, row_data)
    end
    return collect(transpose(hcat(mesh...)))
end

# ==== Main ==== #
# Files
data_name = "data/num-dats.csv"
dt_results = "../results/dt-results.csv"

# Data
train_data, _, _ = create_data("data/num-data.csv", norm = false)

train_datax = train_data[1]
train_datay = train_data[2]

# Decision tree
dt = DecisionTreeClassifier(criterion = "entropy")
mesh = read_meshgrid("../results/meshgrid.csv")
fit!(dt, train_datax, reshape(train_datay, :))
generate_results(dt, dt_results, data_name, mesh)

# Save tree
save_model = "../results/tree.joblib"
joblib.dump(dt, save_model)
