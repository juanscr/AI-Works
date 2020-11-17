# ==== Global behavior ==== #
using CSV
using PyCall
using ScikitLearn
include("modules/Tools.jl")

@pyimport joblib
@sk_import tree: DecisionTreeClassifier

# ==== Main ==== #
# Files
data_name = "data/num-dats.csv"
dt_results = "../results/dt-results.csv"

# Data
train_data, _, _ = create_data("data/num-data.csv")
train_datax = train_data[1]
train_datay = train_data[2]

train_datae, _, _ = create_data("data/embedded-data.csv")
train_dataxe = train_datae[1]
train_dataye = train_datae[2]

# Decision tree
dt = DecisionTreeClassifier(criterion = "entropy")
fit!(dt, train_datax, reshape(train_datay, :))

dte = DecisionTreeClassifier(criterion = "entropy")
fit!(dte, train_dataxe, reshape(train_dataye, :))

# Save tree
save_model = "../results/tree.joblib"
joblib.dump(dt, save_model)

save_model = "../results/tree-emb.joblib"
joblib.dump(dte, save_model)
