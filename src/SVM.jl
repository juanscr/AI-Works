# ==== Global behavior ==== #
using PyCall
using ScikitLearn
include("modules/Tools.jl")

@pyimport joblib
@sk_import svm: SVC

# ==== Main ==== #
# Data
data_name = "data/num-data.csv"
train_data, _, _ = create_data(data_name)
train_datax = train_data[1]
train_datay = train_data[2]

# SVM
kernels = ["linear", "rbf", "poly"]

# Output
prefix = "../results/svm-"
for kernel in kernels
    svm = SVC(kernel = kernel)
    fit!(svm, train_datax, reshape(train_datay, :))
    joblib.dump(svm, string(prefix, kernel, ".joblib"))
end

# Files embedded data
data_name = "data/embedded-data.csv"
train_datae, _, _ = create_data(data_name)
train_dataxe = train_datae[1]
train_dataye = train_datae[2]

# Output
prefix = "../results/svm-emb-"
for kernel in kernels
    svm = SVC(kernel = kernel)
    fit!(svm, train_dataxe, reshape(train_dataye, :))
    joblib.dump(svm, string(prefix, kernel, ".joblib"))
end
