# ==== Global behavior ==== #
using PyCall
using ScikitLearn
include("modules/Tools.jl")

@pyimport joblib
@sk_import svm: SVC

# ==== Main ==== #
# Data
train_data, _, _ = create_data("data/num-data.csv")

train_datax = train_data[1]
train_datay = train_data[2]

# SVM
kernels = ["linear", "rbf", "poly"]

# Files
data_name = "data/num-data.csv"

# Output
prefix = "../results/svm-"
for kernel in kernels
    svm = SVC(kernel = kernel)
    fit!(svm, train_datax, reshape(train_datay, :))
    joblib.dump(svm, string(prefix, kernel, ".joblib"))
end

# Files embedded data
data_name = "data/embedded-data.csv"

# Output
prefix = "../results/svm-emb-"
for kernel in kernels
    svm = SVC(kernel = kernel)
    fit!(svm, train_datax, reshape(train_datay, :))
    joblib.dump(svm, string(prefix, kernel, ".joblib"))
end
