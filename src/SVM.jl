# ==== Global behavior ==== #
using ScikitLearn
include("modules/Tools.jl")

@sk_import svm: SVC

# ==== Functions ==== #
function generate_results(svm, name_file :: String, data_name :: String,
                          kernel :: String, n = 10)
    datax, _ = create_data(data_name, sep = false)
    class = svm.predict(datax)

    # File
    open(name_file, "a") do io
        write(io, string("Kernel,", kernel, "\n"))

        # Classification
        write(io, "Classification\n")
        for num in class
            write(io, string(num, "\n"))
        end

        # Meshgrid
        mesh = meshgrid(datax, n)
        Z = decision_function(svm, mesh)
    end
end

# ==== Main ==== #
# Data
train_data, _, _ = create_data("data/num-data.csv")

train_datax = train_data[1]
train_datay = train_data[2]

# SVM
svm = SVC(kernel = "linear")
fit!(svm, train_datax, reshape(train_datay, :))
generate_results(svm, "../results/svm-results.csv", "data/num-data.csv",
                 "Linear")
