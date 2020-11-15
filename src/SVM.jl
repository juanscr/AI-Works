# ==== Global behavior ==== #
using ScikitLearn
include("modules/Tools.jl")

@sk_import svm: SVC

# ==== Functions ==== #
function generate_meshgrid(data_name :: String, name_file :: String; n = 10)
    datax, _ = create_data(data_name, sep = false)
    mesh = meshgrid(datax, n)
    open(name_file, "w") do io
        write(io, "Meshgrid\n")
        for i in 1:size(mesh, 1)
            for j in 1:size(mesh, 2)
                write(io, string(mesh[i, j], ","))
            end
            write(io, "\n")
        end
    end
    return mesh
end


# ==== Main ==== #
# Data
train_data, _, _ = create_data("data/num-data.csv")

train_datax = train_data[1]
train_datay = train_data[2]

# SVM
kernels = ["linear", "rbf", "poly"]

# Files
data_name = "data/num-dats.csv"
meshgrid_file = "../results/meshgrid.csv"
svm_results = "../results/svm-results.csv"

# Output
mesh = generate_meshgrid(data_name, meshgrid_file, n = 5)
for kernel in kernels
    svm = SVC(kernel = kernel)
    fit!(svm, train_datax, reshape(train_datay, :))
    extra_info = string("Kernel,", kernel)
    generate_results(svm, svm_results, data_name, kernel, mesh,
                     extra_info=extra_info)
end
