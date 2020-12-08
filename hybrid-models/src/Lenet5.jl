# ========== Packages ========== #
using BSON: @save, @load
using CUDAapi
using Flux
using MLDatasets: MNIST
using Plots
using Statistics

import ProgressMeter

# ========== Functions ========== #
function get_image(input_case)
    return reverse(input_case[:, :, 1], dims=2)'
end

function write_file(xtest, ytest, trained_model)
    open("../results/lenet5.csv", "w") do io
        # Real values
        write(io, "Real\n")
        for y in ytest
            write(io, string(y, "\n"))
        end

        # Trained model output
        write(io, "Model output\n")
        for i in 1:size(xtest)[4]
            input = xtest[:, :, :, i]
            y_model = trained_model(input)

            # Printing
            for y in y_model
                write(io, string(y, ","))
            end
            write(io, "\n")
        end
    end
end

function extract_activation(input, trained_model, layers)
    # X-test graph
    input_cpu = input |> cpu
    heatmap(get_image(input_cpu[:, :, 1]), color=:greys)
    savefig("../article/figs/input-act-space.pdf")

    # Plotting activation spaces
    for i in 1:length(layers)
        layer = layers[i]
        act_space = trained_model[1:layer](input) |> cpu

        mean_act = mean(act_space, dims = 3)
        heatmap(get_image(mean_act[:, :, 1]), color=:greys)
        savefig(string("../article/figs/act-space-", i, ".pdf"))
    end
end

function extract_model_info(xtest, ytest, trained_model, layers)
    input = xtest[:, :, :, 2] |> gpu
    extract_activation(input, trained_model, layers)

    # Write file
    trained_model_cpu = trained_model |> cpu
    write_file(xtest, ytest, trained_model_cpu)
end

function train(lenet5, train_loader; η=3e-4, epocs=200, create=true)
    if !create
        @load "lenet5.bson" model
        return model |> gpu
    end

    # Cuda model
    model = lenet5 |> gpu

    # Parameters
    ps = Flux.params(model)
    opt = Flux.Optimise.Descent(η)
    loss(y_appr, y) = Flux.Losses.mse(y_appr, y)

    # Training
    p = ProgressMeter.Progress(epocs)
    for epoc in 1:epocs
        for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(ps) do
                y_appr = model(x)
                loss(y_appr, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        ProgressMeter.next!(p)
    end

    model_cpu = model |> cpu
    @save "lenet5.bson" model_cpu
    return model
end

# ========== Parameters ========== #
# Number of classes
nclass = 10

# Image size
x_pixels = 28
y_pixels = 28
depth = 1
imgsize = (x_pixels, y_pixels, depth)

# Data
xtrain, ytrain = MNIST.traindata(Float64)
xtrain = reshape(xtrain, imgsize..., :)

# Creating batch
ytrain = Flux.onehotbatch(ytrain, 0:9)
train_loader = Flux.Data.DataLoader(xtrain, ytrain, batchsize=128)

# ========== Lenet 5 ========== #
# Layers
C1 = Conv((5, 5), depth => 6, relu)
S2 = MaxPool((2, 2))
C3 = Conv((5, 5), 6 => 16, relu)
S4 = MaxPool((2, 2))
C5 = Dense(256, 120, relu)
F6 = Dense(120, 84, relu)
out = Dense(84, nclass)

# Architecture
aux_reshape1 = x -> reshape(x, imgsize..., :)
aux_reshape2 = x -> reshape(x, :, size(x, 4))
lenet5 = Chain(aux_reshape1, C1, S2, C3, S4, aux_reshape2, C5, F6, out)

# ========== Training ========== #
create = true
model = train(lenet5, train_loader, η=0.01, epocs=200, create=create)

# ========== Exporting info ========== #
# Testing data
xtest, ytest = MNIST.testdata(Float64)
xtest = reshape(xtest, imgsize..., :)

# Layers to extract activation space
layers = collect(2:5)

# Extracting model information
extract_model_info(xtest, ytest, model, layers)
