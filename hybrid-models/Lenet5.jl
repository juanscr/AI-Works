# Packages
using CUDAapi
using Flux
using MLDatasets: MNIST
import ProgressMeter

# Number of classes
nclass = 10

# Image size
x_pixels = 28
y_pixels = 28
depth = 1
imgsize = (x_pixels, y_pixels, depth)

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
lenet5 = Chain(aux_reshape1, C1, S2, C3, S4, aux_reshape2, C5, F6, out);

# Data
xtrain, ytrain = MNIST.traindata(Float64)
xtrain = reshape(xtrain, imgsize..., :)

# Creating batch
ytrain = Flux.onehotbatch(ytrain, 0:9)
train_loader = Flux.Data.DataLoader(xtrain, ytrain, batchsize=128);

# Cuda model
model = lenet5 |> gpu

# Parameters
ps = Flux.params(lenet5)
opt = Flux.Optimise.Descent(3e-4)
loss(y_appr, y) = Flux.Losses.mse(y_appr, y)
epocs = 5

# Training
for epoc in 1:epocs
    p = ProgressMeter.Progress(length(train_loader))

    for (x, y) in train_loader
        x, y = x |> gpu, y |> gpu
        gs = Flux.gradient(ps) do
            y_appr = model(x)
            loss(y_appr, y)
        end
        Flux.Optimise.update!(opt, ps, gs)
        ProgressMeter.next!(p)
    end
end
