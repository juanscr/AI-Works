include("BackPropagation.jl")
include("PlotTools.jl")
include("Tools.jl")

full_data, header = read_data("data/datosimulacionC.csv");
full_data = normalize(full_data);

# Input data: Variables Sp, Xv and OD
full_X = full_data[:, [1, 2, 3, 5, 6]];
N, m = size(full_X);

# Output data: Variable X
full_Y = reshape(full_data[:, 4], (N, 1));

# Training set 60%
training = [];
# Test 20%
test = [];
# Validation 20%
validation = [];

for i in 1:N
    r = rand()
    if r <= 0.6
        append!(training, i)
    elseif r <= 0.8
        append!(test, i)
    else
        append!(validation, i)
    end
end

training_X = full_X[training, :]
training_Y = full_Y[training, :]
test_X = full_X[test, :]
validation_X = full_X[validation, :]

# Parameters
L = [5, 5];
ϕ_sigm(x) = 1.0 ./ (1.0 .+ exp.(-x));
∂ϕ_sigm(x) = ϕ_sigm(x).*(1 .- ϕ_sigm(x));

ϕ = [ϕ_sigm for i in 1:size(L, 1) + 2];
∂ϕ = [∂ϕ_sigm for i in 1:size(L, 1) + 2];

## First Eta
η = -0.2
Vs, Φs, Ws, ∇s, Ξ = nn(training_X, training_Y, L, ϕ, ∂ϕ; η=η, α=0, s=50);

# Average Error
plot_ξav(Ξ, save_fig=true, out_file="Eav.pdf", dir="eta_$η")

# Plot Gradients
plot_∇s(∇s, save_fig=true, out_file="grads.pdf", dir="eta_$η")

# NN-Training Output
Y_nn = zeros(length(training), 1)
for i in 1:length(training)
    Y_nn[i, 1] = propagate(training_X[i, :], Ws, ϕ, 2)[2][end][1]
end
# NN-Test Output
Y_nn_test = zeros(length(test), 1)
for i in 1:length(test)
    Y_nn_test[i, 1] = propagate(test_X[i, :], Ws, ϕ, 2)[2][end][1]
end
# NN-Validation Output
Y_nn_valid = zeros(length(validation), 1)
for i in 1:length(validation)
    Y_nn_valid[i, 1] = propagate(validation_X[i, :], Ws, ϕ, 2)[2][end][1]
end

# NN vs. Real
fig = plotPL(0:1/(length(training)-1):1, training_Y[:, 1], color=:auto, label="Real")
fig = plotPL(0:1/(length(training)-1):1, Y_nn[:, 1], fig, color=:auto, label="NN-Training")
fig = plotPL(0:1/(length(test)-1):1, Y_nn_test[:, 1], fig, color=:auto, label="NN-Test")
fig = plotPL(0:1/(length(validation)-1):1, Y_nn_valid[:, 1], fig, color=:auto, label="NN-Validation")
save(fig, "output.pdf", dir="eta_$η")

## Second Eta
η = -0.9
Vs, Φs, Ws, ∇s, Ξ = nn(training_X, training_Y, L, ϕ, ∂ϕ; η=η, α=0, s=50);

# Average Error
plot_ξav(Ξ, save_fig=true, out_file="Eav.pdf", dir="eta_$η")

# Plot Gradients
plot_∇s(∇s, save_fig=true, out_file="grads.pdf", dir="eta_$η")

# NN-Training Output
Y_nn = zeros(length(training), 1)
for i in 1:length(training)
    Y_nn[i, 1] = propagate(training_X[i, :], Ws, ϕ, 2)[2][end][1]
end
# NN-Test Output
Y_nn_test = zeros(length(test), 1)
for i in 1:length(test)
    Y_nn_test[i, 1] = propagate(test_X[i, :], Ws, ϕ, 2)[2][end][1]
end
# NN-Validation Output
Y_nn_valid = zeros(length(validation), 1)
for i in 1:length(validation)
    Y_nn_valid[i, 1] = propagate(validation_X[i, :], Ws, ϕ, 2)[2][end][1]
end

# NN vs. Real
fig = plotPL(0:1/(length(training)-1):1, training_Y[:, 1], color=:auto, label="Real")
fig = plotPL(0:1/(length(training)-1):1, Y_nn[:, 1], fig, color=:auto, label="NN-Training")
fig = plotPL(0:1/(length(test)-1):1, Y_nn_test[:, 1], fig, color=:auto, label="NN-Test")
fig = plotPL(0:1/(length(validation)-1):1, Y_nn_valid[:, 1], fig, color=:auto, label="NN-Validation")
save(fig, "output.pdf", dir="eta_$η")
