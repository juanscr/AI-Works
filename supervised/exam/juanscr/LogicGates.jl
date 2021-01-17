# ========== Global Beahvior ========== #
include("ActFunctions.jl");
include("Brain.jl");
using LaTeXStrings
using Random

save(x) = savefig(string("figs/", x))

# Fixed seed for testing
Random.seed!(1234)

# ========== Testing ========== #
# Functions
sig = Sigmoid()
tan = Tanh()
linear = Linear()
relu = ReLu()

# Dataset for logic gates
datax = [0 0; 0 1; 1 0; 1 1]
datay = [zeros(size(datax)[1], 1), zeros(size(datax)[1], 1), zeros(size(datax)[1], 1)]
and(x, y) = x & y
or(x, y) = x | y
for i in 1:size(datax)[1]
    datay[1][i, 1] = xor(datax[i, :]...)
    datay[2][i, 1] = and(datax[i, :]...)
    datay[3][i, 1] = or(datax[i, :]...)
end

# Learn data
m = 2
n = 1
l = [2, 2, 2]
brains = [Brain(m, n, l, [sig for i in 1:(length(l) + 1)]),
          Brain(m, n, l, [sig for i in 1:(length(l) + 1)]),
          Brain(m, n, l, [sig for i in 1:(length(l) + 1)])]


# Learning each dataset
grads = []
epocs = [100000, 5000, 5000]
for i in 1:length(brains)
    grad, _, _ = brains[i].learn_data(datax, datay[i], η=0.5, epocs=epocs[i],
                                      α=0.2)
    push!(grads, grad)
end

# Plot gradients
layers = [latexstring("l = ", i + 1) for i in 1:length(brains[1].ω)]
labels = ["xor", "and", "or"]
for i in 1:length(grads)
    for j in 1:size(datax)[1]
        println("Data ", datax[j, 1], ", ", datax[j, 2])
        println(brains[i].propagate(datax[j, :]))
        println()
    end
    println("#########################")
    plot(grads[i],
         label = reshape(layers, 1, length(layers)),
         xlabel = L"$n$",
         ylabel = L"$\sum_i \delta^l_i$")
    save(string("grads-", labels[i], "-mom.pdf"))
end
