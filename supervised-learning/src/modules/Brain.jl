# ========== Global Beahvior ========== #
include("ActFunctions.jl")
using Distributions
using LinearAlgebra
using Random

# ========== Neural Network ========== #
mutable struct Brain
    ϕ::Vector{C1Function}
    ω::Vector{Matrix{Float64}}
    init_ω::Function
    propagate::Function
    learn_data::Function
    bias::Bool

    function Brain(m::Int64, n::Int64, l::Vector{Int64},
                   ϕ::Vector{<:C1Function}; bias = true)
        this = new()
        this.ϕ = ϕ
        this.bias = bias

        # Function to initialize weights
        this.init_ω = function init_ω(m, n, l)
            gen_weight = function(x, y, bias)
                dim_left = x
                if bias
                    dim_left += 1
                end
                return rand(Uniform(-1, 1), (dim_left, y))
            end

            # Initialize weights
            this.ω = [gen_weight(m, l[1], this.bias)]
            for i in 2:length(l)
                push!(this.ω, gen_weight(l[i - 1], l[i], this.bias))
            end
            push!(this.ω, gen_weight(l[end], n, this.bias))
        end
        this.init_ω(m, n, l)

        # Forward propagation
        this.propagate = function propagate(x; give_all = false)
            xᵣ = reshape(x, length(x), 1)
            if this.bias
                xᵣ = [xᵣ; 1]
            end

            # Forward propagation
            z = Vector{Matrix{Float64}}([])
            a = Vector{Matrix{Float64}}([xᵣ])
            i = 1
            for l in 1:length(this.ω)
                push!(z, this.ω[l]' * a[l])
                push!(a, this.ϕ[l].eval.(z[end]))
                a[end] = this.bias && l != length(this.ω) ? [a[end]; 1] : a[end]
            end

            if give_all
                return a, z
            else
                return a[end]
            end
        end

        this.learn_data = function learn_data(x, y; η=0.1, α=0.1, epocs=10)
            grads = zeros(epocs, length(this.ω))
            ΔWs = [zeros(size(this.ω[i])) for i in 1:length(this.ω)]
            instant_error = zeros(size(y))
            average_error = zeros(epocs, size(y, 1))
            for epoc in 1:epocs
                δ = nothing
                for batch in 1:size(x)[1]
                    δ = []

                    ## Forward Phase
                    act_vals, pre_act_vals = this.propagate(x[batch, :],
                                                            give_all = true)
                    error = act_vals[end] - y[batch, :]
                    instant_error[batch, :] = -error

                    ## Backward Phase
                    # Initialization
                    push!(δ, error .* ϕ[end].eval_diff.(pre_act_vals[end]))

                    # Construct each delta
                    for l in (length(pre_act_vals) - 1):-1:1
                        ω_no_bias = this.ω[l + 1][1:(end - 1), :]
                        δₗ = (ω_no_bias * δ[1]) .* ϕ[l].eval_diff.(pre_act_vals[l])
                        pushfirst!(δ, δₗ)
                    end

                    # Gradient descent
                    for l in 1:length(this.ω)
                        ΔW = - η * act_vals[l] * δ[l]' + α * ΔWs[l]
                        this.ω[l] += ΔW
                        ΔWs[l] = ΔW
                    end
                end

                # Outputs
                average_error[epoc, :] = sum(instant_error .^2 / 2, dims = 2)
                grads[epoc, :] = map(sum, δ)
            end
            return grads, average_error
        end

        return this
    end
end
