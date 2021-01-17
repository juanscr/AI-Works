# Continuous function with first derivatives type
abstract type C1Function end

# ========== Activation Functions ========== #
# Sigmoid Activation Function
mutable struct Sigmoid<:C1Function
    eval::Function
    eval_diff::Function

    function Sigmoid()
        this = new()
        this.eval = function eval(x)
            return 1 / (1 + exp(-x))
        end

        this.eval_diff = function eval_diff(x)
            return eval(x) * (1 - eval(x))
        end

        return this
    end
end

# Hyperbolic tangent
mutable struct Tanh<:C1Function
    eval::Function
    eval_diff::Function

    function Tanh()
        this = new()
        this.eval = function eval(x)
            return tanh(x)
        end

        this.eval_diff = function eval_diff(x)
            return 1 - eval(x) ^ 2
        end

        return this
    end
end

# Linear
mutable struct Linear<:C1Function
    eval::Function
    eval_diff::Function

    function Linear()
        this = new()
        this.eval = function eval(x)
            return x
        end

        this.eval_diff = function eval_diff(x)
            return 1
        end

        return this
    end
end

# ReLu
mutable struct ReLu<:C1Function
    eval::Function
    eval_diff::Function

    function ReLu()
        this = new()
        this.eval = function eval(x)
            return maximum([x, 0])
        end

        this.eval_diff = function eval_diff(x)
            return x < 0 ? 0 : 1
        end

        return this
    end
end
