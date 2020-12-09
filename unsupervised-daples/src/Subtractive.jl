include("Tools.jl");

# Algorithm
#   data -> (n, p) dataset matrix
#   c -> Number of clusters
#   dist -> Similarity function
#   rₐ -> Neighborhood parameter
#   ϵ_up -> Upper tolerance for acceptance
#   ϵ_down -> Lower tolerance for rejection
#   n_it -> Maximum number of iterations
#   arg -> Additional parameter for similarity function
#   norm -> Whether the data will be normalized or not
function subtractive_cluster(data, dist, rₐ;
    ϵ_up = 0.5, ϵ_down = 0.15, n_it=10, arg=nothing, norm=true
)
    # Normalize data
    data = norm ? normalize(data) : data
    n, p = size(data)
    rᵦ = 1.5 * rₐ
    # Initial Densities
    D = zeros(n, 1)
    for j = 1:n
        D[j, 1] = sum(exp.(-dist(data,
            reshape(data[j, :], (1, p)), arg).^2 ./ (rₐ/2)^2))
    end
    # Iterate
    evals = []
    update = true
    for k in 1:n_it
        # Find point with maximum density
        ind_max = argmax(D)[1]
        pₖ = reshape(data[ind_max, :], (1, p))
        if k == 1
            global M = D[ind_max]
            global protos = pₖ
            push!(evals, M)
        else
            # Accept/Reject procedure from original paper
            Mₖ = D[ind_max]
            if Mₖ > ϵ_up * M
                protos = vcat(protos, pₖ)
                push!(evals, Mₖ)
            elseif Mₖ < ϵ_down * M
                break
            else
                dmin = minimum(dist(protos, pₖ, arg))
                if (dmin/rₐ + Mₖ/M) >= 1
                    protos = vcat(protos, pₖ)
                    push!(evals, Mₖ)
                    update = true
                else
                    D[ind_max] = 0
                    update = false
                end
            end
        end
        # Updates the density with found cluster
        if update
            D -= D[ind_max] * exp.(-dist(data,
                reshape(data[ind_max, :], (1, p)), arg).^2 ./ (rᵦ/2)^2)
        end
    end
    # Find nearest points to found cluster centers
    U, _ = find_membership(data, dist, protos, arg)
    return data, protos, U, evals
end

# Explore space - Subtractive
function explore_subtractive(data, dist, rₐs; dataset="country", newfile=true)
    s = joinpath(pwd(), "results")
    if ~isdir(s)
        mkdir(s)
    end
    nrₐ = size(rₐs, 1)
    mode = newfile ? "w" : "a"
    outfile = open(joinpath(s, dataset*"_subtractive.res"), mode)
    ks = zeros(nrₐ)
    for i in 1:nrₐ
        rₐ = rₐs[i]
        protos = subtractive_cluster(data, dist, rₐ)[2]
        k = size(protos, 1)
        ks[i] = k
        if i == nrₐ
            write(outfile, string(k))
        else
            write(outfile, string(k)*",")
        end
    end
    close(outfile)
    return ks
end
