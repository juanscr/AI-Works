using Pkg
Pkg.add("GLM")
Pkg.add("LaTeXStrings")
Pkg.add("Plots")

using GLM
using LaTeXStrings
using Plots

# Standardize file saving
function save(fig, out_file; dir="")
    if ~isdir(joinpath(pwd(), "figs"))
        mkdir("figs");
    end
    if ~isdir(joinpath(pwd(), "figs", dir))
        mkdir(joinpath(pwd(), "figs", dir))
    end
    plot_wd = joinpath(pwd(), "figs", dir);
    savefig(fig, joinpath(plot_wd, out_file))
end

# Standardize plots and surfaces
function plotPL(x, y,
    add = nothing;
    z = nothing,
    label = "",
    color = :black,
    lw = 2,
    legend = :right,
    fontsize = 12,
    font = "times",
    xlabel = "",
    ylabel = "",
    zlabel = "",
    st = :surface,
    cam = (-30, 30)
)
    if isnothing(add)
        if isnothing(z)
            global fig = plot(x, y,
                label = label,
                color = color,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize
            )
        else
            global fig = plot(x, y, z,
                label = label,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                zlabel = zlabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                ztickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize,
                st = st
            )
        end
    else
        if isnothing(z)
            plot!(add, x, y,
                label = label,
                color = color,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize
            )
        else
            plot!(add, x, y, z,
                label = label,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                zlabel = zlabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                ztickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize,
                st = st
            )
        end
    end
    if isnothing(add)
        return fig
    else
        return add
    end
end

# Standardize scatters
function scatterPL(x, y,
    add = nothing;
    z = nothing,
    label = "",
    color = :auto,
    lw = 2,
    legend = :right,
    fontsize = 12,
    font = "times",
    xlabel = "",
    ylabel = "",
    zlabel = "",
    st = :surface,
    cam = (-30, 30),
    markersize = 4,
    markershape = :circle,
    markerstrokewidth = 0,
    markersizewidth = 0,
    markeralpha = 1,
    markerstrokealpha = 0
)
    if isnothing(add)
        if isnothing(z)
            global fig = scatter(x, y,
                label = label,
                color = color,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize,
                markersize = markersize,
                markershape = markershape,
                markerstrokewidth = markerstrokewidth,
                markersizewidth = markersizewidth,
                markeralpha = 1,
                markerstrokealpha = 0
            )
        else
            global fig = scatter(x, y, z,
                label = label,
                color = color,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                zlabel = zlabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                ztickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize,
                markersize = markersize,
                markershape = markershape,
                markerstrokewidth = markerstrokewidth,
                markersizewidth = markersizewidth,
                markeralpha = 1,
                markerstrokealpha = 0
            )
        end
    else
        if ~isnothing(z)
            scatter!(add, x, y, z,
                label = label,
                color = color,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize,
                markersize = markersize,
                markershape = markershape,
                markerstrokewidth = markerstrokewidth,
                markersizewidth = markersizewidth,
                markeralpha = 1,
                markerstrokealpha = 0
            )
        else
            scatter!(add, x, y,
                label = label,
                color = color,
                lw = lw,
                legend = legend,
                xlabel = xlabel,
                ylabel = ylabel,
                zlabel = zlabel,
                titlefont = (fontsize, font),
                xtickfont = (fontsize, font),
                ytickfont = (fontsize, font),
                ztickfont = (fontsize, font),
                # guidefont = (fontsize, font),
                legendfontsize = fontsize,
                markersize = markersize,
                markershape = markershape,
                markerstrokewidth = markerstrokewidth,
                markersizewidth = markersizewidth,
                markeralpha = 1,
                markerstrokealpha = 0
            )
        end
    end
    if isnothing(add)
        return fig
    else
        return add
    end
end

# Plot 2D clusters with respective data. Includes sequence of cluster centers
# for K-Means and Fuzzy-C-Means algorithms.
function plot_k_clusters(protos, U, data, out_file; dir="", protos_iter=nothing)
    k = size(protos, 1)
    for j in 1:k
        aux = argmax(U, dims=1)[[x[1] for x in argmax(U, dims=1)] .== j]
        data_cluster = data[[x[2] for x in aux], :]
        if j == 1
            global fig = scatterPL(data_cluster[:, 1], data_cluster[:, 2])
        else
            scatterPL(data_cluster[:, 1], data_cluster[:, 2], fig)
        end
    end
    if ~isnothing(protos_iter)
        n_protos = size(protos_iter, 1)
        alphas = collect(0:1:n_protos) ./ n_protos
        j = 1
        for proto in protos_iter
            scatterPL(proto[:, 1], proto[:, 2], fig,
                color=:grey, markersize=4, markeralpha=alphas[j]
            )
            j += 1
        end
    end
    scatterPL(protos[:, 1], protos[:, 2], fig, color=:black, markersize=7)
    save(fig, out_file, dir=dir)
    return fig
end

# Plot 2D clustered data.
function plot_clustered_data(k, U, data, out_file; dir="")
    for j in 1:k
        aux = argmax(U, dims=1)[[x[1] for x in argmax(U, dims=1)] .== j]
        data_cluster = data[[x[2] for x in aux], :]
        if j == 1
            global fig = scatterPL(data_cluster[:, 1], data_cluster[:, 2])
        else
            scatterPL(data_cluster[:, 1], data_cluster[:, 2], fig)
        end
    end
    save(fig, out_file, dir=dir)
    return fig
end

# Plot 3D scatter of clusters and their centers
function plot_3DClusters(protos, U, data, out_file, labels; dir="",
    plot_protos = true, k = nothing
)
    if isnothing(k)
        k = size(protos, 1)
    end
    for j in 1:k
        aux = argmax(U, dims=1)[[x[1] for x in argmax(U, dims=1)] .== j]
        data_cluster = data[[x[2] for x in aux], :]
        if j == 1
            global fig = scatterPL(data_cluster[:, 1], data_cluster[:, 2],
                z=data_cluster[:, 3]
            )
        else
            scatterPL(data_cluster[:, 1], data_cluster[:, 2], fig,
                z=data_cluster[:, 3]
            )
        end
    end
    if plot_protos
        scatterPL(protos[:, 1], protos[:, 2], fig,
            z=protos[:, 3], color=:black, markersize=5,
            xlabel=labels[1],
            ylabel=labels[2],
            zlabel=labels[3]
        )
    end
    save(fig, out_file, dir=dir)
    return fig
end

# Projects above method for higher dimensions
function plot_nDimGroups(protos, U, sample_data, out_file; dir="")
    n, p = size(data)
    for j in 1:binomial(p, 3)
        k1 = j % p + 1
        k2 = (j+1) % p + 1
        k3 = (j+2) % p + 1
        plot_3DClusters(protos_cmeans[:, k1:sign(k3-k1):k3], U_cmeans,
            data[:, k1:sign(k3-k1):k3],
            out_file*"$j.pdf",
            ("x$k1", "x$k2", "x$k3"),
            dir=dir
        )
    end
end

# Plots a 3D surface of the density of two variables (mount and sub only).
function plot_density(aux, evals, dims; arg=1)
    indexes = [i in dims ? (1:size(evals, i)) : arg for i in 1:length(size(evals))]
    return plotPL(
        aux[dims[1]], aux[dims[2]], z=evals[indexes...],
        xlabel = "x$(dims[1])",
        ylabel = "x$(dims[2])",
        zlabel = "Density"
    )
end

# Exports a GIF with the density destruction (sub and mount only).
function generate_density_gif(aux, evals, out_file; arg=1, fps=1, dir="")
    s = joinpath(pwd(), "figs", dir)
    if dir != "" && ~isdir(s)
        mkdir(s)
    end
    dims_arr = [(i, i+1) for i in 1:(length(aux) - 1)]
    for dims in dims_arr
        anim = @animate for i in 1:size(evals, 1)
            plot_density(aux, evals[i], dims, arg=arg)
        end
        outfile = joinpath(pwd(), "figs", dir,"dims"*string(dims)*"_"*out_file)
        gif(anim, outfile, fps=fps)
    end
end

# Plots the multivariate D-D Plot for the given depths with fitted LM
function ddplot(Z₁, Z₂, filename; dir="", legend=:bottomright)
    # Linear regression
    res = lm(reshape(Z₁, (size(Z₁, 1), 1)), Z₂)
    β₁ = coef(res)[1]
    β₀ = mean(Z₂) - β₁*mean(Z₁)
    maxZ = maximum(Z₁)
    xZ = 0:0.01:maxZ
    yZ = β₀ .+ β₁*xZ
    println("beta0 = ", β₀)
    println("beta1 = ", β₁)
    # Depths scatter
    fig = scatterPL(Z₁, Z₂, color=:black, label="Depths")
    plotPL(xZ, yZ, fig, color=:red, label=L"Z_2=\beta_0+\beta_1 Z_1",
        legend=legend, xlabel=L"Z_1", ylabel=L"Z_2"
    )
    save(fig, filename, dir=dir)
    return fig
end
