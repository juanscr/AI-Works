# using Pkg
# Pkg.add("LaTeXStrings")
# Pkg.add("Plots")

using LaTeXStrings
using Plots

include("BackPropagation.jl")

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

# Plot ξav (average error energy)
function plot_ξav(Ξ; save_fig=false, out_file="", dir="")
    s = size(Ξ, 2)
    fig = plotPL(collect(1:s), reshape(mean(Ξ, dims=1), s),
        xlabel = "Epoch",
        ylabel = L"\xi_{av}"
    )
    save_fig ? save(fig, out_file, dir=dir) : nothing
    return fig
end

function plot_∇s(∇s; save_fig=false, out_file="", dir="")
    s, nδ = size(∇s)
    first = true
    fig = nothing
    for i in 1:nδ
        if first
            fig = plotPL(collect(1:s), ∇s[:, i],
                xlabel="Epoch",
                ylabel=L"\sum\delta_j",
                color=:auto,
                label="Layer $i"
            )
            first = false
        else
            plotPL(collect(1:s), ∇s[:, i], fig,
                xlabel="Epoch",
                ylabel=L"\sum\delta_j",
                color = :auto,
                label="Layer $i"
            )
        end
    end
    plot!(legend = :bottomleft)
    save_fig ? save(fig, out_file, dir=dir) : nothing
    return fig
end

function plot_seed∇(l, S, X, Y, L, ϕ, ∂ϕ; η=0.05, α=0.1, s=10,
    save_fig=false, out_file="", dir=""
)
    seeds = size(S, 1)
    arr∇ = []
    for seed in S
        _, _, _, ∇s, _ = nn(X, Y, L, ϕ, ∂ϕ; η=η, α=α, s=s, seed=seed)
        push!(arr∇, ∇s)
    end
    first = true
    fig = nothing
    for i in 1:seeds
        if first
            fig = plotPL(collect(1:s), arr∇[i][:, l],
                xlabel="Epoch",
                ylabel=L"\sum\delta_j",
                color=:auto,
            )
            first = false
        else
            plotPL(collect(1:s), arr∇[i][:, l], fig,
                xlabel="Epoch",
                ylabel=L"\sum\delta_j",
                color = :auto,
            )
        end
    end
    save_fig ? save(fig, out_file, dir=dir) : nothing
    return fig
end
