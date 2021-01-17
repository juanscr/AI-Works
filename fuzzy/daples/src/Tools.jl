using Pkg
Pkg.add("LaTeXStrings")
using Plots
using LaTeXStrings
using Fuzzy

# 4th-Order Runge-Kutta
function rk4(f, x0, h, T)
    nvars = size(x0, 1)
    arr_size = Int(T/h)
    xs = zeros(ComplexF64, nvars, arr_size)
    xs[:,1] = x0
    t = 0
    ts = zeros(1)
    for i in 2:arr_size
        push!(ts, t)
        x_aux = xs[:, i-1]
        k1 = f(t, x_aux)
        t += h/2
        k2 = f(t, x_aux + k1*h/2)
        k3 = f(t, x_aux + k2*h/2)
        t += h/2
        k4 = f(t, x_aux + k3*h)
        xs[:, i] = x_aux + h*(k1 + 2*k2 + 2*k3 + k4)/6
    end
    return ts, xs
end

# Standard Euler's method
function euler(f, x0, h, T)
    nvars = size(x0, 1)
    arr_size = Int(T/h)
    xs = zeros(ComplexF64, nvars, arr_size)
    xs[:,1] = x0
    t = 0
    ts = zeros(1)
    for i in 2:arr_size
        push!(ts, t)
        x_aux = xs[:, i-1]
        k1 = f(t, x_aux)
        xs[:, i] = x_aux + h*f(t, x_aux)
        t += h
    end
    return ts, xs
end

# Standarize plot format
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
            legendfontsize = fontsize
        )
    else
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
            legendfontsize = fontsize
        )
    end
    if isnothing(add)
        return fig
    else
        return add
    end
end

# Plot array of MFs
function plotMFs(MFs, dom, labels, outfile; n=1000)
    colors = [:blue, :black, :orange, :red, :green]
    t = collect(dom[1]:(dom[2] - dom[1])/n:dom[2])
    i = 1
    for MF in MFs
        if i == 1
            global fig = plotPL(t, map(x -> MF.eval(x), t),
                label = labels[i],
                color = colors[i]
            )
        else
            fig = plotPL(t, map(x -> MF.eval(x), t),
                fig,
                label = labels[i],
                color = colors[i]
            )
        end
        i += 1
    end
    savefig(outfile)
    return fig
end

# Print rules combinations
function print_combinations(outfile)
    us = ["low" "medium" "high"];
    vs = ["low" "medium" "high"];
    xs = ["null" "controlled" "high"];

    open(outfile, "w") do io
        for u in us
            for v in vs
                for x in xs
                    write(io, "$u and $v and $x then \n")
                end
            end
        end
    end
end

# Create rules from file
function set_rules(rules_file, norm)
    rules_file = open(rules_file, "r") do io
        read(io, String)
    end;
    lines = split(rules_file, "\n")[1:end-1];
    for i in 1:length(lines)
        lines[i] = replace(lines[i], "\r" => "")
    end
    rules_u = [];
    rules_v = [];
    for line in lines
        ant = split(line, " then ")[1]
        con = split(line, " then ")[2]
        push!(rules_u, Rule(split(ant, " and "), split(con, " and ")[1], norm))
        push!(rules_v, Rule(split(ant, " and "), split(con, " and ")[2], norm))
    end
    return rules_u, rules_v
end

# Iterate a single step of euler's method for a given u and v
function iter_euler(x_prev, t, u, v; h=0.01)
    k = 10^-5;
    k1 = 1;
    k2 = 0.05;
    α = 0.75;
    k3 = 0.05;
    k4 = 0.01;
    β = 0.1;
    k5 = 0.1;
    f(t, x) = k .+ (1 .+ k1*v)*k2*(x.^α) .- k3*x .- k4*log(1 .+ u)*(x.^β) .- k5*log(1 .+ v)*x
    x_new = x_prev + h*f(t, x_prev)
    t += h
    return t, x_new
end

# Simulates the system with FIS-based controller
function simulate_FISCS(x0, u0, v0, T, fis_u, fis_v; h = 0.01, defuzz = "WTAV")
    # Initialization
    arr_size = Int64(T/h)
    xs = zeros(ComplexF64, arr_size)
    ts = zeros(Float64, arr_size)
    us = zeros(Float64, arr_size)
    vs = zeros(Float64, arr_size)
    xs[1] = x0
    us[1] = u0
    vs[1] = v0
    t = 0
    t, xs[2] = iter_euler(x0, t, u0, v0, h = h)
    # Simulate
    for i in 2:arr_size-1
        ts[i] = t
        # Evaluate FIS for new controls
        in_vals = [us[i-1], vs[i-1], real(xs[i])]
        us[i] = max(us[i-1] + eval_fis(fis_u, in_vals, defuzz), 0)
        vs[i] = max(vs[i-1] + eval_fis(fis_v, in_vals, defuzz), 0)
        # Find next point
        t, xs[i+1] = iter_euler(xs[i], t, us[i], vs[i])
    end
    us[end] = us[end-1]
    vs[end] = vs[end-1]
    ts[end] = t
    return ts, xs, us, vs
end

# Evaluate FISCS for a set of scenarios
function eval_FISCS(x0s, u0s, v0s, Ts, fis_u, fis_v; h = 0.01, defuzz = "WTAV")
    tarr = []
    xarr = []
    uarr = []
    varr = []
    sims = []
    for i in 1:length(x0s)
        ts, xs, us, vs = simulate_FISCS(x0s[i], u0s[i], v0s[i], Ts[i], fis_u,
            fis_v, h = h, defuzz = defuzz
        )
        push!(tarr, ts)
        push!(xarr, xs)
        push!(uarr, us)
        push!(varr, vs)
        push!(sims, [x0s[i] u0s[i] v0s[i]])
    end
    return tarr, xarr, uarr, varr
end

# Generates and exports plots for different scenarios results
function plotEvalFISCS(tarr, xarr, uarr, varr, cwd; token = "")
    for i in 1:length(xarr)
        fig = plotPL(tarr[i], uarr[i], color = :blue, label=L"$u(t)$")
        plotPL(tarr[i], varr[i], fig, color = :orange, label=L"$v(t)$")
        plotPL(tarr[i], real.(xarr[i]), fig, color = :black,
            label=L"$x(t)$", xlabel=L"$t$", legend=:topright
        )
        savefig(joinpath(cwd, token*"_sc$i.pdf"))
    end
    return fig
end
