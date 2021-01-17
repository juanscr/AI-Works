# Install required modules
using Pkg
Pkg.add("Plots")
Pkg.add(url="https://github.com/juanscr/Fuzzy.jl")

# Import modules
using Plots
using Fuzzy

# Import functions from auxiliary file
include("Tools.jl");

# Creates figures folder for output
if ~isdir(joinpath(pwd(), "figs"))
    mkdir("figs");
end
cwd = joinpath(pwd(), "figs");

## Membership functions
# Controllers
contMF_low = SigmoidMF(-20, 0.25, 0);
contMF_med = BellMF(0.25, 4, 0.5);
contMF_high = SigmoidMF(10, 0.6, 1);
in_controllers = [contMF_low, contMF_med, contMF_high];
plotMFs(in_controllers, [0 1], ["Low", "Medium", "High"],
    joinpath(cwd, "contMFs.pdf")
)

# x(t)
xMF_null = TrapezoidalMF(0, 0, 0.1, 0.2);
xMF_controlled = BellMF(0.15, 5, 0.3);
xMF_high = TrapezoidalMF(0.4, 0.5, 1, 1);
in_xs = [xMF_null, xMF_controlled, xMF_high];
plotMFs(in_xs, [0 1], ["Null", "Controlled", "High"],
    joinpath(cwd, "stateMFs.pdf")
)

# Output controllers
# Δs
ω = 0.005;
a = 10/ω;
ΔMF_dec = SigmoidMF(-a, -ω/3, -ω);
ΔMF_con = GaussianMF(0, ω/25);
ΔMF_inc = SigmoidMF(a, ω/3, ω);
out_controllers = [ΔMF_dec, ΔMF_con, ΔMF_inc];
plotMFs(out_controllers, [-ω ω],
    ["Decrease", "Constant", "Increase"], joinpath(cwd, "outMFs.pdf")
)

## Inputs
in_u = Dict();
in_u["low"] = contMF_low;
in_u["medium"] = contMF_med;
in_u["high"] = contMF_high;

in_v = Dict();
in_v["low"] = contMF_low;
in_v["medium"] = contMF_med;
in_v["high"] = contMF_high;

in_x = Dict();
in_x["null"] = xMF_null;
in_x["controlled"] = xMF_controlled;
in_x["high"] = xMF_high;

inputs = [in_u, in_v, in_x];

## Outputs
out_Δu = Dict();
out_Δu["decrease"] = ΔMF_dec;
out_Δu["constant"] = ΔMF_con;
out_Δu["increase"] = ΔMF_inc;

out_Δv = Dict();
out_Δv["decrease"] = ΔMF_dec;
out_Δv["constant"] = ΔMF_con;
out_Δv["increase"] = ΔMF_inc;

## Rules
Tnorm = "A-PROD";
rules_u, rules_v = set_rules("rules", Tnorm);

## FIS Mamdani
fis_u = FISMamdani(inputs, out_Δu, rules_u, (-ω, ω));
fis_v = FISMamdani(inputs, out_Δv, rules_v, (-ω, ω));

## Fuzzy Inference System Control System  (FISCS)
# Scenarios
x0s = [1.0 1 1 0 0 0];
u0s = [0.0 1 0 1 0 1];
v0s = [0.0 0 1 0 1 1];
Ts = [150 150 150 300 300 300];

# Three defuzzification methods
defuzz = ["WTAV", "MOM", "COG"];
for d in defuzz
    # Simulate Scenarios
    ts, xarr, uarr, varr = eval_FISCS(x0s, u0s, v0s, Ts,
        fis_u, fis_v, defuzz = d
    )
    # Generate and export plots
    plotEvalFISCS(ts, xarr, uarr, varr, cwd, token = d)
end

# Surfaces
h = 0.01;
u_dom = collect(0:h:1);
v_dom = collect(0:h:1);
x_dom = collect(0:h:1);
colormap = :inferno;

# Constant u
fu(x, y) = eval_fis(fis_u, [0.5, x, y]);
plot(v_dom, x_dom, fu, st=:surface, c=colormap, camera=(50, 40));
plot!(xlabel=L"$v(t-h)$", ylabel=L"$x(t)$", legend=:outertopright)
savefig(joinpath(cwd, "deltaU_constU.pdf"))

fv(x, y) = eval_fis(fis_v, [0.5, x, y]);
plot(v_dom, x_dom, fv, st=:surface, c=colormap, camera=(50, 40));
plot!(xlabel=L"$v(t-h)$", ylabel=L"$x(t)$", c=grad, legend=:outertopright)
savefig(joinpath(cwd, "deltaV_constU.pdf"))

# Constant v
fu(x, y) = eval_fis(fis_u, [x, 0.5, y]);
plot(u_dom, x_dom, fu, st=:surface, c=colormap, camera=(30, 40));
plot!(xlabel=L"$u(t-h)$", ylabel=L"$x(t)$", c=grad, legend=:outertopright)
savefig(joinpath(cwd, "deltaU_constV.pdf"))

fv(x, y) = eval_fis(fis_v, [x, 0.5, y]);
plot(u_dom, x_dom, fv, st=:surface, c=colormap, camera=(50, 40));
plot!(xlabel=L"$u(t-h)$", ylabel=L"$x(t)$", c=grad, legend=:outertopright)
savefig(joinpath(cwd, "deltaV_constV.pdf"))

# Constant x
fu(x, y) = eval_fis(fis_u, [x, y, 0.5]);
plot(u_dom, v_dom, fu, st=:surface, c=colormap, camera=(50, 40));
plot!(xlabel=L"$u(t-h)$", ylabel=L"$v(t-h)$", c=grad, legend=:outertopright)
savefig(joinpath(cwd, "deltaU_constX.pdf"))

fv(x, y) = eval_fis(fis_v, [x, y, 0.5]);
plot(u_dom, v_dom, fv, st=:surface, c=colormap, camera=(40, 40));
plot!(xlabel=L"$u(t-h)$", ylabel=L"$x(t)$", c=grad, legend=:outertopright)
savefig(joinpath(cwd, "deltaV_constX.pdf"))
