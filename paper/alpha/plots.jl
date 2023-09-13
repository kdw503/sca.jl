using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)

Pkg.activate(".")
using GLMakie, JLD, Colors

z = 0.5
for mtdstr in ["hals","admm"]
    ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(\".\",\"paper\",\"alpha\",\"$(mtdstr)_alpha_vs_avgfits.jld2\"))"))
    submtdstrs = ["_sp_nn"]; mtdstr != "hals" && push!(submtdstrs,"_sp") 
    for submtdstr in submtdstrs
        frpx = "$(mtdstr)$(submtdstr)"; dickeystr = "stat$(submtdstr)"
        @eval ($(Symbol("$(frpx)_rng"))=($(ddsym)["rng"]))
        # @eval ($(Symbol("$(frpx)_means")) = ($(ddsym)["stat$(submtdstr)"][1]))
        # @eval ($(Symbol("$(frpx)_stds")) = ($(ddsym)["stat$(submtdstr)"][2]))
        eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat$(submtdstr)\"][1]"))
        eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat$(submtdstr)\"][2]"))
        @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + z*$(Symbol("$(frpx)_stds"))))
        @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - z*$(Symbol("$(frpx)_stds"))))
    end
end

# ddhals = load("hals_runtime_vs_avgfits.jld2"); rng = ddhals["rng"]
# hals_nn_means = ddhals["stat_nn"][1]; hals_nn_stds = ddhals["stat_nn"][2]
# hals_sp_nn_means = ddhals["stat_sp_nn"][1]; hals_sp_nn_stds = ddhals["stat_sp_nn"][2]
# hals_nn_upper = hals_nn_means + z*hals_nn_stds; hals_nn_lower = hals_nn_means - z*hals_nn_stds
# hals_sp_nn_upper = hals_sp_nn_means + z*hals_sp_nn_stds; hals_sp_nn_lower = hals_sp_nn_means - z*hals_sp_nn_stds

# ddsca = load("sca_runtime_vs_avgfits.jld2"); rng = ddsca["rng"]
# sca_nn_means = ddsca["stat_nn"][1]; sca_nn_stds = ddsca["stat_nn"][2]
# sca_sp_means = ddsca["stat_sp"][1]; sca_sp_stds = ddsca["stat_sp"][2]
# sca_sp_nn_means = ddsca["stat_sp_nn"][1]; sca_sp_nn_stds = ddsca["stat_sp_nn"][2]
# sca_nn_upper = sca_nn_means + z*sca_nn_stds; sca_nn_lower = sca_nn_means - z*sca_nn_stds
# sca_sp_upper = sca_sp_means + z*sca_sp_stds; sca_sp_lower = sca_sp_means - z*sca_sp_stds
# sca_sp_nn_upper = sca_sp_nn_means + z*sca_sp_nn_stds; sca_sp_nn_lower = sca_sp_nn_means - z*sca_sp_nn_stds

# ddadmm = load("admm_runtime_vs_avgfits.jld2"); rng = ddadmm["rng"]
# admm_nn_means = ddadmm["stat_nn"][1]; admm_nn_stds = ddadmm["stat_nn"][2]
# admm_sp_means = ddadmm["stat_sp"][1]; admm_sp_stds = ddadmm["stat_sp"][2]
# admm_sp_nn_means = ddadmm["stat_sp_nn"][1]; admm_sp_nn_stds = ddadmm["stat_sp_nn"][2]
# admm_nn_upper = admm_nn_means + z*admm_nn_stds; admm_nn_lower = admm_nn_means - z*admm_nn_stds
# admm_sp_upper = admm_sp_means + z*admm_sp_stds; admm_sp_lower = admm_sp_means - z*admm_sp_stds
# admm_sp_nn_upper = admm_sp_nn_means + z*admm_sp_nn_stds; admm_sp_nn_lower = admm_sp_nn_means - z*admm_sp_nn_stds


alpha = 0.2; cls = distinguishable_colors(10; lchoices=range(0, stop=50, length=15)); clbs = convert.(RGBA,cls,alpha)

fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "α", ylabel = "average fit", title = "Average Fit Value vs. Running Time",
        xminorgridvisible=true,xminorticks = IntervalsBetween(10),xminorticksvisible=true)

lns = Dict(); bnds=Dict()
for (i,(frpx, lbl)) in enumerate([("hals_sp_nn","HALS"), ("admm_sp","Comp. SMF"), ("admm_sp_nn","Comp. NMF"),
                                #   ("sca_sp","SMF (β=0)"), 
                                #   ("sca_sp_nn","SMF (β=1000)")
                                  ])
    # eval(print("$(frpx)_means"))
    ln = lines!(ax, eval(Symbol("$(frpx)_rng")), eval(Symbol("$(frpx)_means")), color=cls[i], label=lbl)
    bnd = band!(ax, eval(Symbol("$(frpx)_rng")), eval(Symbol("$(frpx)_lower")), eval(Symbol("$(frpx)_upper")), color=clbs[i])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, position = :cb) # halign = :left, valign = :top
save(joinpath(".","paper","alpha","avgfits.png"),fig,px_per_unit=2)


# # hals nn
# lhnn=lines!(ax, rng, hals_nn_means, color=cls[1], label="HALS (α=0)")
# bhnn=band!(ax, rng, hals_nn_lower, hals_nn_upper, color=RGBA(0,0,1,0.2))
# # delete!(ax,lhnn); delete!(ax,bhnn) # or delete!.(ax,[lhnn,bhnn])
# # hals sp nn
# lines!(ax, rng, hals_sp_nn_means, color=:green, label="HALS (α=0.1)")
# band!(ax, rng, hals_sp_nn_lower, hals_sp_nn_upper, color=RGBA(0,1,0,0.2))

# # sca nn
# lines!(ax, rng, sca_nn_means, color=:red, label="SMF (α=0,β=1000)")
# band!(ax, rng, sca_nn_lower, sca_nn_upper, color=RGBA(1,0,0,0.2))
# # sca sp
# lines!(ax, rng, sca_sp_means, color=:magenta, label="SMF (α=100,β=0)")
# band!(ax, rng, sca_sp_lower, sca_sp_upper, color=RGBA(1,0,1,0.2))
# # sca sp nn
# lines!(ax, rng, sca_sp_nn_means, color=:magenta, label="SMF (α=100,β=1000)")
# band!(ax, rng, sca_sp_nn_lower, sca_sp_nn_upper, color=RGBA(1,0,1,0.2))

# # admm nn
# color = (0.5,0.5,0)
# lines!(ax, rng, admm_nn_means, color=RGBA(color...,1), label="Comp. NMF (α=0)")
# band!(ax, rng, admm_nn_lower, admm_nn_upper, color=RGBA(color...,0.2))
# # admm sp nn
# color = (1,0.5,0.5)
# lines!(ax, rng, admm_sp_nn_means, color=RGBA(color...,1), label="Comp. NMF (α=10)")
# band!(ax, rng, admm_sp_nn_lower, admm_sp_nn_upper, color=RGBA(color...,0.2))
# # admm sp
# color = (0.7,0.1,0.5)
# lines!(ax, rng, admm_sp_means, color=RGBA(color...,1), label="Comp. SMF (α=10)")
# band!(ax, rng, admm_sp_lower, admm_sp_upper, color=RGBA(color...,0.2))





















using StatsModels
using Distributions

x = 1:10
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
se = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]

alpha = 0.05
z = quantile(Normal(), 1 - alpha / 2)
lower_bound = y - z * se
upper_bound = y + z * se

fig = GLMakie.Figure()

#----- Axis
# ax1 = Axis(f[1, 1])
# ax2 = Axis(f[1, 2])
# ax3 = Axis(f[2, 2])
#
# <axis label & title>
# ax = Axis(fig[1, 1], xlabel = "x label", xlabelsize=30, ylabel = "y label", title = "Title", titlealign = :center)
# ax.xlabelsize = 40 # to change later
# Axis(f[2, 1], title = L"\sum_i{x_i \times y_i}")
# Axis(f[3, 1], title = rich(
#     "Rich text title",
#     subscript(" with subscript", color = :slategray)
# ))
# xlabelpadding : padding between the xlabel and the ticks or axis.
#
# <link axes>
# linkyaxes!(ax1, ax2)
# linkxaxes!(ax2, ax3)
#
# <xtick format & rotation>
# ax3 = Axis(f[2, 2], title = "Axis 3", xlabel = "x label", xtickformat = "{:.3f}ms", xticklabelrotation = pi/4)
# xtickformat = values -> [rich("$(value^2)", superscript("XY", color = :red)) for value in values])
#
# <twin axis>
# ax1 = Axis(f[1, 1], yticklabelcolor = :blue)
# ax2 = Axis(f[1, 1], yticklabelcolor = :red, yaxisposition = :right)
# hidespines!(ax2)
# hidexdecorations!(ax2)
# lines!(ax1, 0..10, sin, color = :blue)
# lines!(ax2, 0..10, x -> 100 * cos(x), color = :red)
#
# <Limits>
# ax1 = Axis(f[1, 1], limits = (nothing, nothing), title = "(nothing, nothing)")
# ax2 = Axis(f[1, 2], limits = (0, 4pi, -1, 1), title = "(0, 4pi, -1, 1)")
# ax3 = Axis(f[2, 1], limits = ((0, 4pi), nothing), title = "((0, 4pi), nothing)")
# ax4 = Axis(f[2, 2], limits = (nothing, 4pi, nothing, 1), title = "(nothing, 4pi, nothing, 1)")
#
# <aspect ratio>
# ax1 = Axis(f[1, 1], aspect = nothing, title = "nothing")
# ax2 = Axis(f[1, 2], aspect = DataAspect(), title = "DataAspect()")
# ax3 = Axis(f[2, 1], aspect = AxisAspect(1), title = "AxisAspect(1)")
# ax4 = Axis(f[2, 2], aspect = AxisAspect(2), title = "AxisAspect(2)")
#
# <xtick label space>
# xspace = maximum(tight_xticklabel_spacing!, [ax2, ax3])
# ax2.xticklabelspace = xspace
# ax3.xticklabelspace = xspace
#
# <color>
# scatter(x,sin,color=range(0, 1, length=100), colormap = :thermal, markersize=20)
# colors = repeat([:crimson, :dodgerblue, :slateblue1, :sienna1, :orchid1], 20)
# scatter(x,sin,color=colors,markersize=20)
#
# <etc>
# backgroundcolor, {left,right,top,bottom}spinecolor, bottomspnevisible, spinewidth,
# subtitle, subtitlecolor, subtitlefont, subtitlesize, height, width, valign,
# xaxisposition=:bottom, xgridcolor=RGBAf(0, 0, 0, 0.12), xgridstyle, xgridvisible,
# {x,xminor}grid{color,style,visible,width}, {x,xminor}tick{align,color,size,visible,width}
# xminorticks = IntervalsBetween(2); xminorticks = [1,2,3,4]
# xcale = identity, log10, sqrt, Makie.logit(logit function)
#         Makie.Symlog10(10.0) (Symlog10 with linear scaling between -10 and 10)
ax = GLMakie.Axis(fig[1, 1], xlabel = "x label", ylabel = "y label", title = "Title")

#----- lines!
lines!(ax, x, y, color=:blue, label="Plot")

#----- band!, fill_between! and alpha blending
band!(ax, x, lower_bound, upper_bound, color=RGBA(0,0,1,0.2))
#fill_between!(ax, x, lower_bound, upper_bound, color=RGBA(0,0,1,0.2))

xlims!(ax, minimum(x), maximum(x))
ylims!(ax, minimum(lower_bound), maximum(upper_bound))


fig[Axis].ylabel.position = :center
fig[Axis].xlabel.position = :center

display(fig)

axislegend(ax)
save("test.png",fig,px_per_unit=2)


# Legend
lin = lines!(xs, ys, color = :blue)
sca = scatter!(xs, ys, color = :red, markersize = 15)
Legend(f[1, 2], [lin, sca, lin], ["a line", "some dots", "line again"])
Legend(f[2, 1], [lin, sca, lin], ["a line", "some dots", "line again"],
    orientation = :horizontal, tellwidth = false, tellheight = true)




f = GLMakie.Figure()
GLMakie.Axis(f[1, 1])

n, m = 100, 101
t = range(0, 1, length=m)
X = cumsum(randn(n, m), dims = 2)
X = X .- X[:, 1]
μ = vec(mean(X, dims=1)) # mean
lines!(t, μ)              # plot mean line
σ = vec(std(X, dims=1))  # stddev
band!(t, μ + σ, μ - σ)   # plot stddev band


f = GLMakie.Figure()
GLMakie.Axis(f[1, 1])

xs = 1:0.2:10
ys_low = -0.2 .* sin.(xs) .- 0.25
ys_high = 0.2 .* sin.(xs) .+ 0.25

band!(xs, ys_low, ys_high)
band!(xs, ys_low .- 1, ys_high .-1, color = RGBA(0,0,0,0.5))


x = range(0, 10, length=100)
f, ax, l1 = lines(x, sin) # Figure, Axis, Lines
l2 = lines!(ax, x, cos)


x = range(0, 10, length=100)

lines(x, sin, color = :red, label = "sin")
lines!(x, cos, color = :blue, label = "cos")
axislegend()
current_figure()

fig = Figure()
lines(fig[1, 1], x, y, color = :red)
lines(fig[1, 2], x, y, color = :blue)
lines(fig[2, 1:2], x, y, color = :green, linestyle = :dash) # :dot, :dashdot, :dashdotdot, [0.5,1.0,1.5,2.5]

