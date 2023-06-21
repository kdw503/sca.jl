
using GLMakie
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
