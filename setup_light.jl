#using MultivariateStats # for ICA
using LinearAlgebra, JLD2
using FakeCells, AxisArrays, ImageCore, MappedArrays, NMF, Statistics
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using TestData
using LCSVD, CompNMF, IncrementalSVD, TSVD

"""
resol=(800,600); fntsize1 = 30; fntsize2 = 30

fig = Figure(size=resol)
ax = AMakie.Axis(fig[1, 1], limits = ((0,100), nothing), xlabel = "iteration", ylabel = "penalty",
                xlabelsize=fntsize2, ylabelsize=fntsize2, xticklabelsize=fntsize2, yticklabelsize=fntsize2,
                yscale = log10, title = "Penalty Changes")

lines!(ax, f_x10, color=mtdcolors[2], label="i_maxiter=10", linestyle=nothing)
lines!(ax, f_x100, color=mtdcolors[4], label="i_maxiter=100", linestyle=:dash)
lines!(ax, f_x1000, color=mtdcolors[5], label="i_maxiter=1000", linestyle=:dashdot)

axislegend(ax, labelsize=fntsize1, position = :cb)
save(joinpath(subworkpath,"penalty.png"),fig, px_per_unit=2)
"""
lines_plot