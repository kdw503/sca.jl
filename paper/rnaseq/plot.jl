using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","rnaseq")

#include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
#include(joinpath(workpath,"utils.jl"))


#colormap
# Makie.available_gradients()
# Plasma, Inferno, Magma, Cividis, Jet, grays, heat, :Spectral
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# X data
f = Figure(size=(120,300))
ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
joint_limits = (-0.0, 10)
Xs = Array(X[1:402:100562,1:323:32285])
hm1 = heatmap!(ax, Xs', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
hidedecorations!(ax)
save(joinpath(subworkpath,"$(file_name)_X[250,100].png"),f)

mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# LCSVD
fprefix = "WMB-10Xv2-HY-raw_lcsvd_noc500_aw0.0_ah0.005_bw5.0_bh0.0_it100"
dd = load(joinpath(subworkpath,"$(fprefix).jld2"))
W, H, rt1, rt2, fitval = dd["W"], dd["H"], dd["rt1"], dd["rt2"], dd["fitval"]
noc = size(W,2)
for i in 1:4
    f = Figure()
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    rowsizeq = size(W,1)÷4
    rows = (i==4 ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    joint_limits = (-0.04, 0.04)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_W$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,1000000)))
ht1 = hist!(ax,vec(W),bin=5)
save(joinpath(subworkpath,"$(fprefix)_histo1000000_W.png"),f)
num_blocks = 16
for i in 1:num_blocks
    f = Figure(size=(2200,600))
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (-200, 200) # 1024
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_H$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
ht1 = hist!(ax,vec(H),bin=5)
save(joinpath(subworkpath,"$(fprefix)_histo100_H.png"),f)



# HALS
fprefix = "WMB-10Xv2-HY-raw_hals_noc500_a0.1_iter100"
dd = load(joinpath(subworkpath,"$(fprefix).jld2"))
W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
now = size(W,2)
for i in 1:4
    f = Figure()
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    rowsizeq = size(W,1)÷4
    rows = (i==4 ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    @show rows
    joint_limits = (-0.08, 0.08)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)                     # These three
    # scatter!(ax, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_W$i.png"),f)
end
num_blocks = 16
for i in 1:num_blocks
    f = Figure(size=(2200,600))
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (-200, 200) # 1591
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)                     # These three
    # scatter!(ax, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_H$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
ht1 = hist!(ax,vec(H),bin=5)
save(joinpath(subworkpath,"$(fprefix)_histo100_H.png"),f)

