using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))


#======= 3X4 =========================#
fontsize = 30; rowsize = 400 

f = Figure(resolution = (2150, 1350))
rowgap!(f.layout,0)

g1 = f[1, 1] = GridLayout()
g2 = f[2, 1] = GridLayout()
g3 = f[3, 1] = GridLayout()

# Panel SNR
fname = joinpath(subworkpath, "SNR", "avgfits-10db1f15s_all.png")
axa=AMakie.Axis(g1[1,1], title="(a) SNR = -10dB", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axa, label=false); hidespines!(axa); image!(axa, rotr90(load(fname)))
fname = joinpath(subworkpath, "SNRwPP", "avgfits-10db1f15s_all.png")
axb=AMakie.Axis(g1[1,2], title="(b) SNR = -10dB (with preprocessing)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axb, label=false); hidespines!(axb); image!(axb, rotr90(load(fname)));
fname = joinpath(subworkpath, "SNR", "avgfits20db1f15s_all.png")
axc=AMakie.Axis(g1[1,3], title="(c) SNR = 20dB", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axc, label=false); hidespines!(axc); image!(axc, rotr90(load(fname)))
fname = joinpath(subworkpath, "SNR", "avgfits30db1f15s_all.png")
axd=AMakie.Axis(g1[1,4], title="(d) SNR = 30dB", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axd, label=false); hidespines!(axd); image!(axd, rotr90(load(fname)));
rowsize!(g1,1,rowsize); colgap!(g1,0)

# Panel NOC
fname = joinpath(subworkpath, "ncells", "avgfits0db1f10s_all.png")
axe=AMakie.Axis(g2[1,1], title="(e) NOC = 10", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axe, label=false); hidespines!(axe); image!(axe, rotr90(load(fname)))
fname = joinpath(subworkpath, "ncells", "avgfits0db1f20s_all.png")
axf=AMakie.Axis(g2[1,2], title="(f) NOC = 20", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axf, label=false); hidespines!(axf); image!(axf, rotr90(load(fname)));
fname = joinpath(subworkpath, "ncells", "avgfits0db1f30s_all.png")
axf=AMakie.Axis(g2[1,3], title="(g) NOC = 30", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axf, label=false); hidespines!(axf); image!(axf, rotr90(load(fname)))
fname = joinpath(subworkpath, "ncells", "avgfits0db1f50s_all.png")
axh=AMakie.Axis(g2[1,4], title="(h) NOC = 50", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axh, label=false); hidespines!(axh); image!(axh, rotr90(load(fname)));
rowsize!(g2,1,rowsize); colgap!(g2,0)

# Panel size
fname = joinpath(subworkpath, "size", "avgfits0db0.7f15s_all.png")
axi=AMakie.Axis(g3[1,1], title="(i) size = 32 × 16 × 700", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axi, label=false); hidespines!(axi); image!(axi, rotr90(load(fname)))
fname = joinpath(subworkpath, "size", "avgfits0db1f15s_all.png")
axj=AMakie.Axis(g3[1,2], title="(j) size = 40 × 20 × 1000", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axj, label=false); hidespines!(axj); image!(axj, rotr90(load(fname)));
fname = joinpath(subworkpath, "size", "avgfits0db5f15s_all.png")
axk=AMakie.Axis(g3[1,3], title="(k) size = 80 × 40 × 5000", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axk, label=false); hidespines!(axk); image!(axk, rotr90(load(fname)))
fname = joinpath(subworkpath, "size", "avgfits0db10f15s_all.png")
axl=AMakie.Axis(g3[1,4], title="(l) size = 120 × 60 × 10000", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axl, label=false); hidespines!(axl); image!(axl, rotr90(load(fname)));
rowsize!(g3,1,rowsize); colgap!(g3,0)
rowgap!(f.layout,0)

save(joinpath(subworkpath,"scalability","scalability_all_figures(4X3).png"),f,px_per_unit=2)


#======= 3X3 =========================#
fontsize = 30; rowsize = 400 

f = Figure(resolution = (1620, 1350))
rowgap!(f.layout,0)

g1 = f[1, 1] = GridLayout()
g2 = f[2, 1] = GridLayout()
g3 = f[3, 1] = GridLayout()

# Panel SNR
fname = joinpath(subworkpath, "SNR", "avgfits-10db1f15s_all.png")
axa=AMakie.Axis(g1[1,1], title="(a) SNR = -10dB", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axa, label=false); hidespines!(axa); image!(axa, rotr90(load(fname)))
fname = joinpath(subworkpath, "SNRwPP", "avgfits-10db1f15s_all.png")
axb=AMakie.Axis(g1[1,2], title="(b) SNR = -10dB (with preprocessing)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axb, label=false); hidespines!(axb); image!(axb, rotr90(load(fname)));
fname = joinpath(subworkpath, "SNR", "avgfits20db1f15s_all.png")
axc=AMakie.Axis(g1[1,3], title="(c) SNR = 20dB", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axc, label=false); hidespines!(axc); image!(axc, rotr90(load(fname)))
rowsize!(g1,1,rowsize); colgap!(g1,0)

# Panel NOC
fname = joinpath(subworkpath, "ncells", "avgfits0db1f10s_all.png")
axe=AMakie.Axis(g2[1,1], title="(d) NOC = 10", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axe, label=false); hidespines!(axe); image!(axe, rotr90(load(fname)))
fname = joinpath(subworkpath, "ncells", "avgfits0db1f20s_all.png")
axf=AMakie.Axis(g2[1,2], title="(e) NOC = 20", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axf, label=false); hidespines!(axf); image!(axf, rotr90(load(fname)));
fname = joinpath(subworkpath, "ncells", "avgfits0db1f50s_all.png")
axf=AMakie.Axis(g2[1,3], title="(f) NOC = 50", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axf, label=false); hidespines!(axf); image!(axf, rotr90(load(fname)))
rowsize!(g2,1,rowsize); colgap!(g2,0)

# Panel size
fname = joinpath(subworkpath, "size", "avgfits0db1f15s_all.png")
axj=AMakie.Axis(g3[1,1], title="(g) size = 40 × 20 × 1000", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axj, label=false); hidespines!(axj); image!(axj, rotr90(load(fname)));
fname = joinpath(subworkpath, "size", "avgfits0db5f15s_all.png")
axk=AMakie.Axis(g3[1,2], title="(h) size = 80 × 40 × 5000", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axk, label=false); hidespines!(axk); image!(axk, rotr90(load(fname)))
fname = joinpath(subworkpath, "size", "avgfits0db10f15s_all.png")
axl=AMakie.Axis(g3[1,3], title="(i) size = 120 × 60 × 10000", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axl, label=false); hidespines!(axl); image!(axl, rotr90(load(fname)));
rowsize!(g3,1,rowsize); colgap!(g3,0)
rowgap!(f.layout,0)

save(joinpath(subworkpath,"scalability","scalability_all_figures.png"),f,px_per_unit=2)
