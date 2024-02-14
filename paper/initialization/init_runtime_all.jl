using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","initialization")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))

#= This is ten times slower than direct call in the python
using PyCall
@pyimport numpy as npy
@pyimport sklearn.utils.extmath as skue # randomized_svd, squared_norm, randomized_range_finder 

r_ov = 10
rt0 = time()
L = skue.randomized_range_finder(X, size = ncells + r_ov, n_iter = 3)
R = skue.randomized_range_finder(X', size = ncells + r_ov, n_iter = 3)
rt = time()-rt0
=#

if false
    num_experiments = 50; ncellsrng = 4:2:200; factorrng=1:10
else
    num_experiments = 2; ncellsrng = 5:6; factorrng=1:2
end

dataset = :fakecells; inhibitindices=0; bias=0.1; SNR=0; factor = 5; ncells = 15
imgsz0 = (40,20); lengthT0=1000

# ncells
sqfactor = sqrt(factor)
imgsz = (Int(floor(sqfactor*imgsz0[1])),Int(floor(sqfactor*imgsz0[2]))); lengthT = factor*lengthT0; sigma = round(sqfactor*5.0)
isvdmeans=Float64[]; isvdstds=Float64[]
lowrankmeans=Float64[]; lowrankstds=Float64[]
nndsvdmeans=Float64[]; nndsvdstds=Float64[]; rnndsvdmeans=Float64[]; rnndsvdstds=Float64[]
for (iter, ncl) in enumerate(ncellsrng)
    @show ncl; flush(stdout)

isvdrt1s = []; nndsvdrt1s=[]; rnndsvdrt1s=[]
lowrankrt1s=[]; rlowrankrt1s=[]; 
for i in 1:num_experiments
    X, imsz, lhT, ncs, gtncells, datadic = load_data(:fakecells; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=0.1, useCalciumT=true,
            inhibitindices=0, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
    isvdrt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncl; initmethod=:isvd, svdmethod=:isvd)
    lowrankrt1 = @elapsed  CompNMF.compmat(X, Wp, Hp; w=4, rov=10)
    nndsvdrt1 = @elapsed  NMF.nndsvd(X, ncl, variant=:ar, initdata=svd(X))
    rnndsvdrt1 = @elapsed  NMF.nndsvd(X, ncl, variant=:ar)
    push!(isvdrt1s,isvdrt1); push!(nndsvdrt1s,nndsvdrt1); push!(rnndsvdrt1s,rnndsvdrt1)
    push!(lowrankrt1s,lowrankrt1)
end
isvdmean = mean(isvdrt1s); isvdstd = std(isvdrt1s)
lowrankmean = mean(lowrankrt1s); lowrankstd = std(lowrankrt1s)
nndsvdmean = mean(nndsvdrt1s); nndsvdstd = std(nndsvdrt1s)
rnndsvdmean = mean(rnndsvdrt1s); rnndsvdstd = std(rnndsvdrt1s)
push!(isvdmeans,isvdmean); push!(isvdstds,isvdstd)
push!(lowrankmeans,lowrankmean); push!(lowrankstds,lowrankstd)
push!(nndsvdmeans,nndsvdmean); push!(nndsvdstds,nndsvdstd)
push!(rnndsvdmeans,rnndsvdmean); push!(rnndsvdstds,rnndsvdstd)
end
save(joinpath(subworkpath,"ncellsrng_vs_rt1s.jld2"),
            "ncellsrng",ncellsrng, "isvdmeans",isvdmeans,"isvdstds",isvdstds,
            "lowrankmeans",lowrankmeans,"lowrankstds",lowrankstds,
            "nndsvdmeans",nndsvdmeans,"nndsvdstds",nndsvdstds, "rnndsvdmeans",rnndsvdmeans,"rnndsvdstds",rnndsvdstds)

z = 0.5
dd = load(joinpath(subworkpath,"ncellsrng_vs_rt1s.jld2")); ncellsrng = dd["ncellsrng"]
sca_means = dd["isvdmeans"]; sca_stds = dd["isvdstds"]
sca_upper = sca_means + z*sca_stds; sca_lower = sca_means - z*sca_stds
hals_means = dd["nndsvdmeans"]; hals_stds = dd["nndsvdstds"]
hals_upper = hals_means + z*hals_stds; hals_lower = hals_means - z*hals_stds
hals_r_means = dd["rnndsvdmeans"]; hals_r_stds = dd["rnndsvdstds"]
hals_r_upper = hals_r_means + z*hals_r_stds; hals_r_lower = hals_r_means - z*hals_r_stds
admm_means = dd["lowrankmeans"]; admm_stds = dd["lowrankstds"]
admm_upper = admm_means + z*admm_stds; admm_lower = admm_means - z*admm_stds

fig = Figure(resolution=(500,280))
ax1 = AMakie.Axis(fig[1, 1], limits = ((0,100), (-1,40)), xlabel = "number of components", ylabel = "time(sec)")#, title = "Number of components vs. Initialization time")

lin = []
push!(lin,lines!(ax1, ncellsrng, sca_means, color=mtdcolors[2], label="ISVD"))
band!(ax1, ncellsrng, sca_lower, sca_upper, color=mtdcoloras[2])
push!(lin,lines!(ax1, ncellsrng, admm_means, color=mtdcolors[5], label="Compression"))
band!(ax1, ncellsrng, admm_lower, admm_upper, color=mtdcoloras[5])
push!(lin,lines!(ax1, ncellsrng, hals_means, color=mtdcolors[7], linestyle=:dash, label="NNDSVD(SVD)"))
band!(ax1, ncellsrng, hals_lower, hals_upper, color=mtdcoloras[7])
push!(lin,lines!(ax1, ncellsrng, hals_r_means, color=mtdcolors[3], label="NNDSVD(RSVD)"))
band!(ax1, ncellsrng, hals_r_lower, hals_r_upper, color=mtdcoloras[3])
labels = ["ISVD","Compression","NNDSVD(SVD)","NNDSVD(RSVD)"]
#axislegend(ax1, labelsize=20, position = :lt) # halign = :left, valign = :top
fig[:,2] = Legend(fig[:,1],lin,labels)
save(joinpath(subworkpath,"ncellsrng_vs_rt1s.png"),fig,px_per_unit=2)


# factor
isvdmeans=Float64[]; isvdstds=Float64[]
lowrankmeans=Float64[]; lowrankstds=Float64[]
nndsvdmeans=Float64[]; nndsvdstds=Float64[]; rnndsvdmeans=Float64[]; rnndsvdstds=Float64[]
for (iter, factor) in enumerate(factorrng)
    @show factor; flush(stdout)
    sqfactor = sqrt(factor)
    imgsz = (Int(floor(sqfactor*imgsz0[1])),Int(floor(sqfactor*imgsz0[2]))); lengthT = factor*lengthT0; sigma = sqfactor*5.0
    isvdrt1s = []; nndsvdrt1s=[]; rnndsvdrt1s=[]
    lowrankrt1s=[]
    for i in 1:num_experiments
        X, imsz, lhT, ncs, gtncells, datadic = load_data(:fakecells; sigma=5.0, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=0.1, useCalciumT=true,
                inhibitindices=0, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
        isvdrt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=:isvd, svdmethod=:isvd)
        lowrankrt1 = @elapsed  CompNMF.compmat(X, Wp, Hp; w=4, rov=10)
        nndsvdrt1 = @elapsed NMF.nndsvd(X, ncells, variant=:ar, initdata=svd(X))
        rnndsvdrt1 = @elapsed NMF.nndsvd(X, ncells, variant=:ar)
        push!(isvdrt1s,isvdrt1); 
        push!(lowrankrt1s,lowrankrt1)
        push!(nndsvdrt1s,nndsvdrt1); push!(rnndsvdrt1s,rnndsvdrt1)
    end
    isvdmean = mean(isvdrt1s); isvdstd = std(isvdrt1s)
    lowrankmean = mean(lowrankrt1s); lowrankstd = std(lowrankrt1s)
    nndsvdmean = mean(nndsvdrt1s); nndsvdstd = std(nndsvdrt1s)
    rnndsvdmean = mean(rnndsvdrt1s); rnndsvdstd = std(rnndsvdrt1s)
    push!(isvdmeans,isvdmean); push!(isvdstds,isvdstd)
    push!(nndsvdmeans,nndsvdmean); push!(nndsvdstds,nndsvdstd)
    push!(rnndsvdmeans,rnndsvdmean); push!(rnndsvdstds,rnndsvdstd)
    push!(lowrankmeans,lowrankmean); push!(lowrankstds,lowrankstd)
end
save(joinpath(subworkpath,"factorrng_vs_rt1s.jld2"),
            "factorrng",factorrng, "isvdmeans",isvdmeans,"isvdstds",isvdstds,
           "lowrankmeans",lowrankmeans,"lowrankstds",lowrankstds,
            "nndsvdmeans",nndsvdmeans,"nndsvdstds",nndsvdstds, "rnndsvdmeans",rnndsvdmeans,"rnndsvdstds",rnndsvdstds)

z = 0.5
dd = load(joinpath(subworkpath,"factorrng_vs_rt1s.jld2")); factorrng = dd["factorrng"]
sca_means = dd["isvdmeans"]; sca_stds = dd["isvdstds"]
sca_upper = sca_means + z*sca_stds; sca_lower = sca_means - z*sca_stds
admm_means = dd["lowrankmeans"]; admm_stds = dd["lowrankstds"]
admm_upper = admm_means + z*admm_stds; admm_lower = admm_means - z*admm_stds
hals_means = dd["nndsvdmeans"]; hals_stds = dd["nndsvdstds"]
hals_upper = hals_means + z*hals_stds; hals_lower = hals_means - z*hals_stds
hals_r_means = dd["rnndsvdmeans"]; hals_r_stds = dd["rnndsvdstds"]
hals_r_upper = hals_r_means + z*hals_r_stds; hals_r_lower = hals_r_means - z*hals_r_stds

fig = Figure(resolution=(500,280))
ax1 = AMakie.Axis(fig[1, 1], xlabel = "data size (MB)", ylabel = "time(sec)",
    xtickformat = values -> ["$(Int(round(value^2*6.4)))" for value in values]) # factor^2*40*20*1000*64bit(Float64)/8bit/1000000(Mega) MByte

lin = []
push!(lin,lines!(ax1, factorrng, sca_means, color=mtdcolors[2], label="ISVD"))
band!(ax1, factorrng, sca_lower, sca_upper, color=mtdcoloras[2])
push!(lin,lines!(ax1, factorrng, admm_means, color=mtdcolors[5], label="Compression"))
band!(ax1, factorrng, admm_lower, admm_upper, color=mtdcoloras[5])
push!(lin,lines!(ax1, factorrng, hals_means, linestyle=:dash, color=mtdcolors[7], label="NNDSVD(SVD)"))
band!(ax1, factorrng, hals_lower, hals_upper, color=mtdcoloras[7])
push!(lin,lines!(ax1, factorrng, hals_r_means, color=mtdcolors[3], label="NNDSVD(RSVD)"))
band!(ax1, factorrng, hals_r_lower, hals_r_upper, color=mtdcoloras[3])
labels = ["ISVD","Compression","NNDSVD(SVD)","NNDSVD(RSVD)"]
#axislegend(ax1, labelsize=20, position = :lt) # halign = :left, valign = :top
fig[:,2] = Legend(fig[:,1],lin,labels)
save(joinpath(subworkpath,"factorrng_vs_rt1s.png"),fig,px_per_unit=2)



#==== combine all the figures ===========#
fontdize = 25

imga = load(joinpath(subworkpath,"factorrng_vs_rt1s.png"))
imgb = load(joinpath(subworkpath,"ncellsrng_vs_rt1s.png"))
imgc = load(joinpath(subworkpath,"avgfits-10db1f15s_all.png"))
imgd = load(joinpath(subworkpath,"avgfits-10db1f50s_all.png"))
imge = load(joinpath(subworkpath,"avgfits0db1f15s_all.png"))
imgf = load(joinpath(subworkpath,"avgfits0db1f50s_all.png"))

f = Figure(resolution = (1500,800))
ax11=AMakie.Axis(f[1,1], title ="(a) Data sive vs. runtime (NOC=15)", titlesize=fontsize, width=550,aspect = DataAspect())
hidedecorations!(ax11); hidespines!(ax11)
ax12=AMakie.Axis(f[1,2], title="(b) HALS result of -10dB 15NOC", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax12); hidespines!(ax12)
ax13=AMakie.Axis(f[1,3], title="(c) HALS result of -10dB 50NOC", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax13); hidespines!(ax13)
ax21=AMakie.Axis(f[2,1],title="(d) NOC vs. runtime (size = 230MB)", titlesize=fontsize, width=550, aspect = DataAspect())
hidedecorations!(ax21); hidespines!(ax21)
ax22=AMakie.Axis(f[2,2], title="(e) HALS result of 0dB 15NOC", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax22); hidespines!(ax22)
ax23=AMakie.Axis(f[2,3], title="(f) HALS result of 0dB 50NOC", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax23); hidespines!(ax23)
image!(ax11, rotr90(imga)); image!(ax12, rotr90(imgc)); image!(ax13, rotr90(imge))
image!(ax21, rotr90(imgb)); image!(ax22, rotr90(imgd)); image!(ax23, rotr90(imgf))
save(joinpath(subworkpath,"initialization.png"),f)





#=
# factor & ncells
factorrng = 1:10; ncellsrng = 5:5:200; imgsz0 = (40,20); lengthT0 = 200
isvdrt1ss = [];; lowrankrt1ss=[]; nndsvdrt1ss=[]; rlowrankrt1ss=[]; rnndsvdrt1ss=[]
for (iter, factor) in enumerate(factorrng)
    @show factor; flush(stdout)
    imgsz = (imgsz0[1]*factor,imgsz0[2]); lengthT = lengthT0*factor
isvdrt1s = []; lowrankrt1s=[]; nndsvdrt1s=[]; rlowrankrt1s=[]; rnndsvdrt1s=[]
for ncl in ncellsrng
    X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, imgsz=imgsz, bias=bias, lengthT=lengthT, useCalciumT=true,
            inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false);
    X = noisefilter(filter,X)
    isvdrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncl, initmethod=:isvd,poweradjust=initpwradj)
    lowrankrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncl, initmethod=:lowrank_nndsvd,poweradjust=initpwradj,svdmethod=:svd)
    nndsvdrt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncl, variant=:ar, initdata=svd(X))
    rlowrankrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncl, initmethod=:lowrank_nndsvd,poweradjust=initpwradj,svdmethod=:rsvd)
    rnndsvdrt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncl, variant=:ar)
    push!(isvdrt1s,isvdrt1); push!(lowrankrt1s,lowrankrt1); push!(nndsvdrt1s,nndsvdrt1)
    push!(rlowrankrt1s,rlowrankrt1); push!(rnndsvdrt1s,rnndsvdrt1)
end
push!(isvdrt1ss,isvdrt1s); push!(lowrankrt1ss,lowrankrt1s); push!(nndsvdrt1ss,nndsvdrt1s)
push!(rlowrankrt1ss,rlowrankrt1s); push!(rnndsvdrt1ss,rnndsvdrt1s)
end
save(joinpath(subworkpath,"factorrng_ncellsrng_vs_rt1s.jld2"), "imgsz0",imgsz0, "lengthT0",lengthT0,
                "factorrng",factorrng,"ncellsrng",ncellsrng,"isvdrt1ss",isvdrt1ss,"lowrankrt1ss",lowrankrt1ss,
                "nndsvdrt1ss",nndsvdrt1ss,"rlowrankrt1ss",rlowrankrt1ss,"rnndsvdrt1ss",rnndsvdrt1ss)


dd = load(joinpath(subworkpath,"factorrng_ncellsrng_vs_rt1s.jld2"))
imgsz0 = dd["imgsz0"]; lengthT0 = dd["lengthT0"]
factorrng = dd["factorrng"]; ncellsrng = dd["ncellsrng"]
isvdrt1ss = dd["isvdrt1ss"]; lowrankrt1ss = dd["lowrankrt1ss"]; nndsvdrt1ss = dd["nndsvdrt1ss"]
rlowrankrt1ss = dd["rlowrankrt1ss"]; rnndsvdrt1ss = dd["rnndsvdrt1ss"]

fig = Figure(resolution = (500,400))
ax1 = AMakie.Axis(fig[1, 1], xlabel = "ncells/datasize", ylabel = "time(sec)", title = "ncells/datasize vs. Initialization time",
    xtickformat = values -> ["$(value)" for value in values])
fl = length(factorrng); nl = length(ncellsrng)
cls = distinguishable_colors(fl*nl; lchoices=range(0, stop=50, length=20))
for (i, factor) in [(5,5)] # enumerate(factorrng)
    lengthT = lengthT0*factor
    ratiorng = ncellsrng./lengthT
    @show ratiorng
    lines!(ax1, ratiorng, Float32.(isvdrt1ss[i]), color=mtdcolors[2], label="ISVD($factor)")
    lines!(ax1, ratiorng, Float32.(lowrankrt1ss[i]), color=mtdcolors[3], label="COMP($factor)")
    lines!(ax1, ratiorng, Float32.(nndsvdrt1ss[i]), color=mtdcolors[4], label="NNDSVD($factor)")
    lines!(ax1, ratiorng, Float32.(rlowrankrt1ss[i]), color=mtdcolors[7], label="rCOMP($factor)")
    lines!(ax1, ratiorng, Float32.(rnndsvdrt1ss[i]), color=mtdcolors[8], label="rNNDSVD($factor)")
end
# axislegend(ax1, position = :lt) # halign = :left, valign = :top
save(joinpath(subworkpath,"factorrng_ncellsrng_vs_rt1s.png"),fig,px_per_unit=2)

=#