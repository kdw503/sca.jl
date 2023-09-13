using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)
subworkpath = joinpath(workpath,"paper","initialization")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 50; ncellsrng = 4:2:200; factorrng=1:10
else
    num_experiments = 2; ncellsrng = 5:6; factorrng=1:2
end


# ncells

isvdmeans=Float64[]; isvdstds=Float64[]
lowrankmeans=Float64[]; lowrankstds=Float64[]; nndsvdmeans=Float64[]; nndsvdstds=Float64[]
rlowrankmeans=Float64[]; rlowrankstds=Float64[]; rnndsvdmeans=Float64[]; rnndsvdstds=Float64[]
for (iter, ncl) in enumerate(ncellsrng)
    @show ncl; flush(stdout)

isvdrt1s = []; lowrankrt1s=[]; nndsvdrt1s=[]; rlowrankrt1s=[]; rnndsvdrt1s=[]
for i in 1:num_experiments
    X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
            inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false);
    isvdrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncl, initmethod=:isvd,poweradjust=initpwradj)
    lowrankrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncl, initmethod=:lowrank_nndsvd,poweradjust=initpwradj,svdmethod=:svd)
    nndsvdrt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncl, variant=:ar, initdata=svd(X))
    rlowrankrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncl, initmethod=:lowrank_nndsvd,poweradjust=initpwradj,svdmethod=:rsvd)
    rnndsvdrt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncl, variant=:ar)
    push!(isvdrt1s,isvdrt1); push!(lowrankrt1s,lowrankrt1); push!(nndsvdrt1s,nndsvdrt1)
    push!(rlowrankrt1s,rlowrankrt1); push!(rnndsvdrt1s,rnndsvdrt1)
end
isvdmean = mean(isvdrt1s); isvdstd = std(isvdrt1s)
lowrankmean = mean(lowrankrt1s); lowrankstd = std(lowrankrt1s)
nndsvdmean = mean(nndsvdrt1s); nndsvdstd = std(nndsvdrt1s)
rlowrankmean = mean(rlowrankrt1s); rlowrankstd = std(rlowrankrt1s)
rnndsvdmean = mean(rnndsvdrt1s); rnndsvdstd = std(rnndsvdrt1s)
push!(isvdmeans,isvdmean); push!(isvdstds,isvdstd)
push!(lowrankmeans,lowrankmean); push!(lowrankstds,lowrankstd)
push!(nndsvdmeans,nndsvdmean); push!(nndsvdstds,nndsvdstd)
push!(rlowrankmeans,rlowrankmean); push!(rlowrankstds,rlowrankstd)
push!(rnndsvdmeans,rnndsvdmean); push!(rnndsvdstds,rnndsvdstd)
end
save(joinpath(subworkpath,"ncellsrng_vs_rt1s.jld2"),
            "ncellsrng",ncellsrng, "isvdmeans",isvdmeans,"isvdstds",isvdstds,
            "lowrankmeans",lowrankmeans,"lowrankstds",lowrankstds, "nndsvdmeans",nndsvdmeans,"nndsvdstds",nndsvdstds,
            "rlowrankmeans",rlowrankmeans,"rlowrankstds",rlowrankstds, "rnndsvdmeans",rnndsvdmeans,"rnndsvdstds",rnndsvdstds)

#             isvdmeans0=copy(isvdmeans); isvdstds0=copy(isvdstds)
#             lowrankmeans0=copy(lowrankmeans); lowrankstds0=copy(lowrankstds); nndsvdmeans0 =copy(nndsvdmeans); nndsvdstds0= copy(nndsvdstds)
#             rlowrankmeans0=copy(rlowrankmeans); rlowrankstds0= copy(rlowrankstds); rnndsvdmeans0=copy(rnndsvdmeans); rnndsvdstds0= copy(rnndsvdstds)
# dd = load(joinpath(subworkpath,"ncellsrng_vs_rt1s3.jld2")); ncellsrng = dd["ncellsrng"]
# isvdmeans = dd["isvdmeans"]; isvdstds = dd["isvdstds"]
# lowrankmeans = dd["lowrankmeans"]; lowrankstds = dd["lowrankstds"]
# nndsvdmeans = dd["nndsvdmeans"]; nndsvdstds = dd["nndsvdstds"]
# rlowrankmeans = dd["rlowrankmeans"]; rlowrankstds = dd["rlowrankstds"]
# rnndsvdmeans = dd["rnndsvdmeans"]; rnndsvdstds = dd["rnndsvdstds"]

# nndsvdmeans[78] = nndsvdmeans0[2]; nndsvdstds[78] = nndsvdstds0[2]
# rlowrankmeans[32] = rlowrankmeans0[1]; rlowrankstds[32] = rlowrankstds0[1]
# rlowrankmeans[78] = rlowrankmeans0[2]; rlowrankstds[78] = rlowrankstds0[2]
# save(joinpath(subworkpath,"ncellsrng_vs_rt1s4.jld2"),
#             "ncellsrng",ncellsrng, "isvdmeans",isvdmeans,"isvdstds",isvdstds,
#             "lowrankmeans",lowrankmeans,"lowrankstds",lowrankstds, "nndsvdmeans",nndsvdmeans,"nndsvdstds",nndsvdstds,
#             "rlowrankmeans",rlowrankmeans,"rlowrankstds",rlowrankstds, "rnndsvdmeans",rnndsvdmeans,"rnndsvdstds",rnndsvdstds)


z = 0.5
dd = load(joinpath(subworkpath,"ncellsrng_vs_rt1s.jld2")); ncellsrng = dd["ncellsrng"]
sca_means = dd["isvdmeans"]; sca_stds = dd["isvdstds"]
sca_upper = sca_means + z*sca_stds; sca_lower = sca_means - z*sca_stds
admm_means = dd["lowrankmeans"]; admm_stds = dd["lowrankstds"]
admm_upper = admm_means + z*admm_stds; admm_lower = admm_means - z*admm_stds
hals_means = dd["nndsvdmeans"]; hals_stds = dd["nndsvdstds"]
hals_upper = hals_means + z*hals_stds; hals_lower = hals_means - z*hals_stds
admm_r_means = dd["rlowrankmeans"]; admm_r_stds = dd["rlowrankstds"]
admm_r_upper = admm_r_means + z*admm_r_stds; admm_r_lower = admm_r_means - z*admm_r_stds
hals_r_means = dd["rnndsvdmeans"]; hals_r_stds = dd["rnndsvdstds"]
hals_r_upper = hals_r_means + z*hals_r_stds; hals_r_lower = hals_r_means - z*hals_r_stds

fig = Figure()
ax1 = GLMakie.Axis(fig[1, 1], xlabel = "number of components", ylabel = "time(sec)", title = "Number of cells vs. Initialization time")

lines!(ax1, ncellsrng, sca_means, color=mtdcolors[2], label="SVD")
band!(ax1, ncellsrng, sca_lower, sca_upper, color=mtdcoloras[2])
lines!(ax1, ncellsrng, admm_means, color=mtdcolors[3], label="Compression + NNDSVD")
band!(ax1, ncellsrng, admm_lower, admm_upper, color=mtdcoloras[3])
lines!(ax1, ncellsrng, hals_means, color=mtdcolors[4], label="NNDSVD")
band!(ax1, ncellsrng, hals_lower, hals_upper, color=mtdcoloras[4])
lines!(ax1, ncellsrng, admm_r_means, color=mtdcolors[5], linestyle=:dot, label="Compression + NNDSVD(random)")
band!(ax1, ncellsrng, admm_r_lower, admm_r_upper, color=mtdcoloras[5])
lines!(ax1, ncellsrng, hals_r_means, color=mtdcolors[6], linestyle=:dot, label="NNDSVD(random)")
band!(ax1, ncellsrng, hals_r_lower, hals_r_upper, color=mtdcoloras[6])

axislegend(ax1, position = :lt) # halign = :left, valign = :top
save(joinpath(subworkpath,"ncellsrng_vs_rt1s.png"),fig,px_per_unit=2)



# factor

isvdmeans=Float64[]; isvdstds=Float64[]
lowrankmeans=Float64[]; lowrankstds=Float64[]; nndsvdmeans=Float64[]; nndsvdstds=Float64[]
rlowrankmeans=Float64[]; rlowrankstds=Float64[]; rnndsvdmeans=Float64[]; rnndsvdstds=Float64[]
for (iter, factor) in enumerate(factorrng)
    @show factor; flush(stdout)
    imgsz = (40*factor,20); lengthT = 100*factor
isvdrt1s = []; lowrankrt1s=[]; nndsvdrt1s=[]; rlowrankrt1s=[]; rnndsvdrt1s=[]
for i in 1:num_experiments
    X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, imgsz=imgsz, bias=bias, lengthT=lengthT, useCalciumT=true,
            inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false);
    X = noisefilter(filter,X)
    isvdrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=initpwradj)
    lowrankrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:lowrank_nndsvd,poweradjust=initpwradj,svdmethod=:svd)
    nndsvdrt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar, initdata=svd(X))
    rlowrankrt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:lowrank_nndsvd,poweradjust=initpwradj,svdmethod=:rsvd)
    rnndsvdrt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
    push!(isvdrt1s,isvdrt1); push!(lowrankrt1s,lowrankrt1); push!(nndsvdrt1s,nndsvdrt1)
    push!(rlowrankrt1s,rlowrankrt1); push!(rnndsvdrt1s,rnndsvdrt1)
end
isvdmean = mean(isvdrt1s); isvdstd = std(isvdrt1s)
lowrankmean = mean(lowrankrt1s); lowrankstd = std(lowrankrt1s)
nndsvdmean = mean(nndsvdrt1s); nndsvdstd = std(nndsvdrt1s)
rlowrankmean = mean(rlowrankrt1s); rlowrankstd = std(rlowrankrt1s)
rnndsvdmean = mean(rnndsvdrt1s); rnndsvdstd = std(rnndsvdrt1s)
push!(isvdmeans,isvdmean); push!(isvdstds,isvdstd)
push!(lowrankmeans,lowrankmean); push!(lowrankstds,lowrankstd)
push!(nndsvdmeans,nndsvdmean); push!(nndsvdstds,nndsvdstd)
push!(rlowrankmeans,rlowrankmean); push!(rlowrankstds,rlowrankstd)
push!(rnndsvdmeans,rnndsvdmean); push!(rnndsvdstds,rnndsvdstd)
end
save(joinpath(subworkpath,"factorrng_vs_rt1s.jld2"),
            "factorrng",factorrng, "isvdmeans",isvdmeans,"isvdstds",isvdstds,
            "lowrankmeans",lowrankmeans,"lowrankstds",lowrankstds, "nndsvdmeans",nndsvdmeans,"nndsvdstds",nndsvdstds,
            "rlowrankmeans",rlowrankmeans,"rlowrankstds",rlowrankstds, "rnndsvdmeans",rnndsvdmeans,"rnndsvdstds",rnndsvdstds)

z = 0.5
dd = load(joinpath(subworkpath,"factorrng_vs_rt1s.jld2")); factorrng = dd["factorrng"]
sca_means = dd["isvdmeans"]; sca_stds = dd["isvdstds"]
sca_upper = sca_means + z*sca_stds; sca_lower = sca_means - z*sca_stds
admm_means = dd["lowrankmeans"]; admm_stds = dd["lowrankstds"]
admm_upper = admm_means + z*admm_stds; admm_lower = admm_means - z*admm_stds
hals_means = dd["nndsvdmeans"]; hals_stds = dd["nndsvdstds"]
hals_upper = hals_means + z*hals_stds; hals_lower = hals_means - z*hals_stds
admm_r_means = dd["rlowrankmeans"]; admm_r_stds = dd["rlowrankstds"]
admm_r_upper = admm_r_means + z*admm_r_stds; admm_r_lower = admm_r_means - z*admm_r_stds
hals_r_means = dd["rnndsvdmeans"]; hals_r_stds = dd["rnndsvdstds"]
hals_r_upper = hals_r_means + z*hals_r_stds; hals_r_lower = hals_r_means - z*hals_r_stds

fig = Figure(resolution = (500,400))
ax1 = GLMakie.Axis(fig[1, 1], xlabel = "data size (Mbyte)", ylabel = "time(sec)", title = "Data size vs. Initialization time",
    xtickformat = values -> ["$(value^2*8/10)" for value in values])

lines!(ax1, factorrng, sca_means, color=mtdcolors[2], label="SVD")
band!(ax1, factorrng, sca_lower, sca_upper, color=mtdcoloras[2])
lines!(ax1, factorrng, admm_means, linestyle = :dash, color=mtdcolors[3], label="Compression + NNDSVD")
band!(ax1, factorrng, admm_lower, admm_upper, color=mtdcoloras[3])
lines!(ax1, factorrng, hals_means, linestyle = :dashdot, color=mtdcolors[4], label="NNDSVD")
band!(ax1, factorrng, hals_lower, hals_upper, color=mtdcoloras[4])
lines!(ax1, factorrng, admm_r_means, color=mtdcolors[5], linestyle=:dot, label="Compression + NNDSVD(random)")
band!(ax1, factorrng, admm_r_lower, admm_r_upper, color=mtdcoloras[5])
lines!(ax1, factorrng, hals_r_means, color=mtdcolors[6], linestyle=:dot, label="NNDSVD(random)")
band!(ax1, factorrng, hals_r_lower, hals_r_upper, color=mtdcoloras[6])

axislegend(ax1, position = :lt) # halign = :left, valign = :top
save(joinpath(subworkpath,"factorrng_vs_rt1s.png"),fig,px_per_unit=2)



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
ax1 = GLMakie.Axis(fig[1, 1], xlabel = "ncells/datasize", ylabel = "time(sec)", title = "ncells/datasize vs. Initialization time",
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

