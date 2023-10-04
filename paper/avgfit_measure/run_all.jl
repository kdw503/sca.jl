using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","avgfit_measure")

include(joinpath(workpath,"setup_light.jl"))

dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if false
    sca_maxiter = 100; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 100
else
    sca_maxiter = 4; sca_inner_maxiter = 50; sca_ls_maxiter = 2
    admm_maxiter = 2; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 2
end

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# SCA
@show "SCA"
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN = :WH2M2; α = 0.001; β = 0
useRelaxedL1=true; useRelaxedNN=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none

(tailstr,initmethod,α,β) = ("_sp",:isvd,0.001,0.)
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#
α1=α2=α; β1=β2=β
avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=false,
    store_inner_trace=false, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W3,H3)
initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_$(optimmethod)_a$(α1)_b$(β1)_af$(avgfit)_r$(r)_it$(niters)_rt$(rt2)")
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)

# ADMM
@show "ADMM"
mfmethod = :ADMM; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 10; β = 0; usennc=true
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = :none

(tailstr,initmethod,α,usennc) = ("_nn",:lowrank_nndsvd,0.,true)
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#
avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
α1=α2=α; β1=β2=β
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
    store_trace=false, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                    penmetric=penmetric, stparams=stparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed  W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                    penmetric=penmetric, stparams=stparams, cparams=cparams);
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_a$(α1)_af$(avgfit)_it$(niters)_rt$(rt2)")
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)

# HALS
@show "HALS"
W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize)
initmethod = :svd
if initmethod == :svd
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar, initdata=svd(X));
else
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
end
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1
(tailstr,α) = ("_nn",0.)
Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                tol=tol, verbose=false), X, Wcd, Hcd)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
normalizeW!(Wcd,Hcd); W3,H3 = sortWHslices(Wcd,Hcd)
initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_a$(α)_af$(avgfit)_it$(hals_maxiter)_rt$(rt2)")
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)


#====== SPCA =================#
using ScikitLearn

mfmethod = :SPCA

@sk_import decomposition: SparsePCA
α = 0.5; ridge_alpha=0.01; max_iter=500; tol=1e-7
rtspca = @elapsed resultspca = fit_transform!(SparsePCA(n_components=ncells,alpha=α,ridge_alpha=ridge_alpha,max_iter=max_iter,tol=tol,verbose=true),X) 
W1 = copy(resultspca); H1 = W1\X
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W3,H3)
fprex = "$(mfmethod)$(datastr)$(filterstr)"
fname = joinpath(subworkpath,"$(fprex)_a$(α)_af$(avgfit)_rt$(rtspca)")
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)

#====== make avg_fit faster =============#
include(joinpath(workpath,"setup_plot.jl"))

@show "SCA"
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN = :WH2M2; α = 0.001; β = 0.01
useRelaxedL1=true; useRelaxedNN=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none

num_experiments = 2
factor = 1; fovsz = (40,20)
imgsz = (factor*fovsz[1],fovsz[2]); lengthT = factor*1000
maskths = [8, 1.0, 0.5, 0.25]
for iter in 1:num_experiments
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.001,0.)
# maskW=zeros(Bool,imgsz); maskW[1:fovsz[1],1:fovsz[2]].=true; maskW = vec(maskW); maskH = Colon()
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; imgsz=imgsz, fovsz=fovsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
    inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

for maskth in maskths
if maskth < 1.0
    maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
else
    maskW=Colon(); maskH=Colon()
end
step = maskth > 1.0 ? maskth : 8

rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#
α1=α2=α; β1=β2=β
avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust, λ=step,
    useprecond=useprecond, uselv=uselv, maskW=maskW, maskH=maskH)#
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                    penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW[maskW,:], gtH[maskH,:], W1[maskW,:], H1[:,maskH]; clamp=false)
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W3,H3)
initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_$(optimmethod)_a$(α1)_b$(β1)_af$(avgfit)_r$(r)_it$(niters)_rt$(rt2)")
imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false)

f_xs = getdata(trs,:f_x); niters = getdata(trs,:niter); totalniters = sum(niters)
avgfitss = getdata(trs,:avgfits); fxss = getdata(trs,:fxs)
avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
    isempty(afs) && continue
    append!(avgfits,afs); append!(inner_fxs,fxs)
    if iter == 1
        rt2i = 0.
    else
        rt2i = collect(range(start=laps[iter-1],stop=laps[iter],length=length(afs)+1))[2:end].-laps[1]
    end
    append!(rt2s,rt2i)
end
@show length(rt2s); flush(stdout)

dd = Dict()
dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
if true#iter == num_experiments
    metadata = Dict()
    metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
    metadata["alpha"] = α; metadata["beta"] = β; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
end
save(joinpath(subworkpath,"sca$(maskth)_randmask_results$(iter).jld2"),"metadata",metadata,"data",dd)
end
end

using Interpolations

for maskth in maskths
rt2_min = Inf
for iter in 1:num_experiments
    dd = load(joinpath(subworkpath,"sca$(maskth)_randmask_results$(iter).jld2"),"data")
    rt2s = dd["rt2s"]; @show rt2s[end]
    rt2_min = min(rt2_min,rt2s[end])
end
rt2_min = floor(rt2_min, digits=4)
rng = range(0,stop=rt2_min,length=100)

stat_sp=[]; afs=[]
for iter in 1:num_experiments
    dd = load(joinpath(subworkpath,"sca$(maskth)_randmask_results$(iter).jld2"))
    rt2s = dd["data"]["rt2s"]; avgfits = dd["data"]["avgfits"]
    lr = length(rt2s); la = length(avgfits)
    lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; avgfits=avgfits[1:l])
    nodes = (rt2s,)
    itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
    push!(afs,itp(rng))
end
avgfits = hcat(afs...)
means = dropdims(mean(avgfits,dims=2),dims=2)
stds = dropdims(std(avgfits,dims=2),dims=2)
push!(stat_sp,means); push!(stat_sp,stds)
save(joinpath(subworkpath,"sca$(maskth)_randmask_runtime_vs_avgfits.jld2"),"rng",rng, "stat_sp",stat_sp)
end

z = 0.5
ddsca=load(joinpath(subworkpath,"sca8.0_randmask_runtime_vs_avgfits.jld2"))
rng8 = ddsca["rng"]; sca8_sp_means=ddsca["stat_sp"][1]; sca_sp_stds=ddsca["stat_sp"][2]
sca8_sp_upper=sca8_sp_means+z*sca_sp_stds; sca8_sp_lower=sca8_sp_means-z*sca_sp_stds 
ddsca=load(joinpath(subworkpath,"sca1.0_randmask_runtime_vs_avgfits.jld2"))
rng1p0 = ddsca["rng"]; sca1p0_sp_means=ddsca["stat_sp"][1]; sca_sp_stds=ddsca["stat_sp"][2]
sca1p0_sp_upper=sca1p0_sp_means+z*sca_sp_stds; sca1p0_sp_lower=sca1p0_sp_means-z*sca_sp_stds 
ddsca=load(joinpath(subworkpath,"sca0.5_randmask_runtime_vs_avgfits.jld2"))
rng0p5 = ddsca["rng"]; sca0p5_sp_means=ddsca["stat_sp"][1]; sca_sp_stds=ddsca["stat_sp"][2]
sca0p5_sp_upper=sca0p5_sp_means+z*sca_sp_stds; sca0p5_sp_lower=sca0p5_sp_means-z*sca_sp_stds 
ddsca=load(joinpath(subworkpath,"sca0.25_randmask_runtime_vs_avgfits.jld2"))
rng0p25 = ddsca["rng"]; sca0p25_sp_means=ddsca["stat_sp"][1]; sca_sp_stds=ddsca["stat_sp"][2]
sca0p25_sp_upper=sca0p25_sp_means+z*sca_sp_stds; sca0p25_sp_lower=sca0p25_sp_means-z*sca_sp_stds 

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon()
fig = Figure(resolution=(400,300))
maxplottimes = [3.0]
ax = AMakie.Axis(fig[1, 1], limits = ((0,0.3), nothing), xlabel = "time(sec)", ylabel = "average fit", title = "Average Fit Value vs. Running Time")
lines!(ax, rng8[plotrng], sca8_sp_means[plotrng], color=mtdcolors[5], label="no mask 8 step")
band!(ax, rng8[plotrng], sca8_sp_lower[plotrng], sca8_sp_upper[plotrng], color=mtdcoloras[5])
lines!(ax, rng1p0[plotrng], sca1p0_sp_means[plotrng], color=mtdcolors[2], label="no mask")
band!(ax, rng1p0[plotrng], sca1p0_sp_lower[plotrng], sca1p0_sp_upper[plotrng], color=mtdcoloras[2])
lines!(ax, rng0p5[plotrng], sca0p5_sp_means[plotrng], color=mtdcolors[3], label="rand 0.5")
band!(ax, rng0p5[plotrng], sca0p5_sp_lower[plotrng], sca0p5_sp_upper[plotrng], color=mtdcoloras[3])
lines!(ax, rng0p25[plotrng], sca0p25_sp_means[plotrng], color=mtdcolors[4], label="rand 0.25")
band!(ax, rng0p25[plotrng], sca0p25_sp_lower[plotrng], sca0p25_sp_upper[plotrng], color=mtdcoloras[4])

axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits_vs_runtime_randmask.png"),fig,px_per_unit=2)

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon()
fig = Figure(resolution=(400,300))
maxplottimes = [3.0]
ax = AMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "average fit", title = "Average Fit Value vs. Iteration")
lines!(ax, 1:length(sca8_sp_means), sca8_sp_means[plotrng], color=mtdcolors[5], label="no mask 8 step")
band!(ax, 1:length(sca8_sp_means), sca8_sp_lower[plotrng], sca8_sp_upper[plotrng], color=mtdcoloras[5])
lines!(ax, 1:length(sca1p0_sp_means), sca1p0_sp_means[plotrng], color=mtdcolors[2], label="no mask")
band!(ax, 1:length(sca1p0_sp_means), sca1p0_sp_lower[plotrng], sca1p0_sp_upper[plotrng], color=mtdcoloras[2])
lines!(ax, 1:length(sca0p5_sp_means), sca0p5_sp_means[plotrng], color=mtdcolors[3], label="rand 0.5")
band!(ax, 1:length(sca0p5_sp_means), sca0p5_sp_lower[plotrng], sca0p5_sp_upper[plotrng], color=mtdcoloras[3])
lines!(ax, 1:length(sca0p25_sp_means), sca0p25_sp_means[plotrng], color=mtdcolors[4], label="rand 0.25")
band!(ax, 1:length(sca0p25_sp_means), sca0p25_sp_lower[plotrng], sca0p25_sp_upper[plotrng], color=mtdcoloras[4])

axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits_vs_iteration_randmask.png"),fig,px_per_unit=2)

