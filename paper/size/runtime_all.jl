using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","size")

include(joinpath(workpath,"setup_light.jl"))

dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 2
    sca_maxiter = 100; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 400; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 200
else
    num_experiments = 2
    sca_maxiter = 4; sca_inner_maxiter = 2; sca_ls_maxiter = 2
    admm_maxiter = 4; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 4
end
factors = [10]
for iter in 1:num_experiments
    @show iter; flush(stdout)
for factor = factors
    fovsz = (40,20)
    imgsz = (factor*fovsz[1],fovsz[2]); lengthT = factor*1000
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
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# SCA
@show "SCA"
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; α = 0.001; β = 0.01
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
try
for (tailstr,initmethod,α,β) in [("_sp",:isvd,0.001,0.), ("_nn",:nndsvd,0.,0.01), ("_sp_nn",:nndsvd,0.001,0.001)]
    @show tailstr; flush(stdout)
    dd = Dict()
    rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    σ0=s*std(W0) #=10*std(W0)=#
    for β in [0.001,0.01,0.1,1,10,100]
    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
        useprecond=useprecond, uselv=uselv)
    lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    # Mw, Mh = copy(Mw0), copy(Mh0);
    # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
    #                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                     penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    makepositive && flip2makepos!(W3,H3)
    fprex = "sca$(factor)"
    fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false)
    end
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

    dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
        metadata["alpha"] = α; metadata["beta"] = β; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
    end
    save(joinpath(subworkpath,"sca","sca$(factor)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
end
catch e
    save(joinpath(subworkpath,"sca$(factor)_error_$(iter).jld2"),datadic)
    @warn e
    iter -= 1
end
end # for factor
end # for iter


using Interpolations

for factor = factors
rt2_min = Inf
for tailstr in ["_sp","_sp_nn"]
    for iter in 1:num_experiments
        dd = load(joinpath(subworkpath,"sca","sca$(factor)$(tailstr)_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]; @show rt2s[end]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_min = floor(rt2_min, digits=4)
rng = range(0,stop=rt2_min,length=100)

stat_nn=[]; stat_sp=[]; stat_sp_nn=[]
for tailstr in ["_sp","_sp_nn"]
    afs=[]
    for iter in 1:num_experiments
        @show tailstr, iter
        dd = load(joinpath(subworkpath,"sca","sca$(factor)$(tailstr)_results$(iter).jld2"))
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
    tailstr == "_nn" && (push!(stat_nn,means); push!(stat_nn,stds))
    tailstr == "_sp" && (push!(stat_sp,means); push!(stat_sp,stds))
    tailstr == "_sp_nn" && (push!(stat_sp_nn,means); push!(stat_sp_nn,stds))
end
save(joinpath(subworkpath,"sca$(factor)_runtime_vs_avgfits.jld2"),"rng",rng, "stat_nn", stat_nn, "stat_sp", stat_sp, "stat_sp_nn", stat_sp_nn)
end


factors = [20]
for iter in 1:num_experiments
    @show iter; flush(stdout)
for factor = factors
    fovsz = (40,20)
    imgsz = (factor*fovsz[1],fovsz[2]); lengthT = factor*100
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
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# ADMM
@show "ADMM"
mfmethod = :ADMM; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; α = 10; β = 0; usennc=true
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = :none

for (tailstr,initmethod,α,usennc) in [("_nn",:lowrank_nndsvd,0.,true)]#, ("_sp",:lowrank,10.,false), ("_sp_nn",:lowrank_nndsvd,10.,true)]
    @show tailstr; flush(stdout)
    dd = Dict()
    rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    σ0=s*std(W0) #=10*std(W0)=#
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    α1=α2=α; β1=β2=β
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        regSpar=regSpar, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
        store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    # Mw, Mh = copy(Mw0), copy(Mh0);
    # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
    #                                                     penmetric=penmetric, stparams=stparams, cparams=cparams);
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed  W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                        penmetric=penmetric, stparams=stparams, cparams=cparams);
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    fprex = "admm$(factor)"
    fname = joinpath(subworkpath,"$(fprex)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)
    f_xs = getdata(trs,:f_x)
    avgfitss = getdata(trs,:avgfits)
    avgfits = Float64[]
    for (iter,afs) in enumerate(avgfitss)
        isempty(afs) && continue
        append!(avgfits,afs)
    end
    rt2s = collect(range(start=0,stop=rt2,length=length(avgfits)))

    dd["niters"] = niters; dd["totalniters"] = niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = Float64[]
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = 0; metadata["r"] = 0; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
        metadata["alpha"] = α; metadata["beta"] = 0; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
    end
    save(joinpath(subworkpath,"admm","admm$(factor)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end
end # for factor
end # for iter


using Interpolations

for factor = factors
rt2_min = Inf
for tailstr in ["_nn"]#,"_sp","_sp_nn"]
    for iter in 1:num_experiments
        dd = load(joinpath(subworkpath,"admm","admm$(factor)$(tailstr)_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]; @show rt2s[end]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_min = floor(rt2_min, digits=4)
rng = range(0,stop=rt2_min,length=100)

stat_nn=[]; stat_sp=[]; stat_sp_nn=[]
for tailstr in ["_nn"]#,"_sp","_sp_nn"]
    afs=[]
    for iter in 1:num_experiments
        @show tailstr, iter
        dd = load(joinpath(subworkpath,"admm","admm$(factor)$(tailstr)_results$(iter).jld2"))
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
    tailstr == "_nn" && (push!(stat_nn,means); push!(stat_nn,stds))
    tailstr == "_sp" && (push!(stat_sp,means); push!(stat_sp,stds))
    tailstr == "_sp_nn" && (push!(stat_sp_nn,means); push!(stat_sp_nn,stds))
end
save(joinpath(subworkpath,"admm$(factor)_runtime_vs_avgfits.jld2"),"rng",rng, "stat_nn", stat_nn, "stat_sp", stat_sp, "stat_sp_nn", stat_sp_nn)
end




factors = [20]
for iter in 1:num_experiments
    @show iter; flush(stdout)
for factor = factors
    fovsz = (40,20)
    imgsz = (factor*fovsz[1],fovsz[2]); lengthT = factor*100
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
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);

# HALS
@show "HALS"
W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize)
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1
for (tailstr,α) in [("_nn",0.),("_sp_nn",0.1)]
    @show tailstr; flush(stdout)
    dd = Dict()
    # Wcd, Hcd = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
    # result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
    #                 tol=tol, verbose=true), X, Wcd, Hcd; W0=W0, H0=H0, d=diag(D), gtW=gtW, gtH=gtH)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, Wcd, Hcd)
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
    normalizeW!(Wcd,Hcd); W3,H3 = sortWHslices(Wcd,Hcd)
    fprex = "halse$(factor)"
    fname = joinpath(subworkpath,"$(fprex)_a$(α)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)
    rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
    if true#iter == num_experiments
        metadata = Dict()
        metadata["maxiter"] = maxiter; metadata["alpha"] = α
    end
    save(joinpath(subworkpath,"hals","hals$(factor)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end
end # for factor
end # for iter

using Interpolations

for factor = factors
rt2_min = Inf
for tailstr in ["_nn","_sp_nn"]
    for iter in 1:num_experiments
        dd = load(joinpath(subworkpath,"hals","hals$(factor)$(tailstr)_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_min = floor(rt2_min, digits=4)
rng = range(0,stop=rt2_min,length=100)

stat_nn=[]; stat_sp_nn=[]
for tailstr in ["_nn","_sp_nn"]
    afs=[]
    for iter in 1:num_experiments
        dd = load(joinpath(subworkpath,"hals","hals$(factor)$(tailstr)_results$(iter).jld2"))
        rt2s = dd["data"]["rt2s"]; avgfits = dd["data"]["avgfits"]
        nodes = (rt2s,)
        itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
        push!(afs,itp(rng))
    end
    avgfits = hcat(afs...)
    means = dropdims(mean(avgfits,dims=2),dims=2)
    stds = dropdims(std(avgfits,dims=2),dims=2)
    tailstr == "_nn" && (push!(stat_nn,means); push!(stat_nn,stds))
    tailstr == "_sp_nn" && (push!(stat_sp_nn,means); push!(stat_sp_nn,stds))
end
save(joinpath(subworkpath,"hals$(factor)_runtime_vs_avgfits.jld2"),"rng",rng, "stat_nn", stat_nn, "stat_sp_nn", stat_sp_nn)
end
