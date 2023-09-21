using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","cbclface")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :cbclface; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 50
    sca_maxiter = 80; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 400
else
    num_experiments = 2
    sca_maxiter = 2; sca_inner_maxiter = 2; sca_ls_maxiter = 2
    admm_maxiter = 2; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 2
end

αrng = 1:20:1000
for (iter, α) in enumerate(αrng)
    @show iter; flush(stdout)
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
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# SCA
@show "SCA"
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN = :WH2M2
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
#try
for (tailstr,initmethod,β) in [("_sp",:isvd,0.)]
    @show tailstr; flush(stdout)
    dd = Dict()
    rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    σ0=s*std(W0) #=10*std(W0)=#
    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
        useprecond=useprecond, uselv=uselv)
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
    f_xs = getdata(trs,:f_x); niters = getdata(trs,:niter); totalniters = sum(niters)
    avgfitss = getdata(trs,:avgfits); sparseWss = getdata(trs,:sparseWs); fxss = getdata(trs,:fxs)
    avgfits = Float64[]; sparseWs = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
    for (i,(afs,sws,fxs)) in enumerate(zip(avgfitss, sparseWss, fxss))
        isempty(afs) && continue
        append!(avgfits,afs); append!(sparseWs,sws); append!(inner_fxs,fxs)
        if i == 1
            rt2i = 0.
        else
            rt2i = collect(range(start=laps[i-1],stop=laps[i],length=length(afs)+1))[2:end].-laps[1]
        end
        append!(rt2s,rt2i)
    end
    @show length(rt2s); flush(stdout)

    dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["sparseWs"] = sparseWs; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
        metadata["alpha"] = α; metadata["beta"] = β; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
    end
    save(joinpath(subworkpath,"sca","sca$(tailstr)_alpha_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
end
# catch e
#     save(joinpath(subworkpath,"sca_error_$(iter).jld2"),datadic)
#     @warn e
#     iter -= 1
# end
end


using Interpolations

num_expriments=50
rt2_min = Inf
for tailstr in ["_sp"]
    for iter in 1:num_expriments
        dd = load(joinpath(subworkpath,"sca","sca$(tailstr)_alpha_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]; @show rt2s[end]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_minf = floor(rt2_min, digits=2)
rng = range(0,stop=rt2_minf,length=100)

stat_nn1=[]; stat_sp1=[]; stat_sp_nn1=[] # fits
stat_nn2=[]; stat_sp2=[]; stat_sp_nn2=[] # sparse W
for tailstr in ["_sp"]
    afs=[]; sws=[]
    for iter in 1:num_expriments
        @show tailstr, iter
        dd = load(joinpath(subworkpath,"sca","sca$(tailstr)_alpha_results$(iter).jld2"))
        rt2s = dd["data"]["rt2s"]; avgfits = dd["data"]["avgfits"]; sparseWs = dd["data"]["sparseWs"]
        lr = length(rt2s); la = length(avgfits)
        lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; avgfits=avgfits[1:l])
        nodes = (rt2s,)
        itp1 = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
        itp2 = Interpolations.interpolate(nodes, sparseWs, Gridded(Linear()))
        push!(afs,itp1(rng)); push!(sws,itp2(rng))
    end
    avgfits = hcat(afs...); sparseWs = hcat(sws...)
    means1 = dropdims(mean(avgfits,dims=2),dims=2)
    stds1 = dropdims(std(avgfits,dims=2),dims=2)
    means2 = dropdims(mean(sparseWs,dims=2),dims=2)
    stds2 = dropdims(std(sparseWs,dims=2),dims=2)
    tailstr == "_nn" && (push!(stat_nn1,means1); push!(stat_nn1,stds1);
                        push!(stat_nn2,means2); push!(stat_nn2,stds2))
    tailstr == "_sp" && (push!(stat_sp1,means1); push!(stat_sp1,stds1);
                        push!(stat_sp2,means2); push!(stat_sp2,stds2))
    tailstr == "_sp_nn" && (push!(stat_sp_nn1,means1); push!(stat_sp_nn1,stds1);
                        push!(stat_sp_nn2,means2); push!(stat_sp_nn2,stds2))
end
save(joinpath(subworkpath,"sca_cbcl_alpha_runtime_vs_fits.jld2"),"rng",rng,
        "stat_nn1", stat_nn1, "stat_sp1", stat_sp1, "stat_sp_nn1", stat_sp_nn1,
        "stat_nn2", stat_nn2, "stat_sp2", stat_sp2, "stat_sp_nn2", stat_sp_nn2)
