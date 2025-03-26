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

# ARGS =  ["0","20","1","150","120","800",0.1]
# ARGS =  ["0","20","5","100","100","600"0.3]
# ARGS =  ["0","20","10","50","50","400",0.5]
# ARGS = ["0","2","4","2","2","2",0.1]
# julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\size\runtime_all.jl 0 20 150 120 800 0.1
SNR = eval(Meta.parse(ARGS[1])); num_experiments = eval(Meta.parse(ARGS[2]));
factor = eval(Meta.parse(ARGS[3]));
ncells = 15
lcsvd_maxiter = eval(Meta.parse(ARGS[4]));
hals_maxiter = eval(Meta.parse(ARGS[5]));
compnmf_maxiter = eval(Meta.parse(ARGS[6]));
hals_α = eval(Meta.parse(ARGS[7]));

# for factor == 10
# lcsvd_maxiter = 50; hals_maxiter = 50; compnmf_maxiter = 400

# for factor == 1
# lcsvd_maxiter = 150; hals_maxiter = 120; compnmf_maxiter = 800

# for testing
# lcsvd_maxiter = 1; sca_inner_maxiter = 2; sca_ls_maxiter = 2
# hals_maxiter = 1; maskth=0.25
# compnmf_maxiter = 1; admm_inner_maxiter = 0; admm_ls_maxiter = 0

dataset = :fakecells; inhibitindices=0; bias=0.1; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; maskth=0.25
makepositive = true; tol = -1.0;

imgsz0 = (40,20); iter=1
for iter in 1:num_experiments
    @show iter; flush(stdout)
    sqfactor = Int(floor(sqrt(factor)))
    imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncells, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = LCSVD.noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)
# penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; α = 0.005; β = 5.0
# useRelaxedL1=true; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking;
# inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter; poweradjust = :none

mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased

# Result demonstration parameters
usedenoiseW0H0 = true
try
for (tailstr,initmethod,α,β) in [("_sp_nn",:isvd,0.005,5.0),("_nn",:nndsvd,0.,5.0), ("_sp",:isvd,0.005,0.)]# 
    dd = Dict(); tol=-1
    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    # rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    # stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    #     regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    #     useprecond=useprecond, uselv=uselv, maskW=maskW, maskH=maskH)
    # lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    # cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    #     x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    #     store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    # Mw, Mh = copy(Mw0), copy(Mh0);
    # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
    #                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
    σ0=s*std(W0) #=10*std(W0)=#
    r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
          # if this is too big iteration number would be increased
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=usedenoiseW0H0,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    M, N = copy(M0), copy(N0)
    rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
    alg.store_trace = false; alg.store_inner_trace = false
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
    W, H = rst0.W, rst0.H
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W, H; clamp=false)
    normalizeW!(W,H); W3,H3 = sortWHslices(W,H)
    fprex = "$(prefix)$(SNR)db$(factor)$(initmethod)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(rst.niters)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false, verbose=false)
    
    f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niter); totalniters = sum(niters)
    avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
    avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
    for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
        isempty(afs) && continue
        append!(avgfits,afs); append!(inner_fxs,fxs)
        if iter == 1
            rt2i = 0.
        else
            rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
        end
        append!(rt2s,rt2i)
    end

    dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter
        metadata["alpha"] = α; metadata["beta"] = β; metadata["initmethod"] = initmethod
    end
    save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
end
catch e
    fprex = "$(prefix)$(SNR)db$(factor)"
    save(joinpath(subworkpath,prefix,"$(fprex)_error_$(iter).jld2"),datadic)
    @warn e
    iter -= 1
end

# HALS
prefix = "hals"
@show prefix; flush(stdout)
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1
for (tailstr,initmethod,α) in [("_nn",:nndsvd,0.),("_sp_nn",:nndsvd,hals_α)]#
    dd = Dict(); tol=-1
    W1, H1 = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
    result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=true), X, W1, H1; gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
    W1, H1 = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, W1, H1)
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    fprex = "$(prefix)$(SNR)db$(factor)$(initmethod)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false, verbose=false)
    rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
    if true#iter == num_experiments
        metadata = Dict()
        metadata["maxiter"] = maxiter; metadata["alpha"] = α
    end
    save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end

# COMPNMF
prefix = "compnmf"
@show prefix; flush(stdout)
mfmethod = :COMPNMF; maxiter = compnmf_maxiter
for (tailstr,initmethod) in [("_nn",:lowrank_nndsvd)]
    dd = Dict(); tol=-1
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    # rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    # stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    #     regSpar=regSpar, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc,
    #     uselv=uselv, maskW=maskW, maskH=maskH)
    # cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    #     x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
    #     store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    # Mw, Mh = copy(Mw0), copy(Mh0);
    # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
    #                                                     penmetric=penmetric, stparams=stparams, cparams=cparams);
    rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn;
                        gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn;
                        gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
    rt1 += rst0.inittime # add calculation time for compression matrices L and R
    rt2 -= rst0.inittime
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcn, Hcn; clamp=false)
    normalizeW!(Wcn,Hcn); W3,H3 = sortWHslices(Wcn,Hcn)
    fprex = "$(prefix)$(SNR)db$(factor)$(initmethod)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false, verbose=false)

    rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
    if true#iter == num_experiments
        metadata = Dict()
        metadata["maxiter"] = maxiter
    end
    save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end
end # for iter
