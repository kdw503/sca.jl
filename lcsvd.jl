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
using LCSVD

dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 2
    sca_maxiter = 100; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 400; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 200; maskth=0.25
else
    num_experiments = 2
    sca_maxiter = 4; sca_inner_maxiter = 2; sca_ls_maxiter = 2
    admm_maxiter = 4; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 4; maskth=0.25
end

factor=1; fovsz = (40,20)
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
tailstr,initmethod,α,β = ("_sp",:nndsvd,0.005,0.) # ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:nndsvd,0.005,5.0)
α1=α2=α; β1=β2=β
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=:nndsvd, svdmethod=:isvd)
σ0=10*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
mfmethod = :LCSVD; maxiter = 100; tol=1e-7
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

# LCSVD
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, uselv=false, maskW=maskW, maskH=maskH,
    maxiter = maxiter, store_trace = false, store_inner_trace = false, show_trace = true, allow_f_increases = true,
    f_abstol = tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, successive_f_converge=0)
W, H = copy(Wp), copy(Hp); M, N = copy(M0), copy(N0)
rt2 = @elapsed rst = LCSVD.solve!(alg, X, W0, H0, D, M, N, W, H; gtW=gtW, gtH=gtH);
# 0.391060 seconds (197.17 k allocations: 135.467 MiB)

avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W, H; clamp=false)
normalizeW!(W,H); W3,H3 = sortWHslices(W,H)
fprex = "lcsvd$(factor)"
fname = joinpath(workpath,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(rst.niters)_rt$(rt2)")
#imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false)

# SCA
stparams = StepParams(sd_group=:whole, optimmethod=:optim_lbfgs, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=:WH1M2, regNN=:WH2M2, useRelaxedL1=true, σ0=σ0, r=r, poweradjust=:none,
    useprecond=false, uselv=false, maskW=maskW, maskH=maskH)
lsparams = LineSearchParams(method=:ls_BackTracking, α0=1.0, c_1=1e-4, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=50, store_trace=false,
    store_inner_trace=false, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
#@show rt2
W, H = copy(Wp), copy(Hp); Mw, Mh = copy(M0), copy(N0)
rt2 = @elapsed W1, H1, objvals, laps, trs, niter = scasolve!(X, W0, H0, D, Mw, Mh, W, H; gtW=gtW, gtH=gtH,
                             penmetric=:SCA, stparams=stparams, lsparams=lsparams, cparams=cparams);
# 0.817546 seconds (613.23 k allocations: 157.871 MiB, 4.65% gc time, 46.20% compilation time)                             
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
false && flip2makepos!(W3,H3)
fprex = "sca$(factor)"
fname = joinpath(workpath,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(niter)_rt$(rt2)")
#imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)




T=Float64; tol=1e-4; maxiter = 4; inner_maxiter=0; uselv=false
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
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
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp0, Hp0, D = initsemisca(X, ncells, initmethod=:lowrank_nndsvd,poweradjust=initpwradj)

stparams = StepParams(sd_group=:whole, optimmethod=:sca_admm, approx=true, α1=0, α2=0, β1=0, β2=0,
        regSpar=:WH1M2, useRelaxedL1=false, σ0=3.0, r=0.5, poweradjust=:none, useprecond=false, usennc=true,
        uselv=uselv, maskW=Colon(), maskH=Colon())
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
        store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
Mw, Mh = copy(Mw0), copy(Mh0); Wp, Hp = copy(Wp0), copy(Hp0)
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                    penmetric=:SCA, stparams=stparams, cparams=cparams);


L = W0; R = H0; X_tilde=D; W02=nothing; H02=nothing; w0tw0s=nothing; h0h0ts=nothing
isadmm=true; trace = Trace(T); outeriter=100; uselv=false
Mw = copy(Mw0); Mh = copy(Mh0); Wp, Hp = copy(Wp0), copy(Hp0)
state = SCA.initial_state(W0,H0,D,Mw,Mh,Wp,Hp,isadmm,stparams=stparams) 
SCA.minMwMh_whole_admm!(L,R,W02,H02,w0tw0s,h0h0ts,X_tilde,state,trace, outeriter;
        uselv=uselv, stparams=stparams)

Wp, Hp = copy(Wp0), copy(Hp0)
alg = CompNMF.CompressedNMF{T}(maxiter=maxiter, tol=tol, verbose=false)
updater = CompNMF.CompressedNMFUpd{T}(alg.xi, alg.lambda, alg.phi, alg.SCA_penmetric, alg.SCA_αw, alg.SCA_αh)
s = CompNMF.prepare_state(updater, X, Wp, Hp; gtW=gtW[maskW,:], gtH=gtH[maskH,:])
CompNMF.update_wh!(updater, s, X, Wp, Hp)

Wcn, Hcn = copy(Wp0), copy(Hp0);
rt2 = @elapsed CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn;
                    gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcn, Hcn; clamp=false)
normalizeW!(Wcn,Hcn); W3,H3 = sortWHslices(Wcn,Hcn)
fprex = "CompNMF"
fname = joinpath(workpath,"$(fprex)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
#imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false)
