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
using CompNMF

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
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :COMPNMF; maxiter = 400; tol=-1
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
tailstr = "_nn"
dd = Dict()
# Wcd, Hcd = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
# result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
#                 tol=tol, verbose=true), X, Wcd, Hcd; W0=W0, H0=H0, d=diag(D), gtW=gtW, gtH=gtH)
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
rt2 = @elapsed CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn;
                    gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcn, Hcn; clamp=false)
normalizeW!(Wcn,Hcn); W3,H3 = sortWHslices(Wcn,Hcn)
fprex = "halse$(factor)"
fname = joinpath(workpath,"$(fprex)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
#imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false)





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
