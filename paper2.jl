using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))

using ProfileView, BenchmarkTools

dataset = :fakecells; SNR=0; inhibitidx=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitidx)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitidx=inhibitidx, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells); dataset == :fakecells && (gtW = datadic["gtW"]; gtH = datadic["gtH"])
X = noisefilter(filter,X)
save_gtH = false; save_gtH && (close("all"); plot(gtH); savefig("gtH.png"))

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitidx]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# SCA

mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 100; β = 1000; usennc=false
useRelaxedL1=true; useRelaxedNN=true; s=10*0.3^0; σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter=10; inner_maxiter = 50; ls_maxiter = 100
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = (reg == :WH1 && sd_group != :whole) ? :balance : :none

rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#
α1=α2=100; β1=β2=0

stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    reg=reg, useRelaxedL1=useRelaxedL1, useRelaxedNN=useRelaxedNN, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
for maxiter = 1:30
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                    penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    makepositive && flip2makepos!(W3,H3)
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W3, H3; clamp=false)
    imsave_data(dataset,joinpath(workpath,"paper","avgfit_measure","sca_it$(maxiter)_af$(avgfit)_rt$(rt2)"),W3,H3,imgsz,100; saveH=false)
end

# HALS
@show "HALS"
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
tol=-1
for maxiter in 2:2:100
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=0.1, l₁ratio=1,
                    tol=tol, verbose=false), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd)
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
    imsaveW(joinpath(workpath,"paper","avgfit_measure","hals_sp_nn_it$(maxiter)_af$(avgfit)_rt$(rt2).png"),Wcd,imgsz)
end




# IPNewton test
rt1sca = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
Mw0[:,1] .*= -1; Mh0[1,:] .*= -1
Mw0[1,2:end] .= -5*diag(Mw0)[2:end]; W = W0*Mw0; W[W.<0]
Mh0[2:end,1] .= -6*diag(Mh0)[2:end]; H = Mh0*H0; H[H.<0]
normalizeW!(W,H)
imsaveW("paper2.png",W,imgsz)
