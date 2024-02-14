using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","audio")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))

dataset = :audio; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

lcsvd_maxiter = 100; lcsvd_inner_maxiter = 50; lcsvd_ls_maxiter = 100
compnmf_maxiter = 1500; compnmf_inner_maxiter = 0; compnmf_ls_maxiter = 0
hals_maxiter = 100; tol=-1

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
cls = distinguishable_colors(ncells; lchoices=range(0, stop=50, length=15))

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
# X = noisefilter(filter,X)

subtract_bg=false; sbgstr = subtract_bg ? "sbg" : "nosbg"

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    series(Hcd); save(joinpath(subworkpath,"Hr1.png"),current_figure())
    series(gtH[:,inhibitindices]'); save(joinpath(subworkpath,"Hr1_gtH.png"),current_figure())
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# LCSVD

# method="sca"; @show method
# mfmethod = :SCA; initmethod=:nndsvd; penmetric = :SCA; sd_group=:whole; regSpar=:WH1M2; regNN=:WH2M2; α = 100; β = 0
# useRelaxedL1=true; s=10*0.3^0; 
# r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
#       # if this is too big iteration number would be increased
# # Optimization parameters
# tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
# maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# # Result demonstration parameters
# makepositive = true; poweradjust = :none
# σ0=s*std(W0) #=10*std(W0)=#
# β1=β2=β; α1=α2=α
# rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
# stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
#     regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
#     useprecond=useprecond, uselv=uselv)
# lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
# cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
#     x_abstol=tol, successive_f_converge=0, maxiter=sca_maxiter, inner_maxiter=inner_maxiter, store_trace=true,
#     store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
# Mw, Mh = copy(Mw0), copy(Mh0);
# cparams.store_trace = false; cparams.store_inner_trace = false;
# cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
# rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
#                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
# fitval = SCA.fitd(X,W1*H1)
# makepositive && flip2makepos!(W1,H1)
# normalizeW!(W1,H1); W1 .*= 10; H1 ./=10; W3,H3 = sortWHslices(W1,H1)
# #    normalizeW!(W1,H1)
# fprx = "$(mfmethod)$(dataset)_$(initmethod))$(reg)_aw$(α1)_ah$(α2)_b$(β)_it$(sca_maxiter)_fv$(fitval)_rt$(rt2)"
# fig = plotWH_data(dataset,joinpath(subworkpath,fprx),W3,H3; space=10, issave=true)
## series([gtH[:,inhibitindices],Hsca[inhibitindices,:]]; color=cls, labels=["Ground Truth H","Estimated H"]); axislegend(position = :rt)
## save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())
uselv=false; s=10; maxiter = lcsvd_maxiter
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
makepositive = true
α1=α2=α=0.005    #αrng[iter]; 
for (prefix,tailstr,initmethod,β) = [("lcsvd_precon","_sp",:isvd,0.),("lcsvd_precon","_sp",:nndsvd,0.),
                                    ("lcsvd","_sp_nn",:isvd,5.0),("lcsvd","_sp_nn",:nndsvd,5.0)]
    @show prefix
    useprecond = prefix ∈ ["lcsvd_precon","lcsvd_precon_LPF"] ? true : false
    β1=β2=β
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod,
                                    svdmethod=:isvd) #svdmethod=:isvd doesn't work for initmethod=:nndsvd
    W0 = W0[:,1:ncells]; H0 = H0[1:ncells,:]; M0 = M0[1:ncells,1:ncells]; N0 = N0[1:ncells,1:ncells]; D = D[1:ncells,1:ncells]
    σ0=s*std(W0) #=10*std(W0)=#
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond, usedenoiseW0H0=false,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    # M, N = copy(M0), copy(N0)
    # rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
    alg.store_trace = false; alg.store_inner_trace = false
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
    Wlc, Hlc = rst0.W, rst0.H
    fitval = LCSVD.fitd(X,Wlc*Hlc)
    LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
    makepositive && LCSVD.flip2makepos!(Wlc,Hlc)
    Wlc .*= 10; Hlc ./=10; W3,H3 = LCSVD.sortWHslices(Wlc,Hlc)
    fprex = "$(prefix)$(tailstr)_$(initmethod)"
    fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_fv$(fitval)_it$(rst0.niters)_rt$(rt2)")
    fig = TestData.plotWH_data(dataset,fname,W3,H3; space=10, issave=true)
    fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_fv$(fitval)_it$(rst0.niters)_rt$(rt2)_perm")
    fig = TestData.plotWH_data(dataset,fname,W3[:,[2,1,3]],H3[[2,1,3],:]; space=10, issave=true)
end


# CompNMF

# method="admm"; @show method
# mfmethod = :ADMM; initmethod=:lowrank_nndsvd; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 10; β = 0; usennc=true
# useRelaxedL1=true; s=10*0.3^0; 
# r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
#       # if this is too big iteration number would be increased
# # Optimization parameters
# tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
# maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# # Result demonstration parameters
# makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
# poweradjust = :none
# rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
# stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
#     reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
#     α1=α2=α; β1=β2=β
# cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
#     x_abstol=tol, successive_f_converge=0, maxiter=admm_maxiter, inner_maxiter=inner_maxiter,
#     store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
# Mw, Mh = copy(Mw0), copy(Mh0);
# cparams.store_trace = false; cparams.store_inner_trace = false;
# cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
# rt2 = @elapsed  W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
#                                                     penmetric=penmetric, stparams=stparams, cparams=cparams);
# fitval = SCA.fitd(X,W1*H1)
# makepositive && flip2makepos!(W1,H1)
# normalizeW!(W1,H1); W1 .*= 10; H1 ./=10; W3,H3 = sortWHslices(W1,H1)
# fprx = "$(mfmethod)$(dataset)_$(initmethod)_a$(α)_it$(admm_maxiter)_fv$(fitval)_rt$(rt2)"
# fig = plotWH_data(dataset,joinpath(subworkpath,fprx),W3,H3; space=10, issave=true)

prefix="compnmf"; @show prefix
maxiter = compnmf_maxiter; initmethod=:nndsvd
for iter in 1:100
    @show iter
    rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);

    # Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    # result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn;
    #                     gtU=gtW, gtV=gtH, maskU=:, maskV=:)
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
    rt1 += rst0.inittime # add calculation time for compression matrices L and R
    rt2 -= rst0.inittime
    fitval = LCSVD.fitd(X,Wcn*Hcn)
    LCSVD.normalizeW!(Wcn,Hcn)
    Wcn .*= 10; Hcn ./=10; W3,H3 = LCSVD.sortWHslices(Wcn,Hcn)
    fprex = "$(prefix)_$(initmethod)"
    fname = joinpath(subworkpath,"$(fprex)_fv$(fitval)_it$(maxiter)_rt$(rt2)")
    fig = TestData.plotWH_data(dataset,fname,W3,H3; space=10, issave=true)
end

# HALS
prefix="hals"; @show prefix
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
initmethod = :rsvd
if initmethod == :rsvd
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
else
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar, initdata=svd(X));
end

for α in [0, 0.1]
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=hals_maxiter, α=α, l₁ratio=1,
                tol=tol, verbose=false), X, Wcd, Hcd)
    avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
    fitval = LCSVD.fitd(X,Wcd*Hcd)
    LCSVD.normalizeW!(Wcd,Hcd); Wcd .*= 10; Hcd ./=10; W3,H3 = LCSVD.sortWHslices(Wcd,Hcd)
    fprex = "$(prefix)_$(initmethod)"
    fname = joinpath(subworkpath,"$(fprex)_a$(α)_fv$(fitval)_it$(maxiter)_rt$(rt2)")
    fig = TestData.plotWH_data(dataset,fname,W3,H3; space=10, issave=true)
end

prefix="hals"; @show prefix
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
maxiter = hals_maxiter
hals_αrng = range(0,0.5,num_experiments); msess=[]

α = 0
Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
            tol=tol, verbose=false), X, Wcd, Hcd)
fitval = LCSVD.fitd(X,Wcd*Hcd)
LCSVD.normalizeW!(Wcd,Hcd)
fprex = "$(prefix)"
fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_fv$(fitval)_it$(maxiter)_rt$(rt2)")




# Figure
imglcsvd1 = load(joinpath(subworkpath,"audio_lcsvd_sp_isvd_a0.005_b0.0.png"))
imglcsvd2 = load(joinpath(subworkpath,"audio_lcsvd_sp_nn_isvd_a0.005_b5.0.png"))
imgadmm = load(joinpath(subworkpath,"audio_compnmf_nndsvd.png"))
imghals = load(joinpath(subworkpath,"audio_hals_rsvd_a0.1.png"))

fontsize = 30
f = Figure(resolution = (2000,1100))
ax11=AMakie.Axis(f[1,1],title="(a) LCSVD (α=0.005 β=0)", titlesize=fontsize, aspect = DataAspect()); hidespines!(ax11)
hidedecorations!(ax11)
ax12=AMakie.Axis(f[1,2],title="(b) LCSVD (α=0.005 β=5.0)", titlesize=fontsize, aspect = DataAspect()); hidespines!(ax12)
hidedecorations!(ax12)
ax21=AMakie.Axis(f[2,1],title="(c) Compressed NMF", titlesize=fontsize, aspect = DataAspect()); hidespines!(ax21)
hidedecorations!(ax21)
ax22=AMakie.Axis(f[2,2],title="(d) HALS NMF (α=0.1)", titlesize=fontsize, aspect = DataAspect()); hidespines!(ax22)
hidedecorations!(ax22)
image!(ax11, rotr90(imglcsvd1)); image!(ax12, rotr90(imglcsvd2))
image!(ax21, rotr90(imgadmm)); image!(ax22, rotr90(imghals))

save(joinpath(subworkpath,"audio_all_figures.png"),f)
