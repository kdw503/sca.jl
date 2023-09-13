using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)
subworkpath = joinpath(workpath,"paper","audio")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))
using GLMakie

dataset = :audio; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

sca_maxiter = 100; sca_inner_maxiter = 50; sca_ls_maxiter = 100
admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
hals_maxiter = 100

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
cls = distinguishable_colors(ncells; lchoices=range(0, stop=50, length=15))

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

subtract_bg=false; sbgstr = subtract_bg ? "sbg" : "nosbg"

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    series(Hcd); save(joinpath(subworkpath,"Hr1.png"),current_figure())
    series(gtH[:,inhibitindices]'); save(joinpath(subworkpath,"Hr1_gtH.png"),current_figure())
    # bg = fit_background(X);
    # normalizeW!(bg.S,bg.T); imsave_data(dataset,"Wr1",reshape(bg.S,length(bg.S),1),bg.T,imgsz,100; signedcolors=dgwm(), saveH=false)
    # series(bg.T'); save("Hr2.png",current_figure())
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# SCA
method="sca"; @show method
mfmethod = :SCA; initmethod=:nndsvd; penmetric = :SCA; sd_group=:whole; regSpar=:WH1M2; regNN=:WH2M2; α = 100; β = 0
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
σ0=s*std(W0) #=10*std(W0)=#
β1=β2=β; α1=α2=α
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=sca_maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
fitval = SCA.fitd(X,W1*H1)
makepositive && flip2makepos!(W1,H1)
normalizeW!(W1,H1); W1 .*= 10; H1 ./=10; W3,H3 = sortWHslices(W1,H1)
#    normalizeW!(W1,H1)
fprx = "$(mfmethod)$(dataset)_$(initmethod))$(reg)_aw$(α1)_ah$(α2)_b$(β)_it$(sca_maxiter)_fv$(fitval)_rt$(rt2)"
fig = plotWH_data(dataset,joinpath(subworkpath,fprx),W3,H3; space=10, issave=true)

# series([gtH[:,inhibitindices],Hsca[inhibitindices,:]]; color=cls, labels=["Ground Truth H","Estimated H"]); axislegend(position = :rt)
# save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())

# ADMM
method="admm"; @show method
mfmethod = :ADMM; initmethod=:lowrank_nndsvd; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 10; β = 0; usennc=true
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = :none
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
    α1=α2=α; β1=β2=β
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=admm_maxiter, inner_maxiter=inner_maxiter,
    store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed  W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                    penmetric=penmetric, stparams=stparams, cparams=cparams);
fitval = SCA.fitd(X,W1*H1)
makepositive && flip2makepos!(W1,H1)
normalizeW!(W1,H1); W1 .*= 10; H1 ./=10; W3,H3 = sortWHslices(W1,H1)
fprx = "$(mfmethod)$(dataset)_$(initmethod)_a$(α)_it$(admm_maxiter)_fv$(fitval)_rt$(rt2)"
fig = plotWH_data(dataset,joinpath(subworkpath,fprx),W3,H3; space=10, issave=true)


# HALS
method="hals"; @show method
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
initmethod = :svd
if initmethod == :rsvd
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
else
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar, initdata=svd(X));
end
mfmethod = :HALS;  α=0.1; tol=-1

Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=hals_maxiter, α=α, l₁ratio=1,
            tol=tol, verbose=false), X, Wcd, Hcd)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
fitval = SCA.fitd(X,Wcd*Hcd)
normalizeW!(Wcd,Hcd); Wcd .*= 10; Hcd ./=10; W3,H3 = sortWHslices(Wcd,Hcd)
fprx = "$(mfmethod)$(dataset)_$(initmethod)_a$(α)_it$(hals_maxiter)_fv$(fitval)_rt$(rt2)"
fig = plotWH_data(dataset,joinpath(subworkpath,fprx),W3,H3; space=10, issave=true)



# Figure
imgsca1 = load(joinpath(subworkpath,"SCAaudio_isvd_a100_b0_it100_fv0.9922997487378895_rt0.0207727_plot_WH.png"))
imgsca2 = load(joinpath(subworkpath,"SCAaudio_nndsvd_a100_b0_it100_fv0.992302949908781_rt0.0268503_plot_WH.png"))
imgadmm1 = load(joinpath(subworkpath,"ADMMaudio_lowrank_nndsvd_a10_it1500_fv0.8071448761549662_rt0.0813385_plot_WH.png"))
imgadmm2 = load(joinpath(subworkpath,"ADMMaudio_lowrank_nndsvd_a10_it1500_fv0.9882295945463136_rt0.0619883_plot_WH.png"))
imghals = load(joinpath(subworkpath,"HALSaudio_rsvd_a0.1_it100_fv0.9921731304672196_rt0.0153537_plot_WH.png"))

f = Figure(resolution = (1000,900))
ax11=GLMakie.Axis(f[1,1],title="(a) SMF (ISVD init.)", titlesize=20, aspect = DataAspect()); hidespines!(ax11)
hidedecorations!(ax11)
ax12=GLMakie.Axis(f[1,2],title="(b) SMF (NNDSVD init.)", titlesize=20, aspect = DataAspect()); hidespines!(ax12)
hidedecorations!(ax12)
ax21=GLMakie.Axis(f[2,1],title="(d) Compressed NMF", titlesize=20, aspect = DataAspect()); hidespines!(ax21)
hidedecorations!(ax21)
ax22=GLMakie.Axis(f[2,2],title="(e) Compressed NMF", titlesize=20, aspect = DataAspect()); hidespines!(ax22)
hidedecorations!(ax22)
ax3all=GLMakie.Axis(f[3,1:2],title="(c) HALS", titlesize=20, aspect = DataAspect()); hidespines!(ax3all)
hidedecorations!(ax13)
image!(ax11, rotr90(imgsca1)); image!(ax12, rotr90(imgsca2)); image!(ax3all, rotr90(imghals))
image!(ax21, rotr90(imgadmm1)); image!(ax22, rotr90(imgadmm2))
save(joinpath(subworkpath,"audio.png"),f)
