using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)
subworkpath = joinpath(workpath,"paper","cbclface")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))
using GLMakie

dataset = :cbclface; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

sca_maxiter = 400; sca_inner_maxiter = 50; sca_ls_maxiter = 100
admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
hals_maxiter = 200

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
mfmethod = :SCA; initmethod=:isvd; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN = :WH2M2; α = 100; β = 0
useRelaxedL1=true; useRelaxedNN=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
σ0=s*std(W0) #=10*std(W0)=#
β1=β2=β; α1 = α2 = α
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
Wscas = []; rtscas=[]; scamaxiterrng = 4:2:20
for scamaxiter in scamaxiterrng
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                    penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    fitval = SCA.fitd(X,W1*H1)
    makepositive && flip2makepos!(W1,H1)
    normalizeW!(W1,H1)
    fprx = "$(method)$(dataset)_a$(α)_b$(β)_it$(scamaxiter)_fv$(fitval)_rt$(rt2)"
    imsave_data(dataset,joinpath(subworkpath,fprx),W1,H1,imgsz,100; saveH=false)
    imsave_reconstruct(joinpath(subworkpath,fprx),X,W1,H1,imgsz; index=100, gridcols=7, clamp_level=1.0)
    push!(Wscas,W1); push!(rtscas,rt2)
end
# series([gtH[:,inhibitindices],Hsca[inhibitindices,:]]; color=cls, labels=["Ground Truth H","Estimated H"]); axislegend(position = :rt)
# save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())

# ADMM
method="admm"; @show method
mfmethod = :ADMM; initmethod=:lowrank_nndsvd; penmetric = :SCA; sd_group=:whole; α = 0; β = 0; usennc=true
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = :none
α1=α2=α; β1=β2=β
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
Wadmms=[]; rtadmms=[]; admmmaxiterrng = 50:50:500
for admmmaxiter = admmmaxiterrng
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=admmmaxiter, inner_maxiter=inner_maxiter,
        store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed  W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                        penmetric=penmetric, stparams=stparams, cparams=cparams);
    fitval = SCA.fitd(X,W1*H1)
    normalizeW!(W1,H1)
    fprx = "$(method)$(dataset)_a$(α)_it$(admmmaxiter)_fv$(fitval)_rt$(rt2)"
    imsave_data(dataset,joinpath(subworkpath,fprx),W1,H1,imgsz,100; saveH=false)
    imsave_reconstruct(joinpath(subworkpath,fprx),X,W1,H1,imgsz; index=100, gridcols=7, clamp_level=1.0)
    push!(Wadmms,W1); push!(rtadmms,rt2)
    # series([gtH[:,inhibitindices],Hadmm[inhibitindices,:]]; color=cls); save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())
end

# HALS
method="hals"; @show method
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter = hals_maxiter; α=0.6; tol=-1
Whalss=[]; rthalss=[]; halsmaxiterrng = 20:20:100
for halsmaxiter = halsmaxiterrng
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=halsmaxiter, α=α, l₁ratio=1,
                tol=tol, verbose=false), X, Wcd, Hcd)
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
    fitval = SCA.fitd(X,Wcd*Hcd)
    normalizeW!(Wcd,Hcd)
    fprx = "$(method)$(dataset)_a$(α)_it$(halsmaxiter)_fv$(fitval)_rt$(rt2)"
    imsave_data(dataset,joinpath(subworkpath,fprx),Wcd,Hcd,imgsz,100; saveH=false)
    imsave_reconstruct(joinpath(subworkpath,fprx),X,Wcd,Hcd,imgsz; index=100, gridcols=7, clamp_level=1.0)
    push!(Whalss,Wcd); push!(rthalss,rt2)
    # series([gtH[:,inhibitindices],Hhals[inhibitindices,:]]; color=cls); save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())
end

# Figure
mtdcolors = [RGBA{N0f8}(0.00,0.00,0.00,1.0),RGBA{N0f8}(0.00,0.45,0.70,1.0),RGBA{N0f8}(0.90,0.62,0.00,1.0),
             RGBA{N0f8}(0.00,0.62,0.45,1.0),RGBA{N0f8}(0.80,0.47,0.65,1.0),RGBA{N0f8}(0.34,0.71,0.91,1.0),
             RGBA{N0f8}(0.84,0.37,0.00,1.0),RGBA{N0f8}(0.94,0.89,0.26,1.0)]
dtcolors = distinguishable_colors(ncells; lchoices=range(0, stop=50, length=15))
# Input data
gridcols=Int(ceil(sqrt(size(Wscas[1],2))))
imgsca1 = mkimgW(Wscas[1],imgsz,gridcols=gridcols); imgsca2 = mkimgW(Wscas[end],imgsz,gridcols=gridcols)
imgadmm1 = mkimgW(Wadmms[1],imgsz,gridcols=gridcols); imgadmm2 = mkimgW(Wadmms[end],imgsz,gridcols=gridcols)
imghals1 = mkimgW(Whalss[1],imgsz,gridcols=gridcols); imghals2 = mkimgW(Whalss[end],imgsz,gridcols=gridcols)
labels = ["SMF","Compressed NMF","HALS NMF"]
f = Figure(resolution = (900,1500))
ax11=GLMakie.Axis(f[1,1],title=labels[1], titlesize=30, subtitle="maxiter=$(scamaxiterrng[1]), runtime=$(round(rtscas[1],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax11)
ax12=GLMakie.Axis(f[1,2],title=labels[1], titlesize=30, subtitle="maxiter=$(scamaxiterrng[end]), runtime=$(round(rtscas[end],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax12)
ax21=GLMakie.Axis(f[2,1],title=labels[2], titlesize=30, subtitle="maxiter=$(admmmaxiterrng[1]), runtime=$(round(rtadmms[1],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax21)
ax22=GLMakie.Axis(f[2,2],title=labels[2], titlesize=30, subtitle="maxiter=$(admmmaxiterrng[end]), runtime=$(round(rtadmms[end],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax22)
ax31=GLMakie.Axis(f[3,1],title=labels[3], titlesize=30, subtitle="maxiter=$(halsmaxiterrng[1]), runtime=$(round(rthalss[1],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax31)
ax32=GLMakie.Axis(f[3,2],title=labels[3], titlesize=30, subtitle="maxiter=$(halsmaxiterrng[end]), runtime=$(round(rthalss[end],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax32)
image!(ax11, rotr90(imgsca1)); image!(ax12, rotr90(imgsca2))
image!(ax21, rotr90(imgadmm1)); image!(ax22, rotr90(imgadmm2))
image!(ax31, rotr90(imghals1)); image!(ax32, rotr90(imghals2))
save(joinpath(subworkpath,"cbclface.png"),f)
