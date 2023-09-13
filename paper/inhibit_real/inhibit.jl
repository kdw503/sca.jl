using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)
subworkpath = joinpath(workpath,"paper","inhibit_real")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))
using GLMakie

dataset = :inhibit_real; SNR=0; inhibitindices=1; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

sca_maxiter = 400; sca_inner_maxiter = 50; sca_ls_maxiter = 100
admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
hals_maxiter = 200

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
cls = distinguishable_colors(ncells; lchoices=range(0, stop=50, length=15))

inhibitloc = datadic["inhibited_loc"]
inhibit_roi = (inhibitloc[1]-25:inhibitloc[1]+5,inhibitloc[2]-20:inhibitloc[2]+20)
imginhibit = reshape(X,imgsz...,lengthT)[inhibit_roi...,:]
imgsz = size(imginhibit)[1:end-1]; X = reshape(imginhibit,prod(imgsz),lengthT); ncells=50
(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

if savemp4
    encoder_options = (crf=23, preset="medium")
    clamp_level=1.0
    X_max = maximum(abs,X)*clamp_level; Xnor = X./X_max;  X_clamped = clamp.(Xnor,0.,1.)
    Xuint8 = UInt8.(round.(map(clamp01nan, X_clamped)*255)) # Frame dims must be a multiple of two
    imgodd = reshape.(eachcol(Xuint8),imgsz...); imgeven = map(frame->frame[1:end-1,1:end-1], imgodd)
    VideoIO.save(joinpath(subworkpath,"$(dataset).mp4"), imgeven, framerate=30, encoder_options=encoder_options) # compatible with ppt (best)
end

icroi = (11:17,18:24) # (24:28,20:24), (11:17,18:24)
h = Float64[]
for x in eachcol(X)
    push!(h,sum(reshape(x,imgsz...)[icroi...]))
end

for subtract_bg in [false, true]
sbgstr = subtract_bg ? "sbg" : "nosbg"

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,joinpath(subworkpath,"Wr1"),Wcd,Hcd,imgsz,100; gridcols=1, signedcolors=dgwm(), saveH=false)
    series(Hcd); save(joinpath(subworkpath,"Hr1.png"),current_figure())
    # bg = median_background(TiledFactorizations.accumfloat(T),X);
    # normalizeW!(bg.S,bg.T); imsave_data(dataset,"Wr1",reshape(bg.S,length(bg.S),1),bg.T,imgsz,100; signedcolors=dgwm(), saveH=false)
    # series(bg.T'); save("Hr2.png",current_figure())
    Hcd31 = mapwindow(mean, Hcd, (1,31))
    # sigma = std(Hcd)
    # Hclamp = map((a,b) -> (a>b+2sigma) || (a<b-2sigma) ? b : a, Hcd,Hcd31); Hcd31 = mapwindow(mean, Hclamp, (1,31))
    # Hclamp = map((a,b) -> (a>b+2sigma) || (a<b-2sigma) ? b : a, Hcd,Hcd31); Hcd31 = mapwindow(mean, Hclamp, (1,31))
    # Hclamp = map((a,b) -> (a>b+2sigma) || (a<b-2sigma) ? b : a, Hcd,Hcd31); Hcd31 = mapwindow(mean, Hclamp, (1,31))
    f=Figure(resolution = (900,400)); ax=GLMakie.Axis(f[1,1],title="H component",xlabel="time index")
    lines!(ax,Hcd[1,:],label="Hbg"); lines!(ax,Hcd31[1,:],label="Hbg_LPF")
    axislegend(ax, position = :rb)
    save(joinpath(subworkpath,"Hr1_LPF.png"),current_figure())

    hbefore = Float64[]
    for x in eachcol(X)
        push!(hbefore,sum(reshape(x,imgsz...)[icroi...]))
    end
    bg = Wcd*Hcd31; X .-= bg
    # plot sbged H
    h = Float64[]
    for x in eachcol(X)
        push!(h,sum(reshape(x,imgsz...)[icroi...]))
    end
    f=Figure(resolution = (900,400))
    axbefore=GLMakie.Axis(f[1,1],title="Intensity of the inhibited cell")
    axafter=GLMakie.Axis(f[2,1],title="Intensity of the inhibited cell after background subtration" ,xlabel="time index"); linkxaxes!(axbefore, axafter)
    lines!(axbefore,hbefore,label="Hbg"); lines!(axafter,h,label="Hbg"); axislegend(ax, position = :rb)
    save(joinpath(subworkpath,"Hinhibit_sbg.png"),current_figure())
end
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# SCA
method="sca"; @show method
mfmethod = :SCA; initmethod=:isvd; penmetric = :SCA; sd_group=:whole; regSpar = :WH1H2; regNN = :WH2M2
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
σ0=s*std(W0) #=10*std(W0)=#
β1=0.1; β2=0; α1=10; α2=0
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed Wsca, Hsca, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
normalizeW!(Wsca,Hsca)
makepositive && flip2makepos!(Wsca,Hsca)
Wsca, Hsca = sortWHslices(Wsca,Hsca)
#imshowW(Wsca,imgsz,gridcols=10)
fprx = "$(method)$(datastr)_$(sbgstr)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(maxiter)_rt$(rt2)"
imsaveW(joinpath(subworkpath,fprx)*".png",Wsca,imgsz,gridcols=10)
# imsave_data(dataset,joinpath(subworkpath,fprx),Wsca,Hsca,imgsz,100; saveH=false)
# series([gtH[:,inhibitindices],Hsca[inhibitindices,:]]; color=cls, labels=["Ground Truth H","Estimated H"]); axislegend(position = :rt)
# save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())

# ADMM
method="admm"; @show method
mfmethod = :ADMM; initmethod=:lowrank_nndsvd; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 0; β = 0; usennc=true
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
    regSpar=regSpar, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
    store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed  Wadmm, Hadmm, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                    penmetric=penmetric, stparams=stparams, cparams=cparams);
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
normalizeW!(Wadmm,Hadmm)
makepositive && flip2makepos!(Wadmm,Hadmm)
#imshowW(Wadmm,imgsz,gridcols=5)
fprx = "$(method)$(datastr)_$(sbgstr)_a$(α)_it$(maxiter)_rt$(rt2)"
imsaveW(joinpath(subworkpath,fprx)*".png",Wadmm,imgsz,gridcols=10)
#imsave_data(dataset,joinpath(subworkpath,fprx),Wadmm,Hadmm,imgsz,100; saveH=false)
# series([gtH[:,inhibitindices],Hadmm[inhibitindices,:]]; color=cls); save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())

# HALS
method="hals"; @show method
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1; α=0
Whals, Hhals = copy(Wcd0), copy(Hcd0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=α, l₁ratio=1,
                tol=tol, verbose=false), X, Whals, Hhals)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
normalizeW!(Whals,Hhals)
#imshowW(Whals,imgsz,gridcols=10)
fprx = "$(method)$(datastr)_$(sbgstr)_a$(α)_it$(maxiter)_rt$(rt2)"
imsaveW(joinpath(subworkpath,fprx)*".png",Whals,imgsz,gridcols=10)
# imsave_data(dataset,joinpath(subworkpath,fprx),Whals,Hhals,imgsz,100; saveH=false)
# series([gtH[:,inhibitindices],Hhals[inhibitindices,:]]; color=cls); save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())

dtcolors = distinguishable_colors(5; lchoices=range(0, stop=50, length=15))
f=Figure(resolution = (900,400))
axsca=GLMakie.Axis(f[1,1],title="H components of the inhibited cell (SMF)")
axhals=GLMakie.Axis(f[2,1],title="H components of the inhibited cell (HALS)" ,xlabel="time index"); linkxaxes!(axbefore, axafter)
icidxscas = subtract_bg ? [2,4,14] : [21,42,48]
icidxhalss = subtract_bg ? [3,5,30,35] : [3,9,35]
colors = [3,2,4,5]
linsca = [lines!(axsca,Hsca[icidx,:], label="Hsmf[$(icidx),:]", color=dtcolors[colors[i]]) for (i,icidx) in enumerate(icidxscas)]
labelsca = ["Hsmf[$(icidx),:]" for (i,icidx) in enumerate(icidxscas)]
Legend(f[1,2],linsca,labelsca) # axislegend(axsca, position = :lt)
linhals = [lines!(axhals,Hhals[icidx,:], label="Hhals[$(icidx),:]", color=dtcolors[colors[i]]) for (i,icidx) in enumerate(icidxhalss)]
labelhals = ["Hhals[$(icidx),:]" for (i,icidx) in enumerate(icidxhalss)]
Legend(f[2,2],linhals,labelhals) # axislegend(axhals, position = :lt)
save(joinpath(subworkpath,"SMF_and_Hals_inhibitH_$(sbgstr).png"),current_figure())

# Figure
mtdcolors = [RGBA{N0f8}(0.00,0.00,0.00,1.0),RGBA{N0f8}(0.00,0.45,0.70,1.0),RGBA{N0f8}(0.90,0.62,0.00,1.0),
             RGBA{N0f8}(0.00,0.62,0.45,1.0),RGBA{N0f8}(0.80,0.47,0.65,1.0),RGBA{N0f8}(0.34,0.71,0.91,1.0),
             RGBA{N0f8}(0.84,0.37,0.00,1.0),RGBA{N0f8}(0.94,0.89,0.26,1.0)]
# Input data
imggt = mkimgW(gtW,imgsz)
hdata = eachcol(gtH)
labels = ["cell $i" for i in 1:length(hdata)]
f = Figure(resolution = (900,400))
ax11=GLMakie.Axis(f[1,1],title="W component", aspect = DataAspect()); hidedecorations!(ax11)
axall2=GLMakie.Axis(f[:,2],title="H component",xlabel="time index")
image!(ax11, rotr90(imggt))
lin = [lines!(axall2,hd,color=dtcolors[i]) for (i,hd) in enumerate(hdata)]
labels[1] *= " (inhibited)"
f[:,3] = Legend(f[:,2],lin,labels)
save(joinpath(subworkpath,"idx$(inhibitindices)_bias$(bias)_gt.png"),f)

imggt = mkimgW(gtW,imgsz); imgsca = mkimgW(Wsca,imgsz); imgadmm = mkimgW(Wadmm,imgsz); imghals = mkimgW(Whals,imgsz)
# scainhibitindices = (bias == 0.5) && (subtract_bg == false) ? 8 : inhibitindices
hdata = [gtH[:,inhibitindices],Hsca[inhibitindices,:],Hadmm[inhibitindices,:],Hhals[inhibitindices,:]] # Hsca inhibit index setting for plot
labels = ["Ground Truth","SMF","Compressed NMF","HALS NMF"]
f = Figure(resolution = (1000,400))
ax11=GLMakie.Axis(f[1,1],title=labels[1], aspect = DataAspect()); hidedecorations!(ax11)
ax21=GLMakie.Axis(f[2,1],title=labels[2], aspect = DataAspect()); hidedecorations!(ax21)
ax31=GLMakie.Axis(f[3,1],title=labels[3], aspect = DataAspect()); hidedecorations!(ax31)
ax41=GLMakie.Axis(f[4,1],title=labels[4], aspect = DataAspect()); hidedecorations!(ax41)
axall2=GLMakie.Axis(f[:,2],title="Inhibited H component",xlabel="time index")
image!(ax11, rotr90(imggt)); image!(ax21, rotr90(imgsca)); image!(ax31, rotr90(imgadmm)); image!(ax41, rotr90(imghals))
lin = [lines!(axall2,hd,color=mtdcolors[i]) for (i,hd) in enumerate(hdata)]
f[:,3] = Legend(f[:,2],lin,labels)
save(joinpath(subworkpath,"idx$(inhibitindices)_bias$(bias)_$(sbgstr).png"),f)

end # subtract_bg

