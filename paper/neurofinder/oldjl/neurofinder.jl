if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
include(joinpath(workpath,"setup.jl"))
subworkpath = joinpath(workpath,"paper","neurofinder")

dataset = :neurofinder_small; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:neurofinder_small,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

sca_maxiter = 100; sca_inner_maxiter = 50; sca_ls_maxiter = 100
admm_maxiter = 100; admm_inner_maxiter = 0; admm_ls_maxiter = 0
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
    normalizeW!(Wcd,Hcd); imsave_neurofinder(joinpath(subworkpath,"BG_Wr1"),Wcd,Hcd,imgsz,100; gridcols=1, signedcolors=g1wm(), saveH=false)
    series(Hcd); save(joinpath(subworkpath,"BG_Hr1.png"),current_figure())
    Hcd = mapwindow(mean, Hcd, (1,401))
    series(Hcd); save(joinpath(subworkpath,"BG_Hr1_LPF.png"),current_figure())
    bg = Wcd*Hcd; X .-= bg
end
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# SCA
method="sca"; @show method
mfmethod = :SCA; initmethod=:isvd; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 1; β = 0
useRelaxedL1=true; useRelaxedNN=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
σ0=s*std(W0) #=10*std(W0)=#
β1=β2=β; α1 = 0; α2 = 10 # α
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    reg=reg, useRelaxedL1=useRelaxedL1, useRelaxedNN=useRelaxedNN, σ0=σ0, r=r, poweradjust=poweradjust,
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
avgfit, ml, merrval, rerrs = SCA.matchedfitval(X,datadic["cells"], W1, H1; clamp=false)
nodr = SCA.matchedorder(ml,ncells); Wsca, Hsca = W1[:,nodr], H1[nodr,:]; SCA.normalizeW!(Wsca,Hsca)
# SCA.normalizeW!(W1,H1); Wsca,Hsca = sortWHslices(W1,H1)
fprx = "$(method)$(dataset)_a$(α)_b$(β)_it$(sca_maxiter)_fv$(fitval)_afv$(avgfit)_rt$(rt2)"
imsave_data(dataset,joinpath(subworkpath,fprx),Wsca,Hsca,imgsz,100; saveH=false)

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
avgfit, ml, merrval, rerrs = SCA.matchedfitval(X,datadic["cells"], W1, H1; clamp=false)
nodr = SCA.matchedorder(ml,ncells); Wadmm, Hadmm = W1[:,nodr], H1[nodr,:]; SCA.normalizeW!(Wadmm,Hadmm)
fprx = "$(method)$(dataset)_a$(α)_it$(admm_maxiter)_fv$(fitval)_afv$(avgfit)_rt$(rt2)"
imsave_data(dataset,joinpath(subworkpath,fprx),Wadmm,Hadmm,imgsz,100; saveH=false)

# HALS
method="hals"; @show method
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter = hals_maxiter; α=0.0; tol=-1

Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=hals_maxiter, α=α, l₁ratio=1,
            tol=tol, verbose=false), X, Wcd, Hcd)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
fitval = SCA.fitd(X,Wcd*Hcd)
# SCA.normalizeW!(Wcd,Hcd); Wcd,Hcd = sortWHslices(Wcd,Hcd)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(X,datadic["cells"], Wcd, Hcd; clamp=false)
nodr = SCA.matchedorder(ml,ncells); Wcd, Hcd = Wcd[:,nodr], Hcd[nodr,:]; SCA.normalizeW!(Wcd,Hcd)
fprx = "$(method)$(dataset)_a$(α)_it$(hals_maxiter)_fv$(fitval)_afv$(avgfit)_rt$(rt2)"
imsave_data(dataset,joinpath(subworkpath,fprx),Wcd,Hcd,imgsz,100; saveH=false)

plotH(subworkpath,X,datadic["cells"],Hsca,Hadmm,Hcd,1:8000)

function plotH(path,X,GTX,Hsca,Hadmm,Hcd,rng)
    mtdcolors = [RGBA{N0f8}(0.00,0.00,0.00,1.0),RGBA{N0f8}(0.00,0.45,0.70,1.0),RGBA{N0f8}(0.90,0.62,0.00,1.0),
                RGBA{N0f8}(0.00,0.62,0.45,1.0),RGBA{N0f8}(0.80,0.47,0.65,1.0),RGBA{N0f8}(0.34,0.71,0.91,1.0),
                RGBA{N0f8}(0.84,0.37,0.00,1.0),RGBA{N0f8}(0.94,0.89,0.26,1.0)]
    gtncells = length(GTX); ncells = size(Hsca,1)
    for i in 1:ncells
        f = Figure(resolution = (1000,300))
        ax11=AMakie.Axis(f[1,1],title="$i", titlesize=30)
        i <= gtncells && begin
            vecs = collect.(GTX[i][2]); idxs = eltype(vecs[1])[]; foreach(v->append!(idxs,v),vecs)
            gtxi = X[idxs,:]; foreach(c->c.=c.*GTX[i][3],eachcol(gtxi))
            gtH = norm.(eachcol(gtxi))
            lines!(ax11,gtH[rng],color=mtdcolors[1],label="anotated")
        end
        lines!(ax11,Hsca[i,rng],color=mtdcolors[2],label="SMF")
        lines!(ax11,Hadmm[i,rng],color=mtdcolors[3],label="Compresed NMF")
        lines!(ax11,Hcd[i,rng],color=mtdcolors[4],label="HALS")
        axislegend(ax11, position = :rt) # halign = :left, valign = :top
        save(joinpath(path,"neurofinder_H$(i).png"),f)
    end
end

# Figure
dtcolors = distinguishable_colors(ncells; lchoices=range(0, stop=50, length=15))
# Input data
gridcols=Int(ceil(sqrt(size(Wscas[1],2))))
imgsca1 = mkimgW(Wscas[1],imgsz,gridcols=gridcols); imgsca2 = mkimgW(Wscas[end],imgsz,gridcols=gridcols)
imgadmm1 = mkimgW(Wadmms[1],imgsz,gridcols=gridcols); imgadmm2 = mkimgW(Wadmms[end],imgsz,gridcols=gridcols)
imghals1 = mkimgW(Whalss[1],imgsz,gridcols=gridcols); imghals2 = mkimgW(Whalss[end],imgsz,gridcols=gridcols)
labels = ["SMF","Compressed NMF","HALS NMF"]
f = Figure(resolution = (900,1500))
ax11=AMakie.Axis(f[1,1],title=labels[1], titlesize=30, subtitle="maxiter=$(scamaxiterrng[1]), runtime=$(round(rtscas[1],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax11)
ax12=AMakie.Axis(f[1,2],title=labels[1], titlesize=30, subtitle="maxiter=$(scamaxiterrng[end]), runtime=$(round(rtscas[end],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax12)
ax21=AMakie.Axis(f[2,1],title=labels[2], titlesize=30, subtitle="maxiter=$(admmmaxiterrng[1]), runtime=$(round(rtadmms[1],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax21)
ax22=AMakie.Axis(f[2,2],title=labels[2], titlesize=30, subtitle="maxiter=$(admmmaxiterrng[end]), runtime=$(round(rtadmms[end],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax22)
ax31=AMakie.Axis(f[3,1],title=labels[3], titlesize=30, subtitle="maxiter=$(halsmaxiterrng[1]), runtime=$(round(rthalss[1],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax31)
ax32=AMakie.Axis(f[3,2],title=labels[3], titlesize=30, subtitle="maxiter=$(halsmaxiterrng[end]), runtime=$(round(rthalss[end],digits=2))sec",
        subtitlesize=25, aspect = DataAspect())
hidedecorations!(ax32)
image!(ax11, rotr90(imgsca1)); image!(ax12, rotr90(imgsca2))
image!(ax21, rotr90(imgadmm1)); image!(ax22, rotr90(imgadmm2))
image!(ax31, rotr90(imghals1)); image!(ax32, rotr90(imghals2))
save(joinpath(subworkpath,"neurofinder.png"),f)
