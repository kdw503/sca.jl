if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)
include(joinpath(workpath,"setup_light.jl"))
subworkpath = joinpath(workpath,"paper","neurofinder")

dataset = :neurofinder_small; SNR=0; imgsz=(40,20); inhibitindices=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=true; sbgstr = subtract_bg ? "_sbg" : ""

if true
    sca_maxiter = 100; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 100
else
    sca_maxiter = 2; sca_inner_maxiter = 2; sca_ls_maxiter = 2
    admm_maxiter = 2; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 2
end

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, imgsz=imgsz, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,joinpath(subworkpath,"Wr1"),Wcd,Hcd,imgsz,100; 
        gridcols=1, signedcolors=dgwm(), saveH=false)
    Hcd201 = mapwindow(mean, Hcd, (1,201))
#    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*Hcd201; X .-= bg
end

# SCA
@show "SCA"
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN = :WH2M2
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none

(tailstr,initmethod) = ("_sp",:isvd)
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#
α1=10; α2=0; β1=0; β2=0
avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN,
    # reg=reg, useRelaxedNN=useRelaxedNN,
    useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=false,
    store_inner_trace=false, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
# normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W1,H1)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(X,datadic["cells"], W1, H1; clamp=false)
nodr = SCA.matchedorder(ml,ncells); Wsca, Hsca = W1[:,nodr], H1[nodr,:]; SCA.normalizeW!(Wsca,Hsca)
imshowW(Wsca,imgsz,gridcols=5)

initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(sbgstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_$(sd_group)_$(optimmethod)_$(regSpar)_aw$(α1)_ah$(α2)_$(regNN)_bw$(β1)_bh$(β2)_af$(avgfit)_r$(r)_it$(niters)_rt$(rt2)")
imsaveW(fname*".png",Wsca,imgsz,gridcols=5)
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
# imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)
f = plotH_data(fname, H3[[4,6],:]; resolution = (800,400), space=0.2)
# Wsvd, Hsvd = W0[:,nodr], H0[nodr,:]
# imsave_data(dataset,joinpath(subworkpath,fprex*"svd"),Wsvd,Hsvd,imgsz,100; saveH=false)


# ADMM
@show "ADMM"
mfmethod = :ADMM; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 0; β = 0; usennc=false
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = :none

(tailstr,initmethod,α,usennc) = ("_nn",:lowrank_nndsvd,0.,true)
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#
avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
α1=1;α2=α; β1=β2=β
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
    store_trace=false, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                    penmetric=penmetric, stparams=stparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed  W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                    penmetric=penmetric, stparams=stparams, cparams=cparams);
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
# normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W1,H1)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(X,datadic["cells"], W1, H1; clamp=false)
nodr = SCA.matchedorder(ml,ncells); W3, H3 = W1[:,nodr], H1[nodr,:]; SCA.normalizeW!(W3,H3)

initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(sbgstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_$(sd_group)_$(optimmethod)_$(reg)_aw$(α1)_ah$(α2)_af$(avgfit)_r$(r)_it$(niters)_rt$(rt2)")
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
#imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)
f = plotH_data(fname, H3[[4,6],:]; resolution = (800,400), space=0.2)

# HALS
@show "HALS"
W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize)
initmethod = :svd
if initmethod == :svd
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar, initdata=svd(X));
else
    rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
end
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1
regularization=:both # regularization(:both, :transformation(W) :components(H))
(tailstr,α) = ("_nn",0.0)
Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, regularization=regularization, α=α, l₁ratio=1,
                tol=tol, verbose=false), X, Wcd, Hcd) 
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
# normalizeW!(Wcd,Hcd); W3,H3 = sortWHslices(Wcd,Hcd)
avgfit, ml, merrval, rerrs = SCA.matchedfitval(X,datadic["cells"], Wcd, Hcd; clamp=false)
nodr = SCA.matchedorder(ml,ncells); W3, H3 = Wcd[:,nodr], Hcd[nodr,:]; SCA.normalizeW!(W3,H3)

initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(sbgstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_$(regularization)_a$(α)_af$(avgfit)_it$(hals_maxiter)_rt$(rt2)")
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
#imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)
f = plotH_data(fname, H3[[4,6],:]; resolution = (800,400), space=0.2)

#====== SPCA =================#
using ScikitLearn

mfmethod = :SPCA

@sk_import decomposition: SparsePCA
α = 0.5; ridge_alpha=0.01; max_iter=500; tol=1e-7
rtspca = @elapsed resultspca = fit_transform!(SparsePCA(n_components=ncells,alpha=α,ridge_alpha=ridge_alpha,max_iter=max_iter,tol=tol,verbose=true),X) 
W1 = copy(resultspca); H1 = W1\X
avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W3,H3)
fprex = "$(mfmethod)$(datastr)$(filterstr)"
fname = joinpath(subworkpath,"$(fprex)_a$(α)_af$(avgfit)_rt$(rtspca)")
imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=true)




fitd(a,b) = (na=norm(a); nb=norm(b); fitd(a,b,na,nb))
fitd(a,b,na) = (nb=norm(b); fitd(a,b,na,nb))
fitd(a,b,na,nb) = (denom=na^2+nb^2; 1-sum(abs2,a-b)/denom)












#=========== Degeneration of neurofinder ==========================================#
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN = :WH2M2
useRelaxedL1=true; s=10*0.3^0; initmethod = :isvd
r=(0.3)^1
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
makepositive = true; poweradjust = :none

rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0)
α1=10; α2=0; β1=0; β2=0
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=false,
    store_inner_trace=false, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
αw, αh, βw, βh, σ2, _ = SCA.normalize_params!(W0,H0,D,Mw0, Mh0, regSpar, regNN, 1, α1, α2, β1, β2, σ0, 0)
penall0, pen0, sparw0, sparh0, nnw0, nnh0 = SCA.penaltyMwMh(W0,H0,D,0,Mw,Mh,αw,αh,βw,βh,regSpar,stparams.l1l2ratio,stparams.M2power,useRelaxedL1,uselv)
# (7.5090638884831975, 0.6629194979634113, 6.846144390519786, 0, 0, 0)
W1,H1,Mw,Mh = balanceWH!(W0,H0,Mw,Mh)
makepositive && flip2makepos!(W1,H1,Mw,Mh); W3,H3=copy(W1),copy(H1); normalizeW!(W3,H3)
initmethodstr = "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(sbgstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_$(sd_group)_$(optimmethod)_$(regSpar)_aw$(α1)_ah$(α2)_$(regNN)_bw$(β1)_bh$(β2)_af$(avgfit)_r$(r)_it$(niters)_rt$(rt2)")
imgsaverng = 1:30
imsave_data(dataset,fname,W3[:,imgsaverng],H3[imgsaverng,:],imgsz,100; saveH=false)
f = plotH_data(fname, H3[[2,8],:]; resolution = (800,400), space=0.2)

# delete the contaminated cell 
imgw2 = reshape(W1[:,2],imgsz...); imgw2[1:18,1:16].=0; W2 = copy(W1); W2[:,2] = vec(imgw2)
Mwdel = W0\W2; Mhdel = Mwdel\D
W2 = W0*Mwdel; H2 = Mhdel*H0
αw, αh, βw, βh, σ2, _ = SCA.normalize_params!(W0,H0,D,Mwdel, Mhdel, regSpar, regNN, 1, α1, α2, β1, β2, σ0, 0)
penall0, pen0, sparw0, sparh0, nnw0, nnh0 = SCA.penaltyMwMh(W0,H0,D,0,Mwdel,Mhdel,αw,αh,βw,βh,regSpar,stparams.l1l2ratio,stparams.M2power,useRelaxedL1,uselv)
# (7.907378330079086, 4.642023004899269e-30, 7.907378330079086, 0, 0, 0)
normalizeW!(W2,H2)
fname = joinpath(subworkpath,"SCA_init_delete_f$(penall0)")
imsave_data(dataset,fname,W2[:,1:10],H2[1:10,:],imgsz,100; saveH=false)
f = plotH_data(fname, H2[[2,8],:]; resolution = (800,400), space=0.2)

# SCA again
Mw3, Mh3 = copy(Mwdel), copy(Mhdel);
α1 = 10; α2 = 0; β1 = 0; β2 = 0
stparams.useRelaxedL1=useRelaxedL1; stparams.α1 = α1; stparams.α2 = α2; stparams.β1 = β1; stparams.β2 = β2
αw, αh, βw, βh, σ2, _ = SCA.normalize_params!(W0,H0,D,Mw, Mh, regSpar, regNN, 1, α1, α2, β1, β2, σ0, 0)
cparams.plotiterrng=1:0; cparams.plotinneriterrng=1:50; cparams.maxiter = 100; cparams.inner_maxiter=50; cparams.show_trace = false
rt2 = @elapsed W3, H3, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw3, Mh3, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
penall, pen, sparw, sparh, nnw, nnh = SCA.penaltyMwMh(W0,H0,D,0,Mw3,Mh3,αw,αh,βw,βh,regSpar,stparams.l1l2ratio,stparams.M2power,useRelaxedL1,uselv)
# (6.9455182983847745, 0.5494332384180196, 6.3960850599667545, 0, 0, 0)
W3,H3,Mw3,Mh3 = balanceWH!(W0,H0,Mw3,Mh3)
makepositive && flip2makepos!(W3,H3)
W4, H4 = copy(W3), copy(H3); normalizeW!(W4,H4)
initmethodstr = "_delete"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(sbgstr)$(initmethodstr)"
fname = joinpath(subworkpath,"$(fprex)_$(sd_group)_$(optimmethod)_$(regSpar)_aw$(α1)_ah$(α2)_$(regNN)_bw$(β1)_bh$(β2)__f$(penall)_af$(avgfit)_r$(r)_it$(niters)_rt$(rt2)")
imgsaverng = 1:30
imsave_data(dataset,fname,W4[:,imgsaverng],H4[imgsaverng,:],imgsz,100; saveH=false)
f = plotH_data(fname, H4[[2,8],:]; resolution = (800,400), space=0.2)

W02 = W0.^2; H02 = H0.^2; isadmm = false
Mw, Mh = copy(Mwdel), copy(Mhdel); x0 = vcat(vec(Mw'),vec(Mh))
state = SCA.initial_state(W0,H0,D,Mw,Mh,W0*Mw,Mh*H0,isadmm,stparams=stparams,lsparams=lsparams)
fg!, P = SCA.prepare_fg_invert_whole(W0, H0, W02, H02, D, state, stparams)
gradE = zeros(2*p^2); E0 = fg!(1,nothing,x0); fg!(nothing,gradE,x0)


imgdel1 = load(joinpath(subworkpath,"SCA_init_delete_f0.16143130379239573_W_1.png"))
imgdel2 = load(joinpath(subworkpath,"SCA_init_delete_f0.16143130379239573_W_2.png"))
img1 = load(joinpath(subworkpath,"SCA_neurofinder_small_none_sbg_delete_whole_optim_lbfgs_WH1M2_aw10_ah0_WH2M2_bw0_bh0__f7.430181757332959_af0.46554846902884306_r0.3_it100_rt4.2075799_W_1.png"))
img2 = load(joinpath(subworkpath,"SCA_neurofinder_small_none_sbg_delete_whole_optim_lbfgs_WH1M2_aw10_ah0_WH2M2_bw0_bh0__f7.430181757332959_af0.46554846902884306_r0.3_it100_rt4.2075799_W_2.png"))
f = Figure()
ax1 = GLMakie.Axis(f[1,1],aspect=DataAspect()); hidedecorations!(ax1); hidespines!(ax1)
ax2 = GLMakie.Axis(f[2,1],aspect=DataAspect()); hidedecorations!(ax2); hidespines!(ax2)
ax3 = GLMakie.Axis(f[3,1],aspect=DataAspect()); hidedecorations!(ax3); hidespines!(ax3)
ax4 = GLMakie.Axis(f[4,1],aspect=DataAspect()); hidedecorations!(ax4); hidespines!(ax4)
image!(ax1,rotr90(imgdel1))
image!(ax2,rotr90(img1))
image!(ax3,rotr90(imgdel2))
image!(ax4,rotr90(img2))

f = Figure()
ax = GLMakie.Axis(f[1,1])
lindel2 = lines!(ax,H2[2,:])
linsca2 = lines!(ax,H1[2,:])