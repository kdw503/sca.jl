using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","noc_nac")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))

dataset = :fakecells; inhibitindices=0; bias=0.1; SNR = -10; ncells = 15; factor = 1
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
maskth=0.25; makepositive = true; tol=-1
imgsz0 = (40,20)
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

pcb_maxiter = 200; pcb_inner_maxiter = 50; pcb_ls_maxiter = 100
admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
hals_maxiter = 200

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = LCSVD.noisefilter(filter,X,imgsz)

subtract_bg=false; sbgstr = subtract_bg ? "sbg" : "nosbg"

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# PCB
prefix = "PCB"; r=0.3; nac = 0; maxiter = pcb_maxiter
for (tailstr,initmethod,α,β) in [("_sp_nn",:tsvd,0.005,5.0),("_nn",:nndsvd,0.,5.0), ("_sp",:tsvd,0.005,0.)]#
    @show tailstr
    dd = Dict()
    α1=α2=α; β1=β2=β
    useprecond = tailstr == "_sp_nn" ? false : true
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    # init
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells, nac; initmethod=initmethod, svdmethod=:tsvd)
    σ0=std(W0*M0) #=10*std(W0)=#
    # solve!
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond, usedenoiseW0H0=false,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    M, N = copy(M0), copy(N0)
    rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
    alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
    Wlc, Hlc = rst0.W, rst0.H
    # evaluate
    avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wlc, Hlc; clamp=false)
    LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
    fprex = "$(prefix)$(SNR)db$(factor)f$(ncells)_$(nac)s$(initmethod)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", Wlc,Hlc,gtW,gtH,imgsz,100; saveH=false, verbose=false)

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
        metadata["sigma0"] = σ0; metadata["r"] = r; metadata["initmethod"] = initmethod
        metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
        metadata["usedenoiseW0H0"] = usedenoiseW0H0; metadata["denoisefilter"] = alg.denoisefilter; 
        metadata["alpha"] = α; metadata["beta"] = β
    end
    save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
end

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
axsca=AMakie.Axis(f[1,1],title="H components of the inhibited cell (SMF)")
axhals=AMakie.Axis(f[2,1],title="H components of the inhibited cell (HALS)" ,xlabel="time index"); linkxaxes!(axbefore, axafter)
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
ax11=AMakie.Axis(f[1,1],title="W component", aspect = DataAspect()); hidedecorations!(ax11)
axall2=AMakie.Axis(f[:,2],title="H component",xlabel="time index")
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
ax11=AMakie.Axis(f[1,1],title=labels[1], aspect = DataAspect()); hidedecorations!(ax11)
ax21=AMakie.Axis(f[2,1],title=labels[2], aspect = DataAspect()); hidedecorations!(ax21)
ax31=AMakie.Axis(f[3,1],title=labels[3], aspect = DataAspect()); hidedecorations!(ax31)
ax41=AMakie.Axis(f[4,1],title=labels[4], aspect = DataAspect()); hidedecorations!(ax41)
axall2=AMakie.Axis(f[:,2],title="Inhibited H component",xlabel="time index")
image!(ax11, rotr90(imggt)); image!(ax21, rotr90(imgsca)); image!(ax31, rotr90(imgadmm)); image!(ax41, rotr90(imghals))
lin = [lines!(axall2,hd,color=mtdcolors[i]) for (i,hd) in enumerate(hdata)]
f[:,3] = Legend(f[:,2],lin,labels)
save(joinpath(subworkpath,"idx$(inhibitindices)_bias$(bias)_$(sbgstr).png"),f)

