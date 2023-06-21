using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

include("setup.jl")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include("dataset.jl")
include("utils.jl")

using ProfileView, BenchmarkTools

dataset = :fakecells; SNR=0; inhibitidx=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitidx)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

# testdic = Dict()
# A=rand(40,20,1000); img_nl = AxisArray(colorview(Gray, A), :y, :x, :time)
# gtW = rand(40,20,7); gtH = rand(1000,7)

# testdic["gt_ncells"] = 7
# testdic["imgrs"] = Matrix(reshape(img_nl,800,1000))
# testdic["img_nl"] = img_nl
# testdic["gtW"] = gtW
# testdic["gtH"] = gtH
# JLD.save("C:\\Users\\kdw76\\WUSTL\\Work\\Data\\fakecells\\fakecells0_calcium_sz(40, 20)_lengthT1000_J0_SNR0_bias0.1.jld",testdic)
# JLD.load("C:\\Users\\kdw76\\WUSTL\\Work\\Data\\fakecells\\fakecells0_calcium_sz(40, 20)_lengthT1000_J0_SNR0_bias0.1.jld")

data_sca_nn = Dict[]; data_sca_sp = Dict[]; data_sca_sp_nn = Dict[]
data_admm_nn = Dict[]; data_admm_sp = Dict[]; data_admm_sp_nn = Dict[]
data_hals_nn = Dict[]; data_hals_sp_nn = Dict[]

for iter in 1:30
    @show iter
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

maxiterstr = "it$(maxiter)"
optimmethodstr = (optimmethod ∈ [:SB_newton, :SB_lbfgs, :SB_cg] || !useRelaxedL1) ? "$optimmethod" : "$(optimmethod)_σ$(s)r$(r)"
β1 = β2 = β; α1 = α; α2 = α; # α2 = 0
scaparamstr = "_$(reg)_$(sd_group)_$(optimmethodstr)_$(maxiterstr)_init$(inner_maxiter)_αw$(α1)_αh$(α2)_β$(β)"
ddsca["SCA_alpha0"] = σ0; ddsca["SCA_r"] = r; ddsca["SCA_inner_maxiter"] = inner_maxiter;

figi = GLMakie.Figure()
axi = GLMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "fit value", title = "Fit values",
                    xlabelsize=12, ylabelsize=12, titlesize=12)
figt = GLMakie.Figure()
axt = GLMakie.Axis(fig[1, 1], xlabel = "time", ylabel = "fit value", title = "Fit values",
                    xlabelsize=12, ylabelsize=12, titlesize=12)
for (tailstr,initmethod,α,β) in [("_nn",:nndsvd,0.,1000.), ("_sp",:isvd,100.,0.), ("_sp_nn",:nndsvd,100.,1000.)]
    @show tailstr
    ddsca_nn = Dict()
    initmethodstr = "_$(initmethod)"
    fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
    rt1sca = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

    α1=α2=α; β1=β2=β
    avgfits=Float64[]; fxsss=Float64[]; rt2s=Float64[]
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        reg=reg, useRelaxedL1=useRelaxedL1, useRelaxedNN=useRelaxedNN, σ0=σ0, r=r, poweradjust=poweradjust,
        useprecond=useprecond, uselv=uselv)
    lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    Mw, Mh = copy(Mw0), copy(Mh0);
    rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; penmetric=penmetric,
                                            stparams=stparams, lsparams=lsparams, cparams=cparams);
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; penmetric=penmetric,
                                            stparams=stparams, lsparams=lsparams, cparams=cparams);
    omstr = stparams.optimmethod == :sca_lbfgs ? "_lbfgs" : chop(String(stparams.optimmethod),head=5,tail=0)
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    makepositive && flip2makepos!(W3,H3)
    fxs = getdata(trs,:f_x); fxss = getdata(trs,:fxs); niters = getdata(trs,:niter)
    push!(fxsss,fxss) # inner_trace
    lfxss, totaliters = length.(fxss), sum(niters)

    avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
    imsave_data(dataset,"$(fprex)$(omstr)_a$(α1)_b$(β1)_it$(niters)_tit$(totaliters)_fit$(avgfit)_rt$(rt2)",W3,H3,imgsz,100; saveH=false)

    Mwss = getdata(trs,:Mws); Mhss = getdata(trs,:Mhs);
    Mwsall = []; Mhsall = []; rt2sall = Float64[]
    for (iter,(Mwsi,Mhsi)) in enumerate(zip(Mwss,Mhss))
        @show iter-1
        isempty(Mwsi) && continue
        for (inner_iter,(Mw, Mh)) in enumerate(zip(Mwsi,Mhsi))
            @show inner_iter
            push!(Mwsall,Mw); push!(Mhsall,Mh)
        end
        if iter == 1
            rt2 = 0.
        else
            rt2 = collect(range(start=laps[iter-1],stop=laps[iter],length=length(Mwsi)+1))[2:end].-laps[1]
        end
        append!(rt2sall,rt2)
    end
    @show length(rt2sall)
    for (i,(Mw,Ms,rt2)) in enumerate(zip(Mwsall,Mhsall,rt2sall))
        i%2 != 1 && continue
        @show i
        W3=W0*Mw; H3 = Mh*H0
        normalizeW!(W3,H3)
        avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
#            avgfit = fitdold(X,W3*H3)
        push!(avgfits,avgfit)
        push!(rt2s,rt2)
    end

    ddsca["sca_initmtd$(tailstr)"] = initmethod; ddsca["sca_alpha$(tailstr)"] = α; ddsca["sca_beta$(tailstr)"] = β
    ddsca["sca_maxiter$(tailstr)"] = maxiter; ddsca["sca_niters$(tailstr)"] = niters;
    ddsca["sca_rt2s$(tailstr)"] = rt2s; ddsca["sca_avgfits$(tailstr)"]=avgfits
    lines!(axi, avgfits); lines!(axt, rt2s, avgfits)
end
fnprfx = "sca_methods_vs_fitvals_it$(maxiter)"
for withlegend = [false, true]
    for ax in [axi,axt]
        withlegend && ax.legend(["SMF NN","SMF Sparse", "SMF NN Sparse"], fontsize = 12)
    end
    save("$(fnprfx)_iter_$(withlegend).png",figi)
    save("$(fnprfx)_time_$(withlegend).png",figt)
end
close("all")
save("$(fnprfx).jld",ddsca)

# ADMM
ddadmm = Dict()

mfmethod = :ADMM; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 10; β = 0; usennc=true
useRelaxedL1=true; s=10*0.3^0; σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter=1000; inner_maxiter = 50; ls_maxiter = 500
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = (reg == :WH1 && sd_group != :whole) ? :balance : :none

maxiterstr = "it$(maxiter)"
optimmethodstr = (optimmethod == :sca_admm) ? "" : "_optim"
β1 = β2 = β; α1 = α; α2 = α; # α2 = 0
scaparamstr = "_$(reg)_$(sd_group)_$(optimmethodstr)_$(maxiterstr)_init$(inner_maxiter)_αw$(α1)_αh$(α2)_β$(β)"
uselogscale = true

figi = GLMakie.Figure()
axi = GLMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "fit value", title = "Fit values",
                    xlabelsize=12, ylabelsize=12, titlesize=12)
figt = GLMakie.Figure()
axt = GLMakie.Axis(fig[1, 1], xlabel = "time", ylabel = "fit value", title = "Fit values",
                    xlabelsize=12, ylabelsize=12, titlesize=12)
for (tailstr,initmethod,α,usennc) in [("_nn",:lowrank_nndsvd,0.,true), ("_sp",:lowrank,10.,false), ("_sp_nn",:lowrank_nndsvd,10.,true)]
    @show tailstr
    initmethodstr = "_$(initmethod)"
    fprex = "$(mfmethod)$(optimmethodstr)$(datastr)$(filterstr)$(initmethodstr)"
    rt1sca = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    avgfits=[]; fxsss=[]
    α1=α2=α; β1=β2=β
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
        store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    Mw, Mh = copy(Mw0), copy(Mh0);
    rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; penmetric=penmetric,
                                        stparams=stparams, cparams=cparams);
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed  W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; penmetric=penmetric,
                                        stparams=stparams, cparams=cparams);
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    makepositive && flip2makepos!(W3,H3)
    avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
    imsave_data(dataset,"$(fprex)_a$(α1)_it$(niters)_fit$(avgfit)_rt$(rt2)",W3,H3,imgsz,100; saveH=false)

    Wss = getdata(trs,:Ws); Hss = getdata(trs,:Hs);
    fxs = getdata(trs,:f_x)

    for (iter,(Ws,Hs)) in enumerate(zip(Wss,Hss))
        @show iter-1
        isempty(Ws) && continue
        for (inner_iter,(W3, H3)) in enumerate(zip(Ws,Hs))
            @show inner_iter
            normalizeW!(W3,H3)
            avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
            # avgfit = fitdold(X,W3*H3)
            push!(avgfits,avgfit)
        end
    end
    rt2s = range(start=0,stop=rt2,length=length(avgfits))
    ddadmm["admm_initmtd$(tailstr)"] = initmethod; ddadmm["admm_alpha$(tailstr)"] = α
    ddadmm["admm_maxiter$(tailstr)"] = maxiter; ddadmm["admm_niters$(tailstr)"] = niters;
    ddadmm["admm_rt2s$(tailstr)"] = rt2s; ddadmm["admm_avgfits$(tailstr)"]=avgfits
    lines!(axi, avgfits); lines!(axt, rt2s, avgfits)
end
fnprfx = "admm($(optimmethod))_vs_fitvals_it$(maxiter)"
for withlegend = [false, true]
    for ax in [axi,axt]
        withlegend && ax.legend(["ADMM NN","ADMM Sparse", "ADMM NN Sparse"], fontsize = 12)
    end
    save("$(fnprfx)_iter_$(withlegend).png",figi)
    save("$(fnprfx)_time_$(withlegend).png",figt)
end
close("all")
save("$(fnprfx).jld",ddadmm)


# HALS
ddhals = Dict()

rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter=10; 
fprex = "$(mfmethod)$(datastr)$(filterstr)"

figi = GLMakie.Figure()
axi = GLMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "fit value", title = "Fit values",
                    xlabelsize=12, ylabelsize=12, titlesize=12)
figt = GLMakie.Figure()
axt = GLMakie.Axis(fig[1, 1], xlabel = "time", ylabel = "fit value", title = "Fit values",
                    xlabelsize=12, ylabelsize=12, titlesize=12)
for (tailstr,α) in [("_nn",0.),("_sp_nn",0.1)]
    avgfits=[]
    Wcd, Hcd = copy(Wcd0), copy(Hcd0); normalizeW!(Wcd,Hcd)
    avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
    push!(avgfits,avgfit)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=true), X, Wcd, Hcd)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); W3,H3 = sortWHslices(Wcd,Hcd)
    avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
    imsave_data(dataset,"$(fprex)_a$(α)_it$(result.niters)_fit$(avgfit)_rt$(rt2)",W3,H3,imgsz,100; saveH=false)

    for (iter,(W3,H3)) in enumerate(zip(result.Ws,result.Hs))
        @show iter
        normalizeW!(W3,H3)
        avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
        push!(avgfits,avgfit)
    end
    rt2s = range(start=0,stop=rt2,length=length(avgfits))
    ddhals["hals_alpha$(tailstr)"] = α
    ddhals["hals_maxiter$(tailstr)"] = maxiter; ddhals["hals_niters$(tailstr)"] = result.niters;
    ddhals["hals_rt2s$(tailstr)"] = rt2s; ddhals["hals_avgfits$(tailstr)"]=avgfits
    lines!(axi, avgfits); lines!(axt, rt2s, avgfits)
end
fnprfx = "hals_vs_fitvals_it$(maxiter)"
for withlegend = [false, true]
    for ax in [axi,axt]
        withlegend && ax.legend(["HALS NN", "HALS NN Sparse"], fontsize = 12)
    end
    save("$(fnprfx)_iter_$(withlegend).png",figi)
    save("$(fnprfx)_time_$(withlegend).png",figt)
end
close("all")
save("$(fnprfx).jld",ddhals)








Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rtcd100 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=100, α=0.1, l₁ratio=1,
                tol=tol), $X, $Wcd, $Hcd)

@show niters, rtssca, rtscd50, rtscd100

rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
α = 0.0; halsmaxiter = 50
Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rtcd = @elapsed result, Ws, Hs = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=halsmaxiter, α=α, l₁ratio=1, tol=tol, verbose=true), X, Wcd, Hcd)
normalizeW!(Wcd,Hcd); W3,H3 = sortWHslices(Wcd,Hcd)
avgfitval, ml, mfitval, rerrs = matchedfitval(gtW, gtH, W3, H3)

imsave_data(dataset,"HALS_$(SNR)dB_a$(α)_it$(halsmaxiter)_rt$(rtcd)",W3,H3,imgsz,100; saveH=true)

# Profile View
Mw, Mh = copy(Mw0), copy(Mh0)
W1, H1, objvals, trs, niters = scasolve!(X, W0, H0, d, Mw, Mh; penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0)
@profview scasolve!(X, W0, H0, d, Mw, Mh; penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);

# Cthulhu
@descend scasolve!(X, W0, H0, d, Mw, Mh; penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams)



# save images
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W3,H3)
omstr = stparams.optimmethod == :sca_lbfgs ? "_lbfgs" : chop(String(stparams.optimmethod),head=5,tail=0)
imsave_data(dataset,"$(mfmethod)$(omstr)_$(SNR)dB_a$(α1)_it$(niters)_rt$(rt2)",W3,H3,imgsz,100; saveH=true)
fdts=getdata(trs,:sympen); fsps=getdata(trs,:sparseW)+getdata(trs,:sparseH)
xdiffss=getdata(trs,:x_abs); fxss = fdts + fsps
plot_convergence("$(mfmethod)$(omstr)_$(SNR)dB_a$(α1)_it$(niters)_rt$(rt2)",xdiffss,fxss; title="convergence (SCA)")
plotH_data(dataset,"$(mfmethod)$(omstr)_$(SNR)dB_a$(α1)_it$(niters)_rt$(rt2)",H3)

fxss=getdata(trs,:fxs)





# ADMM
initmethod = :lowrank_nmf
rt1sca = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar)
save_init_figure = false; save_init_figure && begin
    normalizeW!(Wp,Hp); W2,H2 = copy(Wp), copy(Hp) # sortWHslices(Wp,Hp)
    imsave_data(dataset,"$(initmethod)$(datastr)$(filterstr)_rti$(rt1sca)",W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
end

mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 10; β = 0; usennc=false
useRelaxedL1=true; s=10*0.3^0; σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=1e-8; optimmethod = :sca_admm_optim_lbfgs; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter=10; scamaxiter = 10; halsmaxiter = 50; spcamaxiter=maxiter; inner_maxiter = 200; ls_maxiter = 500
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = (reg == :WH1 && sd_group != :whole) ? :balance : :none

maxiterstr = mfmethod == :HALS ? "it$(halsmaxiter)" : "it$(scamaxiter)"
initmethodstr = mfmethod == :HALS ? "_nndsvd" : mfmethod == :SPCA ? "" : "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
optimmethodstr = (optimmethod ∈ [:SB_newton, :SB_lbfgs, :SB_cg] || !useRelaxedL1) ? "$optimmethod" : "$(optimmethod)_σ$(s)r$(r)"
β1 = β2 = β; α1 = α; α2 = α; # α2 = 0
scaparamstr = "_$(reg)_$(sd_group)_$(optimmethodstr)_$(maxiterstr)_init$(inner_maxiter)_αw$(α1)_αh$(α2)_β$(β)"
uselogscale = true

figi, axi = plt.subplots(1,1, figsize=(5,4))
#ax.set_title(yaxisstr)
axi.set_ylabel("log10(penalty)",fontsize = 12)
axi.set_xlabel("iteration",fontsize = 12)

figt, axt = plt.subplots(1,1, figsize=(5,4))
#ax.set_title(yaxisstr)
axt.set_ylabel("log10(penalty)",fontsize = 12)
axt.set_xlabel("time",fontsize = 12)

#for α in [0.1,1,10,100,1000]
for (optimmethod,useRelaxedL1,useprecond,α,innermaxiter,usennc) in [#=(:optim_lbfgs,true,true,100,50,false),=#
                                                            (:sca_admm,false,false,0,100,true),
                                                            (:sca_admm_optim_lbfgs,false,false,0,100,true)]
    α1=α2=α; β1=β2=β
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
    lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=scamaxiter, inner_maxiter=inner_maxiter,
        store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    Mw, Mh = copy(Mw0), copy(Mh0);
    rt2 = @elapsed W1, H1, objvals, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh; penmetric=penmetric, stparams=stparams,
                                        lsparams=lsparams, cparams=cparams);
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    makepositive && flip2makepos!(W3,H3)
    fxs = getdata(trs,:f_x)
    # calculate MSD
    avgfitval, ml, mfitval, rerrs = matchedfitval(gtW, gtH, W3, H3)

    imsave_data(dataset,"$(fprex)$(optimmethod)_a$(α1)_b$(β1)_r$(r)_it$(niters)_rt$(rt2)",W3,H3,imgsz,100; saveH=false)

    iterrng = 0:niters; trng = iterrng/niters*rt2
    # fxs ./= fx0
    uselogscale && (fxs=log10.(fxs))
    length(fxs) != 0 && begin
        axi.plot(iterrng[2:end],fxs[2:end]); axt.plot(trng[2:end],fxs[2:end])
    end
end
title = "w_sep_vs_wo_sep_nn"
for withlegend = [false, true]
    for ax in [axi,axt]
        withlegend && ax.legend([#="SCA",=# "With W H seperation","W/O  W H seperation"], fontsize = 12)
    end
    figi.savefig("$(title)_$(initmethod)_$(optimmethod)_it$(maxiter)_iter_$(withlegend).png")
    figt.savefig("$(title)_$(initmethod)_$(optimmethod)_it$(maxiter)_time_$(withlegend).png")
end
close("all")



using BoxTrees, TileTrees, TileTreesGUI, ImageView

## Generate some data to visualize (if you already have a TileTree, this step is
## not necessary )

# Create a tile tree with two rectangular tiles, each "active" only a
# small portion of the time
t1, t2 = zeros(200), zeros(200)
t1[rand(1:200, 30)] = rand(30)   # active for 30/200 time points
t2[rand(1:200, 30)] = rand(30)
ttree = TileTree([Tile((Span(16,50), Span(1,80)), ones(35,80), t1),
                  Tile((Span(1,40), Span(71,100)), ones(40,30), t2)])

## Visualize the TileTree

# Create the rendering object
ttc = TileTreeColor(ttree, (0,1), (0,1))
# View the rendering
imshow(ttc)

tileaction = initialize_triage_action(ttree)
tdct = triage_gui(ttree, tileaction, Array(ttree), ttree)
