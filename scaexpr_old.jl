"""
TODO list
"""
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

plt.ioff()

# @sk_import decomposition: SparsePCA

dataset = :fakecells; SNR=0; inhibitidx=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitidx)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitidx=inhibitidx, save_gtimg=true, save_maxSNR_X=false, save_X=false);

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
rt1sca = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar)
save_init_figure = false; save_init_figure && begin
    normalizeW!(Wp,Hp); W2,H2 = sortWHslices(Wp,Hp)
    imsave_data(dataset,"$(initmethod)$(datastr)$(filterstr)_rti$(rt1sca)",W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
end

method = :SCA; initmethod = :isvd; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 100; β = 0
useRelaxedL1=true; s=10; σ0=s*std(W0) #=10*std(W0)=#
r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=1e-6; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking
maxiter=50; scamaxiter = 50; halsmaxiter = 50; spcamaxiter=maxiter; inner_maxiter = 50; ls_maxiter = 50
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1

penaltystr = uselogscale ? "log10(penalty)" : "penalty"
if isplotxandg
    xdiffstr = uselogscale ? "log10(x difference)" : "x difference"
    maxgxstr = uselogscale ? "log10(maximum(g(x)))" : "maximum(g(x))"
end
figs = []; axes = []
yaxisstrs = [("log(f(x))"),("x_diff"),("maximum(g(x))")]; xaxisstrs = ["iterations","time"]
postfixstrs = [("log(f(x))"),("x_diff"),("maximum(g(x))")]
for yaxisstr in yaxisstrs[1:plotnum], xaxisstr in xaxisstrs
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.set_title(yaxisstr)
    ax.set_ylabel(penaltystr,fontsize = 12)
    ax.set_xlabel(xaxisstr,fontsize = 12)
    push!(figs,fig); push!(axes,ax)
    # axis.axis(axis) # [xamin, xamax, yamin, yamax]
end
paramvec_sca = [(:SCA,:whole,:WH1,20,200,penmetric)] #map(a->(:SCA,a,:column,:W1), [100,150])
paramvec_oca = [] #[(:OCA,:component,:W1,100,500,:HALS)] #map(a->(:SCA,a,:column,:W1), [100,150])
paramvec_hals = [(:HALS,:column,:WH1,100,0.1,penmetric)]
paramvec_spca = [] #[(:SPCA,:whole,:W1,100,0.5,:HALS)]
paramsvec = [("method","sd_group","reg","maxiter","α","penmetric"),paramvec_sca...,paramvec_oca...,paramvec_hals...,paramvec_spca...]
αidxs = findall(x->x=="α", paramsvec[1])
SCA_αw = isempty(αidxs) ? α : paramvec_sca[1][αidxs[1]]; SCA_αh = SCA_αw
HALS_α = isempty(αidxs) ? α : paramvec_hals[1][αidxs[1]]; HALS_l1ratio = l₁ratio = 1. # 0.5
pvs = copy(paramsvec[2:end])
deleteat!(pvs,findall(x->x[1]==:SPCA,pvs))
plot_dt_sp = false
legendstrs = plot_dt_sp ? map(param->param[1]==:HALS ? ["$(param[1]) all","$(param[1]) data","$(param[1]) sparseness"] :
                        ["$(param[1]) all","$(param[1]) data","$(param[1]) sparseness"], pvs) :
                        map(param->param[1]==:HALS ? ["$(param[1])"] :
                        ["$(param[1]), $(paramsvec[1][2])=$(param[2])"], pvs) 
legendstrs = vcat(legendstrs...)
title = join([paramsvec[1]...],"_vs_")
for (idx,(method, sd_group, reg, maxiter, α, penmetric)) = enumerate(paramsvec[2:end])
    poweradjust = (reg == :WH1 && sd_group != :whole) ? :balance : :none
    scamaxiter=maxiter; halsmaxiter=maxiter; spcamaxiter=maxiter
    @show method, poweradjust
    # paramsvec = [(:ls_BackTracking,10),(:ls_BackTracking,50),(:ls_BackTracking,100),(:ls_BackTracking,500),(:ls_BackTracking,1000),(:ls_BackTracking,5000)] #,
    # legendstrs = map(innerloop->"reg=$(innerloop[1]),lsmethod=$(innerloop[2])", paramsvec)
    maxiterstr = method == :HALS ? "it$(halsmaxiter)" : "it$(scamaxiter)"
    initmethodstr = method == :HALS ? "_nndsvd" : method == :SPCA ? "" : "_$(initmethod)"
    fprex = "$(method)$(datastr)$(filterstr)$(initmethodstr)"

    rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    Mw0, Mh0 = copy(Mw), copy(Mh)
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
    Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
    optimmethodstr = (optimmethod ∈ [:SB_newton, :SB_lbfgs, :SB_cg] || !useRelaxedL1) ? "$optimmethod" : "$(optimmethod)_σ$(s)r$(r)"
    β1 = β2 = β; α1 = α; α2 = α; # α2 = 0
    scaparamstr = "_$(reg)_$(sd_group)_$(optimmethodstr)_$(maxiterstr)_init$(inner_maxiter)_αw$(α1)_αh$(α2)_β$(β)"
    if method == :SCA
        Mw, Mh = copy(Mw0), copy(Mh0)
        stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
            reg=reg, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust, useprecond=true)
        lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=1, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
            store_inner_trace=false, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:5)
        W1, H1, objvals, trs, niters = scasolve!(X, W0, H0, d, Mw, Mh; HALS_α=HALS_α, HALS_l1ratio=HALS_l1ratio,
                    penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
        fdts=getdata(trs,:sympen); fsps=getdata(trs,:sparseW)+getdata(trs,:sparseH)
        xdiffss=getdata(trs,:x_abs); gnorms=getdata(trs,:gnorm); totalniters=sum(getdata(trs,:niter))
        fx0 = objvals[1]
        Mw, Mh = copy(Mw0), copy(Mh0)
        cparams.store_trace = false; cparams.show_trace=false
        rt = @elapsed scasolve!(X, W0, H0, d, Mw, Mh; penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
        fnprefix = "$(fprex)_rti$(rt1)$(scaparamstr)_its$(niters)_$(totalniters)_rt$(rt)"
        rt = initmethod == :isvd ? 2.95 : 2.5
    elseif method == :OCA
        Mw, Mh = copy(Mw0), copy(Mh0)
        stparams = StepParams(sd_group=sd_group, optimmethod=:optim_lbfgs, approx=true, α1=α1, α2=0,
                β1=β1, β2=0, reg=reg, l1l2ratio=l1l2ratio, useRelaxedL1=true, σ0=100*std(W0)^2, r=0.1)
        lsparams = LineSearchParams(method=ls_method, α0=1.0, maxiter=1000, show_lsplot=true,
                iterations_to_show=[1])
        cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                x_abstol=tol, successive_f_converge=2, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
                store_inner_trace=false, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:0)
        rt = @elapsed W1, objvals, trs, niters = ocasolve!(W0, Mw, d; stparams=stparams,
                                                                        lsparams=lsparams, cparams=cparams);
        fdts=getdata(trs,:sympen); fsps=getdata(trs,:sparseW)+getdata(trs,:sparseH)
        xdiffss=getdata(trs,:x_abs); gnorms=getdata(trs,:gnorm); totalniters=sum(getdata(trs,:niter))
        H1 = W1\X
        fnprefix = "$(fprex)_rti$(rt1)$(scaparamstr)_its$(niters)_$(totalniters)_rt$(rt)"
        fx0 = objvals[1]
    elseif method == :HALS
        W1, H1 = copy(Wcd0), copy(Hcd0)

        result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=halsmaxiter, α=α, l₁ratio=l₁ratio,
                    tol=tol, verbose=true, SCA_penmetric=penmetric, SCA_αw=SCA_αw, SCA_αh=SCA_αh), X, W1, H1)
        @show result.converged
        niters=result.niters; fdts=copy(result.objvalues); fsps=copy(result.sparsevalues)
        xdiffss=[]; gnorms=[]
        fx0 = result.objvalues[1]
        W1, H1 = copy(Wcd0), copy(Hcd0)
        rt = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=halsmaxiter, α=α, l₁ratio=l₁ratio,
                                        tol=tol), X, W1, H1)
        fnprefix = "$(fprex)_rti$(rt1cd)_a$(α)_it$(halsmaxiter)_rt$(rt)"
        rt = 1.76
    elseif method == :SPCA
        ridge_alpha = 0.01
        rt = @elapsed W1 = fit_transform!(SparsePCA(n_components=ncells,alpha=α,ridge_alpha=ridge_alpha,
                                                        max_iter=spcamaxiter,tol=tol,verbose=true),X) 
        H1 = W1\X
        fnprefix = "$(fprex)_a$(α)_ra$(ridge_alpha)_it$(spcamaxiter)_rt$(rt)"
    else
        error("Unsupported method $(method)")
    end
    save_figure && begin
        makepositive && flip2makepos!(W1,H1)
        normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
        W3, H3, _ = dataset == :fakecells ? match_order_to_gt(W1,H1,gtW,gtH) : (W2,H2,0)
        fnprefixgt = dataset == :fakecells ? fnprefix*"_gt" : fnprefix
        imsave_data(dataset,fnprefixgt, W3,H3,imgsz,100; signedcolors=dgwm(), saveH=false)
        imsave_data(dataset,fnprefixgt*"_bal", balanceWH!(W3,H3)...,imgsz,100; signedcolors=dgwm(), saveH=false)
        dataset == :fakecells && inhibitidx != 0  && plotW([gtH[:,inhibitidx]';H3[inhibitidx,:]']',
            fnprefix*"_H$(inhibitidx).png"; title="", xlbl="t", ylbl="H", legendstrs=["gtH","H"], issave=true,
            legendloc=1, axis=:auto)
        dataset == :cbclface && imsave_reconstruct(fnprefix,X,W3,H3,imgsz; index=100)
    end
    if method != :SPCA
        @show niters, length(fdts)
        iterrng = 0:niters; trng = iterrng/niters*rt
        fxss = fdts + fsps
    #    fxss ./= fx0
        uselogscale && (fxss=log10.(fxss); fdts=log10.(fdts); fsps=log10.(fsps); xdiffss=log10.(xdiffss); gnorms=log10.(gnorms))
        length(fxss) != 0 && begin
            axes[1].plot(iterrng,fxss); axes[2].plot(trng,fxss)
            if plot_dt_sp
                axes[1].plot(iterrng,fdts); axes[2].plot(trng,fdts)
                axes[1].plot(iterrng,fsps); axes[2].plot(trng,fsps)
            end
        end
        plotnum > 1 && length(xdiffss) != 0 && (axes[3].plot(iterrng,xdiffss); axes[4].plot(trng,xdiffss))
        plotnum > 1 && length(gnorms) != 0 && (axes[5].plot(iterrng,gnorms); axes[6].plot(trng,gnorms))
    end
end
for withlegend = [false, true]
    for ax in axes
        withlegend && ax.legend(legendstrs, fontsize = 12)
    end
    for (i,fig) in enumerate(figs)
        fig.savefig("$(datastr)_$(title)_$(postfixstrs[(i+1)÷2])_$(xaxisstrs[(i+1)%2+1])_rxl1_$(initmethod)_$(penmetric)_it$(maxiter)_$(withlegend).png")
    end
end
close("all")


α = 200
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α, α2=α,
    β1=0, β2=0, λ=0, reg=:WH1, l1l2ratio=0.5, M2power=1, useRelaxedL1=true,
    σ0=σ0, r=r, hfirst=true, processorder=:none, poweradjust=:none, useprecond=true)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter, show_lsplot=false,
    iterations_to_show=[15])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=1, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
    store_inner_trace=false, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:5)

Mw, Mh = copy(Mw0), copy(Mh0)
cparams.show_trace=true
rt = @elapsed W, H, objvals, traces, nitersum= scasolve!(X, W0, H0, d, Mw, Mh; penmetric=:SCA, stparams=stparams, lsparams=lsparams, cparams=cparams);
rts = 0
for i in 1:100
    Mw, Mh = copy(Mw0), copy(Mh0)
    cparams.store_trace = false; cparams.show_trace=false
    rt = @elapsed scasolve!(X, W0, H0, d, Mw, Mh; penmetric=:SCA, stparams=stparams, lsparams=lsparams, cparams=cparams);
    rts += rt
end
rts/100

rts = 0
for i in 1:100
    W1, H1 = copy(Wcd0), copy(Hcd0)
    rt = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=50, α=0.1, l₁ratio=1.0,
                                    tol=tol), X, W1, H1)
    rts += rt
end
rts/100



using BenchmarkTools

makepositive && flip2makepos!(W1,H1)
normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
W3, H3, _ = dataset == :fakecells ? match_order_to_gt(W1,H1,gtW,gtH) : (W2,H2,0)
fnprefixgt = dataset == :fakecells ? fnprefix*"_gt" : fnprefix
imsave_data(dataset,fnprefixgt, W3,H3,imgsz,100; signedcolors=dgwm(), saveH=false)

@btime MwMh=Mw*Mh
307.285 ns (1 allocation: 1.98 KiB)
@btime mul!(MwMh,Mw,Mh)
212.568 ns (0 allocations: 0 bytes)

D = Diagonal(rand(p)); B = rand(p,p)
@btime MwMh=Mw*Mh-D
595.025 ns (2 allocations: 3.97 KiB)
@btime begin MwMh.=D;mul!(MwMh,Mw,Mh,1,-1) end
1.031 μs (3 allocations: 64 bytes)
@btime begin mul!(MwMh,Mw,Mh); Ei0 = MwMh-D end
487.865 ns (1 allocation: 1.98 KiB)
@btime begin mul!(MwMh,Mw,Mh); Ei0 = MwMh-B end
490.130 ns (1 allocation: 1.98 KiB)
@btime begin mul!(MwMh,Mw,Mh); for i in 1:15 MwMh[i,i]-=D[i,i] end end
1.335 μs (45 allocations: 720 bytes)
@btime begin mul!(MwMh,Mw,Mh); MwMh -= D end
Error
@btime begin mul!(MwMh,Mw,Mh); view(MwMh,diagind(MwMh)) .-= diag(D) end
1.016 μs (7 allocations: 480 bytes)

@btime W=W0*Mw
7.716 μs (2 allocations: 93.80 KiB)
@btime mul!(W,W0,Mw)
4.329 μs (0 allocations: 0 bytes)

@btime H=Mh*H0
13.645 μs (2 allocations: 117.23 KiB)
@btime mul!(H,Mh,H0)
9.844 μs (0 allocations: 0 bytes)

@btime begin mul!(D,A,B); vec(D) end
97.935 ns (2 allocations: 80 bytes)
@btime vec(A*B)
133.347 ns (3 allocations: 272 bytes)

@btime norm(W0,2)^2
3.575 μs (2 allocations: 32 bytes)
@btime sum(abs2,W0)
1.296 μs (1 allocation: 16 bytes)

:W1
before(0.001856983), after(0.003608701)
:W1 useRelaxedL1
before(0.263810921), after(0.094833703)


stparams.useRelaxedL1 = false
Mw, Mh = copy(Mw0), copy(Mh0)
rt2 = @belapsed  W1, H1, objvals, trs, nitersum, fxss, xdiffss, ngss = scasolve!(X, W0, H0, d, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);

# datasets = [:fakecells,:cbclface,:orlface,:onoffnatural,:natural,:urban,:neurofinder]
# method = :sca, :fastsca, :oca, :fastoca or :cd
# regularization = Fast SCA(:W1M2,:M1,:M2,:W1,:W2), SCA(:W1M2,:W1Mn) or OCA()
# sd_group = :whole, :component, :column or :pixel
#           dataset,   SNRs, maxiter, inner_maxiter, βs, αs,    method,   nclsrng, reg, sd_group, useRelaxedL1, initpwradj
#ARGS =  ["[:neurofinder]","[60]","1","5000","[0]","[10]",":OCA","[40]",":W1",":component", "true", ":balance"]
#ARGS =  ["[:cbclface]","[60]","1","5000","[0]","[10]",":SCA","[40]",":WH1",":whole", "false", ":wh_normalize"]
#ARGS =  ["[:cbclface]","[60]","10","500","[0]","[10]",":SCA","[40]",":WH1",":whole", "false", ":wh_normalize"]
datasets = eval(Meta.parse(ARGS[1])); SNRs = eval(Meta.parse(ARGS[2]))
maxiter = eval(Meta.parse(ARGS[3])); inner_maxiter = eval(Meta.parse(ARGS[4]))
βs = eval(Meta.parse(ARGS[5])) # be careful not to have spaces in the argument
αs = eval(Meta.parse(ARGS[6])); αs = αs
method = eval(Meta.parse(ARGS[7])); nclsrng = eval(Meta.parse(ARGS[8]));
regularization = eval(Meta.parse(ARGS[9]))
sd_group = eval(Meta.parse(ARGS[10])); useRelaxedL1 = eval(Meta.parse(ARGS[11]))
initpwradj = eval(Meta.parse(ARGS[12]))

optimmethod = :optim_lbfgs; approx = true; M2power = 1
ls_method = :ls_BackTracking
rectify = :none # (rectify,β)=(:pinv,0) (cbyc_sd method)
order = regularization ∈ [:M1, :WH1,:W1M2] ? 1 : 2; regWonly = false
initmethod = :isvd; initfn = SCA.nndsvd2
useMedianFilter = false
pwradj = :none; tol=1e-6 # -1 means don't use convergence criterion
show_trace=false; show_inner_trace = false; savefigure = true
plotiterrng=1:0; plotinneriterrng=1:0
makepositive = false

@show datasets, SNRs, maxiter
@show βs, αs, method, αs, nclsrng
flush(stdout)
dataset = datasets[1]; SNR = SNRs[1]; β = βs[1]; β1 = β; β2 = β;
α = αs[1]; α1 = α; α2 = regWonly ? 0. : α; α = αs[1]; ncls = nclsrng[1]
for dataset in datasets
    rt1s=[]; rt2s=[]; mssds=[]; mssdHs=[]
    for SNR in SNRs # this can be change ncellss, factors
        # Load data
        X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure))
        SNRstr = dataset == :fakecells ? "_$(SNR)dB" : ""
        # Median filter
        if useMedianFilter
            rsimg = reshape(X,imgsz...,lengthT)
            rsimgm = mapwindow(median!, rsimg, (3,3,1));
            X = reshape(rsimgm,*(imgsz...),lengthT)
        end
        for ncls in nclsrng
            ncells = dataset ∈ [:fakecells, :neurofinder] ? ncls : ncells0
            medfilterstr = useMedianFilter ? "_med" : ""
            initdatastr = "_$(dataset)$(SNRstr)$(medfilterstr)_nc$(ncells)"
            if method ∈ [:SCA, :OCA]
                @show initmethod, initpwradj
                # Initialize
                rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,
                                                        poweradjust=initpwradj, initfn=initfn, α1=αs[1])
                initstr = "$(initmethod)"
                fprefix0 = "Init$(initdatastr)_$(initstr)_$(initpwradj)_rt"*@sprintf("%1.2f",rt1)
                Mw0, Mh0 = copy(Mw), copy(Mh)
                W1,H1 = copy(W0*Mw), copy(Mh*H0); normalizeW!(W1,H1)
                savefigure && imsave_data(dataset,fprefix0,W1,H1,imgsz,lengthT; saveH=false)
                if dataset == :fakecells
                    savefigure && imsave_data_gt(dataset,fprefix0,W1,H1,datadic["gtW"],datadic["gtH"],
                                                    imgsz,100; saveH=true)
                elseif dataset ∈ [:urban, :cbclface, :orlface]
                    # Xrecon = W3*H3[:,100]; @show Xrecon[1:10]
                    savefigure && imsave_reconstruct(fprefix0,X,W1,H1,imgsz; index=100)
                end
                fprefix1 = "$(initdatastr)_$(initstr)_$(initpwradj)"
                push!(rt1s,rt1)
                for β in βs
                    β1 = β; β2 = β
                    for α in αs
                        @show SNR, ncells, β, α
                        α1 = α; α2 = regWonly ? 0. : α
                        flush(stdout)
                        paramstr="_Reg$(regularization)_βw$(β1)_βh$(β2)_αw$(α1)_αh$(α2)"
                        fprefix2 = "$(method)$(fprefix1)_$(sd_group)_$(optimmethod)"*paramstr
                        Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
                        if method ∈ [:SCA] # Fast Symmetric Component Analysis
                            stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=approx,
                                    α1=α1, α2=α2, β1=β1, β2=β2, reg=regularization, order=order, M2power=M2power,
                                    useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
                                    poweradjust=pwradj, rectify=rectify)
                            lsparams = LineSearchParams(method=ls_method, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_lsplot=false,
                                    iterations_to_show=[15])
                            cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                                    x_abstol=tol, successive_f_converge=1, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
                                    show_trace=show_trace, show_inner_trace=show_inner_trace, plotiterrng=1plotiterrng,
                                    plotinneriterrng=plotinneriterrng)
                            rt2 = @elapsed W1, H1, objvals, trs, nitersum, fxss, xdiffss, ngss = scasolve!(W0, H0, d, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
                                        x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
                        elseif method ∈ [:OCA] # Fast Orthogonal Component Analysis
                            stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=approx, 
                                    α1=α1, α2=0, β1=β1, β2=0, reg=regularization, order=order, 
                                    useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=0.1)
                            lsparams = LineSearchParams(method=ls_method, c=0.5, α0=1.0, ρ=0.5, maxiter=50, show_lsplot=true,
                                    iterations_to_show=[1])
                            cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                                    x_abstol=tol, successive_f_converge=2, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
                                    show_trace=true, show_inner_trace=show_inner_trace, plotiterrng=plotiterrng, plotinneriterrng=plotinneriterrng)
                            rt2 = @elapsed W1, objvals, trs = ocasolve!(W0, Mw, d; stparams=stparams, lsparams=lsparams, cparams=cparams);
                            x_abss, xw_abss, xh_abss, f_xs, f_rel, orthogpen, regW, regH = getdata(trs)
                            H1 = W1\X
                        end
                        normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
                        # save W and H image data and plot
                        iter = length(trs)-1
                        fprefix3 = fprefix2*"_tol$(tol)_iter$(iter)_nits$(nitersum)_rt"*@sprintf("%1.2f",rt2)
                        if savefigure
                            makepositive && flip2makepos!(W3,H3)
                            imsave_data(dataset,fprefix3,W3,H3,imgsz,100; saveH=true)
                            length(f_xs) > 1 && begin
                                method ∈ [:OCA] ? plot_convergence(fprefix3,x_abss,f_xs; title="convergence (OCA)") : 
                                        plot_convergence(fprefix3,x_abss,xw_abss,xh_abss,f_xs; title="convergence (SCA)")
                            end
                            plotH_data(dataset,fprefix3,H3)
                            close("all")
                        end
                        push!(rt2s,rt2)
                        save(fprefix3*".jld","W0",W0,"H0",H0,"Mw",Mw,"Mh",Mh)

                        if dataset == :fakecells
                            # make W components be positive mostly
                            flip2makepos!(W3,H3)
                            gtW = datadic["gtW"]; gtH = datadic["gtH"]
                            # calculate MSD
                            mssd, ml, ssds = matchednssda(gtW,W3)
                            mssdH = ssdH(ml, gtH,H3')
                            push!(mssds,mssd); push!(mssdHs, mssdH)
                            if savefigure
                                imsave_data_gt(dataset,fprefix2*"_gt", W3,H3,datadic["gtW"],datadic["gtH"],imgsz,100; saveH=true)
                            end
                        elseif dataset ∈ [:urban, :cbclface, :orlface]
                            # Xrecon = W3*H3[:,100]; @show Xrecon[1:10]
                            savefigure && imsave_reconstruct(fprefix3,X,W3,H3,imgsz; index=100)
                        end
                    end
                end
            elseif method == :CD # :cd
                rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
                fprefix0 = "CDinit$(initdatastr)_normalize_rt"*@sprintf("%1.2f",rt1)
                Wcd0 = copy(Wcd); Hcd0 = copy(Hcd)
                W1,H1 = copy(Wcd), copy(Hcd); normalizeW!(W1,H1)
                savefigure && imsave_data(dataset,fprefix0,W1,H1,imgsz,lengthT; saveH=false)
                fprefix1 = "CD$(initdatastr)_normalize"
                push!(rt1s,rt1)
                Wcd0 = copy(Wcd); Hcd0 = copy(Hcd)
                for α in αs # best α = 0.1
                    @show α
                    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
                    usingNMF = true
                    if usingNMF
                        paramstr="_α$(α)"
                        fprefix2 = fprefix1*"_usingNMF"*paramstr
                        rt2 = @elapsed result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=0.5, tol=tol), X, Wcd, Hcd)
                        iter = result.niters; converged = result.converged
                        fprefix3 = fprefix2*"_tol$(tol)_iter$(iter)_$(converged)_rt"*@sprintf("%1.2f",rt2)
                    else
                        cdorder=1; cdpwradj=:none; cdα1=α; cdα2=0
                        stparams = StepParams(α1=cdα1, α2=cdα2, order=cdorder, hfirst=true, processorder=:none, poweradjust=cdpwradj,
                                            rectify=:truncate) 
                        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                                    x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
                        paramstr="_L$(cdorder)_αw$(cdα1)_αh$(cdα2)"
                        fprefix2 = fprefix1*"_$(cdpwradj)"*paramstr
                        rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
                        iter = length(trs)
                        fprefix3 = fprefix2*"_iter$(iter)_rt"*@sprintf("%1.2f",rt2)
                        x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
                        savefigure && plot_convergence(fprefix3,x_abss,xw_abss,xh_abss,f_xs; title="convergence (CD)")
                    end
                    normalizeW!(Wcd,Hcd); Wcd1,Hcd1 = sortWHslices(Wcd,Hcd)
                    if savefigure
                        imsave_data(dataset,fprefix3,Wcd1,Hcd1,imgsz,100; saveH=true)
                        plotH_data(dataset,fprefix3,Hcd1)
                        close("all")
                    end
                    push!(rt2s,rt2)
                    save("$(fprefix3).jld","Wcd",Wcd1,"Hcd",Hcd1)

                    if dataset == :fakecells
                        gtW = datadic["gtW"]; gtH = datadic["gtH"]
                        # calculate MSD
                        mssd, ml, ssds = matchednssd(gtW,Wcd1)
                        mssdH = ssdH(ml, gtH,Hcd1')
                        push!(mssds,mssd); push!(mssdHs, mssdH)
                        # reorder according to GT image
                        savefigure && imsave_data_gt(dataset,fprefix3*"_gt", W3,H3,datadic["gtW"],
                                            datadic["gtH"],imgsz,100; saveH=true)
                    elseif dataset ∈ [:urban, :cbclface, :orlface]
                        # Xrecon = Wcd1*Hcd1[:,100]; @show Xrecon[1:10]
                        savefigure && imsave_reconstruct(fprefix3,X,Wcd1,Hcd1,imgsz; index=100)
                    end
                end # for α
            end # if method
        end # for ncells
    end # for SNR
    if length(SNRs) > 1
        save("$(method)_SNR_vs_MSSD.jld","SNRs",SNRs,"rt1s",rt1s,"rt2s",rt2s,"mssds",mssds,"mssdHs",mssdHs)
    elseif length(nclsrng) > 1
        save("$(method)_NOC_vs_MSSD_$(SNR)dB.jld","nclsrng",nclsrng,"rt1s",rt1s,"rt2s",rt2s,"mssds",mssds,"mssdHs",mssdHs)
    end
#    rts = rt1s+rt2s
    # TODO : plot SNRs vs. rts
    # TODO : plot ncells vs. rts
    # TODO : plot factors vs. rts
end

dataset = :fakecells; SNR=0; initmethod=:isvd; initpwradj=:wh_normalize
useFilter = dataset ∈ [:neurofinder,:fakecells] ? true : false
filter=:meanT # :medT, :medS
filterstr = useFilter ? "_$(filter)" : ""
datastr = dataset == :fakecells ? "_fc$(SNR)dB" : "_$(dataset)"
save_figure = true; makepositive = false

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, bias=0.1, useCalciumT=true,
                                                        save_maxSNR_X=false, save_X=false);
ncells = ncells0

# XmedT = mapwindow(median!, X, (1,3)) # just for each row
# XmeanT = mapwindow(mean, X, (1,3)) # just for each row
# rsimg = reshape(X,imgsz...,lengthT)
# rsimgm = mapwindow(median!, rsimg, (3,3,1))
# XmedS = reshape(rsimgm,*(imgsz...),lengthT)
# X = vcat(X,XmedT,XmeanT,XmedS)
# options = (crf=23, preset="medium")
# clamp_level=1.0; X_max = maximum(abs,X)*clamp_level; Xnor = X./X_max;  X_clamped = clamp.(Xnor,0.,1.)
# Xuint8 = UInt8.(round.(map(clamp01nan, X_clamped)*255))
# VideoIO.save("$(datastr)_$(SNR)dB_allprepro_original_medT_meanT_medS.mp4", reshape.(eachcol(Xuint8),imgsz[1],4*imgsz[2]), framerate=30, encoder_options=options)

if useFilter
    if filter == :medT
        X = mapwindow(median!, X, (1,3)) # just for each row
    end
    if filter == :meanT
        X = mapwindow(mean, X, (1,3)) # just for each row
    end
    if filter == :medS
        rsimg = reshape(X,imgsz...,lengthT)
        rsimgm = mapwindow(median!, rsimg, (3,3,1))
        X = reshape(rsimgm,*(imgsz...),lengthT)
    end
end
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
Mw0, Mh0 = copy(Mw), copy(Mh)
rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
save_figure && begin
    normalizeW!(Wp,Hp); W2,H2 = sortWHslices(Wp,Hp)
    imsave_data(dataset,"$(initmethod)$(datastr)$(filterstr)_rti$(rt1)",W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
end

sd_group=:component; reg = :W1; M2power=1
poweradjust=:balance
#sd_group=:column; reg = :W1M2
l1l2ratio = 0.1
α = 500; β = 0; λ = 0.1
tol=1e-6; uselogscale=true; save_figure=true; optimmethod = :SB_newton
ls_method = :ls_BackTracking
scamaxiter = 300; halsmaxiter = 300; inner_maxiter = 20; ls_maxiter = 500
mtds = [:SCA, :HALS] # :HALS
isplotxandg = false; plotnum = isplotxandg ? 3 : 1

penaltystr = uselogscale ? "log10(penalty)" : "penalty"
if isplotxandg
    xdiffstr = uselogscale ? "log10(x difference)" : "x difference"
    maxgxstr = uselogscale ? "log10(maximum(g(x)))" : "maximum(g(x))"
end
figfx0, axisfx0 = plt.subplots(1,1, figsize=(5,4))  # SCA vs HALS fx iter
axisfx0.set_title("log(f(x))")
axisfx0.set_ylabel(penaltystr,fontsize = 12)
axisfx0.set_xlabel("iterations",fontsize = 12)
figfx0t, axisfx0t = plt.subplots(1,1, figsize=(5,4))  # SCA vs HALS fx time
axisfx0t.set_title("log(f(x))")
axisfx0t.set_ylabel(penaltystr,fontsize = 12)
axisfx0t.set_xlabel("time",fontsize = 12)
for method = mtds
    @show method
    figs = []; axes = []
    yaxisstrs = [("log(f(x))"),("x_diff"),("maximum(g(x))")]; xaxisstrs = ["iterations","time"]
    for yaxisstr in yaxisstrs[1:plotnum], xaxisstr in xaxisstrs
        fig, ax = plt.subplots(1,1, figsize=(5,4))
        ax.set_title(yaxisstr)
        ax.set_ylabel(penaltystr,fontsize = 12)
        ax.set_xlabel(xaxisstr,fontsize = 12)
        push!(figs,fig); push!(axes,ax)
        # axis.axis(axis) # [xamin, xamax, yamin, yamax]
    end
    # innerloopvec = [(:ls_BackTracking,10),(:ls_BackTracking,50),(:ls_BackTracking,100),(:ls_BackTracking,500),(:ls_BackTracking,1000),(:ls_BackTracking,5000)] #,
    # legendstrs = map(innerloop->"reg=$(innerloop[1]),lsmethod=$(innerloop[2])", innerloopvec)
    maxiterstr = method == :HALS ? "it$(halsmaxiter)" : "it$(scamaxiter)"
    fprex = "$(method)$(datastr)$(filterstr)"

    innerloopvec = [("scamaxiter","α"),(300,100)] #,
    legendstrs = map(innerloop->"$(innerloopvec[1][1])=$(innerloop[1]), $(innerloopvec[1][2])=$(innerloop[2])", innerloopvec[2:end])
    title = join([innerloopvec[1]...],"_vs_")
    for (idx,(scamaxiter, α)) = enumerate(innerloopvec[2:end])

    # innerloopvec = [("ls_maxiter"),(0),(500)] #,
    # legendstrs = map(innerloop->"$(innerloopvec[1][1])=$(innerloop[1]), ", innerloopvec[2:end])
    # title = ("withLS_vs_woLS")
    # for (idx,(ls_maxiter)) = enumerate(innerloopvec[2:end])

    # innerloopvec = [("sd_group","reg"),(:component,:W1M2)] #,(:whole,:WH1)
    # legendstrs = map(innerloop->"$(innerloopvec[1][1])=$(innerloop[1]), $(innerloopvec[1][2])=$(innerloop[2])", innerloopvec[2:end])
    # title = ("whole_vs_column")
    # for (idx,(sd_group, reg)) = enumerate(innerloopvec[2:end])

        rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
        Mw0, Mh0 = copy(Mw), copy(Mh)
        rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
        Wcd0, Hcd0 = copy(Wcd), copy(Hcd)

        scaparamstr = "_$(reg)_$(sd_group)_$(optimmethod)_$(maxiterstr)_lsit$(ls_maxiter)_α$(α)_β$(β)"
        β1 = β2 = β; α1 = α2 = α
        if method == :SCA
            Mw, Mh = copy(Mw0), copy(Mh0)
            stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true,
                α1=α1, α2=α2, β1=β1, β2=β2, λ=λ, reg=reg, l1l2ratio=l1l2ratio, order=1, M2power=M2power,
                useRelaxedL1=true, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
                poweradjust=poweradjust, rectify=:none)
            lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter, show_lsplot=false,
                iterations_to_show=[15])
            cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                x_abstol=tol, successive_f_converge=1, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=false,
                store_inner_trace=false, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:5)
            rt = @elapsed W1, H1, objvals, trs, nitersum, fxss, xdiffss, ngss = scasolve!(W0, H0, d, Mw, Mh; stparams=stparams,
                                                                            lsparams=lsparams, cparams=cparams);
            save_figure && begin
                normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
                makepositive && flip2makepos!(W2,H2)
                imsave_data(dataset,"$(fprex)_rti$(rt1)$(scaparamstr)_rt$(rt)",
                            W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
            end
            fx0 = objvals[1]
        elseif method == :OCA
            Mw, Mh = copy(Mw0), copy(Mh0)
            stparams = StepParams(sd_group=sd_group, optimmethod=:optim_lbfgs, approx=true,
                    α1=α1, α2=0, β1=β1, β2=0, reg=reg, l1l2ratio=l1l2ratio, order=1,
                    useRelaxedL1=true, σ0=100*std(W0)^2, r=0.1)
            lsparams = LineSearchParams(method=ls_method, α0=1.0, maxiter=1000, show_lsplot=true,
                    iterations_to_show=[1])
            cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                    x_abstol=tol, successive_f_converge=2, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
                    store_inner_trace=false, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:0)
            rt = @elapsed W1, objvals, trs, nitersum, fxss, xdiffss, ngss = ocasolve!(W0, Mw, d; stparams=stparams,
                                                                            lsparams=lsparams, cparams=cparams);
            save_figure && begin
                H1 = W1\X
                normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
                makepositive && flip2makepos!(W2,H2)
                imsave_data(dataset,"$(fprex)_rti$(rt1)$(scaparamstr)_rt$(rt)",
                            W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
            end
            fx0 = objvals[1]
        elseif method == :HALS
            α = 0.1
            Wcd, Hcd = copy(Wcd0), copy(Hcd0)
            rt = @elapsed result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=halsmaxiter, α=α, l₁ratio=0.5,
                                            tol=tol), X, Wcd, Hcd)
            @show result.converged
            nitersum=result.niters; fxss=copy(result.objvalues); xdiffss=copy(result.objvalues); ngss=copy(result.objvalues)
            save_figure && begin
                normalizeW!(Wcd,Hcd); Wcd1,Hcd1 = sortWHslices(Wcd,Hcd)
                imsave_data(dataset,"$(fprex)_rti$(rt1cd)_a$(α)_rt$(rt)",Wcd1,Hcd1,imgsz,100; saveH=false)
            end
            fx0 = result.objvalues[1]
        else
            error("Unsupported method $(method)")
        end
        @show nitersum, length(fxss)
        iterrng = 0:nitersum; trng = iterrng/nitersum*rt
        fxss ./= fx0
        uselogscale && (fxss=log10.(fxss); xdiffss=log10.(xdiffss); ngss=log10.(ngss))
        length(fxss) != 0 && begin
            axes[1].plot(iterrng,fxss); axes[2].plot(trng,fxss)
            idx == 1 && (axisfx0.plot(iterrng,fxss); axisfx0t.plot(trng,fxss))
        end
        plotnum > 1 && length(xdiffss) != 0 && (axes[3].plot(iterrng,xdiffss); axes[4].plot(trng,xdiffss))
        plotnum > 1 && length(ngss) != 0 && (axes[5].plot(iterrng,ngss); axes[6].plot(trng,ngss))
    end
    for ax in axes
        ax.legend(legendstrs, fontsize = 12)
    end
    for (i,fig) in enumerate(figs)
        fig.savefig("$(fprex)_$(title)_$(maxiterstr)_$(yaxisstrs[(i+1)÷2])_$(xaxisstrs[(i+1)%2+1]).png")
    end
end
axisfx0.legend(String.(mtds), fontsize = 12)
axisfx0t.legend(String.(mtds), fontsize = 12)
figfx0.savefig("SCA_vs_HALS$(datastr)$(filterstr)_its$(scamaxiter)_ith$(halsmaxiter)_iter.png")
figfx0t.savefig("SCA_vs_HALS$(datastr)$(filterstr)_its$(scamaxiter)_ith$(halsmaxiter)_time.png")
close("all")




ncells = 800
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,
poweradjust=initpwradj, initfn=initfn, α1=αs[1])
initstr = "$(initmethod)"
fprefix0 = "Init$(initdatastr)_$(initstr)_$(initpwradj)_rt"*@sprintf("%1.2f",rt1)
Mw0, Mh0 = copy(Mw), copy(Mh)
W1,H1 = copy(W0*Mw), copy(Mh*H0); normalizeW!(W1,H1)
imsaveW("SNR-20.png",W1,imgsz;gridcols=40)

stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=approx,
α1=α1, α2=α2, β1=β1, β2=β2, reg=regularization, order=order, M2power=M2power,
useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
poweradjust=pwradj, rectify=rectify)

D = Diagonal(d); fw=gh=Matrix(1.0I,0,0)
normw2 = norm(W0)^2; normh2 = norm(H0)^2
normwp = norm(W0,stparams.order)^stparams.order; normhp = norm(H0,stparams.order)^stparams.order
sndpow = sqrt(norm(d))^stparams.M2power
(βw, βh) = (stparams.β1/normw2, stparams.β2/normh2)
Msparse = stparams.poweradjust ∈ [:M1,:M2]
(αw, αh) = Msparse ?                     (stparams.α1, stparams.α2) :
           stparams.poweradjust==:W1M2 ? (stparams.α1/normwp/sndpow, stparams.α2/normhp/sndpow) :
                                         (stparams.α1/normwp, stparams.α2/normhp)

SCA.penaltyMwMh(Mw,Mh,W0,H0,D,fw,gh,βw,βh,αw,αh,0,Msparse,stparams.order; reg=stparams.reg,
                M2power=stparams.M2power, useRelaxedL1=stparams.useRelaxedL1)



#===== QP solver ===============#
using OSQP

# minimize        0.5 x' P x + q' x
# subject to      l <= A x <= u

function QPsolve(P,q,A,u,l)
    options = Dict(
        :verbose => false,
        :eps_abs => 1e-09,
        :eps_rel => 1e-09,
        :check_termination => 1,
        :polish => false,
        :max_iter => 4000,
        :rho => 0.1,
        :adaptive_rho => false,
        :warm_start => true,
    )

    model = OSQP.Model()
    OSQP.setup!(
        model; P =P, q = q, A = A, l = l, u = u, options...,
    )
    results = OSQP.solve!(model)
    results.x, results.info.obj_val # results.y (parameter for dual problem)
end

P = [11.0 0.0; 0.0 0.0]
q = [3.0; 4]
A = [-1 0; 0 -1; -1 -3; 2 5; 3 4]
u = [0.0; 0.0; -15; 100; 80]
l = -Inf * ones(length(u))

QPsolve(sparse(P),q,sparse(A),u,l)

# test with SCA
W = W0*Mw; b = vec(W)
m,p = size(W0)
Aw = zeros(m*p,p^2)
SCA.FM2A!(W0,Aw)
P = 2*sparse(Aw'Aw); q = -2*Aw'b
A = sparse(Aw); l = zeros(m*p); u = Inf*ones(m*p)
rt = @elapsed x, pen = QPsolve(P,q,A,u,l)
Mwsol = reshape(x,p,p); W1 = W0*Mwsol; H1 = W1\X
normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
imsave_data(dataset,"test_rt$(rt)",W2,H2,imgsz,100; saveH=false)

#====== SPCA =================#
using ScikitLearn

dataset = :neurofinder; SNR=-10
useMedianFilter = true; medT = true; medS = false
datastr = dataset == :fakecells ? "_fc$(SNR)dB" : "_$(dataset)"

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, useCalciumT=true, save_maxSNR_X=false);
ncells = ncells0

@sk_import decomposition: PCA
rtpca = @elapsed resultpca = fit_transform!(PCA(n_components=ncells,tol=0),X) 
W0 = copy(reslutpca); H0 = W0\X
normalizeW!(W0,H0); W2,H2 = sortWHslices(W0,H0)
imsave_data(dataset,"pca_rt$(rtpca)",W2,H2,imgsz,100; saveH=false)

@sk_import decomposition: SparsePCA
alpha = 0.5; ridge_alpha=0.01; max_iter=500; tol=1e-7
for ridge_alpha in [0,0.01,0.05,0.1,0.5,1.0,5]
    rtspca = @elapsed resultspca = fit_transform!(SparsePCA(n_components=ncells,alpha=alpha,ridge_alpha=ridge_alpha,max_iter=max_iter,tol=tol,verbose=true),X) 
    W0 = copy(resultspca); H0 = W0\X
    normalizeW!(W0,H0); W2,H2 = sortWHslices(W0,H0)
    fprefix = "SPCA_$(datastr)_a$(alpha)_ra$(ridge_alpha)_tol$(tol)_miter$(max_iter)_rt$(rtspca)"
    imsave_data(dataset,fprefix,W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
    makepositive && flip2makepos!(W2,H2)
    imsave_data_gt(dataset,fprefix*"_gt", W2,H2,gtW,gtH,imgsz,100; saveH=true)
    W3, H3, _ = match_order_to_gt(W2,H2,gtW,gtH)
    inhibitidx != 0  && plotW([gtH[:,inhibitidx]';H3[inhibitidx,:]']', fprefix*"_inhibitH.png"; title="", xlbl="t", ylbl="H", legendstrs=["gtH","H"],
            issave=true, legendloc=1, axis=:auto)
end
# Parameters:
    # n_componentsint, default=None
    # Number of sparse atoms to extract. If None, then n_components is set to n_features.

    # alphafloat, default=1
    # Sparsity controlling parameter. Higher values lead to sparser components.

    # ridge_alphafloat, default=0.01
    # Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method.

    # max_iterint, default=1000
    # Maximum number of iterations to perform.

    # tolfloat, default=1e-8
    # Tolerance for the stopping condition.

    # method{‘lars’, ‘cd’}, default=’lars’
    # Method to be used for optimization. lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso). Lars will be faster if the estimated components are sparse.

    # n_jobsint, default=None
    # Number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

    # U_initndarray of shape (n_samples, n_components), default=None
    # Initial values for the loadings for warm restart scenarios. Only used if U_init and V_init are not None.

    # V_initndarray of shape (n_components, n_features), default=None
    # Initial values for the components for warm restart scenarios. Only used if U_init and V_init are not None.

    # verboseint or bool, default=False
    # Controls the verbosity; the higher, the more messages. Defaults to 0.

    # random_stateint, RandomState instance or None, default=None
    # Used during dictionary learning. Pass an int for reproducible results across multiple function calls. See Glossary.

# Attributes:
    # components_ndarray of shape (n_components, n_features)
    # Sparse components extracted from the data.

    # error_ndarray
    # Vector of errors at each iteration.

    # n_components_int
    # Estimated number of components.

    # New in version 0.23.

    # n_iter_int
    # Number of iterations run.

    # mean_ndarray of shape (n_features,)
    # Per-feature empirical mean, estimated from the training set. Equal to X.mean(axis=0).

    # n_features_in_int
    # Number of features seen during fit.

    # New in version 0.24.

    # feature_names_in_ndarray of shape (n_features_in_,)
    # Names of features seen during fit. Defined only when X has feature names that are all strings.

    # New in version 1.0.
using ForwardDiff, BenchmarkTools, Test

for (m,n,p) in [(800,1000,15),(361, 2429, 49)]
    @show p
    p = p; m = 500; n = 1000#; k=rand(1:p)
    Mw = rand(p,p).-0.5; Mh = rand(p,p).-0.5; D = Diagonal(rand(p))
    W0 = rand(m,p).-0.5; H0 = rand(p,n).-0.5; W=W0*Mw; H=Mh*H0
    fw = gh = Matrix(1.0I,p,p); x0 = vec(Mw)
    αw=rand(); αh=rand(); βw=100*rand(); βh=100*rand(); σ2=rand(); l1l2ratio=0.1; M2power=1
    
    # Relaxed W1
    T = Float64
    w0tw0s = Array{T}(undef,m,p,p); h0h0ts = Array{T}(undef,n,p,p)
    for i = 1:m
        w0tw0s[i,:,:] .= W0[i,:]*W0[i,:]'
    end
    for i = 1:n
        h0h0ts[i,:,:] .= H0[:,i]*H0[:,i]'
    end
    w0tw0s = reshape(w0tw0s,m,p^2); h0h0ts = reshape(h0h0ts,n,p^2)
    W02 = W0.^2; H02 = H0.^2
    gEiT=0; gEwsT=0; gEhsT=0; gdEiT=0; gdEswT=0; gdEshT=0
    for repeat = 1:1000
        tmpmat = Matrix{T}(undef,p,p) 
        gEiT += @elapsed gradEi, Ei0 = SCA.gradEi(tmpmat,Mw,Mh,D; approx=true)
        gEwsT += @elapsed grad, E0 = SCA.gradRelxEs1(tmpmat,W0, W, σ2; approx=true)
        gEhsT += @elapsed grad, E0 = SCA.gradRelxEs1(tmpmat,H0', H', σ2; approx=true)
        gdEiT += @elapsed dHessEi = SCA.diagHessEi(Mw,Mh)
        gdEswT += @elapsed SCA.diagHessRelxEs1(W02,W,σ2)
        gdEshT += @elapsed SCA.diagHessRelxEs1(H02',H',σ2)
    end
    halssummnp = m*n*p+(m+n)*p^2; summnp = p^3 + (m+n)*p^2; sum = gEiT + gEwsT + gEhsT
    @show halssummnp, summnp, p^3, m*(p^2+p), n*(p^2+p), p^2, m*(p^2+p), n*(p^2+p)
    @show sum, gEiT, gEwsT, gEhsT, gdEiT, gdEswT, gdEshT
end

using ForwardDiff
p=5
W0 = rand(10,p).-0.5
Mw = rand(p,p); Mh=rand(5,5); d=rand(5)
Ei(Mw,Mh,d) = sum(abs2,Mw*Mh-Diagonal(d))
Es(W0,Mw) = sum(sqrt.((W0*Mw).^2 .+0.1))
E(W0,Mw,Mh,d; alpha=1) = Ei(Mw,Mh,d) + alpha*Es(W0,Mw)
E(x) = (Mw = reshape(x[1:p^2],p,p)'; Mh = reshape(x[p^2+1:2p^2],p,p); E(W0,Mw,Mh,d; alpha=1))
x0 = vcat(vec(Mw'),vec(Mh))
s = ForwardDiff.gradient(E,x0)
function Eialpha_coeffs(SwSh,SwMh,MwSh,x,step,MwMhmD,p)
    Mw = reshape(x[1:p^2],p,p)';      Sw = reshape(step[1:p^2],p,p)'
    Mh = reshape(x[p^2+1:2p^2],p,p);  Sh = reshape(step[p^2+1:2p^2],p,p)
    mul!(SwSh,Sw,Sh); mul!(SwMh,Sw,Mh); mul!(MwSh,Mw,Sh); SwMhmMwSh = SwMh-MwSh
    sum(abs2,SwSh), 2sum(SwSh.*SwMhmMwSh), 2sum(SwSh.*MwMhmD)+sum(abs2,SwMhmMwSh), 2sum(SwMhmMwSh*MwMhmD), sum(abs2,MwMhmD)
end

mutable struct RepeatedView{T}<:AbstractVector
    sv::SubArray{T,1}
    repeat::Int
end
function repeated_view(v::AbstracVector{T}, indices, repeat) where T
   return RepeatedView{T}(SubArray(v, indices),repeat)
end
function Base.getindex(A::RepeatedView, inds...)
    length(inds)>1 && error("only indexing for Vector is supported")
    ind = inds[1]%
    checkbounds(A, )  # Check if indices are within bounds
    linear_index = linearize(indices, strides(A))
    return A[linear_index]
end

# Custom preconditioning
mutable struct RepeatedDiagonal
    diagw
    diagh
end
 
function ldiv!(pgr, P::RepeatedDiagonal, gr)
    p=length(gr)
    idxw(i)=(i-1)*p+1:i*p
    foreach(i -> copy!(pgr[idxw(i)], gr[idxw(i)]./P.diagw), 1:p)
    idxh(i)=(i-1)*p+1+p^2:i*p+p^2
    foreach(i -> copy!(pgr[idxh(i)], gr[idxh(i)]./P.diagh), 1:p)
end 
function LinearAlgebra.dot(x, P, y)
    p=length(y); s=zero(eltype(x))
    idxw(i)=(i-1)*p+1:i*p
    foreach(i -> s += dot(x[idx], y[idx]./P.diagw), 1:p)
    idxh(i)=(i-1)*p+1+p^2:i*p+p^2
    foreach(i -> s += dot(x[idx], y[idx]./P.diagh), 1:p)
    s
end
