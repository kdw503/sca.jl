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



# datasets = [:fakecells,:cbclface,:orlface,:onoffnatural,:natural,:urban,:neurofinder]
# method = :sca, :fastsca, :oca, :fastoca or :cd
# objective = :normal, :pw, :weighted or :weighted2
# regularization = Fast SCA(:W1M2,:M1,:M2,:W1,:W2), SCA(:W1M2,:W1Mn) or OCA()
# sd_group = :whole, :component, :column or :pixel
#           dataset,   SNRs, initmethod, maxiter, αs, βs, optimmethod,   reg, sd_group, useRelaxedL1
#ARGS =  ["[:fakecells]","[0]",":isvd", "500","[100]","[0]",":SB_lbfgs", ":W1M2",":col", "false"]
datasets = eval(Meta.parse(ARGS[1])); SNRs = eval(Meta.parse(ARGS[2]))
initmethod = eval(Meta.parse(ARGS[3])); maxiter = eval(Meta.parse(ARGS[4]));
αs = eval(Meta.parse(ARGS[5])); βs = eval(Meta.parse(ARGS[6]))
optimmethod = eval(Meta.parse(ARGS[7])); reg = eval(Meta.parse(ARGS[8]))
sd_group = eval(Meta.parse(ARGS[9])); useRelaxedL1 = eval(Meta.parse(ARGS[10]))

dataset=datasets[1]; SNR = SNRs[1]; initpwradj=:wh_normalize
useFilter = dataset ∈ [:neurofinder,:fakecells] ? true : false
filter=:meanT # :medT, :medS
filterstr = useFilter ? "_$(filter)" : ""
datastr = dataset == :fakecells ? "_fc$(SNR)dB" : "_$(dataset)"
save_figure = true; makepositive = false

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, bias=0.1, useCalciumT=true,
                                                        save_maxSNR_X=false, save_X=false);
ncells = ncells0

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

method = :SCA; objective = :normal
M2power=1; l1l2ratio = 0.1
r=0.1 # decaying rate for relaxed L1
α = αs[1]; β = βs[1]; λ = 0.1
tol=-1; uselogscale=true; save_figure=true
ls_method = :ls_BackTracking
scamaxiter = maxiter; halsmaxiter = maxiter; inner_maxiter = 50; ls_maxiter = 500
isplotxandg = false; plotnum = isplotxandg ? 3 : 1

penaltystr = uselogscale ? "log10(penalty)" : "penalty"
if isplotxandg
    xdiffstr = uselogscale ? "log10(x difference)" : "x difference"
    maxgxstr = uselogscale ? "log10(maximum(g(x)))" : "maximum(g(x))"
end
figs = []; axes = []
yaxisstrs = [("log(f(x)/f(x0))"),("x_diff"),("maximum(g(x))")]; xaxisstrs = ["iterations","time"]
postfixstrs = [("log(f(x))"),("x_diff"),("maximum(g(x))")]
for yaxisstr in yaxisstrs[1:plotnum], xaxisstr in xaxisstrs
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.set_title(yaxisstr)
    ax.set_ylabel(penaltystr,fontsize = 12)
    ax.set_xlabel(xaxisstr,fontsize = 12)
    push!(figs,fig); push!(axes,ax)
    # axis.axis(axis) # [xamin, xamax, yamin, yamax]
end
paramsvec = [("method","α","optimmethod"),(:SCA,5000,optimmethod),(:SCA,10000,optimmethod),(:SCA,50000,optimmethod),
                (:SCA,100000,optimmethod)]
legendstrs = map(param->param[1]==:HALS ? "$(paramsvec[1][1])=$(param[1])" :
                                          "$(paramsvec[1][1])=$(param[1]), $(paramsvec[1][2])=$(param[2])", paramsvec[2:end])
title = join([paramsvec[1]...],"_vs_")
for (idx,(method, α, optimmethod)) = enumerate(paramsvec[2:end])
    poweradjust = reg == :W1 ? :balance : :none
    @show method, poweradjust
    # paramsvec = [(:ls_BackTracking,10),(:ls_BackTracking,50),(:ls_BackTracking,100),(:ls_BackTracking,500),(:ls_BackTracking,1000),(:ls_BackTracking,5000)] #,
    # legendstrs = map(innerloop->"reg=$(innerloop[1]),lsmethod=$(innerloop[2])", paramsvec)
    maxiterstr = method == :HALS ? "it$(halsmaxiter)" : "it$(scamaxiter)"
    initmethodstr = method == :HALS ? "_nndsvd" : "_$(initmethod)"
    fprex = "$(method)$(datastr)$(filterstr)$(initmethodstr)"

    # rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    # Mw0, Mh0 = copy(Mw), copy(Mh)
    # rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
    # Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
    optimmethodstr = (optimmethod ∈ [:SB_newton, :SB_lbfgs, :SB_cg] || !useRelaxedL1) ? "$optimmethod" : "$(optimmethod)_dr$(r)"
    scaparamstr = "_$(reg)_$(sd_group)_$(optimmethodstr)_$(maxiterstr)_lsit$(ls_maxiter)_α$(α)_β$(β)_λ$(λ)"
    β1 = β2 = β; α1 = α2 = α
    if method == :SCA
        Mw, Mh = copy(Mw0), copy(Mh0)
        stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true,
            α1=α1, α2=α2, β1=β1, β2=β2, λ=λ, reg=reg, l1l2ratio=l1l2ratio, order=1, M2power=M2power,
            useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=r, hfirst=true, processorder=:none,
            poweradjust=poweradjust, rectify=:none, objective=objective)
        lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter, show_lsplot=false,
            iterations_to_show=[15])
        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=1, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=false,
            store_inner_trace=false, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:5)
        rt = @elapsed W1, H1, objvals, trs, nitersum, fxss, xdiffss, ngss = scasolve!(W0, H0, d, Mw, Mh; stparams=stparams,
                                                                        lsparams=lsparams, cparams=cparams);
        save_figure && begin
            balanceWH!(W1,H1); W2,H2 = sortWHslices(W1,H1)
            makepositive && flip2makepos!(W2,H2)
            imsave_data(dataset,"$(fprex)_rti$(rt1)$(scaparamstr)_rt$(rt)",
                        W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
        end
        fx0 = objvals[1]
    elseif method == :OCA
        Mw, Mh = copy(Mw0), copy(Mh0)
        stparams = StepParams(sd_group=sd_group, optimmethod=:optim_lbfgs, approx=true,
                α1=α1, α2=0, β1=β1, β2=0, reg=reg, l1l2ratio=l1l2ratio, order=1,
                useRelaxedL1=true, σ0=100*std(W0)^2, r=0.1, objective=:normal)
        lsparams = LineSearchParams(method=ls_method, α0=1.0, maxiter=1000, show_lsplot=true,
                iterations_to_show=[1])
        cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                x_abstol=tol, successive_f_converge=2, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
                store_inner_trace=false, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:0)
        rt = @elapsed W1, objvals, trs, nitersum, fxss, xdiffss, ngss = ocasolve!(W0, Mw, d; stparams=stparams,
                                                                        lsparams=lsparams, cparams=cparams);
        save_figure && begin
            H1 = W1\X
            balanceWH!(W1,H1); W2,H2 = sortWHslices(W1,H1)
            makepositive && flip2makepos!(W2,H2)
            imsave_data(dataset,"$(fprex)_rti$(rt1)$(scaparamstr)_rt$(rt)",
                        W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
        end
        fx0 = objvals[1]
    elseif method == :HALS
        Wcd, Hcd = copy(Wcd0), copy(Hcd0)
        rt = @elapsed result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=halsmaxiter, α=α, l₁ratio=0.5,
                                        tol=tol), X, Wcd, Hcd)
        @show result.converged
        nitersum=result.niters; fxss=copy(result.objvalues); xdiffss=copy(result.objvalues); ngss=copy(result.objvalues)
        save_figure && begin
            balanceWH!(Wcd,Hcd); Wcd1,Hcd1 = sortWHslices(Wcd,Hcd)
            imsave_data(dataset,"$(fprex)_rti$(rt1cd)_a$(α)_it$(halsmaxiter)_rt$(rt)",Wcd1,Hcd1,imgsz,100; saveH=false)
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
    end
    plotnum > 1 && length(xdiffss) != 0 && (axes[3].plot(iterrng,xdiffss); axes[4].plot(trng,xdiffss))
    plotnum > 1 && length(ngss) != 0 && (axes[5].plot(iterrng,ngss); axes[6].plot(trng,ngss))
end
for withlegend = [false, true]
    for ax in axes
        withlegend && ax.legend(legendstrs, fontsize = 12)
    end
    for (i,fig) in enumerate(figs)
        fig.savefig("$(title)_$(initmethod)_$(reg)_$(sd_group)_$(optimmethod)_$(postfixstrs[(i+1)÷2])_$(xaxisstrs[(i+1)%2+1])_rxl1_it$(maxiter)_$(withlegend).png")
    end
end
close("all")
