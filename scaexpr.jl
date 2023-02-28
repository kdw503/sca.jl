"""
TODO list
- gradient and hessian of cbyc :W1 sparseness should be documented
- power weighted version-2 need to be implemented
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

# datasets = [:fakecells,:cbclface,:orlface,:onoffnatural,:natural,:urban,:neurofinder]
# method = :sca, :fastsca, :oca, :fastoca or :cd
# objective = :normal, :pw, :weighted or :weighted2
# regularization = Fast SCA(:W1M2,:M1,:M2,:W1,:W2), SCA(:W1M2,:W1Mn) or OCA()
# sd_group = :whole, :component, :column or :pixel
#           dataset,   SNRs, maxiter, inner_maxiter, λs, βs,    method,   nclsrng, objective, reg, sd_group, useRelaxedL1, initpwradj
#ARGS =  ["[:neurofinder]","[60]","1","5000","[0]","[10]",":OCA","[40]",":normal",":W1",":component", "true", ":balance"]
#ARGS =  ["[:cbclface]","[60]","1","5000","[0]","[10]",":SCA","[40]",":normal",":WH1",":whole", "false", ":wh_normalize"]
#ARGS =  ["[:cbclface]","[60]","10","500","[0]","[10]",":SCA","[40]",":normal",":WH1",":whole", "false", ":wh_normalize"]
datasets = eval(Meta.parse(ARGS[1])); SNRs = eval(Meta.parse(ARGS[2]))
maxiter = eval(Meta.parse(ARGS[3])); inner_maxiter = eval(Meta.parse(ARGS[4]))
λs = eval(Meta.parse(ARGS[5])) # be careful not to have spaces in the argument
βs = eval(Meta.parse(ARGS[6])); αs = βs
method = eval(Meta.parse(ARGS[7])); nclsrng = eval(Meta.parse(ARGS[8]));
objective = eval(Meta.parse(ARGS[9])); regularization = eval(Meta.parse(ARGS[10]))
sd_group = eval(Meta.parse(ARGS[11])); useRelaxedL1 = eval(Meta.parse(ARGS[12]))
initpwradj = eval(Meta.parse(ARGS[13]))

optimmethod = :optim_lbfgs; approx = true; M2power = 1
ls_method = :ls_BackTracking
rectify = :none # (rectify,λ)=(:pinv,0) (cbyc_sd method)
order = regularization ∈ [:M1, :WH1,:W1M2] ? 1 : 2; regWonly = false
initmethod = :isvd; initfn = SCA.nndsvd2
useMedianFilter = false
pwradj = :none; tol=1e-6 # -1 means don't use convergence criterion
show_trace=true; show_inner_trace = true; savefigure = true
plotiterrng=1:1; plotinneriterrng=1:1

@show datasets, SNRs, maxiter
@show λs, βs, method, αs, nclsrng
flush(stdout)
dataset = datasets[1]; SNR = SNRs[1]; λ = λs[1]; λ1 = λ; λ2 = λ;
β = βs[1]; β1 = β; β2 = regWonly ? 0. : β; α = αs[1]; ncls = nclsrng[1]
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
                                                        poweradjust=initpwradj, initfn=initfn, β1=βs[1])
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
                for λ in λs
                    λ1 = λ; λ2 = λ
                    for β in βs
                        @show SNR, ncells, λ, β
                        β1 = β; β2 = regWonly ? 0. : β
                        flush(stdout)
                        paramstr="_Obj$(objective)_Reg$(regularization)_λw$(λ1)_λh$(λ2)_βw$(β1)_βh$(β2)"
                        fprefix2 = "$(method)$(fprefix1)_$(sd_group)_$(optimmethod)"*paramstr
                        Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
                        if method ∈ [:SCA] # Fast Symmetric Component Analysis
                            stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=approx,
                                    β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=regularization, order=order, M2power=M2power,
                                    useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
                                    poweradjust=pwradj, rectify=rectify, objective=objective)
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
                                    β1=β1, β2=0, λ1=λ1, λ2=0, reg=regularization, order=order, 
                                    useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=0.1, objective=objective)
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
                for α in βs # best α = 0.1
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
                        cdorder=1; cdpwradj=:none; cdβ1=α; cdβ2=0
                        stparams = StepParams(β1=cdβ1, β2=cdβ2, order=cdorder, hfirst=true, processorder=:none, poweradjust=cdpwradj,
                                            rectify=:truncate) 
                        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                                    x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
                        paramstr="_L$(cdorder)_βw$(cdβ1)_βh$(cdβ2)"
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

dataset = :fakecells; SNR=-20; initmethod=:isvd; initpwradj=:wh_normalize
useMedianFilter = true; medT = true; medS = false
medstr = useMedianFilter ? medT ? medS ? "_medT_medS" : "_medT" : "medT" : ""
datastr = dataset == :fakecells ? "_fc$(SNR)dB" : "_$(dataset)"

λ1 = λ2 = 1000; β1 = β2 = 1; α = 0.1
X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, useCalciumT=true, save_maxSNR_X=true);
ncells = ncells0
if useMedianFilter
    if medT
        X = mapwindow(median!, X, (1,3)) # just for each row
    end
    if medS
        rsimg = reshape(X,imgsz...,lengthT)
        rsimgm = mapwindow(median!, rsimg, (3,3,1))
        X = reshape(rsimgm,*(imgsz...),lengthT)
    end
end
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
Mw0, Mh0 = copy(Mw), copy(Mh)
rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
Wcd0, Hcd0 = copy(Wcd), copy(Hcd)

tol=1e-7; uselogscale=true; save_figure=true
methods = [:OCA, :SCA]
maxitervec = [(10,500)]
penaltystr = uselogscale ? "log10(penalty)" : "penalty"
xdiffstr = uselogscale ? "log10(x difference)" : "x difference"
maxgxstr = uselogscale ? "log10(maximum(g(x)))" : "maximum(g(x))"
legendstrs = map(maxiters->"maxiter=$(maxiters[1]),inner_maxiter=$(maxiters[2])", maxitervec)
figfx0, axisfx0 = plt.subplots(1,1, figsize=(5,4))
axisfx0.set_title("f(x)")
axisfx0.set_ylabel(penaltystr,fontsize = 12)
axisfx0.set_xlabel("iterations",fontsize = 12)
figfx0t, axisfx0t = plt.subplots(1,1, figsize=(5,4))
axisfx0t.set_title("f(x)")
axisfx0t.set_ylabel(penaltystr,fontsize = 12)
axisfx0t.set_xlabel("time",fontsize = 12)
for method = methods
    @show optimmethod
    figfx, axisfx = plt.subplots(1,1, figsize=(5,4))
    axisfx.set_title("f(x)")
    axisfx.set_ylabel(penaltystr,fontsize = 12)
    figfxt, axisfxt = plt.subplots(1,1, figsize=(5,4))
    axisfxt.set_title("f(x)")
    axisfxt.set_ylabel(penaltystr,fontsize = 12)
    figxdiff, axisxdiff = plt.subplots(1,1, figsize=(5,4))
    axisxdiff.set_title("x_diff")
    axisxdiff.set_ylabel(xdiffstr,fontsize = 12)
    figxdifft, axisxdifft = plt.subplots(1,1, figsize=(5,4))
    axisxdifft.set_title("x_diff")
    axisxdifft.set_ylabel(xdiffstr,fontsize = 12)
    figngx, axisngx = plt.subplots(1,1, figsize=(5,4))
    axisngx.set_title("maximum(g(x))")
    axisngx.set_ylabel(maxgxstr,fontsize = 12)
    figngxt, axisngxt = plt.subplots(1,1, figsize=(5,4))
    axisngxt.set_title("maximum(g(x))")
    axisngxt.set_ylabel(maxgxstr,fontsize = 12)
    for ax in [axisfx,axisxdiff,axisngx]
        ax.set_xlabel("iterations",fontsize = 12)
    end
    for ax in [axisfxt,axisxdifft,axisngxt]
        ax.set_xlabel("time",fontsize = 12)
    end
    # ax.axis(axis) # [xamin, xamax, yamin, yamax]
    for (idx,(maxiter,inner_maxiter)) = enumerate(maxitervec)
        @show maxiter, inner_maxiter
        if method == :SCA
            Mw, Mh = copy(Mw0), copy(Mh0)
            stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=approx,
                β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=regularization, order=order, M2power=M2power,
                useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
                poweradjust=pwradj, rectify=rectify, objective=objective)
            lsparams = LineSearchParams(method=ls_method, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_lsplot=false,
                iterations_to_show=[15])
            cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                x_abstol=tol, successive_f_converge=1, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
                store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:0)
            rt = @elapsed W1, H1, objvals, trs, nitersum, fxss, xdiffss, ngss = scasolve!(W0, H0, d, Mw, Mh; stparams=stparams,
                                                                            lsparams=lsparams, cparams=cparams);
            save_figure && begin
                normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
                imsave_data(dataset,"SCA$(datastr)$(medstr)_SNR$(SNR)_rti$(rt1)_bw$(β1)_lw$(λ1)_rt$(rt)",W2,H2,imgsz,100; saveH=false)
            end
        elseif method == :OCA
            Mw, Mh = copy(Mw0), copy(Mh0)
            stparams = StepParams(sd_group=:component, optimmethod=optimmethod, approx=approx,
                    β1=β1, β2=0, λ1=λ1, λ2=0, reg=regularization, order=order,
                    useRelaxedL1=true, σ0=100*std(W0)^2, r=0.1, objective=objective)
            lsparams = LineSearchParams(method=ls_method, c=0.5, α0=1.0, ρ=0.5, maxiter=50, show_lsplot=true,
                    iterations_to_show=[1])
            cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                    x_abstol=tol, successive_f_converge=2, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
                    store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:0)
            rt = @elapsed W1, objvals, trs, nitersum, fxss, xdiffss, ngss = ocasolve!(W0, Mw, d; stparams=stparams,
                                                                            lsparams=lsparams, cparams=cparams);
            save_figure && begin
                H1 = W1\X
                normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
                imsave_data(dataset,"OCA$(datastr)$(medstr)_SNR$(SNR)_rti$(rt1)_bw$(β1)_lw$(λ1)_rt$(rt)",W2,H2,imgsz,100; saveH=false)
            end
        elseif method == :CD
            Wcd, Hcd = copy(Wcd0), copy(Hcd0)
            rt = @elapsed result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=50, α=α, l₁ratio=0.5,
                                            tol=tol), X, Wcd, Hcd)
            @show result.converged
            nitersum=result.niters; fxss=copy(result.objvalues); xdiffss=copy(result.objvalues); ngss=copy(result.objvalues)
            save_figure && begin
                normalizeW!(Wcd,Hcd); Wcd1,Hcd1 = sortWHslices(Wcd,Hcd)
                imsave_data(dataset,"CD$(datastr)$(medstr)_SNR$(SNR)_rti$(rt1cd)_rt$(rt)",Wcd1,Hcd1,imgsz,100; saveH=false)
            end
        end
        iterrng = 1:nitersum
        trng = iterrng/nitersum*rt
        uselogscale && (fxss=log10.(fxss); xdiffss=log10.(xdiffss); ngss=log10.(ngss))
        axisfx.plot(iterrng,fxss)
        idx == 1 && axisfx0.plot(iterrng,fxss)
        axisxdiff.plot(iterrng,xdiffss)
        axisngx.plot(iterrng,ngss)
        axisfxt.plot(trng,fxss)
        idx == 1 && axisfx0t.plot(trng,fxss)
        axisxdifft.plot(trng,xdiffss)
        axisngxt.plot(trng,ngss)
    end
    for ax in [axisfx,axisxdiff,axisngx,axisfxt,axisxdifft,axisngxt]
        ax.legend(legendstrs, fontsize = 12)
    end
    figfx.savefig("$(optimmethod)_fx.png")
    figxdiff.savefig("$(optimmethod)_xdiff.png")
    figngx.savefig("$(optimmethod)_maxgx.png")
    figfxt.savefig("$(optimmethod)_fx_time.png")
    figxdifft.savefig("$(optimmethod)_xdiff_time.png")
    figngxt.savefig("$(optimmethod)_maxgx_time.png")
end
axisfx0.legend(String.(methods), fontsize = 12)
axisfx0t.legend(String.(methods), fontsize = 12)
figfx0.savefig("optim_fx.png")
figfx0t.savefig("optim_fx_time.png")
close("all")









ncells = 800
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp, d = initsemisca(X, ncells, initmethod=initmethod,
poweradjust=initpwradj, initfn=initfn, β1=βs[1])
initstr = "$(initmethod)"
fprefix0 = "Init$(initdatastr)_$(initstr)_$(initpwradj)_rt"*@sprintf("%1.2f",rt1)
Mw0, Mh0 = copy(Mw), copy(Mh)
W1,H1 = copy(W0*Mw), copy(Mh*H0); normalizeW!(W1,H1)
imsaveW("SNR-20.png",W1,imgsz;gridcols=40)

stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=approx,
β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=regularization, order=order, M2power=M2power,
useRelaxedL1=useRelaxedL1, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
poweradjust=pwradj, rectify=rectify, objective=objective)

D = Diagonal(d); fw=gh=Matrix(1.0I,0,0)
normw2 = norm(W0)^2; normh2 = norm(H0)^2
normwp = norm(W0,stparams.order)^stparams.order; normhp = norm(H0,stparams.order)^stparams.order
sndpow = sqrt(norm(d))^stparams.M2power
(λw, λh) = (stparams.λ1/normw2, stparams.λ2/normh2)
Msparse = stparams.poweradjust ∈ [:M1,:M2]
(βw, βh) = Msparse ?                     (stparams.β1, stparams.β2) :
           stparams.poweradjust==:W1M2 ? (stparams.β1/normwp/sndpow, stparams.β2/normhp/sndpow) :
                                         (stparams.β1/normwp, stparams.β2/normhp)

SCA.penaltyMwMh(Mw,Mh,W0,H0,D,fw,gh,λw,λh,βw,βh,0,Msparse,stparams.order; reg=stparams.reg,
                M2power=stparams.M2power, objective=stparams.objective, useRelaxedL1=stparams.useRelaxedL1)