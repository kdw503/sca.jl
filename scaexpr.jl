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
makepositive = false

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



dataset = :fakecells; SNR=20; initmethod=:isvd; initpwradj=:wh_normalize
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

sd_group=:component; reg = :W1M2
#sd_group=:column; reg = :W1M2
l1l2ratio = 0.1; 
λ = 0; β = 10; α = 0.1
tol=1e-7; uselogscale=true; save_figure=true; optimmethod = :optim_lbfgs
ls_method = :ls_BackTracking
scamaxiter = 100; halsmaxiter = 100; inner_maxiter = 50; ls_maxiter = 500
methods = [:SCA] # :HALS
isplotxandg = false; plotnum = isplotxandg ? 3 : 1

penaltystr = uselogscale ? "log10(penalty)" : "penalty"
if isplotxandg
    xdiffstr = uselogscale ? "log10(x difference)" : "x difference"
    maxgxstr = uselogscale ? "log10(maximum(g(x)))" : "maximum(g(x))"
end
figfx0, axisfx0 = plt.subplots(1,1, figsize=(5,4))  # SCA vs HALS fx iter
axisfx0.set_title("f(x)")
axisfx0.set_ylabel(penaltystr,fontsize = 12)
axisfx0.set_xlabel("iterations",fontsize = 12)
figfx0t, axisfx0t = plt.subplots(1,1, figsize=(5,4))  # SCA vs HALS fx time
axisfx0t.set_title("f(x)")
axisfx0t.set_ylabel(penaltystr,fontsize = 12)
axisfx0t.set_xlabel("time",fontsize = 12)
for method = methods
    @show method
    figs = []; axes = []
    yaxisstrs = [("f(x)"),("x_diff"),("maximum(g(x))")]; xaxisstrs = ["iterations","time"]
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

    innerloopvec = [("λ","β"),(0,50),(0,10)] #,
    legendstrs = map(innerloop->"$(innerloopvec[1][1])=$(innerloop[1]), $(innerloopvec[1][2])=$(innerloop[2])", innerloopvec[2:end])
    title = join([innerloopvec[1]...],"_vs_")
    for (idx,(λ, β)) = enumerate(innerloopvec[2:end])

    # innerloopvec = [("ls_maxiter"),(0),(500)] #,
    # legendstrs = map(innerloop->"$(innerloopvec[1][1])=$(innerloop[1]), ", innerloopvec[2:end])
    # title = ("withLS_vs_woLS")
    # for (idx,(ls_maxiter)) = enumerate(innerloopvec[2:end])

    # innerloopvec = [("sd_group","reg"),(:component,:W1M2)] #,(:whole,:WH1)
    # legendstrs = map(innerloop->"$(innerloopvec[1][1])=$(innerloop[1]), $(innerloopvec[1][2])=$(innerloop[2])", innerloopvec[2:end])
    # title = ("whole_vs_column")
    # for (idx,(sd_group, reg)) = enumerate(innerloopvec[2:end])
        scaparamstr = "_$(reg)_$(sd_group)_$(optimmethod)_$(maxiterstr)_lsit$(ls_maxiter)_b$(β)_l$(λ)"
        λ1 = λ2 = λ; β1 = β2 = β
        if method == :SCA
            Mw, Mh = copy(Mw0), copy(Mh0)
            stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true,
                β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=reg, l1l2ratio=l1l2ratio, order=1, M2power=1,
                useRelaxedL1=false, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
                poweradjust=:none, rectify=:none, objective=:normal)
            lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter, show_lsplot=false,
                iterations_to_show=[15])
            cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                x_abstol=tol, successive_f_converge=1, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
                store_inner_trace=true, show_trace=true, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:5)
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
                    β1=β1, β2=0, λ1=λ1, λ2=0, reg=reg, l1l2ratio=l1l2ratio, order=1,
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
                normalizeW!(W1,H1); W2,H2 = sortWHslices(W1,H1)
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
axisfx0.legend(String.(methods), fontsize = 12)
axisfx0t.legend(String.(methods), fontsize = 12)
figfx0.savefig("SCA_vs_HALS$(datastr)$(filterstr)_its$(scamaxiter)_ith$(halsmaxiter)_iter.png")
figfx0t.savefig("SCA_vs_HALS$(datastr)$(filterstr)_its$(scamaxiter)_ith$(halsmaxiter)_time.png")
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
alpha = 2; ridge_alpha=0.01; max_iter=100; tol=1e-5
rtspca = @elapsed resultspca = fit_transform!(SparsePCA(n_components=ncells,alpha=alpha,ridge_alpha=ridge_alpha,max_iter=max_iter,tol=tol,verbose=true),X) 
W0 = copy(resultspca); H0 = W0\X
normalizeW!(W0,H0); W2,H2 = sortWHslices(W0,H0)
imsave_data(dataset,"SPCA_$(datastr)_a$(alpha)_ra$(ridge_alpha)_tol$(tol)_miter$(max_iter)_rt$(rtspca)",W2,H2,imgsz,100;
        signedcolors=dgwm(), saveH=false)

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