using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
subworkpath = joinpath(workpath,"paper","inhibit_real")

dataset = :inhibit_real; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = nothing # dataset ∈ [:neurofinder,:neurofinder_small,:fakecells] ? :meanST : nothing
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

lcsvd_maxiter = 200; lcsvd_inner_maxiter = 50; lcsvd_ls_maxiter = 100
compnmf_maxiter = 1500; compnmf_inner_maxiter = 0; compnmf_ls_maxiter = 0
hals_maxiter = 200; tol=-1; maskth = 0.25; makepositive = true; mxabs = 0.15; gridcols=10

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset);

inhibitloc = datadic["inhibited_loc"]
inhibit_roi = (inhibitloc[1]-25:inhibitloc[1]+5,inhibitloc[2]-20:inhibitloc[2]+20)
imginhibit = reshape(X,imgsz...,lengthT)[inhibit_roi...,:]
icroi = (11:15,18:22) # inhibited cell
# imginhibitmean = dropdims(sum(imginhibit,dims=3),dims=3)
# mn = minimum(imginhibitmean); imginhibitmean .-= mn
# mx = maximum(imginhibitmean); imginhibitmean ./= mx
# save(joinpath(subworkpath,"inhibit_mean.png"),imginhibitmean)

imgsz = size(imginhibit)[1:end-1]; X = reshape(imginhibit,prod(imgsz),lengthT); ncells=50
(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

filterstr = filter === nothing ? "" : "_$(filter)"
if filter == :meanST 
    X = LCSVD.noisefilter(:meanS,X,imgsz;filterlength=3)
    X = LCSVD.noisefilter(:meanT,X,imgsz;filterlength=3)
elseif filter ∈ [:meanS]
    X = LCSVD.noisefilter(:meanS,X,imgsz;filterlength=3)
elseif filter == :meanT
    X = LCSVD.noisefilter(:meanT,X,imgsz;filterlength=3)
end

savemp4 = false
if savemp4
    encoder_options = (crf=23, preset="medium")
    clamp_level=1.0
    X_max = maximum(abs,X)*clamp_level; Xnor = X./X_max;  X_clamped = clamp.(Xnor,0.,1.)
    Xuint8 = UInt8.(round.(map(clamp01nan, X_clamped)*255)) # Frame dims must be a multiple of two
    imgodd = reshape.(eachcol(Xuint8),imgsz...); imgeven = map(frame->frame[1:end-1,1:end-1], imgodd)
    VideoIO.save(joinpath(subworkpath,"$(dataset).mp4"), imgeven, framerate=30, encoder_options=encoder_options) # compatible with ppt (best)
end

subtract_bg = false; subtract_bgstr = subtract_bg ? "_subbglpf" : ""
if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=60, α=0), X, Wcd, Hcd)
    LCSVD.normalizeW!(Wcd,Hcd)
    # imsave_data(dataset,joinpath(subworkpath,"Wbg"),Wcd,Hcd,imgsz,100;gridcols=1, 
    #         scalemtd=:fixed, mxabs=0.2, signedcolors=TestData.dgwm(), saveH=false)
    save(joinpath(subworkpath,"Wbg.png"),reshape((Wcd.-0.011)./0.02314,imgsz...))
    plotH_data(joinpath(subworkpath,"Hbg"), Hcd)
    # bg = median_background(TiledFactorizations.accumfloat(T),X);
    # normalizeW!(bg.S,bg.T); imsave_data(dataset,"Wr1",reshape(bg.S,length(bg.S),1),bg.T,imgsz,100; signedcolors=dgwm(), saveH=false)
    # series(bg.T'); save("Hr2.png",current_figure())
    Hcd31 = mapwindow(mean, Hcd, (1,31)) # mean filter
    # sigma = std(Hcd)
    # Hclamp = map((a,b) -> (a>b+2sigma) || (a<b-2sigma) ? b : a, Hcd,Hcd31); Hcd31 = mapwindow(mean, Hclamp, (1,31))
    # Hclamp = map((a,b) -> (a>b+2sigma) || (a<b-2sigma) ? b : a, Hcd,Hcd31); Hcd31 = mapwindow(mean, Hclamp, (1,31))
    # Hclamp = map((a,b) -> (a>b+2sigma) || (a<b-2sigma) ? b : a, Hcd,Hcd31); Hcd31 = mapwindow(mean, Hclamp, (1,31))
    plotH_data(joinpath(subworkpath,"Hbg_LPF"), vcat(Hcd,Hcd31), resolution = (800,200), ytickformat="{:.3f}",
        xticksvisible=false, xlabelvisible=false, labels = ["Hbg","Hbg_LPF"])

    # GT H of ROI
    hbefore = Float64[]
    for x in eachcol(X)
        push!(hbefore,sum(reshape(x,imgsz...)[icroi...]))
    end
    bg = Wcd*Hcd31; Xsbg = X-bg
    X = Xsbg

    # plot sbged H
    h = Float64[]
    for x in eachcol(X)
        push!(h,sum(reshape(x,imgsz...)[icroi...]))
    end
    plotH_data(joinpath(subworkpath,"Hinhibit_sbbefore"), hbefore', resolution = (800,200), ytickformat="{:.3f}",
        xticksvisible=false, xlabelvisible=false, show_legend=false)
    plotH_data(joinpath(subworkpath,"Hinhibit_sbafter"), h', resolution = (800,200), ytickformat="{:.3f}",
        xticksvisible=false, xlabelvisible=false, show_legend=false)
    # f=AMakie.Figure(resolution = (900,400))
    # axbefore=AMakie.Axis(f[1,1],title="Intensity of the inhibited cell")
    # axafter=AMakie.Axis(f[2,1],title="Intensity of the inhibited cell after background subtration" ,xlabel="time index"); linkxaxes!(axbefore, axafter)
    # lines!(axbefore,hbefore,label="Hbg"); lines!(axafter,h,label="Hbg"); # axislegend(axafter, position = :rb)
    # save(joinpath(subworkpath,"Hinhibit_sbg.png"),current_figure())
end

# LCSVD
uselv=false; s=10; maxiter = lcsvd_maxiter
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
#    try
for (prefix,tailstr,initmethod,α,β,inhibit_indices) in [#("lcsvd_precon","_sp",:isvd,0.005,0.0),
                                        ("lcsvd","_sp_nn",:nndsvd,0.05,50.0,[1,8,10])
                                        ]
    @show prefix, tailstr
    useprecond = prefix ∈ ["lcsvd_precon","lcsvd_precon_LPF"] ? true : false
    usedenoiseW0H0 = prefix ∈ ["lcsvd_precon_LPF","lcsvd_LPF"] ? true : false
    dd = Dict()
    α1=α; α2=0; β1=β; β2=0
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    # rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    # stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    #     regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    #     useprecond=useprecond, uselv=uselv, maskW=maskW, maskH=maskH)
    # lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    # cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    #     x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    #     store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    # Mw, Mh = copy(Mw0), copy(Mh0);
    # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
    #                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd) 
                                            # svdmethod = :isvd doesn't work for initmethod == :nndsvd
    W0 = W0[:,1:ncells]; H0 = H0[1:ncells,:]; M0 = M0[1:ncells,1:ncells]; N0 = N0[1:ncells,1:ncells]; D = D[1:ncells,1:ncells]
    σ0=s*std(W0) #=10*std(W0)=#
    alg = LCSVD.LinearCombSVD(eltype(X);α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    # M, N = copy(M0), copy(N0)
    # rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
    alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
    Wlc, Hlc = rst0.W, rst0.H
    fitval = LCSVD.fitd(X,Wlc*Hlc)
    makepositive && LCSVD.flip2makepos!(Wlc,Hlc)
    LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
    fprex = "$(prefix)$(filterstr)$(subtract_bgstr)$(initmethod)_nc$(ncells)"
    # precondstr = useprecond ? "_precond" : ""
    # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
    fname = joinpath(subworkpath,prefix,"$(fprex)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_f$(fitval)_it$(rst0.niters)_rt$(rt2)")
    imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false, scalemtd=:fixed, mxabs=mxabs, gridcols=gridcols)
#    inhibit_indices = [9]
    for inhibit_index = inhibit_indices
        plotH_data(fname*"_$inhibit_index", Hlc[inhibit_index,:]'; resolution = (800,200), ytickformat="{:.3f}", xticksvisible=false,
            xlabelvisible=false, show_legend=false)
    end
    # Xrecon = zeros(eltype(X),size(X)...)
    # for index in inhibit_indices
    #     Xrecon += Wlc[:,index]*Hlc[index,:]'
    # end
    # rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(Xrecon, 1; initmethod=initmethod, svdmethod=:svd) 
    #                                         # svdmethod = :isvd doesn't work for initmethod == :nndsvd
    # σ0=s*std(W0) #=10*std(W0)=#
    # alg = LCSVD.LinearCombSVD(eltype(X);α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond, usedenoiseW0H0=usedenoiseW0H0,
    #     denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, store_trace = true,
    #     store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    #     f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    # # M, N = copy(M0), copy(N0)
    # # rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
    # alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
    # M, N = copy(M0), copy(N0)
    # rt2 = @elapsed rst0 = LCSVD.solve!(alg, Xrecon, W0, H0, D, M, N);
    # Wlc, Hlc = rst0.W, rst0.H
    # plotH_data(fname*"_recon", Hlc; resolution = (800,400))

    # f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niter); totalniters = sum(niters)
    # avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
    # avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
    # for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
    #     isempty(afs) && continue
    #     append!(avgfits,afs); append!(inner_fxs,fxs)
    #     if iter == 1
    #         rt2i = 0.
    #     else
    #         rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
    #     end
    #     append!(rt2s,rt2i)
    # end
    # dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    # dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    # if true#iter == num_experiments
    #     metadata = Dict()
    #     metadata["sigma0"] = σ0; metadata["r"] = r; metadata["initmethod"] = initmethod
    #     metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
    #     metadata["usedenoiseW0H0"] = usedenoiseW0H0; metadata["denoisefilter"] = alg.denoisefilter; 
    #     metadata["alpha"] = α; metadata["beta"] = β
    # end
    # save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results.jld2"),"metadata",metadata,"data",dd)
    # GC.gc()
end
# catch e
#     fprex = "$(prefix)$(SNR)db$(factor)f$(ncells)s"
#     save(joinpath(subworkpath,prefix,"$(fprex)_error_$(iter).jld2"),datadic)
#     @warn e
# end

# HALS
prefix = "hals"
rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar)
maxiter = hals_maxiter
for (tailstr,initmethod,α,inhibit_indices) in [("_nn",:nndsvd,0.,[1,2]),("_sp_nn",:nndsvd,0.001,[1,2])]#
    dd = Dict()
    # W1, H1 = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
    # result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
    #                 tol=tol, verbose=true), X, W1, H1; gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
    W1, H1 = copy(Wcd0), copy(Hcd0); T = eltype(Wcd0)
    rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{T}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, W1, H1)
    fitval = LCSVD.fitd(X,W1*H1); Whls, Hhls = W1, H1
    LCSVD.normalizeW!(Whls,Hhls)#; W1,H1 = LCSVD.sortWHslices(W1,H1)
    fprex = "$(prefix)$(filterstr)$(subtract_bgstr)$(initmethod)_nc$(ncells)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_f$(fitval)_it$(maxiter)_rt$(rt2)")
    imsave_data(dataset,fname,Whls,Hhls,imgsz,100; saveH=false, scalemtd=:fixed, mxabs=mxabs, gridcols=gridcols)
    for inhibit_index = inhibit_indices
        plotH_data(fname*"_$inhibit_index", Hhls[inhibit_index,:]'; resolution = (800,200), ytickformat="{:.3f}", xticksvisible=false,
            xlabelvisible=false, show_legend=false)
    end
    # rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    # dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1cd; dd["rt2s"] = rt2s
    # dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
    # if true#iter == num_experiments
    #     metadata = Dict()
    #     metadata["maxiter"] = maxiter; metadata["alpha"] = α
    # end
    # save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end

# COMPNMF
prefix = "compnmf"; maxiter = compnmf_maxiter
for (tailstr,initmethod,maxiter) in [#=("_nn",:lowrank_nndsvd,100),=#("_nn",:lowrank_nndsvd,100)]
    dd = Dict()
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    # rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    # stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    #     regSpar=regSpar, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc,
    #     uselv=uselv, maskW=maskW, maskH=maskH)
    # cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    #     x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
    #     store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    # Mw, Mh = copy(Mw0), copy(Mh0);
    # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
    #                                                     penmetric=penmetric, stparams=stparams, cparams=cparams);
    rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
    # Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    # result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn;
    #                     gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
    Wcn, Hcn = copy(Wcn0), copy(Hcn0); T = eltype(Wcn0)
    rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{T}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
    rt1 += rst0.inittime # add calculation time for compression matrices L and R
    rt2 -= rst0.inittime
    fitval = LCSVD.fitd(X,Wcn*Hcn)
    LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    fprex = "$(prefix)$(filterstr)$(subtract_bgstr)$(initmethod)_nc$(ncells)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_f$(fitval)_it$(maxiter)_rt$(rt2)")
    imsave_data(dataset,fname,Wcn,Hcn,imgsz,100; saveH=false, scalemtd=:fixed, mxabs=mxabs, gridcols=gridcols)
    inhibit_indices = [7]
    plotH_data(fname, Hcn[inhibit_indices,:]; resolution = (800,200), ytickformat="{:.0e}", xticksvisible=true,
        xlabelvisible=false, show_legend=false, labels=string.(inhibit_indices))
    ytickformat()
    # rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    # dd["niters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    # dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
    # if true#iter == num_experiments
    #     metadata = Dict()
    #     metadata["maxiter"] = maxiter
    # end
    # save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end



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



#========= Both with and without subtraction background ======================#
fontsize = 30; fontsize2=20

f = Figure(resolution = (1820, 1165))
rowgap!(f.layout,0)

gtop = f[1, 1] = GridLayout()
gmiddle = f[2, 1] = GridLayout()
gbottom = f[3, 1] = GridLayout()
gbottomleft = gbottom[1, 1] = GridLayout()
gbottomright = gbottom[1, 2] = GridLayout()
gcde = gbottomleft[1, 1] = GridLayout()
gf = gbottomleft[1, 2] = GridLayout()
gghi = gbottomright[1, 1] = GridLayout()
gj = gbottomright[1, 2] = GridLayout()

fname = joinpath(subworkpath,"imgcut.png")
axa=AMakie.Axis(gtop[1,1], title="(a) Mean image of
the region of interest", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axa); hidespines!(axa); image!(axa, rotr90(load(fname)))
fname = joinpath(subworkpath,"inhibit_mean_cell.png")
axb=AMakie.Axis(gtop[1,2],title="(b) Background of
inhibited neuron area", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axb); hidespines!(axb); image!(axb, rotr90(load(fname)));
colsize!(gtop,1,300); colsize!(gtop,2,300); colgap!(gtop,150)#; rowsize!(gtop,1,350); rowgap!(gtop,10)

Label(gmiddle[1,1],"(c) Without background subtraction", font="Arial bold", fontsize=fontsize, width=900, padding=(10,2,0,0))
Label(gmiddle[1,2],"(d) With background subtraction", font="Arial bold", fontsize=fontsize, width=900, padding=(10,2,0,0)) # g1p5[1,1,Bottom()]
foreach(i->rowsize!(gmiddle,i,10),1:3)


# Panel (c~e)
fname = joinpath(subworkpath, "lcsvd", "lcsvdnndisvd_nc50+10_aw0.05_ah0_bw50.0_bh0_f0.8169233_it200_rt3.3354302_W.png")
axc=AMakie.Axis(gcde[1,1], subtitle="LCSVD", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axc, label=false); hidespines!(axc); image!(axc, rotr90(load(fname)))
fname = joinpath(subworkpath, "compnmf", "compnmflowrank_nndsvd_nc50_f0.8283547_it50_rt0.1104385237060547_W.png")
axd=AMakie.Axis(gcde[2,1], subtitle="Compressed NMF", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axd, label=false); hidespines!(axd); image!(axd, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "halsnndsvd_nc50_a0.0_f0.82841855_it200_rt0.7570591_W.png")
axe=AMakie.Axis(gcde[3,1], subtitle="HALS NMF", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axe, label=false); hidespines!(axe); image!(axe, rotr90(load(fname)));
colsize!(gcde,1,450); foreach(i->rowsize!(gcde,i,200),1:3); rowgap!(gcde,0)

# Panel f
fname = joinpath(subworkpath, "Hinhibit_sbbefore_plot_H.png")
axf1=AMakie.Axis(gf[1,1], subtitle = "Sum of intensities in the inhibited neuron region", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axf1, label=false); hidespines!(axf1); image!(axf1, rotr90(load(fname)))
fname = joinpath(subworkpath, "lcsvd", "lcsvdnndisvd_nc50+10_aw0.05_ah0_bw50.0_bh0_f0.8169233_it200_rt3.1558125_plot_H_8.png")
axf2=AMakie.Axis(gf[2,1], subtitle="LCSVD (cell 8)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axf2, label=false); hidespines!(axf2); image!(axf2, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvdnndisvd_nc50+10_aw0.05_ah0_bw50.0_bh0_f0.8169233_it200_rt3.1558125_plot_H_9.png")
axf3=AMakie.Axis(gf[3,1], subtitle="LCSVD (cell 9)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axf3, label=false); hidespines!(axf3); image!(axf3, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "halsnndsvd_nc50_a0.0_f0.82841855_it200_rt0.7570591_plot_H_8.png")
axf4=AMakie.Axis(gf[4,1], subtitle="HALS NMF (cell 8)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axf4, label=false); hidespines!(axf4); image!(axf4, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "halsnndsvd_nc50_a0.0_f0.82841855_it200_rt0.7570591_plot_H_19.png")
axf5=AMakie.Axis(gf[5,1], subtitle="HALS NMF (cell 19)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axf5, label=false); hidespines!(axf5); image!(axf5, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "halsnndsvd_nc50_a0.0_f0.82841855_it200_rt0.7570591_plot_H_33.png")
axf6=AMakie.Axis(gf[6,1], subtitle="HALS NMF (cell 33)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axf6, label=false); hidespines!(axf6); image!(axf6, rotr90(load(fname)));
colsize!(gf,1,400); foreach(i->rowsize!(gf,i,100),1:5); rowsize!(gf,6,120); rowgap!(gf,0)

# Panel (g~i)
fname = joinpath(subworkpath, "lcsvd", "lcsvd_subbglpfnndisvd_nc50+10_aw0.05_ah0_bw50.0_bh0_f0.4946435_it200_rt2.1994236_W.png")
axg=AMakie.Axis(gghi[1,1],subtitle="LCSVD", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axg, label=false); hidespines!(axg); image!(axg, rotr90(load(fname)))
fname = joinpath(subworkpath, "compnmf", "compnmf_subbglpflowrank_nndsvd_nc50_f-0.9999596_it50_rt0.12781850871887207_W.png")
axh=AMakie.Axis(gghi[2,1], subtitle="Compressed NMF", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axh, label=false); hidespines!(axh); image!(axh, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_subbglpfnndsvd_nc50_a0.0_f0.41323447_it200_rt1.1138178_W.png")
axi=AMakie.Axis(gghi[3,1], subtitle="HALS NMF", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axi, label=false); hidespines!(axi); image!(axi, rotr90(load(fname)));
colsize!(gghi,1,450); foreach(i->rowsize!(gghi,i,200),1:3); rowgap!(gghi,0)

# Panel j
fname = joinpath(subworkpath, "Hbg_LPF_plot_H.png")
axj1=AMakie.Axis(gj[1,1], subtitle = "Original and LP Filtered H comp. of Background", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj1, label=false); hidespines!(axj1); image!(axj1, rotr90(load(fname)))
fname = joinpath(subworkpath, "Hinhibit_sbafter_plot_H.png")
axj2=AMakie.Axis(gj[2,1], subtitle="Sum of intensities in the inhibited neuron region", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj2, label=false); hidespines!(axj2); image!(axj2, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvd_subbglpfnndisvd_nc50+10_aw0.05_ah0_bw50.0_bh0_f0.4946435_it200_rt2.1994236_plot_H_1.png")
axj3=AMakie.Axis(gj[3,1], subtitle="LCSVD (cell 1)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj3, label=false); hidespines!(axj3); image!(axj3, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvd_subbglpfnndisvd_nc50+10_aw0.05_ah0_bw50.0_bh0_f0.4946435_it200_rt2.1994236_plot_H_8.png")
axj4=AMakie.Axis(gj[4,1], subtitle="LCSVD (cell 8)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj4, label=false); hidespines!(axj4); image!(axj4, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "compnmf_subbglpflowrank_nndsvd_nc50_f-0.9999596_it50_rt0.12781850871887207_plot_H.png")
axj5=AMakie.Axis(gj[5,1], subtitle="compressed NMF (cell 7)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj5, label=false); hidespines!(axj5); image!(axj5, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_subbglpfnndsvd_nc50_a0.0_f0.41323447_it200_rt1.1138178_plot_H.png")
axj6=AMakie.Axis(gj[6,1], subtitle="HALS NMF (cell 3)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj6, label=false); hidespines!(axj6); image!(axj6, rotr90(load(fname)));
colsize!(gj,1,400); foreach(i->rowsize!(gj,i,100),1:5); rowsize!(gj,6,114); rowgap!(gj,0)

save(joinpath(subworkpath,"inhibit_real_all_figures.png"),f,px_per_unit=2)



#========= Only with subtraction background ======================#
fontsize = 30; fontsize2=20

f = Figure(resolution = (1150, 1370))
rowgap!(f.layout,0)

gtop = f[1, 1] = GridLayout()
gmiddle = f[2, 1] = GridLayout()
gbottom = f[3, 1] = GridLayout()
gghi = gbottom[1, 1] = GridLayout()
gj = gbottom[1, 2] = GridLayout()

fname = joinpath(subworkpath,"imgcut.png")
axa=AMakie.Axis(gtop[1,1], title="(a) Mean image of
the region of interest", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axa); hidespines!(axa); image!(axa, rotr90(load(fname)))
fname = joinpath(subworkpath,"inhibit_mean_cell.png")
axb=AMakie.Axis(gtop[1,2],title="(b) Background of
inhibited neuron area", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axb); hidespines!(axb); image!(axb, rotr90(load(fname)));
colsize!(gtop,1,300); colsize!(gtop,2,300); colgap!(gtop,150); rowsize!(gtop,1,300); rowgap!(gtop,10)

Label(gmiddle[1,1],"(c) Result W and H factors", font="Arial bold", fontsize=fontsize, width=900, padding=(300,2,0,0))
rowsize!(gmiddle,1,10)

# Panel (g~i)
fname = joinpath(subworkpath, "lcsvd", "lcsvd_subbglpfnndsvd_nc50_aw0.05_ah0_bw10.0_bh0_f0.78900945_it200_rt3.9929448_W.png")
axg=AMakie.Axis(gghi[1,1],subtitle="LCSVD", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axg, label=false); hidespines!(axg); image!(axg, rotr90(load(fname)))
fname = joinpath(subworkpath, "compnmf", "compnmf_subbglpflowrank_nndsvd_nc50_f-0.9999596_it50_rt0.12781850871887207_W.png")
axh=AMakie.Axis(gghi[2,1], subtitle="Compressed NMF", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axh, label=false); hidespines!(axh); image!(axh, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_subbglpfnndsvd_nc50_a0.0_f0.41323447_it200_rt1.1138178_W.png")
axi=AMakie.Axis(gghi[3,1], subtitle="HALS NMF", subtitlesize=fontsize, aspect = DataAspect())
hidedecorations!(axi, label=false); hidespines!(axi); image!(axi, rotr90(load(fname)));
colsize!(gghi,1,580); rowgap!(gghi,0) # foreach(i->rowsize!(gghi,i,200),1:3); 

# Panel j
fname = joinpath(subworkpath, "Hbg_LPF_plot_H.png")
axj1=AMakie.Axis(gj[1,1], subtitle = "Original and LP Filtered H comp. of Background", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj1, label=false); hidespines!(axj1); image!(axj1, rotr90(load(fname)))
fname = joinpath(subworkpath, "Hinhibit_sbafter_plot_H.png")
axj2=AMakie.Axis(gj[2,1], subtitle="Sum of intensities in the inhibited neuron region", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj2, label=false); hidespines!(axj2); image!(axj2, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvd_subbglpfnndsvd_nc50_aw0.05_ah0_bw10.0_bh0_f0.78900945_it200_rt3.9929448_1_plot_H.png")
axj3=AMakie.Axis(gj[3,1], subtitle="LCSVD (cell 1)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj3, label=false); hidespines!(axj3); image!(axj3, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvd_subbglpfnndsvd_nc50_aw0.05_ah0_bw10.0_bh0_f0.78900945_it200_rt3.9929448_8_plot_H.png")
axj4=AMakie.Axis(gj[4,1], subtitle="LCSVD (cell 8)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj4, label=false); hidespines!(axj4); image!(axj4, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "compnmf_subbglpflowrank_nndsvd_nc50_f-0.9999596_it50_rt0.12781850871887207_plot_H.png")
axj5=AMakie.Axis(gj[5,1], subtitle="compressed NMF (cell 7)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj5, label=false); hidespines!(axj5); image!(axj5, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_subbglpfnndsvd_nc50_a0.0_f0.41323447_it200_rt1.1138178_plot_H.png")
axj6=AMakie.Axis(gj[6,1], subtitle="HALS NMF (cell 3)", subtitlesize=fontsize2, aspect = DataAspect())
hidedecorations!(axj6, label=false); hidespines!(axj6); image!(axj6, rotr90(load(fname)));
colsize!(gj,1,520); rowsize!(gj,6,143); rowgap!(gj,0) # foreach(i->rowsize!(gj,i,100),1:5); 

save(joinpath(subworkpath,"inhibit_real_all_figures.png"),f,px_per_unit=2)

