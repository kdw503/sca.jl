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
subworkpath = joinpath(workpath,"paper","neurofinder")

dataset = :neurofinder_small; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = :meanST # dataset ∈ [:neurofinder,:neurofinder_small,:fakecells] ? :meanST : nothing
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

lcsvd_maxiter = 100; lcsvd_inner_maxiter = 50; lcsvd_ls_maxiter = 100
compnmf_maxiter = 100; compnmf_inner_maxiter = 0; compnmf_ls_maxiter = 0
hals_maxiter = 100; tol=-1; maskth = 0.25; makepositive = true; mxabs = 0.15

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset);
ncells = 25
# Xm = dropdims(mean(X,dims=2),dims=2)
# mn = minimum(Xm); Xm .-= mn; mx = maximum(Xm); Xm ./= mx
# save(joinpath(subworkpath,"neurofinder.02.00.cut100250_small_sqrt_meanimg.png"), reshape(Xm, imgsz...))

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset ∈ [:fakecells] ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

filterstr = filter === nothing ? "" : "_$(filter)"
if filter == :meanST 
    X = LCSVD.noisefilter(:meanS,X,imgsz;filterlength=3)
    X = LCSVD.noisefilter(:meanT,X,imgsz;filterlength=3)
elseif filter ∈ [:meanS]
    X = LCSVD.noisefilter(:meanS,X,imgsz;filterlength=3)
elseif filter == :meanT
    X = LCSVD.noisefilter(:meanT,X,imgsz;filterlength=3)
end

subtract_bg = true; subtract_bgstr = subtract_bg ? "_subbglpf" : ""
rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
# Hcdlpf = Hcd
# Hcdlpf = mean(Hcd)
Hcdlpf = mapwindow(mean, Hcd, (1,31))
bg = Wcd*Hcdlpf; Xsbg = X - bg
# bg = Wcd*fill(mean(Hcd),1,n); Xsbg = X - bg
if subtract_bg
    X = Xsbg
end

# LCSVD
uselv=false; s=10; maxiter = lcsvd_maxiter
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
#    try
for (prefix,tailstr,initmethod,α,β) in [#("lcsvd_precon","_sp",:isvd,0.005,0.0),
                                #("lcsvd_precon","_nn",:nndsvd,0.,1.0),
                                #("lcsvd","_sp_nn",:nndsvd,0.005,1.0),
                                ("lcsvd","_sp_nn",:nndsvd,0.005,0.1)
                                ]
    @show prefix, tailstr
    useprecond = prefix ∈ ["lcsvd_precon","lcsvd_precon_LPF"] ? true : false
    usedenoiseW0H0 = prefix ∈ ["lcsvd_precon_LPF","lcsvd_LPF"] ? true : false
    dd = Dict()
    α1=α2=α; β1=β2=β
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
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond, usedenoiseW0H0=usedenoiseW0H0,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, store_trace = true,
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
    avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(Xsbg,datadic["cells"], Wlc, Hlc; clamp=false)
    nodr = LCSVD.matchedorder(ml,ncells); switches = []#[(10,18)](18,9),(24,14)
    isempty(switches) || foreach(t -> (a=nodr[t[1]]; nodr[t[1]]=nodr[t[2]]; nodr[t[2]]=a), switches)
    Wlc, Hlc = Wlc[:,nodr], Hlc[nodr,:]; LCSVD.normalizeW!(Wlc,Hlc)#; W1,H1 = LCSVD.sortWHslices(W1,H1)
    isempty(switches) || ((avgfit, _) = LCSVD.matchedfitval(Xsbg,datadic["cells"], Wlc, Hlc; clamp=false, ordered=true))
#    nodr = LCSVD.matchedorder(ml,ncells); Wlc, Hlc = Wlc[:,nodr], Hlc[nodr,:]; LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
    fprex = "$(prefix)$(filterstr)$(subtract_bgstr)$(initmethod)_nc$(ncells)"
    # precondstr = useprecond ? "_precond" : ""
    # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_b$(β)_f$(fitval)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
    imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false, scalemtd=:fixed, mxabs=mxabs)
    # TestData.imsave_data_gt(dataset,fname*"_gt", Wlc,Hlc,gtW,gtH,imgsz,100; saveH=false, scalemtd = :fixed, mxabs=mxabs, verbose=false)

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
for (tailstr,initmethod,α) in [("_nn",:nndsvd,0.),("_sp_nn",:nndsvd,0.001)]#
    dd = Dict()
    # W1, H1 = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
    # result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
    #                 tol=tol, verbose=true), X, W1, H1; gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
    W1, H1 = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, W1, H1)
    fitval = LCSVD.fitd(X,W1*H1)
    avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(Xsbg,datadic["cells"], W1, H1; clamp=false)
    nodr = LCSVD.matchedorder(ml,ncells); switches = []#[(10,18)]
    isempty(switches) || foreach(t -> (a=nodr[t[1]]; nodr[t[1]]=nodr[t[2]]; nodr[t[2]]=a), switches)
    Whls, Hhls = W1[:,nodr], H1[nodr,:]; LCSVD.normalizeW!(Whls,Hhls)#; W1,H1 = LCSVD.sortWHslices(W1,H1)
    isempty(switches) || ((avgfit, _) = LCSVD.matchedfitval(Xsbg,datadic["cells"], Whls, Hhls; clamp=false, ordered=true))
    fprex = "$(prefix)$(filterstr)$(subtract_bgstr)$(initmethod)_nc$(ncells)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_f$(fitval)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
    imsave_data(dataset,fname,Whls,Hhls,imgsz,100; saveH=false, scalemtd=:fixed, mxabs=mxabs)
#    TestData.imsave_data_gt(dataset,fname*"_gt", Whls,Hhls,gtW,gtH,imgsz,100; saveH=false, scalemtd = :fixed, mxabs=mxabs, verbose=false)
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
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
    rt1 += rst0.inittime # add calculation time for compression matrices L and R
    rt2 -= rst0.inittime
    fitval = LCSVD.fitd(X,Wcn*Hcn)
    avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(Xsbg,datadic["cells"], Wcn, Hcn; clamp=false)
    nodr = LCSVD.matchedorder(ml,ncells); Wcn, Hcn = Wcn[:,nodr], Hcn[nodr,:]; LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    fprex = "$(prefix)$(filterstr)$(subtract_bgstr)$(initmethod)_nc$(ncells)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_f$(fitval)_af$(avgfit)_it$(maxiter)_rt$(rt2)")
    imsave_data(dataset,fname,Wcn,Hcn,imgsz,100; saveH=false, scalemtd=:fixed, mxabs=mxabs)
#    TestData.imsave_data_gt(dataset,fname*"_gt", Wcn,Hcn,gtW,gtH,imgsz,100; saveH=false, scalemtd = :fixed, mxabs=mxabs, verbose=false)

    # rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    # dd["niters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    # dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
    # if true#iter == num_experiments
    #     metadata = Dict()
    #     metadata["maxiter"] = maxiter
    # end
    # save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end


#=========== layout 1 =================================#
include(joinpath(workpath,"setup_plot.jl"))

fontsize = 30; imgsize = 310; gapsize = 5

f = Figure(resolution = (2100,1000))

g11 = f[1, 1] = GridLayout()
g1p5 = f[1, 2] = GridLayout()
g12 = f[1, 3] = GridLayout()
g13 = f[1, 4] = GridLayout()
g14 = f[1, 5] = GridLayout()
g15 = f[1, 6] = GridLayout()
g16 = f[1, 7] = GridLayout()

fname = joinpath(subworkpath,"neurofinder.02.00.cut100250_small_sqrt_meanimg.png")
ax11=AMakie.Axis(g11[1,1],title="(a)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax11); hidespines!(ax11); image!(ax11, rotr90(load(fname)));
fname = joinpath(subworkpath,"neurofinder.02.00.cut100250_small_sqrt_corrimg_gt_with_number.png")
ax21=AMakie.Axis(g11[2,1], aspect = DataAspect())
hidedecorations!(ax21); hidespines!(ax21); image!(ax21, rotr90(load(fname)));
colsize!(g11,1,imgsize) # colsize!(gridlayout,column_number,size) size is same as GLMakie.Fixed(size)
rowsize!(g11,1,imgsize) # rowsize!(gridlayout,row_number,size) size is same as GLMakie.Fixed(size)
rowsize!(g11,2,imgsize) # row 1 plus row 2 size are much smaller than height of figure. So they are stick together
                        # with rowgap and located in the vertical center
rowgap!(g11,gapsize)

Label(g1p5[1,1],"Original input", fontsize=fontsize, rotation=pi/2, padding=(2,2,-50,0)) # g1p5[1,1,Bottom()]
Label(g1p5[2,1],"Preprocessed input", fontsize=fontsize, rotation=pi/2, padding=(2,2,10,0))
Label(g1p5[3,1],"Background
subtracted input", fontsize=fontsize, rotation=pi/2, padding=(2,2,20,0))
foreach(i->rowsize!(g1p5,i,imgsize),1:3)

fname = joinpath(subworkpath, "lcsvd_precon", "lcsvd_preconisvd_nc25_a0.005_b0.0_f0.5577611936901041_af0.5506787477126039_it100_rt6.7638123_gt_W.png")
ax12=AMakie.Axis(g12[1,1],title="(b)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax12); hidespines!(ax12); image!(ax12, rotr90(load(fname)))
fname = joinpath(subworkpath, "lcsvd_precon", "lcsvd_precon_meanSTisvd_nc25_a0.005_b0.0_f0.811942641687603_af0.7822030509925142_it100_rt7.9654936_gt_W.png")
ax22=AMakie.Axis(g12[2,1], aspect = DataAspect())
hidedecorations!(ax22); hidespines!(ax22); image!(ax22, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd_precon", "sbg_lpfT", "lcsvd_precon_meanST_subbglpfisvd_nc25_a0.005_b0.0_f0.7113050975511606_af0.8081575529917868_it100_rt7.3482019_gt_W.png")
ax32=AMakie.Axis(g12[3,1], aspect = DataAspect())
hidedecorations!(ax32); hidespines!(ax32); image!(ax32, rotr90(load(fname)));
gl=g12; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "lcsvd", "lcsvdnndsvd_nc25_a0.005_b0.1_f0.5553556791079356_af0.5518840632255515_it100_rt11.4283836_gt_W.png")
ax13=AMakie.Axis(g13[1,1],title="(c)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax13); hidespines!(ax13); image!(ax13, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvd_meanSTnndsvd_nc25_a0.005_b0.1_f0.8107100476137238_af0.7510079152231729_it100_rt20.1313502_gt_W.png")
ax23=AMakie.Axis(g13[2,1], aspect = DataAspect())
hidedecorations!(ax23); hidespines!(ax23); image!(ax23, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "sbg_lpfT", "lcsvd_meanST_subbglpfnndsvd_nc25_a0.005_b0.1_f0.6699229292315947_af0.7662525717170353_it100_rt18.4458246_gt_W.png")
ax33=AMakie.Axis(g13[3,1], aspect = DataAspect())
hidedecorations!(ax33); hidespines!(ax33); image!(ax33, rotr90(load(fname)));
gl=g13; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "compnmf", "compnmflowrank_nndsvd_nc25_f0.5551290037736494_af0.5692559827785831_it100_rt1.0367607316070533_W.png")
ax14=AMakie.Axis(g14[1,1],title="(d)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax14); hidespines!(ax14); image!(ax14, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "compnmf_meanSTlowrank_nndsvd_nc25_f0.8125122185012364_af0.6995833110397979_it100_rt0.5300679057220492_W.png")
ax24=AMakie.Axis(g14[2,1], aspect = DataAspect())
hidedecorations!(ax24); hidespines!(ax24); image!(ax24, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "sbg_lpfT", "compnmf_meanST_subbglpflowrank_nndsvd_nc25_f0.5345823678756119_af0.6548338612282834_it100_rt0.9598744771118177_W.png")
ax34=AMakie.Axis(g14[3,1], aspect = DataAspect())
hidedecorations!(ax34); hidespines!(ax34); image!(ax34, rotr90(load(fname)));
gl=g14; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "hals", "halsnndsvd_nc25_a0.0_f0.5602236872422826_af0.580751371769372_it100_rt12.451775_W.png")
ax15=AMakie.Axis(g15[1,1],title="(e)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax15); hidespines!(ax15); image!(ax15, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_meanSTnndsvd_nc25_a0.0_f0.8131191402699999_af0.7361413740458617_it100_rt13.0424639_W.png")
ax25=AMakie.Axis(g15[2,1], aspect = DataAspect())
hidedecorations!(ax25); hidespines!(ax25); image!(ax25, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "sbg_lpfT", "hals_meanST_subbglpfnndsvd_nc25_a0.0_f0.615633771593345_af0.7657844863189028_it100_rt13.1420257_W.png")
ax35=AMakie.Axis(g15[3,1], aspect = DataAspect())
hidedecorations!(ax35); hidespines!(ax35); image!(ax35, rotr90(load(fname)));
gl=g15; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "hals", "halsnndsvd_nc25_a0.001_f0.5577553479799731_af0.5537833622451933_it100_rt12.3802724_W.png")
ax16=AMakie.Axis(g16[1,1],title="(f)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax16); hidespines!(ax16); image!(ax16, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_meanSTnndsvd_nc25_a0.001_f0.8117235995246221_af0.7726442854762007_it100_rt12.6269606_W.png")
ax26=AMakie.Axis(g16[2,1], aspect = DataAspect())
hidedecorations!(ax26); hidespines!(ax26); image!(ax26, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "sbg_lpfT", "hals_meanST_subbglpfnndsvd_nc25_a0.001_f0.6027381534611983_af0.7711908145966705_it100_rt12.8125903_W.png")
ax36=AMakie.Axis(g16[3,1], aspect = DataAspect())
hidedecorations!(ax36); hidespines!(ax36); image!(ax36, rotr90(load(fname)));
gl=g16; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

save(joinpath(subworkpath,"neurofinder_all_figures.png"),f,px_per_unit=2)


#=========== layout 2 =================================#
fontsize = 30; imgsize = 310; gapsize = 5

f = Figure(resolution = (1760,1380))

gtop = f[1, 1] = GridLayout()
gbottom = f[2, 1] = GridLayout()
g21 = gbottom[1, 1] = GridLayout()
g22 = gbottom[1, 2] = GridLayout()
g23 = gbottom[1, 3] = GridLayout()
g24 = gbottom[1, 4] = GridLayout()
g25 = gbottom[1, 5] = GridLayout()
g26 = gbottom[1, 6] = GridLayout()

Label(gtop[1,1],"(a)", font="Arial bold", fontsize=fontsize, rotation=0, padding=(2,2,2,2)) # g21[1,1,Bottom()]
fname = joinpath(subworkpath,"neurofinder.02.00.cut100250_small_sqrt_meanimg.png")
ax11=AMakie.Axis(gtop[1,2], aspect = DataAspect())
hidedecorations!(ax11); hidespines!(ax11); image!(ax11, rotr90(load(fname)));
fname = joinpath(subworkpath,"neurofinder.02.00.cut100250_small_sqrt_corrimg_gt_with_number.png")
ax21=AMakie.Axis(gtop[1,3], aspect = DataAspect())
hidedecorations!(ax21); hidespines!(ax21); image!(ax21, rotr90(load(fname)));
colsize!(gtop,1,20); colsize!(gtop,2,300); colsize!(gtop,3,300); rowsize!(gtop,1,300); colgap!(gtop,50)

Label(g21[1,1],"Original input", fontsize=fontsize, rotation=pi/2, padding=(2,2,-50,0)) # g21[1,1,Bottom()]
Label(g21[2,1],"Preprocessed input", fontsize=fontsize, rotation=pi/2, padding=(2,2,10,0))
Label(g21[3,1],"Background
subtracted input", fontsize=fontsize, rotation=pi/2, padding=(2,2,20,0))
foreach(i->rowsize!(g21,i,imgsize),1:3)

fname = joinpath(subworkpath, "lcsvd_precon", "lcsvd_preconisvd_nc25_a0.005_b0.0_f0.5577611936901041_af0.5506787477126039_it100_rt6.7638123_gt_W.png")
ax12=AMakie.Axis(g22[1,1],title="(b)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax12); hidespines!(ax12); image!(ax12, rotr90(load(fname)))
fname = joinpath(subworkpath, "lcsvd_precon", "lcsvd_precon_meanSTisvd_nc25_a0.005_b0.0_f0.811942641687603_af0.7822030509925142_it100_rt7.9654936_gt_W.png")
ax22=AMakie.Axis(g22[2,1], aspect = DataAspect())
hidedecorations!(ax22); hidespines!(ax22); image!(ax22, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd_precon", "sbg_lpfT", "lcsvd_precon_meanST_subbglpfisvd_nc25_a0.005_b0.0_f0.7113050975511606_af0.8081575529917868_it100_rt7.3482019_gt_W.png")
ax32=AMakie.Axis(g22[3,1], aspect = DataAspect())
hidedecorations!(ax32); hidespines!(ax32); image!(ax32, rotr90(load(fname)));
gl=g22; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "lcsvd", "lcsvdnndsvd_nc25_a0.005_b0.1_f0.8013417232686422_af0.5677123608999161_it100_rt11.4248592_W.png")
ax13=AMakie.Axis(g23[1,1],title="(c)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax13); hidespines!(ax13); image!(ax13, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvd_meanSTnndsvd_nc25_a0.005_b0.1_f0.9862046178055132_af0.7470044631337701_it100_rt18.4467597_W.png")
ax23=AMakie.Axis(g23[2,1], aspect = DataAspect())
hidedecorations!(ax23); hidespines!(ax23); image!(ax23, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "sbg_lpfT", "lcsvd_meanST_subbglpfnndsvd_nc25_a0.005_b0.1_f0.8936234121608055_af0.7565992026607785_it100_rt14.2074591_W.png")
ax33=AMakie.Axis(g23[3,1], aspect = DataAspect())
hidedecorations!(ax33); hidespines!(ax33); image!(ax33, rotr90(load(fname)));
gl=g23; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "compnmf", "compnmflowrank_nndsvd_nc25_f0.5551290037736494_af0.5692559827785831_it100_rt1.0367607316070533_W.png")
ax14=AMakie.Axis(g24[1,1],title="(d)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax14); hidespines!(ax14); image!(ax14, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "compnmf_meanSTlowrank_nndsvd_nc25_f0.8125122185012364_af0.6995833110397979_it100_rt0.5300679057220492_W.png")
ax24=AMakie.Axis(g24[2,1], aspect = DataAspect())
hidedecorations!(ax24); hidespines!(ax24); image!(ax24, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "sbg_lpfT", "compnmf_meanST_subbglpflowrank_nndsvd_nc25_f0.5345823678756119_af0.6548338612282834_it100_rt0.9598744771118177_W.png")
ax34=AMakie.Axis(g24[3,1], aspect = DataAspect())
hidedecorations!(ax34); hidespines!(ax34); image!(ax34, rotr90(load(fname)));
gl=g24; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "hals", "halsnndsvd_nc25_a0.0_f0.5602236872422826_af0.580751371769372_it100_rt12.451775_W.png")
ax15=AMakie.Axis(g25[1,1],title="(e)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax15); hidespines!(ax15); image!(ax15, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_meanSTnndsvd_nc25_a0.0_f0.8131191402699999_af0.7361413740458617_it100_rt13.0424639_W.png")
ax25=AMakie.Axis(g25[2,1], aspect = DataAspect())
hidedecorations!(ax25); hidespines!(ax25); image!(ax25, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "sbg_lpfT", "hals_meanST_subbglpfnndsvd_nc25_a0.0_f0.615633771593345_af0.7657844863189028_it100_rt13.1420257_W.png")
ax35=AMakie.Axis(g25[3,1], aspect = DataAspect())
hidedecorations!(ax35); hidespines!(ax35); image!(ax35, rotr90(load(fname)));
gl=g25; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

fname = joinpath(subworkpath, "hals", "halsnndsvd_nc25_a0.001_f0.5577553479799731_af0.5537833622451933_it100_rt12.3802724_W.png")
ax16=AMakie.Axis(g26[1,1],title="(f)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax16); hidespines!(ax16); image!(ax16, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_meanSTnndsvd_nc25_a0.001_f0.8117235995246221_af0.7726442854762007_it100_rt12.6269606_W.png")
ax26=AMakie.Axis(g26[2,1], aspect = DataAspect())
hidedecorations!(ax26); hidespines!(ax26); image!(ax26, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "sbg_lpfT", "hals_meanST_subbglpfnndsvd_nc25_a0.001_f0.6027381534611983_af0.7711908145966705_it100_rt12.8125903_W.png")
ax36=AMakie.Axis(g26[3,1], aspect = DataAspect())
hidedecorations!(ax36); hidespines!(ax36); image!(ax36, rotr90(load(fname)));
gl=g26; colsize!(gl,1,imgsize); foreach(i->rowsize!(gl,i,imgsize),1:3); rowgap!(gl,5)

save(joinpath(subworkpath,"neurofinder_all_figures.png"),f,px_per_unit=2)






#=

fontsize = 30

f = Figure(resolution = (2000,1000))

g11 = f[1, 1] = GridLayout()
g12 = f[1, 2] = GridLayout()
g13 = f[1, 3] = GridLayout()
g14 = f[1, 4] = GridLayout()
g15 = f[1, 5] = GridLayout()
g16 = f[1, 6] = GridLayout()

ax1t=AMakie.Axis(g11[1,1], aspect = DataAspect())
hidedecorations!(ax1t); hidespines!(ax1t); 
fname = joinpath(subworkpath,"neurofinder.02.00.cut100250_small_sqrt_meanimg.png")
ax11=AMakie.Axis(g11[2,1],title="(a)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax11); hidespines!(ax11); image!(ax11, rotr90(load(fname)));
fname = joinpath(subworkpath,"neurofinder.02.00.cut100250_small_sqrt_corrimg_gt_with_number.png")
ax21=AMakie.Axis(g11[3,1], aspect = DataAspect())
hidedecorations!(ax21); hidespines!(ax21); image!(ax21, rotr90(load(fname)));
ax1b=AMakie.Axis(g11[4,1], aspect = DataAspect())
hidedecorations!(ax1b); hidespines!(ax1b)
colsize!(g11,1,300) # colsize!(gridlayout,column_number,size) size is same as GLMakie.Fixed(size)
rowsize!(g11,2,300) # rowsize!(gridlayout,row_number,size) size is same as GLMakie.Fixed(size)
rowsize!(g11,3,300)
colgap!(g11,10); rowgap!(g11,10)

fname = joinpath(subworkpath, "lcsvd_precon", "lcsvd_preconisvd_nc25_a0.005_b0.0_f0.5577611936901041_af0.5506787477126039_it100_rt6.7638123_gt_W.png")
ax12=AMakie.Axis(g12[1,1],title="(b)", titlesize=fontsize, ylabel="Original input", ylabelsize=fontsize, aspect = DataAspect())
hidedecorations!(ax12, label=false); hidespines!(ax12); image!(ax12, rotr90(load(fname)))
fname = joinpath(subworkpath, "lcsvd_precon", "lcsvd_precon_meanSTisvd_nc25_a0.005_b0.0_f0.811942641687603_af0.7822030509925142_it100_rt7.9654936_gt_W.png")
ax22=AMakie.Axis(g12[2,1], ylabel="Preprocessed input", ylabelsize=fontsize, aspect = DataAspect())
hidedecorations!(ax22, label=false); hidespines!(ax22); image!(ax22, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd_precon", "sbg_lpfT", "lcsvd_precon_meanST_subbglpfisvd_nc25_a0.005_b0.0_f0.7113050975511606_af0.8081575529917868_it100_rt7.3482019_gt_W.png")
ax32=AMakie.Axis(g12[3,1], ylabel="Background
subtracted input", ylabelsize=fontsize, aspect = DataAspect())
hidedecorations!(ax32, label=false); hidespines!(ax32); image!(ax32, rotr90(load(fname)));
rowgap!(g12,10)

fname = joinpath(subworkpath, "lcsvd", "lcsvdnndsvd_nc25_a0.005_b0.1_f0.5553556791079356_af0.5518840632255515_it100_rt11.4283836_gt_W.png")
ax13=AMakie.Axis(g13[1,1],title="(c)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax13); hidespines!(ax13); image!(ax13, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "lcsvd_meanSTnndsvd_nc25_a0.005_b0.1_f0.8107100476137238_af0.7510079152231729_it100_rt20.1313502_gt_W.png")
ax23=AMakie.Axis(g13[2,1], aspect = DataAspect())
hidedecorations!(ax23); hidespines!(ax23); image!(ax23, rotr90(load(fname)));
fname = joinpath(subworkpath, "lcsvd", "sbg_lpfT", "lcsvd_meanST_subbglpfnndsvd_nc25_a0.005_b0.1_f0.6699229292315947_af0.7662525717170353_it100_rt18.4458246_gt_W.png")
ax33=AMakie.Axis(g13[3,1], aspect = DataAspect())
hidedecorations!(ax33); hidespines!(ax33); image!(ax33, rotr90(load(fname)));
rowgap!(g13,10)

fname = joinpath(subworkpath, "compnmf", "compnmflowrank_nndsvd_nc25_f0.5551290037736494_af0.5692559827785831_it100_rt1.0367607316070533_W.png")
ax14=AMakie.Axis(g14[1,1],title="(d)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax14); hidespines!(ax14); image!(ax14, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "compnmf_meanSTlowrank_nndsvd_nc25_f0.8125122185012364_af0.6995833110397979_it100_rt0.5300679057220492_W.png")
ax24=AMakie.Axis(g14[2,1], aspect = DataAspect())
hidedecorations!(ax24); hidespines!(ax24); image!(ax24, rotr90(load(fname)));
fname = joinpath(subworkpath, "compnmf", "sbg_lpfT", "compnmf_meanST_subbglpflowrank_nndsvd_nc25_f0.5345823678756119_af0.6548338612282834_it100_rt0.9598744771118177_W.png")
ax34=AMakie.Axis(g14[3,1], aspect = DataAspect())
hidedecorations!(ax34); hidespines!(ax34); image!(ax34, rotr90(load(fname)));
rowgap!(g14,10)

fname = joinpath(subworkpath, "hals", "halsnndsvd_nc25_a0.0_f0.5602236872422826_af0.580751371769372_it100_rt12.451775_W.png")
ax15=AMakie.Axis(g15[1,1],title="(e)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax15); hidespines!(ax15); image!(ax15, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_meanSTnndsvd_nc25_a0.0_f0.8131191402699999_af0.7361413740458617_it100_rt13.0424639_W.png")
ax25=AMakie.Axis(g15[2,1], aspect = DataAspect())
hidedecorations!(ax25); hidespines!(ax25); image!(ax25, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "sbg_lpfT", "hals_meanST_subbglpfnndsvd_nc25_a0.0_f0.615633771593345_af0.7657844863189028_it100_rt13.1420257_W.png")
ax35=AMakie.Axis(g15[3,1], aspect = DataAspect())
hidedecorations!(ax35); hidespines!(ax35); image!(ax35, rotr90(load(fname)));
rowgap!(g15,10)

fname = joinpath(subworkpath, "hals", "halsnndsvd_nc25_a0.001_f0.5577553479799731_af0.5537833622451933_it100_rt12.3802724_W.png")
ax16=AMakie.Axis(g16[1,1],title="(f)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(ax16); hidespines!(ax16); image!(ax16, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "hals_meanSTnndsvd_nc25_a0.001_f0.8117235995246221_af0.7726442854762007_it100_rt12.6269606_W.png")
ax26=AMakie.Axis(g16[2,1], aspect = DataAspect())
hidedecorations!(ax26); hidespines!(ax26); image!(ax26, rotr90(load(fname)));
fname = joinpath(subworkpath, "hals", "sbg_lpfT", "hals_meanST_subbglpfnndsvd_nc25_a0.001_f0.6027381534611983_af0.7711908145966705_it100_rt12.8125903_W.png")
ax36=AMakie.Axis(g16[3,1], aspect = DataAspect())
hidedecorations!(ax36); hidespines!(ax36); image!(ax36, rotr90(load(fname)));
rowgap!(g16,10)

save(joinpath(subworkpath,"all_figures.png"),f,px_per_unit=2)

# text!(300,10,text="center", align=(:center,:center),fontsize=30) # works in the latest axis
# text!(ax,300,10,text="center", align=(:center,:center),fontsize=30) # works only in the axis
# Label(gmiddletop[1,1],"test", font="Arial bold", fontsize=fontsize, rotation=0, padding=(0,0,0,0)) # works in the GridLayout

=#


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
