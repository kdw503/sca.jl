using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","dandi")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

fprefix = "dandi1174_sub-Q_ophys"

dd = load(joinpath(subworkpath,"$(fprefix).jld2"))
gtW = dd["gtW"]; data_ann_mean = dd["data_ann_mean"]

bgmtd = :nosbg
fnameX = joinpath(subworkpath,"$(fprefix)_thin_X.jld2")
gtH = gtW\load(fnameX,"Xsbgh"); LCSVD.normalizeW!(gtW,gtH)
X = bgmtd == :sbgmh ? load(fnameX,"Xsbgmh") :
    bgmtd == :sbgh ? load(fnameX,"Xsbgh") :
    bgmtd == :nosbg ? load(fnameX,"Xwbg") :
    error("Unknown bgmtd $(bgmtd)")

noc = bgmtd == :nosbg ? 41 : 40 # true=27
nac = 0
imgsz = size(data_ann_mean); lengthT = size(X,2)
# https://docs.dandiarchive.org/example-notebooks/tutorials/cosyne_2023/advanced_asset_search/
# Need to find wide field single photon imaging data
resultpath = joinpath(subworkpath,"$(fprefix)_thin")

# PCB initialization
initmethod = :nndsvd
rt0 = @elapsed U, Vt, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(Vt'); N0t = copy(N0')
rt11 = 0
if initmethod == :sbc
    try
        rt11 = @elapsed M0 = sbc(U)
    catch e
        save(joinpath(resultpath,"sbc_error$(iter).jld2"),"U",W0)
        error("SBC failed with error: $(e)")
    end
    rt12 = @elapsed N0 = M0\D
    rt13 = @elapsed LCSVD.balanceWH!(M0, N0)
    rtinit = rt0+rt11+rt12+rt12
else
    rtinit = rt0
end
Winit = U*M0; Hinit = N0*Vt
LCSVD.normalizeW!(Winit,Hinit)
fvinit = LCSVD.fitd(X,Winit*Hinit)
(bgidx, space) = bgmtd == :nosbg ? (argmax(abs.(Hinit[:,1])),-10) : (0,-10)
Winit, Hinit = Winit[:,setdiff(1:noc,bgidx)], Hinit[setdiff(1:noc,bgidx),:]
nodrinit, matchlistinit, fitvalsinit, _ = LCSVD.matchedorder(gtW, gtH, Winit, Hinit, bgmtd == :nosbg ? noc-1 : noc; clamp=false, sdsr=2, tdsr=10)
afvinit = sum(fitvalsinit)/length(fitvalsinit)
Winit, Hinit = Winit[:,nodrinit], Hinit[nodrinit,:]
fprex = "PCBinit_$(bgmtd)_$(initmethod)_nac$(nac)"
LCSVD.flip2makepos!(Winit,Hinit;mask=:topNpix)
fnameinit = joinpath(resultpath,"$(fprex)_f$(fvinit)_af$(afvinit)_rt$(rtinit)") 
imsave_data(:ocpi,fnameinit,Winit,Hinit,imgsz,100; # Hinit[setdiff(1:noc,bgidx),:]
        gridcols=8, borderwidth=4, saveH=false, verbose=false, signedcolors=TestData.dgwdm())
plotH_data(fnameinit, Hinit, figsize=(1400,1095),space=space,legend_position=:rc,
    ylabelvisible = false, ygridvisible=false, yticksvisible=false, yticklabelsvisible=false)

# PCB with penalties
(α, β) = (0.005,0.0)#(0.005, 5.0)# (1e-6,1e-6)
α1 = α2 = α; β1 = β2 = β
r = 0.3; useprecond = false
tol = 1e-5; inner_tol = 1e-6; maxiter = Int(ceil(log(eps(eltype(X)))/log(r))); inner_maxiter = 1000
maskth = 0.25; maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    # α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    r=r, useprecond=useprecond, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, inner_maxiter = inner_maxiter,
    store_trace = false, store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol,
    f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
state = LCSVD.prepare_state(U, V, M0, N0t, alg)
updater = LCSVD.LinearCombSVDUpd{eltype(D)}(D, state, noc, alg) # normalize parameters and prepare
Eall, Es, Rw, Rh, Nw, Nh = LCSVD.penaltyMN_wholeparams(U, V, D, M0, N0t, updater.σw2, updater.σh2, updater, alg)
nαw = updater.αw; nαh = updater.αh; nβw = updater.βw; nβh = updater.βh;
Ersw = Rw/nαw; Ersh = Rh/nαh; Ernw = Nw/nβw; Ernh = Nh/nβh;
@show α, β, nαw, nαh, nβw, nβh, Ersw, Ersh, Ernw, Ernh, Eall, Es, Rw, Rh, Nw, Nh

# PCB
(α, β) = bgmtd == :nosbg ? (1e-6,5e-6) : (0.0001,0.05)
usecalparams = false
for usecalparams in [false, true]
@show usecalparams
for (α, β) in [(1e-13,1e-13),(0.,0.)]
# for β=0.04:0.02:0.1, α=0.00000:0.00005:0.0003
    α1 = α2 = α; β1 = β2 = β
    if usecalparams
        factor = 1.
        α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = α*factor; α2vec[1] = α*factor
        β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = β*factor; β2vec[1] = β*factor
    else
        α1vec = α2vec = β1vec = β2vec = eltype(α1)[]
    end
    r = 0.3; useprecond = false
    tol = 1e-5; inner_tol = 1e-6; maxiter = Int(ceil(log(eps(eltype(X)))/log(r))); inner_maxiter = 1000
    maskth = 0.25; maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
        α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
        r=r, useprecond=useprecond, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, inner_maxiter = inner_maxiter,
        store_trace = false, store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol,
        f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
    M1, N1t = copy(M0), copy(N0t)
    rt2pcb = @elapsed rst_pcb = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH)

    Wpcb, Hpcb = rst_pcb.W, rst_pcb.Ht'
    LCSVD.normalizeW!(Wpcb,Hpcb);
    fvpcb = LCSVD.fitd(X,Wpcb*Hpcb)
    (bgidx, space) = bgmtd == :nosbg ? (argmax(abs.(Hpcb[:,1])),-10) : (0,-10)
    Wpcb, Hpcb = Wpcb[:,setdiff(1:noc,bgidx)], Hpcb[setdiff(1:noc,bgidx),:]
    nodrpcb, matchlistpcb, fitvalspcb, _ = LCSVD.matchedorder(gtW, gtH, Wpcb, Hpcb, bgmtd == :nosbg ? noc-1 : noc; clamp=false, sdsr=2, tdsr=10)
    afvpcb = sum(fitvalspcb)/length(fitvalspcb)
    @show α, β, fvpcb, afvpcb, rt2pcb

    Wpcb, Hpcb = Wpcb[:,nodrpcb], Hpcb[nodrpcb,:]
    #fv = LCSVD.fitd(X,Wpcb*Hpcb)
    fprexpcb = "PCB_$(bgmtd)_$(initmethod)_nac$(nac)"
    LCSVD.flip2makepos!(Wpcb,Hpcb;mask=:topNpix)
    fnamepcb = usecalparams ? joinpath(resultpath,"$(fprexpcb)_av[1]$(α1vec[1])_av[2]$(α1vec[2])_bv[1]$(β1vec[1])_bv[2]$(β1vec[2])_f$(fvpcb)_af$(afvpcb)_tol$(tol)_it$(rst_pcb.niters)_rt$(rt2pcb)") :
                        joinpath(resultpath,"$(fprexpcb)_a$(α)_b$(β)_f$(fvpcb)_af$(afvpcb)_tol$(tol)_it$(rst_pcb.niters)_rt$(rt2pcb)") 
    imsave_data(:ocpi,fnamepcb,Wpcb,Hpcb,imgsz,100; gridcols=8, borderwidth=4, saveH=false, verbose=false,
                signedcolors=TestData.dgwdm())
    plotH_data(fnamepcb, Hpcb, figsize=(1400,1095),space=space,legend_position=:rc,
        ylabelvisible = false, ygridvisible=false, yticksvisible=false, yticklabelsvisible=false)
end
end

# HALS
αhls = 0.1
for αhls in [0.,0.1]
    @show αhls
    rt1hls = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, noc, variant=:ar)
    maxiter = 100; tol=1e-6
    Whls, Hhls = copy(Wcd0), copy(Hcd0)
    rt2hls = @elapsed rst_hls = NMF.solve!(NMF.CoordinateDescent{eltype(Whls)}(maxiter=maxiter, α=αhls, l₁ratio=1,
                    tol=tol, verbose=false), X, Whls, Hhls)
    LCSVD.normalizeW!(Whls,Hhls)
    fvhls = LCSVD.fitd(X,Whls*Hhls)
    (bgidx, space) = bgmtd == :nosbg ? (argmax(abs.(Hhls[:,1])),-10) : (0,-10)
    Whls, Hhls = Whls[:,setdiff(1:noc,bgidx)], Hhls[setdiff(1:noc,bgidx),:]
    nodrhls, matchlisthls, fitvalshls, _ = LCSVD.matchedorder(gtW, gtH, Whls, Hhls, bgmtd == :nosbg ? noc-1 : noc; clamp=false, sdsr=2, tdsr=10)
    afvhls = sum(fitvalshls)/length(fitvalshls)
    Whls, Hhls = Whls[:,nodrhls], Hhls[nodrhls,:]
    fprexhls = "HALS_$(bgmtd)"
    fnamehls = joinpath(resultpath,"$(fprexhls)_a$(αhls)_f$(fvhls)_af$(afvhls)_it$(rst_hls.niters)_rtinit$(rt1hls)_rt$(rt2hls)")
    imsave_data(:ocpi,fnamehls,Whls,Hhls,imgsz,100; gridcols=8, borderwidth=4, saveH=false, verbose=false,
                signedcolors=TestData.dgwdm())
    (bgidx, space) = bgmtd == :nosbg ? (argmax(abs.(Hhls[:,1])),-40) : (0,-10)
    plotH_data(fnamehls, Hhls, figsize=(1400,1095),space=space,legend_position=:rc,
        ylabelvisible = false, ygridvisible=false, yticksvisible=false, yticklabelsvisible=false)
end
