using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","spca")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using ScikitLearn
@sk_import decomposition: SparsePCA

#=========== with FakeCells ==============#
dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.1
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))
bias = 0.1

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncs, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
(m,n,p) = (size(X)...,ncs)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

subtract_bg = false
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    bg = W*fill(mean(H),1,n)
    X .-= bg
end
normX2 = norm(X)^2

# PCA

prefix = "pcb"
noc = ncs; nac = 0
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,0.)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

rtsvd = @elapsed Usvd, H0svd, M0, N0, Wp, Hp, Dsvd = LCSVD.initpcb(X, noc, nac; initmethod=:svd, svdmethod=:svd)
Vsvd = copy(H0svd')
fitsvd = LCSVD.fitd(X,Usvd*Dsvd*H0svd)
Xvsvd = X*Vsvd*inv(Vsvd'*Vsvd)*Vsvd'; pve_svd = norm(Xvsvd)^2/normX2
W0svd = copy(Usvd); LCSVD.normalizeW!(W0svd,H0svd); sw_svd = norm(W0svd,1)

rttsvd = @elapsed Utsvd, H0tsvd, M0, N0, Wp, Hp, Dtsvd = LCSVD.initpcb(X, noc, nac; initmethod=:tsvd, svdmethod=:tsvd)
Vtsvd = copy(H0tsvd')
fittsvd = LCSVD.fitd(X,Utsvd*Dtsvd*H0tsvd)
Xvtsvd = X*Vtsvd*inv(Vtsvd'*Vtsvd)*Vtsvd'; pve_tsvd = norm(Xvtsvd)^2/normX2
W0tsvd = copy(Utsvd); LCSVD.normalizeW!(W0tsvd,H0tsvd); sw_tsvd = norm(W0tsvd,1)

rtisvd = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
fitisvd = LCSVD.fitd(X,U*D*H0)
Xv = X*V*inv(V'*V)*V'; pve_isvd = norm(Xv)^2/normX2
W0 = copy(U); LCSVD.normalizeW!(W0,H0); sw_isvd = norm(W0,1)

# irlba initialization for SMA
using RCall

@rput X
rtirlba = @elapsed R"""
# library(scRNAseq)
# library(magrittr)
# library(SingleCellExperiment)
# library(dplyr)
library(epca)
library(Matrix) # as.matrix
# library(ggplot2)
  x = scale(x = X,
            center = F,
            scale = F)
  # s = RSpectra::svds(x, k)
  s = irlba::irlba(x, 9, tol = 1e-10)
  z = s$u
  b = diag(s$d)
  y = s$v
"""
@rget z
@rget b
@rget y
fitirlba = LCSVD.fitd(X,z*b*y')
Xy = X*y*inv(y'*y)*y'; pve_irlba = norm(Xy)^2/normX2
Score = z*b; Loading = Array(y'); LCSVD.normalizeW!(Score,Loading); sw_irlba = norm(Score,1)

save(joinpath(subworkpath,"svd_methods.jld2"),"fitsvd",fitsvd,"pvesvd",pve_svd,"swsvd",sw_svd,"rtsvd",rtsvd,
    "fittsvd",fittsvd,"pvetsvd",pve_tsvd,"swtsvd",sw_tsvd,"rttsvd",rttsvd,
    "fitisvd",fitisvd,"pveisvd",pve_isvd,"swisvd",sw_isvd,"rtisvd",rtisvd,
    "fitirlba",fitirlba,"pveirlba",pve_irlba,"swirlba",sw_irlba,"rtirlba",rtirlba)


rtisvd = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=:tsvd, svdmethod=:tsvd)
V = copy(H0'); N0t = copy(N0')

αrng = 0.0002:0.0002:0.02
optim_method = :lbfgs#4o3norm
fits=[]; afits = []; sws = []; rts = []; pves = []
for (aidx, α) in enumerate(αrng)
    @show aidx
    β1 = β2= β; α1 = α2 = α
    β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
    α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
    r=0.3; useprecond=false; uselv=false; tol=1e-6
    maxiter = optim_method == :lbfgs4o3norm ? 1 : Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter
    inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100))
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
        #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
        r=r, useprecond=useprecond, usedenoiseW0H0=false, optim_method = optim_method,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = false,
        store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
    M1, N1t = copy(M0), copy(N0t)
    rst1 = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
    alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
    M1, N1t = copy(M0), copy(N0t)
    rt2 = @elapsed LCSVD.solve!(alg, X, U, V, D, M1, N1t);

    W1, H1 = rst1.W, rst1.Ht'
    LCSVD.normalizeW!(W1,H1);
    # avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
    afv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
    fv = LCSVD.fitd(X,W1*H1)
    sw = norm(W1,1)
    Xy = X*H1'*inv(H1*H1')*H1; pve = norm(Xy)^2/normX2
    nodr = LCSVD.matchedorder(ml,noc); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
    LCSVD.flip2makepos!(Wlc1,Hlc1)
    fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)_$(initmethod)_$(optim_method)"
    regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_a$(α)_b$(β)"
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
    imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)
    push!(fits,fv); push!(afits,afv); push!(sws,sw); push!(rts,rt2); push!(pves,pve)
end
save(joinpath(subworkpath,"fits_afits_sws_PCB_tsvd_$(optim_method).jld2"),"fits",fits,"afits",afits,"sws",sws,"rts",rts,"pves",pves)

dd = load(joinpath(subworkpath,"fits_afits_sws_PCB_$(optim_method).jld2"))
fits = dd["fits"]
afits = dd["afits"]
sws = dd["sws"]
pves = dd["pves"]

# fit vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "Fit", title = "")
pcb = scatter!(ax, sws, fits, color = :green, markersize = 15, label="PCB")
# scatter!(ax, [(x, y) for x in sws for y in fits], color=:red, strokecolor=:black, strokewidth=1)
axislegend(ax, position = :rt) # halign = :left, valign = :top
save(joinpath(subworkpath,"sws_vs_fits_PCB__$(optim_method).png"),f,px_per_unit=2)

#  fit vs. αrng
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "α", ylabel = "Fit", title = "")
pcb = scatter!(ax, αrng, fits, color = :green, markersize = 15, label="PCB")
axislegend(ax, position = :rt)
save(joinpath(subworkpath,"αrng_vs_fits_PCB_$(optim_method).png"),f,px_per_unit=2)

#  fit vs. αrng
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "α", ylabel = "∥W∥₁", title = "")
pcb = scatter!(ax, αrng, sws, color = :green, markersize = 15, label="PCB")
axislegend(ax, position = :rt)
save(joinpath(subworkpath,"αrng_vs_sws_PCB_$(optim_method).png"),f,px_per_unit=2)

#  runtime vs. αrng
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "α", ylabel = "runtime (sec)", title = "")
pcb = scatter!(ax, αrng, rts, color = :green, markersize = 15, label="PCB")
axislegend(ax, position = :rt)
save(joinpath(subworkpath,"αrng_vs_runtime_PCB_$(optim_method).png"),f,px_per_unit=2)

# PVE vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "PVE", title = "")
pcb = scatter!(ax, sws, pves, color = :green, markersize = 15, label="PCB")
# scatter!(ax, [(x, y) for x in sws for y in fits], color=:red, strokecolor=:black, strokewidth=1)
axislegend(ax, position = :rt) # halign = :left, valign = :top
save(joinpath(subworkpath,"sws_vs_pves_PCB__$(optim_method).png"),f,px_per_unit=2)

# Legend(f[1, 2], [lin, sca, lin], ["a line", "some dots", "line again"])
# Legend(f[2, 1], [lin, sca, lin], ["a line", "some dots", "line again"],
#     orientation = :horizontal, tellwidth = false, tellheight = true)


# HALS
prefix="hals"; @show prefix
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1_hals = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar);
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
mfmethod = :HALS; αhals=0.1; maxiter = 60; tol=-1
αrng = 0.:0.002:0.2
fits_hals=[]; afits_hals = []; sws_hals = []; rts_hals = []; pves_hals = []
for (aidx, αhals) in enumerate(αrng)
    @show aidx
    W, H = copy(Whals0), copy(Hhals0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=αhals, l₁ratio=1,
                    tol=tol, verbose=false), X, W, H)
    LCSVD.normalizeW!(W,H);
    afv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
    fv = LCSVD.fitd(X,W*H)
    sw = norm(W,1)
    Xy = X*H'*inv(H*H')*H; pve = norm(Xy)^2/normX2
    nodr = LCSVD.matchedorder(ml,noc); Whals, Hhals = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
    fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)"
    fname = joinpath(subworkpath,"$(fprex)_a$(αhals)_f$(fv)_af$(afv)_it$(maxiter)_rti$(rt1_hals)_rt$(rt2)")
    imsave_data(dataset,fname,Whals,Hhals,imgsz,100; saveH=false, scalemtd=:maxcol)
    push!(fits_hals,fv); push!(afits_hals,afv); push!(sws_hals,sw); push!(rts_hals,rt2); push!(pves_hals,pve)
end
save(joinpath(subworkpath,"fits_afits_sws_HALS.jld2"),"fits_hals",fits_hals,"afits_hals",afits_hals,
            "sws_hals",sws_hals, "rt1_hals",rt1_hals,"rts_hals",rts_hals,"pves_hals",pves_hals)

# COMPNMF
prefix = "compnmf"
@show prefix; flush(stdout)
mfmethod = :COMPNMF; maxiter = 1000
(tailstr,initmethod) = ("_nn",:lowrank_nndsvd)
dd = Dict(); tol=-1
rt1_cnmf = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, noc, variant=:ar);
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn)
W, H = copy(Wcn0), copy(Hcn0);
rt_cnmf = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, W, H)
rt1_cnmf += rst0.inittime # add calculation time for compression matrices L and R
rt_cnmf -= rst0.inittime
LCSVD.normalizeW!(W,H);
afit_cnmf, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
fit_cnmf = LCSVD.fitd(X,W*H)
sw_cnmf = norm(W,1)
Xy = X*H'*inv(H*H')*H; pve_cnmf = norm(Xy)^2/normX2
nodr = LCSVD.matchedorder(ml,noc); Wcn, Hcn = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)"
fname = joinpath(subworkpath,"$(fprex)_f$(fit_cnmf)_af$(afit_cnmf)_it$(rst0.niters)_rti$(rt1_cnmf)_rt$(rt_cnmf)")
imsave_data(dataset,fname,Wcn,Hcn,imgsz,100; saveH=false, scalemtd=:maxcol)
save(joinpath(subworkpath,"fits_afits_sws_CompressedNMF.jld2"),"fit_cnmf",fit_cnmf,"afit_cnmf",afit_cnmf,
            "sw_cnmf",sw_cnmf, "rt1_cnmf",rt1_cnmf,"rt_cnmf",rt_cnmf,"pve_cnmf",pve_cnmf)

# SPCA
prefix = "spca"
@show prefix; flush(stdout)
makepositive = true
α = 0.5; ridge_alpha=0.01; max_iter=100; tol=0
rtspca = @elapsed resultspca = fit_transform!(SparsePCA(n_components=ncells,alpha=α,ridge_alpha=ridge_alpha,max_iter=max_iter,tol=tol,verbose=true),X) 
W = copy(resultspca); H = W\X
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
nodr = LCSVD.matchedorder(ml,ncells); Wspca, Hspca = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
makepositive && LCSVD.flip2makepos!(Wspca,Hspca,mask=:topNpix)
fprex = "$(prefix)$(SNR)db_bias$(bias)_$(sbgstr)"
fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_af$(avgfit)_it$(max_iter)_rt$(rtspca)")
imsave_data(dataset,fname,Wspca,Hspca,imgsz,100; saveH=false)


# SMA
using RCall

R"library(epca)"

@rput X

# SCA : Sparse Component Analysis (sparsity is applied to only H(Y')) + maximize(∥Z'XY∥₂)
prefix = "sca"
rt2 = @elapsed R"factors_sca <-sca(X, k=15)" # default gamma = sqrt(p*k)=sqrt(1000*15)
@rget factors_sca
rW = factors_sca[:z]; rH = Array(factors_sca[:y]'); rW = X/rH
LCSVD.normalizeW!(rW,rH); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, rW); fitval = LCSVD.fitd(X,rW*rH)
nodr = LCSVD.matchedorder(ml,ncells); Wsca, Hsca = rW[:,nodr], rH[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
LCSVD.flip2makepos!(Wsca,Hsca); # Wsca[:,1:4] .*= -1; Hsca[1:4,:] .*= -1
fprex = "$(prefix)$(SNR)db_bias$(bias)_$(sbgstr)"
fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_rt$(rt2)")
imsave_data(dataset,fname,Wsca,Hsca,imgsz,100; saveH=false, scalemtd=:maxcol)
plotH_data(fname*"_Hinhibit",Hsca[1:7,:]; space=0.,ylabel="",ytickformat="{:.2f}")

# SMA : Sparse Matrix Approximation (sparsity is applied to both W(Z is nXk) and H(Y' is kXp))
prefix = "sma"
@rput X
fits_sma=[]; afits_sma = []; sws_sma = []; rts_sma = []; pves_sma = []
grng = 20:4:324
for (aidx, g) in enumerate(grng) # 110 is optimum
    @rput g
    @show aidx
    rt2 = @elapsed R"factors_sma <-sma(X, k=15, gamma=g)" # gamma_z=sqrt(p*k) and gamma_is default
    @rget factors_sma
    rW = factors_sma[:z]; rH = Array(factors_sma[:y]'); b = factors_sma[:b]; rH = b*rH
    LCSVD.normalizeW!(rW,rH)
    afv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, rW, rH; clamp=false)
    fv = LCSVD.fitd(X,rW*rH)
    sw = norm(rW,1)
    Xy = X*rH'*inv(rH*rH')*rH; pve = norm(Xy)^2/normX2
    nodr = LCSVD.matchedorder(ml,noc); Wsma, Hsma = rW[:,nodr], rH[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
    LCSVD.flip2makepos!(Wsma,Hsma); # Wsca[:,5:7] .*= -1; Hsca[5:7,:] .*= -1
    fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)"
    fname = joinpath(subworkpath,"$(fprex)_f$(fv)_g$(g)_rt$(rt2)")
    imsave_data(dataset,fname,Wsma,Hsma,imgsz,100; saveH=false, scalemtd=:maxcol)
    # plotH_data(fname*"_Hinhibit",Hsca[1:7,:]; space=0.,ylabel="",ytickformat="{:.2f}")
    push!(fits_sma,fv); push!(afits_sma,afv); push!(sws_sma,sw); push!(rts_sma,rt2); push!(pves_sma,pve)
end
save(joinpath(subworkpath,"fits_afits_sws_sma.jld2"),"grng",grng,"fits",fits_sma,"afits",afits_sma,"sws",sws_sma,"rts",rts_sma,"pves",pves_sma)

dd_sma = load(joinpath(subworkpath,"fits_afits_sws_sma.jld2"))
fits_sma = dd_sma["fits"]
afits_sma = dd_sma["afits"]
sws_sma = dd_sma["sws"]
rts_sma = dd_sma["rts"]
pves_sma = dd_sma["pves"]

# fit vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "Fit", title = "")
sma = scatter!(ax, sws_sma, fits_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"fits_vs_sws_SMA.png"),f,px_per_unit=2)

#  fit vs. αrng
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "γ", ylabel = "Fit", title = "")
sma = scatter!(ax, grng, fits_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"grng_vs_fits_SMA.png"),f,px_per_unit=2)

#  fit vs. αrng
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "γ", ylabel = "∥W∥₁", title = "")
sma = scatter!(ax, grng, sws_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"grng_vs_sws_SMA.png"),f,px_per_unit=2)

#  runtime vs. αrng
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "γ", ylabel = "runtime (sec)", title = "")
sma = scatter!(ax, grng, rts_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"grng_vs_runtime_SMA.png"),f,px_per_unit=2)

R"vignette(\"epca\")"


##### Plot PCB vs. SMA
# load pcb result
dd = load(joinpath(subworkpath,"fits_afits_sws_PCB_$(optim_method).jld2"))
fits = dd["fits"]
afits = dd["afits"]
sws = dd["sws"]
pves = dd["pves"]

# load hals result
dd_hals = load(joinpath(subworkpath,"fits_afits_sws_HALS.jld2"))
fits_hals = dd_hals["fits_hals"]
afits_hals = dd_hals["afits_hals"]
sws_hals = dd_hals["sws_hals"]
rt1_hals = dd_hals["rt1_hals"]
rts_hals = dd_hals["rts_hals"]
pves_hals = dd_hals["pves_hals"]

# load cnmf result
dd_cnmf = load(joinpath(subworkpath,"fits_afits_sws_CompressedNMF.jld2"))
fit_cnmf = dd_cnmf["fit_cnmf"]
afit_cnmf = dd_cnmf["afit_cnmf"]
sw_cnmf = dd_cnmf["sw_cnmf"]
rt1_cnmf = dd_cnmf["rt1_cnmf"]
rt_cnmf = dd_cnmf["rt_cnmf"]
pve_cnmf = dd_cnmf["pve_cnmf"]

# load sma result
dd_sma = load(joinpath(subworkpath,"fits_afits_sws_sma.jld2"))
fits_sma = dd_sma["fits"]
afits_sma = dd_sma["afits"]
sws_sma = dd_sma["sws"]
rts_sma = dd_sma["rts"]
pves_sma = dd_sma["pves"]

dd_svd = load(joinpath(subworkpath,"svd_methods.jld2"))
fitsvd = dd_svd["fitsvd"]
pve_svd = dd_svd["pvesvd"]
sw_svd = dd_svd["swsvd"]
rtsvd = dd_svd["rtsvd"]
fittsvd = dd_svd["fittsvd"]
pve_tsvd = dd_svd["pvetsvd"]
sw_tsvd = dd_svd["swtsvd"]
rttsvd = dd_svd["rttsvd"]
fitisvd = dd_svd["fitisvd"]
pve_isvd = dd_svd["pveisvd"]
sw_isvd = dd_svd["swisvd"]
rtisvd = dd_svd["rtisvd"]
fitirlba = dd_svd["fitirlba"]
pve_irlba = dd_svd["pveirlba"]
sw_irlba = dd_svd["swirlba"]
rtirlba = dd_svd["rtirlba"]

# fit vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "Fit", title = "")
pcb = scatter!(ax, sws, fits, color = :green, markersize = 15, label="PCB")
sma = scatter!(ax, sws_sma, fits_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"sws_vs_fits_SMA_n_PCB_tsvd_$(optim_method).png"),f,px_per_unit=2)

# fit vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "Average fit", title = "")
pcb = scatter!(ax, sws, afits, color = :green, markersize = 15, label="PCB")
sma = scatter!(ax, sws_sma, afits_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"sws_vs_afits_SMA_n_PCB_tsvd_$(optim_method).png"),f,px_per_unit=2)

# PVE vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "PVE", title = "")
sma = scatter!(ax, sws_sma, pves_sma, color = :magenta, markersize = 15, label="SMA")
hals = scatter!(ax, sws_hals, pves_hals, color = :orange, markersize = 15, label="HALS")
cnmf = scatter!(ax, sw_cnmf, pve_cnmf, color = :pink, markersize = 15, label="CNMF")
#svd = scatter!(ax, sw_svd, pve_svd, color = :black, markersize = 15, label="SVD")
isvd = scatter!(ax, sw_isvd, pve_isvd, color = :blue, markersize = 15, label="ISVD")
tsvd = scatter!(ax, sw_tsvd, pve_tsvd, color = :red, markersize = 15, label="TSVD") # same as SVD
irlba = scatter!(ax, sw_irlba, pve_irlba, color = :cyan, markersize = 15, label="IRLBA")
pcb = scatter!(ax, sws, pves, color = :green, markersize = 15, label="PCB")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"sws_vs_PVE_of_SVD_SMA_HALS_CNMF_PCB_tsvd_$(optim_method).png"),f,px_per_unit=2)

f = AMakie.Figure()
axMain = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "PVE", title = "",ytickcolor = :white, yticklabelcolor = :white, ygridvisible = false,
leftspinecolor = :white, rightspinecolor = :white,
bottomspinecolor = :white, topspinecolor   = :white)
ylims!(axMain, (0.3, 0.8))
axLeft = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "PVE", width = Relative(0.33), halign = 0.0, xticks =[50, 290])
hidexdecorations!(axLeft)
xlims!(axLeft, (50, 290))
ylims!(axLeft, (0.3, 0.8))
axRight = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "PVE", width = Relative(0.33), halign = 0.0, xticks =[330, 350])
hidexdecorations!(axRight)
xlims!(axRight, (330, 350))
ylims!(axRight, (0.3, 0.8))
for ax ∈ [axLeft, axRight]
    pcb = scatter!(ax, sws, pves, color = :green, markersize = 15, label="PCB")
    sma = scatter!(ax, sws_sma, pves_sma, color = :magenta, markersize = 15, label="SMA")
    isvd = scatter!(ax, sw_isvd, pve_isvd, color = :red, markersize = 15, label="ISVD")
end
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"sws_vs_PVE_SMA_n_PCB_$(optim_method).png"),f,px_per_unit=2)

function makeTestFigure()
	fig = Figure(resolution = (150, 150), fontsize = 10)

	axisMain = AMakie.Axis(fig[1, 1],
		xlabel = "x Label", ylabel = "y Label",
		ytickcolor = :white, yticklabelcolor = :white, ygridvisible = false,
		leftspinecolor = :white, rightspinecolor = :white,
		bottomspinecolor = :white, topspinecolor   = :white
	)
	xlims!(axisMain, (-1, 6.))

	axisLow = AMakie.Axis(fig[1, 1],
		ylabel = "Lorem Ipsum", ylabelcolor = RGBA(1.0, 1.0, 1.0, 1.0),
    ylabelpadding = 2.0, yticklabelsize = 8, height = Relative(0.33), valign = 0.0,
		topspinecolor = :white, yticks = [0, 1]
	)
	hidexdecorations!(axisLow)
	xlims!(axisLow, (-1.,    6.))
	ylims!(axisLow, (-0.2,-0.2+1.5))

	axisHigh = AMakie.Axis(fig[1, 1],
		ylabel = "Lorem Ipsum", ylabelcolor = RGBA(1.0, 1.0, 1.0, 1.0),
    ylabelpadding = 2.0, yticklabelsize = 8, height = Relative(0.55), valign = 1.0,
		bottomspinecolor = :white, yticks = [9, 10, 11]
	)
	hidexdecorations!(axisHigh)
	xlims!(axisHigh, (-1,   6.))
	ylims!(axisHigh, (8.5, 11.))

	boxcar(t) = t > 0. && t < 5. ? 10. : 0.
	ts = range(-1.0, 6., length = 101)
	for axis ∈ [axisLow, axisHigh]
		lines!(axis, ts, boxcar.(ts))
	end

	fig
end

#==================== RCall examples ================================#
using RCall

# Load the ggplot2 library from R
R"library(ggplot2)"
# Create a sample data frame in R
R"my_data <- data.frame(x = c(1, 2, 3, 4, 5), y = c(2, 4, 6, 8, 10))"
# Create a ggplot object and display the plot
R"ggplot(my_data, aes(x = x, y = y)) + geom_point()"

# Create an array in Julia
julia_array = rand(5)
# Transfer the array to R
@rput julia_array
# Now you can use 'julia_array' in R
R"print(julia_array)"
# Execute some R code to modify the array
R"julia_array <- julia_array * 2"
R"r_array <- runif(10)"
# Retrieve an object from R to Julia
@rget julia_array
@rget r_array
R"""
# This is a multiline R code block
x <- c(1, 2, 3, 4, 5)
y <- x^2
print(y)
"""