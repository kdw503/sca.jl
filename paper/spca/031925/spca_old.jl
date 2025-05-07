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

Usvd, H0svd, _ = LCSVD.initpcb(X, noc, nac; initmethod=:svd, svdmethod=:svd)
Vsvd = copy(H0svd')
Xvsvd = X*Vsvd*inv(Vsvd'*Vsvd)*Vsvd'; pve_svd = norm(Xvsvd)^2/normX2
W0svd = copy(Usvd); LCSVD.normalizeW!(W0svd,H0svd); sw_svd = norm(W0svd,1)

rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
Xv = X*V*inv(V'*V)*V'; pve_isvd = norm(Xv)^2/normX2
W0 = copy(U); LCSVD.normalizeW!(W0,H0); sw_isvd = norm(W0,1)

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
save(joinpath(subworkpath,"fits_afits_sws_PCB_$(optim_method).jld2"),"fits",fits,"afits",afits,"sws",sws,"rts",rts,"pves",pves)

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
rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, ncells, variant=:ar);
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
mfmethod = :HALS; αhals=0.1; maxiter = 60; tol=-1
W, H = copy(Whals0), copy(Hhals0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, W, H)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
nodr = LCSVD.matchedorder(ml,ncells); Whals, Hhals = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
makepositive && LCSVD.flip2makepos!(Whals,Hhals)
fprex = "$(prefix)$(SNR)db_bias$(bias)_$(sbgstr)"
fname = joinpath(subworkpath,"$(fprex)_a$(αhals)_f$(fitval)_af$(avgfit)_it$(maxiter)_rti$(rt1)_rt$(rt2)")
imsave_data(dataset,fname,Whals,Hhals,imgsz,100; saveH=false, scalemtd=:maxcol)

# COMPNMF
prefix = "compnmf"
@show prefix; flush(stdout)
mfmethod = :COMPNMF; maxiter = compnmf_maxiter
(tailstr,initmethod) = ("_nn",:lowrank_nndsvd)
dd = Dict(); tol=-1
rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn)
W, H = copy(Wcn0), copy(Hcn0);
rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, W, H)
rt1 += rst0.inittime # add calculation time for compression matrices L and R
rt2 -= rst0.inittime
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
nodr = LCSVD.matchedorder(ml,ncells); Wcn, Hcn = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
makepositive && LCSVD.flip2makepos!(Wcn,Hcn)
fprex = "$(prefix)$(SNR)db_bias$(bias)_$(sbgstr)"
fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_af$(avgfit)_it$(rst0.niters)_rti$(rt1)_rt$(rt2)")
imsave_data(dataset,fname,Wcn,Hcn,imgsz,100; saveH=false, scalemtd=:maxcol)

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

# fit vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "Fit", title = "")
pcb = scatter!(ax, sws, fits, color = :green, markersize = 15, label="PCB")
sma = scatter!(ax, sws_sma, fits_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"sws_vs_fits_SMA_n_PCB_$(optim_method).png"),f,px_per_unit=2)

# fit vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "Average fit", title = "")
pcb = scatter!(ax, sws, afits, color = :green, markersize = 15, label="PCB")
sma = scatter!(ax, sws_sma, afits_sma, color = :magenta, markersize = 15, label="SMA")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"sws_vs_afits_SMA_n_PCB_$(optim_method).png"),f,px_per_unit=2)

# PVE vs. L1 norm of W
f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "PVE", title = "")
pcb = scatter!(ax, sws, pves, color = :green, markersize = 15, label="PCB")
sma = scatter!(ax, sws_sma, pves_sma, color = :magenta, markersize = 15, label="SMA")
isvd = scatter!(ax, sw_isvd, pve_isvd, color = :red, markersize = 15, label="ISVD")
axislegend(ax, position = :rb)
save(joinpath(subworkpath,"sws_vs_PVE_SMA_n_PCB_$(optim_method).png"),f,px_per_unit=2)

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

#=============== seqRNA ===============#

R"""
library(scRNAseq)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)
library(epca)
"""

R"""
dat <- BaronPancreasData()
dim(dat)
gene.select <- !!apply(counts(dat), 1, sd)
label.select <- colData(dat) %>%
data.frame() %>%
dplyr::count(label) %>%
filter(n > 100)
dat1 <- dat[gene.select, colData(dat)$label %in% label.select$label]
"""

R"""
count <- counts(dat1)
label <- setNames(factor(dat1$label), colnames(dat1))
scar <- sca(t(count), k = 9, gamma = 12,
               center = F, scale = F,
               epsilon = 1e-3)
n.gene <- apply(!!scar$loadings, 2, sum) # sum of non zero loadings number for each gene
ngene_sma <- n.gene
Wsma <- as.matrix(scar$scores)
Hsma <- as.matrix(scar$loadings)
"""
@rget Wsma
@rget Hsma
@rget ngene_sma
@rget label
label = Array(label)

rtirlba = @elapsed R"""  ## initialize
  x = scale(x = t(count),
            center = F,
            scale = F)
  # s = RSpectra::svds(x, k)
  s = irlba::irlba(x, 9, tol = 1e-10)
  z = s$u
  b = diag(s$d)
  y = s$v
  score = sqrt(sum(s$d ^ 2))
  diff = c(z = Inf, y = Inf)
"""

rtsma = @elapsed R"""
factors_sma <- sca(t(count), k = 9, gamma = Inf,
               center = F, scale = F,
               epsilon = 1e-3)
               """
@rget factors_sma
rW = factors_sma[:scores]; rH = Array(factors_sma[:loadings]')
#LCSVD.normalizeW!(rW,rH)
fv = LCSVD.fitd(X,rW*rH)
Xy = X*rH'*inv(rH*rH')*rH; pve = norm(Xy)^2/normX2 # 0.9433741435634796
LCSVD.normalizeW!(rW,rH); sw = norm(rW,1) # 317.95

i=1; @show H[i,:][H[i,:].>0.1]; @show H[i,:][H[i,:].<-0.1];  @show gn[H[i,:].>0.1]; @show gn[H[i,:].<-0.1];

R"""
library(ggplot2)
scar$scores %>%
  reshape2::melt(varnames = c("cell", "PC"),
                 value.name = "scores") %>%
  mutate(PC = factor(PC), label = label[cell]) %>%
  ggplot(aes(PC, scores / 1000, fill = PC)) +
  geom_boxplot(color = "grey30", outlier.shape = NA,
               show.legend = FALSE) +
  labs(x = "gene PC", y = bquote("scores ("~10^3~")")) +
  scale_x_discrete(labels = 1:9) +
  facet_wrap(~ label, nrow = 3) +
  scale_fill_brewer(palette = "Set3") +
  theme_classic()
"""

# @rget scar
# W = scar[:scores]; H = scar[:loadings]; label = scar[:label]
#R"Xr = unname(count)"
R"library(Matrix)"
R"count_matrix_dense <- as.matrix(count)"
Xr = rcopy(R"count_matrix_dense")
save(joinpath(subworkpath,"Xr.jld2"),"Xr",Xr, "label", label)

ddseq = load(joinpath(subworkpath,"Xr.jld2"))
Xr = ddseq["Xr"]; X = Array(Xr')
label = ddseq["label"]

initmethod=:isvd; svdmethod=:isvd
#initmethod=:svd; svdmethod=:svd
rtisvd = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(Xr, noc, nac; initmethod=initmethod, svdmethod=svdmethod)
V = copy(H0'); N0t = copy(N0')
β1 = β2= 5.; α1 = α2 = 0.005
r=0.3; tol=1e-6
maxiter = Int(ceil(log(eps(eltype(Xr)))/log(r))) #lcsvd_maxiter
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    r=r, useprecond=false, usedenoiseW0H0=false, optim_method = :lbfgs,
    uselv=false, maxiter = maxiter, inner_maxiter = 1000, store_trace = false,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rtpcb = @elapsed rst1 = LCSVD.solve!(alg, Xr, U, V, D, M1, N1t);
W1, H1 = rst1.W, rst1.Ht'
LCSVD.flip2makepos!(W1,H1)
LCSVD.normalizeW!(W1,H1)
Wpcb, Htpcb = Array(H1'), W1
#d = LCSVD.normalizeWH!(W1,H1)
R"gn<-as.matrix(rownames(dat1))"; @rget gn # gene setNames
gns = []; gnnegs = []; ngene = []; nneggene = []
for i in 1:9
    v = Htpcb[:,i][Htpcb[:,i].>0.1]
    push!(ngene,v)
    gv = gn[Htpcb[:,i].>0.1]
    push!(gns,gv)
    v = Htpcb[:,i][Htpcb[:,i].< -0.1]
    push!(nneggene,v)
    gv = gn[Htpcb[:,i].< -0.1]
    push!(gnnegs,gv)
end

ngene = [1, # INS
        1,  # SST
        11, # CELA3A,CELA3B,CLPS,CPA1,CTRB1,CTRB2,PLA2G1B,PRSS1,PRSS2,REG1A(0.63),REG1B
        2,  # GCG(0.95),TTR(0.27)
        1,  # PPY
        1,  # IAPP
        9,  # CELA2A,CELA3A(0.33),CELA3B,CLPS,CPA1,CTRB1,PLA2G1B,PRSS1,PRSS2
        17, # ACTG1,EEF1A1(0.29),FTH1,FTL,GAPDH,RPL10,RPL13,RPL13A,RPL37A,RPL41,RPL7A,RPS12,RPS2,TIMP1,TMSB4X,TPT1,TTR
        4]  # COL1A1,CPA1,REG1B,TIMP1(0.29)
nneggene = [0,
        0,
        0,
        0,
        0,
        0,
        3, # REG1A(-0.72),REG1B,REG3A
        0,
        4]  # ACTG1,CTRB1,CTRB2(-0.73),SERPINA3

save(joinpath(subworkpath,"Baron","Result_sp_s0324.jld2"),"X",X, "cell_type_label", label, "gene_name", gn,
    "Wpcb", Wpcb, "Htpcb", Htpcb, "ngene", ngene, "nneggene", nneggene, "rtisvd", rtisvd,"rtpcb", rtpcb,
    "Wsma", Wsma, "Hsma", Hsma, "ngene_sma",ngene_sma, "rtsma", rtsma)

save(joinpath(subworkpath,"Baron","Result_sp_nn_s0322.jld2"),"X",X, "cell_type_label", label, "gene_name", gn,
    "Wpcb", Wpcb, "Htpcb", Htpcb, "ngene", ngene, "nneggene", nneggene, "gns",gns, "gnnegs", gnnegs, "rtisvd", rtisvd,"rtpcb", rtpcb,
    "Wsma", Wsma, "Hsma", Hsma, "ngene_sma",ngene_sma, "rtsma", rtsma)

@rput Wpcb
R"""
library(ggplot2)
Wpcb %>%
  reshape2::melt(varnames = c("cell", "PC"),
                 value.name = "scores") %>%
  mutate(PC = factor(PC), label = label[cell]) %>%
  ggplot(aes(PC, scores / 1000, fill = PC)) +
  geom_boxplot(color = "grey30", outlier.shape = NA,
               show.legend = FALSE) +
  labs(x = "gene PC", y = bquote("scores ("~10^3~")")) +
  scale_x_discrete(labels = 1:9) +
  facet_wrap(~ label, nrow = 3) +
  scale_fill_brewer(palette = "Set3") +
  theme_classic()
"""

Wisvd = V*D
@rput Wisvd
R"""
library(ggplot2)
Wisvd %>%
  reshape2::melt(varnames = c("cell", "PC"),
                 value.name = "scores") %>%
  mutate(PC = factor(PC), label = label[cell]) %>%
  ggplot(aes(PC, scores / 1000, fill = PC)) +
  geom_boxplot(color = "grey30", outlier.shape = NA,
               show.legend = FALSE) +
  labs(x = "gene PC", y = bquote("scores ("~10^3~")")) +
  scale_x_discrete(labels = 1:9) +
  facet_wrap(~ label, nrow = 3) +
  scale_fill_brewer(palette = "Set3") +
  theme_classic()
"""

mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# mtd = "pcb"; Whm = Wpcb; Hhm = Htpcb'
mtd = "sma"; Whm = Wsma; Hhm = Hsma'
noc = size(Whm,2)
Wdivision = 2
sz = (262, 654) # f.scene.viewport.val
for i in 1:Wdivision
    f = Figure(size=sz)
    ax = AMakie.Axis(f[1, 1], xaxisposition=:top, xlabelsize=10, xticklabelsize=10, xgridvisible=false, xticks = collect(1:noc))
    rowsizeq = size(Whm,1)÷Wdivision
    rows = (i==Wdivision ? size(Whm,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    joint_limits = (-maximum(Whm), maximum(Whm))
    hm1 = heatmap!(ax, Whm[rows,:]', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"Baron","$(mtd)_heatmap_W$i.png"),f)
end
# f = Figure()
# ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,1000000)))
# ht1 = hist!(ax,vec(Whm),bin=5)
# save(joinpath(subworkpath,"pcb_histo1000000_W.png"),f)
Hdivision = 2#16
hr = 0.1
sz = (840, 207) # f.scene.viewport.val
for i in 1:Hdivision
    f = Figure(size=sz)
    ax = AMakie.Axis(f[1, 1], xaxisposition=:top, ylabelsize=10, yticklabelsize=10, ygridvisible=false, yticks = collect(1:noc))
    colsizeq = size(Hhm,2)÷Hdivision
    cols = colsizeq*(i-1)+1:(i==Hdivision ? size(Hhm,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (-maximum(Hhm)*hr, maximum(Hhm)*hr)
    hm1 = heatmap!(ax, Hhm[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidexdecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"Baron","$(mtd)_heatmap_H$i.png"),f)
end
# f = Figure()
# ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
# ht1 = hist!(ax,vec(Hhm),bin=5)
# save(joinpath(subworkpath,"pcb_histo100_H.png"),f)

#==================== Clustering ===================================#
using NeighborhoodClustering, StatsBase

dd = load(joinpath(subworkpath,"Baron","Result_sp_s0321.jld2"))

X = dd["X"]
label = dd["cell_type_label"]
gn = dd["gene_name"]
Wpcb = dd["Wpcb"]
Htpcb = dd["Htpcb"]
ngene = dd["ngene"]
nneggene = dd["nneggene"]
rtisvd = dd["rtisvd"]
rtpcb = dd["rtpcb"]
Wsma = dd["Wsma"]
Hsma = dd["Hsma"]
ngene_sma = dd["ngene_sma"]
rtsma = dd["rtsma"]


dd = load(joinpath(subworkpath,"Baron","Result_sp_nn_s0321.jld2"))

X = dd["X"]
label = dd["cell_type_label"]
gn = dd["gene_name"]
Wpcb = dd["Wpcb"]
Htpcb = dd["Htpcb"]
ngene = dd["ngene"]
nneggene = dd["nneggene"]
gns = dd["gns"]
gnnegs = dd["gnnegs"]
rtisvd = dd["rtisvd"]
rtpcb = dd["rtpcb"]
Wsma = dd["Wsma"]
Hsma = dd["Hsma"]
ngene_sma = dd["ngene_sma"]
rtsma = dd["rtsma"]

pvalue = 0.0001;
clust = cluster(Wpcb', pvalue)

pvalue=0.0000001; nresample = 5
for pvalue in [0.000001, 0.0000001]
    for nresample in [3]
        clust = cluster_resample(Wpcb', nresample, pvalue)
        save(joinpath(subworkpath,"Baron","Clustering_p$(pvalue)_n$(nresample)_sp_nn_s0322.jld2"),"clust", clust, "pvalue", pvalue, "nresample", nresample)
        clust_sma = cluster_resample(Wsma', nresample, pvalue)
        @show maximum(clust)
    end
end
countmap(clust)
countmap(label)

for i in 1:maximum(clust)
    idx = clust .== i
    celltype = mode(label[idx])
    cmap = countmap(label[idx])
    @show i, celltype, cmap
end

Wpcbn = copy(Wpcb)
for r in eachrow(Wpcbn)
    n = norm(r)
    r ./= n
end
pvalue=0.0000001; nresample = 3
clust = cluster_resample(Wpcbn', nresample, pvalue)
maximum(clust)
countmap(clust)

Wsman = copy(Wsma)
for r in eachrow(Wsman)
    n = norm(r)
    r ./= n
end
pvalue=1e-15; nresample = 50
clust_sma = cluster_resample(Wsman', nresample, pvalue)
maximum(clust_sma)
countmap(clust_sma)

save(joinpath(subworkpath,"Baron","Clustering_p$(pvalue)_n$(nresample)_sp_nn_s0322.jld2"),"clust", clust, "clust_sma", clust_sma, "pvalue", pvalue, "nresample", nresample)
