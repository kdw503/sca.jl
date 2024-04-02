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

dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.1
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"

lcsvd_maxiter = 150
compnmf_maxiter = 1000
hals_maxiter = 150

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncells, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
ncells = ncs
(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
gtfname = "fakecells_calcium_sz$(imgsz)_lengthT$(lengthT)_SNR$(SNR)_bias$(bias)"
imsave_data(dataset,joinpath(subworkpath,gtfname),gtW,gtH',imgsz,100; saveH=false)
plotH_data(joinpath(subworkpath,gtfname),gtH'; space=0.,ylabel="",ytickformat="{:.2f}")
X = LCSVD.noisefilter(filter,X,imgsz)

subtract_bg = false
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    LCSVD.normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
    plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
    bg = W*fill(mean(H),1,n); X .-= bg
end

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)

mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; maxiter = 10; tol=-1 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
makepositive = true
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,0.0)# ("_sp",:isvd,0.005,.0) ,("_nn",:nndsvd,0.,5.0)
β1 = β2= β; α1 = α2 = α
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, store_trace = false,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
M, N = copy(M0), copy(N0)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
W, H = rst0.W, rst0.H
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)

# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
nodr = LCSVD.matchedorder(ml,ncells); Wlc, Hlc = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
makepositive && LCSVD.flip2makepos!(Wlc,Hlc,mask=:topNpix)
fprex = "$(prefix)$(SNR)db_bias$(bias)_$(sbgstr)"
fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_f$(fitval)_af$(avgfit)_it$(rst0.niters)_rti$(rt1)_rt$(rt2)")
imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false)
#continue

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
rt2 = @elapsed R"factors_sma <-sma(X, k=15)" # gamma_z=sqrt(p*k) and gamma_is default
@rget factors_sma
rW = factors_sma[:z]; rH = Array(factors_sma[:y]'); b = factors_sma[:b]; rH = b*rH
LCSVD.normalizeW!(rW,rH); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, rW); fitval = LCSVD.fitd(X,rW*rH)
nodr = LCSVD.matchedorder(ml,ncells); Wsca, Hsca = rW[:,nodr], rH[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
LCSVD.flip2makepos!(Wsca,Hsca); # Wsca[:,5:7] .*= -1; Hsca[5:7,:] .*= -1
fprex = "$(prefix)$(SNR)db_bias$(bias)_g80_$(sbgstr)"
fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_rt$(rt2)")
imsave_data(dataset,fname,Wsca,Hsca,imgsz,100; saveH=false, scalemtd=:maxcol)
plotH_data(fname*"_Hinhibit",Hsca[1:7,:]; space=0.,ylabel="",ytickformat="{:.2f}")


R"vignette(\"epca\")"