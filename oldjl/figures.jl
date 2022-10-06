using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\documents\\NMFinitSVD\\figures")
elseif Sys.isunix()
    cd("/home/daewoo/work/documents/NMFinitSVD/figures")
end

Pkg.activate(".")

using Images, Colors
using SymmetricComponentAnalysis
using StaticArrays, IntervalSets, LinearAlgebra, Optim, NMF, BenchmarkTools
using FakeCells, AxisArrays, MappedArrays, JLD, Printf, PositiveFactorizations
using ForwardDiff, Calculus, ImageAxes, RandomizedLinAlg
using ImageCore, ImageView, Dates
#using ProfileView, Profile # ProfileView conflicts with ImageView 

save("dummy.png",rand(2,2)) # If we didn't call this line before `using PyPlot`,
                            # ImageMagick is used when saving image. That doesn't
                            # properly close the file after saving.
using PyPlot

include(joinpath(pkgdir(SymmetricComponentAnalysis),"test","testdata.jl"))
include(joinpath(pkgdir(SymmetricComponentAnalysis),"test","testutils.jl"))

function plotL(penL1, penL2, penL3, rng, bcs=nothing; showtrace = false, vminL1=nothing, vmaxL1=nothing,
        vminL2=nothing, vmaxL2=nothing, vminL3=nothing, vmaxL3=nothing, title1 = "poca2(W*[1 b;c 1+bc])",
        title2 = "poca2([1 b;c 1+bc]\\H)", title3 = "pocapair([1 b;c 1+bc],W,H)")

    tickbs = [rng[1], rng[1]/2, 0, rng[end]/2, rng[end]]
    tickblabels = [string(rng[1]), string(rng[1]/2), "0", string(rng[end]/2), string(rng[end])]
    tickcs = [rng[1], rng[1]/2, 0, rng[end]/2, rng[end]]
    tickclabels = [string(rng[1]), string(rng[1]/2), "0", string(rng[end]/2), string(rng[end])]
    extent = (rng[1],rng[end],rng[1],rng[end])
    fig, axs = plt.subplots(1, 6, figsize=(16, 4), gridspec_kw=Dict("width_ratios"=>[1,0.3,1,0.3,1,0.1]))

    ax = axs[1]
    vmaxL1===nothing && (vmaxL1 = ceil((minimum(penL1[.!isnan.(penL1)])+1)*15)/10)
    vminL1===nothing && (vminL1 = floor(minimum(penL1[.!isnan.(penL1)])*10)/10)
    #@show vmaxL1, vminL1
    himg1 = ax.imshow(reverse(penL1; dims=1); extent = extent, vmin=vminL1, vmax=vmaxL1)
    ax.set_xlabel("c")
    ax.set_ylabel("b")
    ax.set_xticks(tickcs)
    ax.set_yticks(tickbs)
    ax.set_xticklabels(tickclabels)
    ax.set_yticklabels(tickblabels)
    ax.set_title(title1)
    (showtrace && bcs !==nothing) && ax.plot(bcs[:,2],bcs[:,1],color=(1,0,0))

    ax = axs[2]
    plt.colorbar(himg1, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)

    ax = axs[3]
    vmaxL2===nothing && (vmaxL2 = ceil((minimum(penL2[.!isnan.(penL2)])+1)*15)/10)
    vminL2===nothing && (vminL2 = floor(minimum(penL2[.!isnan.(penL2)])*10)/10)
    #@show vmaxL2, vminL2
    himg2 = ax.imshow(reverse(penL2; dims=1); extent = extent, vmin=vminL2, vmax=vmaxL2)
    ax.set_xlabel("c")
    ax.set_xticks(tickcs)
    ax.set_yticks(tickbs)
    ax.set_xticklabels(tickclabels)
    ax.set_yticklabels(tickblabels)
    ax.set_title(title2)
    (showtrace && bcs !==nothing) && ax.plot(bcs[:,2],bcs[:,1],color=(1,0,0))

    ax = axs[4]
    plt.colorbar(himg2, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)

    ax = axs[5]
    vmaxL3===nothing && (vmaxL3 = ceil((minimum(penL3[.!isnan.(penL3)])+1)*15)/10)
    vminL3===nothing && (vminL3 = floor(minimum(penL3[.!isnan.(penL3)])*10)/10)
    #@show vmaxL3, vminL3
    himg3 = ax.imshow(reverse(penL3; dims=1); extent = extent, vmin=vminL3, vmax=vmaxL3)
    ax.set_xlabel("c")
    ax.set_xticks(tickcs)
    ax.set_yticks(tickbs)
    ax.set_xticklabels(tickclabels)
    ax.set_yticklabels(tickblabels)
    ax.set_title(title3)
    (showtrace && bcs !==nothing) && ax.plot(bcs[:,2],bcs[:,1],color=(1,0,0))

    ax = axs[6]
    plt.colorbar(himg3, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)
    axs
end

function plot_semipen(pen1, rng, xys=nothing; showtrace = false, vminL1=nothing, vmaxL1=nothing, title1 = "||I-Mw*Mh||^2")

    tickxs = [rng[1], rng[end]/4, rng[end]/2, rng[end]*3/4, rng[end]]
    tickxlabels = [string(rng[1]), string(rng[end]/4), string(rng[end]/2), string(rng[end]*3/4), string(rng[end])]
    tickys = [rng[1], rng[end]/4, rng[end]/2, rng[end]*3/4, rng[end]]
    tickylabels = [string(rng[1]), string(rng[end]/4), string(rng[end]/2), string(rng[end]*3/4), string(rng[end])]
    extent = (rng[1],rng[end],rng[1],rng[end])
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw=Dict("width_ratios"=>[1,0.3]))

    ax = axs[1]
    himg1 = ax.imshow(reverse(pen1; dims=1); extent = extent)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(tickxs)
    ax.set_yticks(tickys)
    ax.set_xticklabels(tickxlabels)
    ax.set_yticklabels(tickylabels)
    ax.set_title(title1)
    (showtrace && xys !==nothing) && ax.plot(xys[:,2],xys[:,1],color=(1,0,0))

    ax = axs[2]
    plt.colorbar(himg1, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)
    axs
end

function plot_semipen2(pen1, rng1, rng2, xys=nothing; showtrace = false, vminL1=nothing, vmaxL1=nothing, title1 = "||I-Mw*Mh||^2")

    tickxs = [rng1[1], rng1[1]+(rng1[end]-rng1[1])/4, rng1[1]+(rng1[end]-rng1[1])/2, rng1[1]+(rng1[end]-rng1[1])*3/4, rng1[end]]
    tickxlabels = [@sprintf("%1.2f",tick) for tick in tickxs]
    tickys = [rng2[1], rng2[1]+(rng2[end]-rng2[1])/4, rng2[1]+(rng2[end]-rng2[1])/2, rng2[1]+(rng2[end]-rng2[1])*3/4, rng2[end]]
    tickylabels = [@sprintf("%1.2f",tick) for tick in tickys]
    extent = (rng1[1],rng1[end],rng2[1],rng2[end])
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw=Dict("width_ratios"=>[1,0.3]))

    vmaxL1==nothing && (vmaxL1 = ceil((minimum(pen1[.!isnan.(pen1)])+1)*15)/10)
    vminL1==nothing && (vminL1 = floor(minimum(pen1[.!isnan.(pen1)])*10)/10)
    ax = axs[1]
    himg1 = ax.imshow(reverse(pen1; dims=1); extent = extent, vmin=vminL1, vmax=vmaxL1)
    ax.set_xlabel("θ1")
    ax.set_ylabel("θ2")
    ax.set_xticks(tickxs)
    ax.set_yticks(tickys)
    ax.set_xticklabels(tickxlabels)
    ax.set_yticklabels(tickylabels)
    ax.set_title(title1)
    (showtrace && xys !==nothing) && ax.plot(xys[:,2],xys[:,1],color=(1,0,0),marker="x")

    ax = axs[2]
    plt.colorbar(himg1, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)
    axs
end

function balance_power(W, X)
    H = W\X
    sqpwrsw = sqrt.(norm.(eachcol(W)))
    sqpwrsh = sqrt.(norm.(eachrow(H)))
    balfacs = sqpwrsh./sqpwrsw
    Wn = W*Diagonal(balfacs)  
    Hn = Diagonal(1 ./balfacs)*H  
    Wn, Hn
end

# Dataset 0 : orthogonality = 0.0
gtW = [0.962224   0; 0  0.0595547]
gtH = [0.0270566  0; 0  0.49868]

# Dataset 1 : orthogonality = 0.002126245342057767,  gtM = [ 1.0  0.067171; 0.668001  -0.0448669],  [-1.0 -0.0665589; 0.660016  -0.0444727]
gtW = [0.962224   0.0242178; 0.0116875  0.0595547]
gtH = [0.0270566  0.0198188; 0.0155136  0.49868]

# Dataset 2 : orthognality = 0.22341729948373742, gtM = [ -1.0  -1.51289;  -0.655804   0.442679]
# gtW = [0.584652  0.0508351; 0.447497  0.969165]
# gtH = [0.581042  0.317224;  0.299005  0.836679]

 # Dataset 3 : orthognality = 0.019759032621335056
# gtW = [0.269469   0.048049; 0.0044171  0.0599257]
# gtH = [0.0662107  0.0484804; 0.0101173  0.831084]
X = gtW*gtH
U = svd(X).U
W0 = copy(U); H0 = W0\X
Wn, Hn = balance_power(W0,X)

stparams = StepParams(γ=0., β=50, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W, H, objvals, trs = scasolve(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);

bcs = zeros(length(Ms[idx]),3)
for (i,M) in enumerate(Ms[idx])
    M ./= abs(M[1,1])
    bcs[i,1] = M[1,2]
    bcs[i,2] = M[2,1]
    bcs[i,3] = M[2,2]
    @show 1+bcs[i,1]*bcs[i,2], bcs[i,3]
end

showtrace = false
rng = -3:0.01:3
M2(b,c) = [1 b; c 1+b*c] #[1 b; c 1+b*c]
pocapair0(M, W, H) = sca2(W*M) * sca2(svd(M)\H)
penW=[sca2(W0*M2(b,c)) for b in rng, c in rng]
penH=[sca2(svd(M2(b,c))\H0) for b in rng, c in rng]
penpair=[scapair(M2(b,c),W0,H0) for b in rng, c in rng]
# Dataset 0
# POCA.plotL(penW,penH,penpair,rng;vmaxL1=0.5,vminL1=0,vmaxL2=0.01,vminL2=0,vmaxL3=0.004,vminL3=0)
# POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng; vmaxL1=2,vminL1=-5, vmaxL2=-1, vminL2=-5,vmaxL3 = -1, vminL3=-5)
# Dataset 1
# POCA.plotL(penW,penH,penpair,rng,bcs;vmaxL1=poca2(W0),vminL1=0,vmaxL2=0.002,vminL2=0,vmaxL3=0.002,vminL3=0,showtrace=showtrace)
# POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng,bcs; vmaxL1=1,vminL1=-0.2,vmaxL2=-2.0,vminL2=-5.0,vmaxL3 = -2.0, vminL3=-5.0,showtrace=showtrace)
plotL(penW,penH,penpair,rng;vmaxL1=sca2(W0),vminL1=0,vmaxL2=0.002,vminL2=0,vmaxL3=0.002,vminL3=0,showtrace=showtrace)
plotL(log10.(penW),log10.(penH),log10.(penpair),rng; vmaxL1=1,vminL1=-0.2,vmaxL2=-2.0,vminL2=-5.0,vmaxL3 = -2.0, vminL3=-5.0,showtrace=showtrace)
# Dataset 2
# POCA.plotL(penW,penH,penpair,rng,bcs;vmaxL1=poca2(W0),vminL1=0,vmaxL2=4,vminL2=0,vmaxL3=4,vminL3=0)
# POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng,bcs; vmaxL2=2, vminL2=-0.2,vmaxL3 = 2, vminL3=-0.2)
plotL(penW,penH,penpair,rng;vmaxL1=sca2(W0),vminL1=0,vmaxL2=4,vminL2=0,vmaxL3=4,vminL3=0)
plotL(log10.(penW),log10.(penH),log10.(penpair),rng; vmaxL2=2, vminL2=-0.2,vmaxL3 = 2, vminL3=-0.2)


showtrace = false
rng = 0:0.01:4
function pen(Mw0, Mh0, dMwi, dMhi, x, y)
    norm(I-Mw0*(I+x*dMwi)*(I+y*dMhi)*Mh0)^2
end
penM=[pen(Mw0, Mh0, dMwi, dMhi, x, y) for x in rng, y in rng]
conWH=[sca2(Wn*Mw0*(I+x*dMwi))+sca2((I+y*dMhi)*Mh0*Hn) for x in rng, y in rng]
plot_semipen(penM, rng, title1 = "||I-Mw0*(I+x*dMw)*(I+y*dMh)*Mh0||^2")
plot_semipen(log.(penM), rng, title1 = "||I-Mw0*(I+x*dMw)*(I+y*dMh)*Mh0||^2")
plot_semipen(log.(conWH), rng, title1 = "c1(Mw0*(I+x*dMw))+c2((I+y*dMh)*Mh0)")


W = [1 1; 1 -1]; H = [ones(1, 9); 0.01*sin.(2*π*(0:8)/8)']; invW = svd(W)\I
# X = rand(2,10000); F = svd(X); W = F.U[:,1:2]; H = W\X; invW = svd(W)\I
# Wgt = rand(2,2); Hgt = rand(2,10000); X = Wgt*Hgt; F = svd(X); W = F.U[:,1:2]; H = W\X; invW = svd(W)\I
rI = [1., 0., 0., 1.]
WMw(θ1, θ2) = [cos(θ1) cos(θ2); sin(θ1) sin(θ2)]
Mwf(θ1, θ2) = invW*WMw(θ1, θ2)
Awf(θ1, θ2) = [ Mwf(θ1, θ2)[1,1] Mwf(θ1, θ2)[1,2] 0. 0.;
                Mwf(θ1, θ2)[2,1] Mwf(θ1, θ2)[2,2] 0. 0.;
                0. 0. Mwf(θ1, θ2)[1,1] Mwf(θ1, θ2)[1,2];
                0. 0. Mwf(θ1, θ2)[2,1] Mwf(θ1, θ2)[2,2]]
xhf(θ1, θ2) = svd(Awf(θ1, θ2))\rI
penf(θ1, θ2) = norm(rI - Awf(θ1, θ2)*xhf(θ1, θ2))^2
conWf(θ1, θ2) = sca2(W*Mwf(θ1, θ2))
conHf(θ1, θ2) = sca2(reshape(xhf(θ1, θ2),(2,2))*H)

θ1rng = -pi:0.01:pi; θ2rng = -pi:0.01:pi
#θ1rng = 0:0.001:pi/2; θ2rng = 0:0.001:pi/2

penM=[penf(θ1,θ2) for θ1 in θ1rng, θ2 in θ2rng]
conW=[conWf(θ1, θ2) for θ1 in θ1rng, θ2 in θ2rng]
conH=[conHf(θ1, θ2) for θ1 in θ1rng, θ2 in θ2rng]
conWH=conW+conH
# plot_semipen2(penM, θ1rng, vmaxL1=1.5, vminL1 = 0, title1 = "||I-Mw(θ1,θ2)*Mh||^2")
# plot_semipen2(conWH, θ1rng, title1 = "c1(θ1,θ2)+c2(θ1,θ2)")
# plot_semipen2(log.(penM), θ1rng, title1 = "log(||I-Mw(θ1,θ2)*Mh||^2)")
plot_semipen2(log.(conW), θ1rng, θ2rng, title1 = "log(c1(θ1,θ2))")
plot_semipen2(log.(conH), θ1rng, θ2rng, title1 = "log(c2(θ1,θ2))")
plot_semipen2(log.(conWH), θ1rng, θ2rng, [θ1 θ2], showtrace=true, title1 = "log(c1(θ1,θ2)+c2(θ1,θ2))")

#=== MIT-CBCL-face dataset = (19,19,2429) ==#

function interpolate(img,n)
    imginter = zeros(size(img).*(n+1))
    offset = CartesianIndex(1,1)
    for i in CartesianIndices(img)
        ii = (i-offset)*n+i
        for j in CartesianIndices((n+1,n+1))
            imginter[ii+j-offset] = img[i]
        end
    end
    imginter
end

using Printf
path = "C:\\Users\\kdw76\\WUSTL\\Work\\Data\\MIT-CBCL-face\\face.train\\train\\face"
nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
X = zeros(nRow*nCol,nFace)
for i in 1:2429
    fname = "face"*@sprintf("%05d",i)*".pgm"
    img = load(joinpath(path,fname))
    X[:,i] = vec(img)
end

ncells = 49; clamp_level=0.5; bordersz=1

rt01 = @elapsed W, H = initWH(X, ncells; svd_method=:svd)
Wn, Hn = balance_power(W,X);

# semisymmetric
stparams = StepParams(r=0.1, option=1, recttype=:rectcolumnonly, optim_method=:unconstrained, penaltytype=:semisymmetric_cbyc)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=200, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = scasolve3(Wn, Hn, X; stparams=stparams, cparams=cparams);
#imshowW(W1,imgsz,bordersz=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(W2_clamped, imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_CG.png", W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imshowW(W2,imgsz, gridcols=7, bordersz=bordersz)
imsaveW("Face_CG_color.png", W2,imgsz, gridcols=7, bordersz=bordersz)

# CG Orthogonality
stparams = StepParams(γ=0., β=5, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = scasolve(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,border=true); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(W2_clamped, imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_CG.png", W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imshowW(W2,imgsz, gridcols=7, bordersz=bordersz)
imsaveW("Face_CG_color.png", W2,imgsz, gridcols=7, bordersz=bordersz)
i = 100
ImageView.imshow(map(clamp01nan, reshape(X[:,i],imgsz...)))
save("Face_Org$i.png", map(clamp01nan, reshape(X[:,i],imgsz...)))
ImageView.imshow(map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Face_CG$i.png", map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Face_CG_clamped$i.png", map(clamp01nan, reshape(clamp.(W1,0.,Inf)*clamp.(H1[:,i],0.,Inf),imgsz...)))
ImageView.imshow(map(clamp01nan, reshape(H1[:,i],7,7)))
save("Face_CG_H$i.png", interpolate(map(clamp01nan, reshape(H1[:,i],7,7)),3))
intplimg = interpolate(reshape(H1[:,i],7,7),3)
imshowW(reshape(intplimg,length(intplimg),1),size(intplimg))
imsaveW("Face_CG_H_color$i.png",reshape(intplimg,length(intplimg),1),size(intplimg))

# AC Sparsity
stparams = StepParams(β=0.1, r=1, order=0, optim_method=:constrained, penaltytype=:ac_symmetric_sparsity_sum);
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=0.5, ρ=0.5, maxiter=100);
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true);
rt = @elapsed W1, H1, objvals, trs, penW, penH = scasolve(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_AC.png", W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_AC_color.png", W2,imgsz, gridcols=7, bordersz=bordersz)
i = 100
ImageView.imshow(map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Face_AC$i.png", map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Face_AC_clamped$i.png", map(clamp01nan, reshape(clamp.(W1,0.,Inf)*clamp.(H1[:,i],0.,Inf),imgsz...)))
ImageView.imshow(map(clamp01nan, reshape(H1[:,i],7,7)))
save("Face_AC_H$i.png", interpolate(map(clamp01nan, reshape(H1[:,i],7,7)),3))
intplimg = interpolate(reshape(H1[:,i],7,7),3)
imshowW(reshape(intplimg,length(intplimg),1),size(intplimg))
imsaveW("Face_AC_H_color$i.png",reshape(intplimg,length(intplimg),1),size(intplimg))

# BC Sparsity
stparams = StepParams(β=0.1, r=1, optim_method=:both_constrained, penaltytype=:ac_symmetric_sparsity_sum) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=0.5, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = scasolve(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_BC.png", W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_BC_color.png", W2,imgsz, gridcols=7, bordersz=bordersz)
i = 100
ImageView.imshow(map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Face_BC$i.png", map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Face_BC_clamped$i.png", map(clamp01nan, reshape(clamp.(W1,0.,Inf)*clamp.(H1[:,i],0.,Inf),imgsz...)))
ImageView.imshow(map(clamp01nan, reshape(H1[:,i],7,7)))
save("Face_BC_H$i.png", interpolate(map(clamp01nan, reshape(H1[:,i],7,7)),3))
intplimg = interpolate(reshape(H1[:,i],7,7),3)
imshowW(reshape(intplimg,length(intplimg),1),size(intplimg))
imsaveW("Face_BC_H_color$i.png",reshape(intplimg,length(intplimg),1),size(intplimg))

# Newton skew symmetric
stparams = StepParams(γ=0., β=0, order=0, optim_method=:newton, useprecond=true, skew=true, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = scasolve(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_Skew.png", W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)

# Semi symmetric
stparams = StepParams(γ=0., β=50, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:semisymmetricWH_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=5e-1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:2))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = scasolve2(Wn, Hn, X; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,border=true); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_SemiWH.png", W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Face_SemiWH_color.png", W2,imgsz, gridcols=7, bordersz=bordersz)

# CD
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.0 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd); Wcd_max = maximum(abs,Wcd)*clamp_level; Wcd_clamped = clamp.(Wcd,0.,Wcd_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(Wcd,imgsz, gridcols=7, colors=colors, borderval=Wcd_max, bordersz=bordersz)
imsaveW("Face_CD_a$(α).png", Wcd,imgsz, gridcols=7, colors=colors, borderval=Wcd_max, bordersz=bordersz)
i = 100
ImageView.imshow(map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Face_CD$(i)_a$(α).png", map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
ImageView.imshow(map(clamp01nan, reshape(Hcd[:,i],7,7)))
save("Face_CD_H$(i)_a$(α).png", interpolate(map(clamp01nan, reshape(Hcd[:,i],7,7)),3))

Wsol = copy(Wcd); Hsol = copy(Hcd)
Wsol[Wsol.<0].=0; Hsol[Hsol.<0].=0
Mw = Wn\Wsol; Mh = Hsol/Hn
Wsol = Wn*Mw; Hsol = Mh*Hn
@show sca2(Wsol), sca2(Hsol)

normalizeWH!(Wsol,Hsol); Wsol_max = maximum(abs,Wsol)*clamp_level; Wsol_clamped = clamp.(Wsol,0.,Wsol_max)
colors = (colorant"white", colorant"white", colorant"black")
imshowW(Wsol,imgsz, gridcols=7, colors=colors, borderval=Wsol_max, bordersz=bordersz)
imsaveW("Face_sol_from_Mw.png", Wsol,imgsz, gridcols=7, colors=colors, borderval=Wsol_max, bordersz=bordersz)

#=== Hyperspectral-Urban dataset = (307,307,210) ==#
using MAT

function plotH(H, figsz=(7,8))
    (hrow,hcol) = size(H)
    tickxs = collect(0:50:hcol-1)
    fig, axs = plt.subplots((hrow-1)÷2+1, 2, figsize=figsz, gridspec_kw=Dict("width_ratios"=>[1,1]))

    for i in 1:hrow
        ax = axs[i]
        ax.plot(0:hcol-1, H[i,:])
        ax.set_xticks(tickxs)
        # ax.set_yticks(tickds)
        # ax.set_xticklabels(tickmlabels)
        # ax.set_yticklabels(tickdlabels)
        ax.set_title("$i")
    end
end

vars = matread("C:\\Users\\kdw76\\WUSTL\\Work\\Data\\Hyperspectral-Urban\\Urban_R162.mat")
Y=vars["Y"]; nRow = Int(vars["nRow"]); nCol = Int(vars["nCol"]); nBand = 162; imgsz = (nRow, nCol)
maxvalue = maximum(Y)
X = Array(Y')./maxvalue;  # reinterpret(N0f8,UInt8(255))=1.0N0f8 but vars["Y"] has maximum 1000
img = reshape(X, (nRow, nCol, nBand));
ncells = 7; clamp_level=0.2; bordersz=5

rt01 = @elapsed W, H = initWH(X, ncells; svd_method=:svd)
Wn, Hn = balance_power(W,X);

# CG Orthogonality
stparams = StepParams(γ=0., β=5, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = scasolve(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"black", colorant"black", colorant"white")
imshowW(W2_clamped[:,2:end],imgsz, gridcols=2, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Urban_CG.png", W2_clamped[:,2:end],imgsz, gridcols=2, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Urban_CG_color.png", W2,imgsz, gridcols=2, bordersz=bordersz)
i = 100
ImageView.imshow(map(clamp01nan, reshape(X[:,i],imgsz...)))
save("Urban_Org$i.png", map(clamp01nan, reshape(X[:,i],imgsz...)))
ImageView.imshow(map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Urban_CG$i.png", map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Urban_CG_clamped$i.png", map(clamp01nan, reshape(clamp.(W1,0.,Inf)*clamp.(H1[:,i],0.,Inf),imgsz...)))

plotH(H2[2:end,:])
plt.savefig("Urban_CG_H.png")

# AC Sparsity
stparams = StepParams(β=0.1, r=1, order=0, optim_method=:constrained, penaltytype=:ac_symmetric_sparsity_sum);
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=0.5, ρ=0.5, maxiter=100);
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true);
rt = @elapsed W1, H1, objvals, trs, penW, penH = scasolve(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"black", colorant"black", colorant"white")
imshowW(W2_clamped[:,2:end],imgsz, gridcols=2, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Urban_AC.png", W2_clamped[:,2:end],imgsz, gridcols=2, colors=colors, borderval=W2_max, bordersz=bordersz)
imsaveW("Urban_AC_color.png", W2,imgsz, gridcols=2, bordersz=bordersz)
i = 100
ImageView.imshow(map(clamp01nan, reshape(X[:,i],imgsz...)))
save("Urban_Org$i.png", map(clamp01nan, reshape(X[:,i],imgsz...)))
ImageView.imshow(map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Urban_AC$i.png", map(clamp01nan, reshape(W1*H1[:,i],imgsz...)))
save("Urban_AC_clamped$i.png", map(clamp01nan, reshape(clamp.(W1,0.,Inf)*clamp.(H1[:,i],0.,Inf),imgsz...)))

plotH(H2[2:end,:])
plt.savefig("Urban_AC_H.png")

# CD
ncells=6
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.1 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5, regularization=:transformation), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd); Wcd_max = maximum(abs,Wcd)*clamp_level; Wcd_clamped = clamp.(Wcd,0.,Wcd_max)
colors = (colorant"black", colorant"black", colorant"white")
imshowW(Wcd_clamped,imgsz, gridcols=2, colors=colors, borderval=Wcd_max, bordersz=bordersz)
imsaveW("Urban_CD_regH_a$(α).png", Wcd_clamped,imgsz, gridcols=2, colors=colors, borderval=Wcd_max, bordersz=bordersz)
i = 100
ImageView.imshow(map(clamp01nan, reshape(Wcd*Hcd[:,i],imgsz...)))
save("Urban_CD_regH_$(i)_a$(α).png", map(clamp01nan, reshape(Wcd*Hcd[:,i],imgsz...)))

plotH(Hcd)
plt.savefig("Urban_CD_regH_H_a$(α).png")



data = "Face" # "Urban"
colors = (colorant"white", colorant"white", colorant"black")
# Mulitiplicative 0.17sec
α = 0.1
runtime1 = @elapsed W2, H2 = NMF.nndsvd(X, ncells, variant=:ar)
runtime2 = @elapsed NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse,maxiter=200, lambda_h=α), X, W2, H2)
normalizeWH!(W2,H2); W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
colors = (colorant"white", colorant"white", colorant"black")
imsaveW("$(data)_Multiplicative_W_a$(α).png", W2,imgsz, gridcols=7, colors=colors, borderval=W2_max, bordersz=bordersz)
plotH(H2)
plt.savefig("$(data)_Multiplicative_H_a$(α).png",W2)

# NaiveALS : 0.14sec
α = 0.1
runtime1 = @elapsed W4, H4 = NMF.nndsvd(X, ncells, variant=:ar)
runtime2 = @elapsed NMF.solve!(NMF.ProjectedALS{Float64}(maxiter=200, lambda_h=α), X, W4, H4)
normalizeWH!(W4,H4); W4_max = maximum(abs,W4)*clamp_level; W4_clamped = clamp.(W4,0.,W6_max)
imsaveW("$(data)_NaiveALS_W.png", W4,imgsz, gridcols=7, colors=colors, borderval=W4_max, bordersz=bordersz)
plotH(H4)
plt.savefig("$(data)_NaiveALS_H_a$(α).png")

# CoordinateDescent : 25.1sec
α = 0.1
runtime1 = @elapsed W8, H8 = NMF.nndsvd(X, ncells, variant=:ar)
runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000,α=α,regularization=:transformation), X, W8, H8)
normalizeWH!(W8,H8); W8_max = maximum(abs,W8)*clamp_level; W8_clamped = clamp.(W8,0.,W8_max)
imsaveW("$(data)_CoordinateDescent_W_a$(α).png", W8,imgsz, gridcols=7, colors=colors, borderval=W8_max, bordersz=bordersz)
plotH(H8)
plt.savefig("$(data)_CoordinateDescent_H_a$(α).png")

# Greedy CoordinateDescent : 0.95sec
α = 0.1
runtime1 = @elapsed W10, H10 = NMF.nndsvd(X, ncells, variant=:ar)
runtime2 = @elapsed NMF.solve!(NMF.GreedyCD{Float64}(maxiter=200, lambda_h=α), X, W10, H10) # maxiter=50
normalizeWH!(W10,H10); W10_max = maximum(abs,W10)*clamp_level; W10_clamped = clamp.(W10,0.,W10_max)
imsaveW("$(data)_GreedyCD_W_a$(α).png", W10,imgsz, gridcols=7, colors=colors, borderval=W10_max, bordersz=bordersz)
plotH(H10)
plt.savefig("$(data)_GreedyCD_H_a$(α).png")

# ProjectedGDALS : 599.5sec 
runtime1 = @elapsed W6, H6 = NMF.nndsvd(X, ncells, variant=:ar)
runtime2 = @elapsed NMF.solve!(NMF.ALSPGrad{Float64}(maxiter=200, tolg=1.0e-6), X, W6, H6)
normalizeWH!(W6,H6); W6_max = maximum(abs,W6)*clamp_level; W6_clamped = clamp.(W6,0.,W6_max)
imsaveW("$(data)_ProjectedGDALS_W.png", W6,imgsz, gridcols=7, colors=colors, borderval=W6_max, bordersz=bordersz)
plotH(H6)
plt.savefig("$(data)_ProjectedGDALS_H.png")

