
using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

Pkg.activate(".")

#using MultivariateStats # for ICA
using Images, Convex, SCS, LinearAlgebra, Printf, Colors
using FakeCells, AxisArrays, ImageCore, MappedArrays
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using Clustering # kmean, 
using SymmetricComponentAnalysis
SCA = SymmetricComponentAnalysis

Images.save("dummy.png",rand(2,2)) # If we didn't call this line before `using PyPlot`,
                            # ImageMagick is used when saving image. That doesn't
                            # properly close the file after saving.
using PyPlot
scapath = joinpath(dirname(pathof(SymmetricComponentAnalysis)),"..")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))

plt.ioff()

function convex!(Mw,Mh,W0,H0,sd_group,order,λ1,λ2,β1,β2,maxiter,SNR; poweradjust=:none, fprefix="", tol=1e-6)
    sd_group ∉ [:column, :ac, :pixel] && error("Unsupproted sd_group")
    rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,order;
                    poweradjust=poweradjust, fprefix=fprefix, sd_group=sd_group, SNR=SNR, store_trace=true)
    Mw, Mh, f_xs, x_abss, iter, rt2, trs
end

function ssca!(Mw,Mh,W0,H0,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none, tol=1e-6)
    if sd_group == :column
        sd_grp = "col"
        method = :cbyc_uc
    elseif sd_group == :component
        sd_grp = "comp"
        method = :ac_simple
    else
        error("Unsupported method")
    end
    sd_group ∉ [:column, :ac, :pixel] && error("Unsupproted sd_group")
    stparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=:WkHk, order=order, hfirst=true,
            processorder=:none, poweradjust=poweradjust, method=method, rectify=:truncate) 
    lsparams = LineSearchParams(method=:none, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
            iterations_to_show=[499,500])
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
    iter = length(trs)
    Mw, Mh, f_xs, x_abss, iter, rt2, trs
end

function hals!(Wcd, Hcd, α1, α2, maxiter;  tol=1e-5)
    stparams = StepParams(β1=α1, β2=α2, hfirst=true, processorder=:none, poweradjust=:none,
            rectify=:truncate) 
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
    rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
    x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
    iter = length(trs)
    Wcd, Hcd, f_xs, x_abss, iter, rt2, trs
end

function nmf(X, ncells; tol=1e-5) # for rank-1
    stparams = StepParams(β1=0.1, β2=0.1, hfirst=true, processorder=:none, poweradjust=:none,
            rectify=:truncate) 
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=2, maxiter=100, store_trace=true, show_trace=true)
    rt1 = @elapsed W0, H0, Mw, Mh, Wcd, Hcd = initsemisca(X, ncells, poweradjust=:normalize, use_nndsvd=true)
    rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
    Wcd, Hcd
end

function pwradjust!(W0, H0, Wp, Hp, poweradjust)
    if poweradjust==:normalize
        SCA.normalizeWH!(W0,H0)
        SCA.normalizeWH!(Wp,Hp)
    elseif poweradjust==:balance
        SCA.balance_power!(W0,H0)
        SCA.balance_power!(Wp,Hp)
    else
        error("Unknown power adjustment method")
    end
end

function disp_n_save(W1, H1, W2, H2, f_xs, x_abss, iter, rt1, rt2, SNR; imgsz=(40,20), fprefix1="", fprefix2="")
    initimgfname = fprefix1*"_rt$(rt1).png"
    nmfimgfname = fprefix2*"_itr$(iter)_rt$(rt2).png"
    pltfname = fprefix2*"_itr$(iter)_rt$(rt2)_plt.png"
    jldfname = fprefix2*"_itr$(iter)_rt$(rt2).jld"
    normalizeWH!(W1,H1); normalizeWH!(W2,H2)
    if SNR == "face"
        clamp_level=0.5;
        signedcolors = (colorant"green1", colorant"white", colorant"magenta")
        W1_max = maximum(abs,W1)*clamp_level
        imsaveW(initimgfname, W1, imgsz, gridcols=7, colors=signedcolors, borderval=W1_max, borderwidth=1)
        W2_max = maximum(abs,W2)*clamp_level
        imsaveW(nmfimgfname, W2, imgsz, gridcols=7, colors=signedcolors, borderval=W2_max, borderwidth=1)
    else
        imsaveW(initimgfname, sortWHslices(W1,H1)[1],imgsz,borderwidth=1)
        imsaveW(nmfimgfname,sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
    end
    ax = plotW([log10.(x_abss) log10.(f_xs)], pltfname; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
    save(jldfname, "f_xs", f_xs, "x_abss", x_abss, "rt1", rt1, "rt2", rt2)
end

#======== Prameters ============#

#ARGS =  ["[\"face\"]","1","1","true",":column","[1000]","[10]","\"nndsvd\"","\"convex\""] # convex, ssca, hals
#ARGS =  ["[10]","1","1","true",":column","[1000]","[10]","\"nndsvd\"","\"convex\""] # convex, ssca, hals

SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
init_method = eval(Meta.parse(ARGS[8])); mf_method = eval(Meta.parse(ARGS[9]))
λ1=λ2=λs[1]; β1=βs[1]; β2= Wonly ? 0 : βs[1]

#======== NMF Initialization ============#
SNR = SNRs[1]; initpwradj=:balance
if SNR == "face"
    filepath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
    nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
    X = zeros(nRow*nCol,nFace)
    for i in 1:nFace
        fname = "face"*@sprintf("%05d",i)*".pgm"
        img = load(joinpath(filepath,fname))
        X[:,i] = vec(img)
    end
    ncells = 49;  borderwidth=1
    fprefix0 = "Wuc_face"
    # W1,H1 = copy(W0), copy(H0); normalizeWH!(W1,H1)
    # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
    # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W1,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
    # normalizeWH!(Wp,Hp)
    # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wp,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
else
    imgsize = (40,20); lengthT=1000; jitter=0; SNR=10
    X, imgsz, ncells, fakecells_dic, _ = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
    fprefix0 = "Wuc_$(SNR)dB"
end
m, n = size(X); p = ncells
rt1 = @elapsed W0, H0, _ = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=false)
fprefix1 = fprefix0*"_init-$(init_method)"
fprefix2 = fprefix1*"_$(mf_method)_$(sd_group)_L$(order)_lw$(λ1)_lh$(λ2)_bw$(β1)_bh$(β2)"

# SVD
if init_method == "svd"
    rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=false)
end

# NNDSVD
if init_method == "nndsvd"
    rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
end

# Random-C : average of q columns of the k largest column of X.
if init_method == "randc"
    k = Int(round(n/2)); q = Int(round(k/100));
    sindices = sortperm(norm.(eachcol(X)),rev=true)[1:k]
    Wp = zeros(size(W0)...)
    for i = 1:p
        rindices = rand(1:k, q)
        Wp[:,i] .= sum(X[:,sindices[rindices]],dims=2)./q
    end
    Hp = Wp\X
end

# Xcol Rand
if init_method == "xcol"
    q=10; Wp = zeros(size(X,1),ncells)
    for i in 1:p
        rindices = rand(1:1000,q)
        Wp[:,i] = sum(X[:,rindices],dims=2)./q
    end
    Hp = Wp\X
end

# CUR-based
# statistically(calculated importance) selected columns of X.
# (normalized statistical leverage score, spectral angular distance,
# or symmetrized KL-divergence)

if init_method == "curbased"
end

# k-means
# W as centroids of the best k clusters of data, H as the distances from each
# data point to every centroid.
if init_method == "k-mean"
    W1,H1 = copy(W0), copy(H0)
    R = kmeans(X, ncells; maxiter=200, display=:iter)
    a = assignments(R)
    Wp = zeros(m,p)
    for i in 1:ncells
        num = sum(a.==i)
        wpi = sum(X[:,a.==i],dims=2)/num
        Wp[:,i] = wpi
    end
    Hp = Wp\X
end

# rank-1
# rank-1 NMF for the clustered disjoint set
if init_method == "rank1"
    R = kmeans(X, ncells; maxiter=200, display=:iter)
    a = assignments(R)
    #= average
    wsum = zeros(m, ncells); ncnt=zeros(ncells)
    for (i,ci) in enumerate(a)
        wsum[:,ci] .+= X[:,i]
        ncnt[ci] += 1
    end
    Wp = wsum/Diagonal(ncnt)
    =#
    Wp = zeros(m,ncells)
    for i in 1:ncells
        wpi, _ = nmf(X[:,a.==i],1) # rank-1 nmf
        Wp[:,i] = wpi
    end
    Hp = Wp\X
end

# SPA(Successive Projection Algorithm)
# initialize 𝑆=𝑋 and succesively project 𝑆 to the 𝑝_𝑗=max(𝑠_𝑗 )⁡‖𝑠_𝑗 ‖^2
# and update 𝑆 ← 𝑆−proj(𝑝_𝑗,𝑆) then 𝑊=[𝑝_1,𝑝_2,…,𝑝_𝑟]
if init_method == "spa"
    Wp=zeros(m,0)
    S=copy(X)
    for i in 1:ncells
        maxi = argmax(norm.(eachcol(S)))
        si = S[:,maxi]
        global Wp = hcat(Wp,si)
        global S -= si*(si'./norm(si)^2*S)
    end
    Hp = Wp\X
end

# HC (Hierarchical Clustering)
# merge dependent rows of X until r cluster is left and set it as a H
# initialization
if init_method == "hc"
end

# SC (Subtracting Clustering)
# calculate the potentials pi of being cluster center of every point by
# summing gaussian distances with neighbors. Then choose the maximum
# point as 1st cluster center and subtract a potential from each
# remaining neighbor with the center. Repeat until remaining pi < 0.15p1
if init_method == "sc"
end

# nnICA
# W as the absolute or truncation of the ICA component of X.
if init_method == "nnica"
    ica = fit(ICA, X', ncells; do_whiten = true, maxiter = 1000, tol = 1e0, mean = nothing)
    Wp = Array(transform(ica, X')'); Wp[Wp.<0].=0
    Hp = Wp\X
end

# common part
mh_option = 2
W1, H1 = copy(Wp), copy(Hp)
pwradjust!(W0, H0, Wp, Hp, initpwradj)
Mw = W0\Wp; Mh = mh_option == 1 ? Wp\W0 : Hp/H0

Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(W1-W0*Mw)^2
@show Edata, Em, Ew
if mf_method == "convex"
    Mw, Mh, f_xs, x_abss,iter, rt2, trs = convex!(Mw,Mh,W0,H0,sd_group,order,λ1,λ2,β1,β2,maxiter,SNR;
                                            fprefix=fprefix2, poweradjust=:none, tol=1e-6)
    W2,H2 = copy(W0*Mw), copy(Mh*H0)
elseif mf_method == "ssca"
    Mw, Mh, f_xs, x_abss,iter, rt2, trs = ssca!(Mw,Mh,W0,H0,sd_group,order,λ1,λ2,β1,β2,maxiter;
                                            poweradjust=:none, tol=1e-6)
    W2,H2 = copy(W0*Mw), copy(Mh*H0)
elseif mf_method == "hals"
    W2, H2, f_xs, x_abss, iter, rt2, trs = hals!(W1, H1, β1, β2, maxiter;  tol=1e-5)
else
    error("$(mf_method) is unkown matrix factorization method.")
end
disp_n_save(W1, H1, W2, H2, f_xs, x_abss, iter, rt1, rt2, SNR; imgsz=imgsz, fprefix1=fprefix1, fprefix2=fprefix2)

#=
fnndsvd = "W_10dB_nndsvd_CVX_col_L1_lw0_lh0_bw1.0_bh0_itr50_rt325.248027427.jld"
dd = load(fnndsvd)
fnndsvd_xs = dd["f_xs"]
xnndsvd_abss = dd["x_abss"]

frandc = "W_10dB_randc_CVX_col_L1_lw0_lh0_bw1.0_bh0_itr50_rt357.689826804.jld"
dd = load(frandc)
frandc_xs = dd["f_xs"]
xrandc_abss = dd["x_abss"]

frank1 = "W_10dB_rank1_CVX_col_L1_lw0_lh0_bw1.0_bh0_itr50_rt355.388599356.jld"
dd = load(frank1)
frank1_xs = dd["f_xs"]
xrank1_abss = dd["x_abss"]

scale = :log10
fig, ax = plt.subplots(1,1, figsize=(5,4))
Wscale = scale == :log10 ? log10.(W) : W
ax.plot([fnndsvd_xs frandc_xs frank1_xs])
ax.set_title("f(x)")
ax.legend(["NNDSVD","Ranc C","Rank 1"], fontsize = 12, loc=1)
xlabel("iteration",fontsize = 12)
ylabel("log(penalty)",fontsize = 12)
savefig("allfx.png")


#====== compare with SSCA ===================#
# rank-1
# rank-1 NMF for the clustered disjoint set
init_method = "rank1"
using Clustering
R = kmeans(X, ncells; maxiter=200, display=:iter)
a = assignments(R)
Wp = zeros(m,ncells)
for i in 1:ncells
    wpi, _ = nmf(X[:,a.==i],1) # rank-1 nmf
    Wp[:,i] = wpi
end
Mw = W0\Wp; Mh = Wp\W0
#Wp, Hp = balanced_WH(Wp, X); Mw = W0\Wp; Mh = Hp/H0
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method).png",Wp,imgsz,borderwidth=1)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method).png",W0*Mw,imgsz,borderwidth=1)
Mw0, Mh0 = copy(Mw), copy(Mh)

sd_group=:column; order=1; λ1=0; λ2=0; β1=1.0; β2=0; maxiter=50
# Convex
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)
# SSCA
Mw, Mh = copy(Mw0), copy(Mh0)
ssca!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)

=#