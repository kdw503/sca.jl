
function convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; imgsz=(40,20), poweradjust=:normalize)
    sd_grp = sd_group == :column ? "col" : sd_group == :component ? "comp" : "pix" 
    fprefix = "W_$(SNR)dB_$(init_method)_CVX_$(sd_grp)_L$(order)_lw$(λ1)_lh$(λ2)_bw$(β1)_bh$(β2)"
    sd_group ∉ [:column, :ac, :pixel] && error("Unsupproted sd_group")
    rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter = minMwMh(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,order;
                    poweradjust=poweradjust, fprefix=fprefix, sd_group=sd_group,SNR=SNR)
    save(fprefix*"_itr$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
    W2,H2 = copy(W0*Mw), copy(Mh*H0)
    normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
    imsaveW(fprefix*"_itr$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
    ax = plotW([log10.(x_abss) log10.(f_xs)], fprefix*"_plt.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
end

function ssca!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; imgsz=(40,20), poweradjust=:normalize, tol=1e-6)
    if sd_group == :column
        sd_grp = "col"
        method = :cbyc_uc
    elseif sd_group == :component
        sd_grp = "comp"
        method = :ac_simple
    else
        error("Unsupported method")
    end
    fprefix = "W_$(SNR)dB_$(init_method)_SCA_$(sd_grp)_L$(order)_lw$(λ1)_lh$(λ2)_bw$(β1)_bh$(β2)"
    sd_group ∉ [:column, :ac, :pixel] && error("Unsupproted sd_group")
    stparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=:WkHk, order=order, hfirst=true,
            processorder=:none, poweradjust=:balance, method=method, rectify=:truncate) 
    lsparams = LineSearchParams(method=:none, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
            iterations_to_show=[499,500])
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); iter = length(trs)
    imsaveW(fprefix*"_itr$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
    x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
    ax = plotW([log10.(x_abss) log10.(f_xs)], fprefix*"_plt.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
    save(fprefix*"_itr$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
    normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
    imsaveW(fprefix*"_itr$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
end

function nmf(X, ncells; tol=1e-5)
    stparams = StepParams(β1=0.1, β2=0.1, hfirst=true, processorder=:none, poweradjust=:none,
            rectify=:truncate) 
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=2, maxiter=100, store_trace=true, show_trace=true)
    rt1 = @elapsed W0, H0, Mw, Mh, Wcd, Hcd = initsemisca(X, ncells, balance=false, use_nndsvd=true)
    rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
    Wcd, Hcd
end

#======== NMF Initialization ============#
sd_group=:column; order=1; λ1=0; λ2=0; β1=1.0; β2=0; maxiter=50

imgsize = (40,20); lengthT=1000; jitter=0; SNR = 10
X, imgsz, ncells, fakecells_dic, _ = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                    fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
m, n = size(X); p = ncells = 15

# SVD
init_method = "svd"
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, balance=true, use_nndsvd=false)
Mw0, Mh0 = copy(Mw), copy(Mh); W1,H1 = copy(W0), copy(H0); 
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("W0_$(SNR)dB_init-$(init_method)_rt$(rt1).png",sortWHslices(W1,H1)[1],imgsz,borderwidth=1)
for β in [2.0, 4.0]
    β1 = β; β2 = 0
    convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter)
end

# NNDSVD
init_method = "nndsvd"
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, balance=true, use_nndsvd=true)
Mw0, Mh0 = copy(Mw), copy(Mh);#W1,H1 = copy(W0), copy(H0); 
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method).png",Wp,imgsz,borderwidth=1)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method).png",W0*Mw,imgsz,borderwidth=1)
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)

# Random-C : average of q columns of the k largest column of X.
init_method = "randc"
k = Int(round(n/2)); q = Int(round(k/100));
sindices = sortperm(norm.(eachcol(X)),rev=true)[1:k]
Wp = zeros(size(W0)...)
for i = 1:p
    rindices = rand(1:k, q)
    Wp[:,i] .= sum(X[:,sindices[rindices]],dims=2)./q
end
Mw = W0\Wp; Mh = Wp\W0
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method)_k$(k)_q$(q).png",Wp,imgsz,borderwidth=1)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method)_k$(k)_q$(q)png",W0*Mw,imgsz,borderwidth=1)
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)

# Xcol Rand
init_method = "xcol"
q=10; Wp = zeros(size(W0)...)
for i in 1:15
    rindices = rand(1:1000,q)
    Wp[:,i] = sum(X[:,rindices],dims=2)./q
end
Mw = W0\Wp; Mh = Wp\W0
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method)_q$(q).png",Wp,imgsz,borderwidth=1)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method)_q$(q).png",W0*Mw,imgsz,borderwidth=1)
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)

# CUR-based
# statistically(calculated importance) selected columns of X.
# (normalized statistical leverage score, spectral angular distance,
# or symmetrized KL-divergence)

init_method = "curbased"

# k-means
# W as centroids of the best k clusters of data, H as the distances from each
# data point to every centroid.
init_method = "k-mean"
using Clustering
R = kmeans(X, ncells; maxiter=200, display=:iter)
a = assignments(R)
for i in 1:ncells
    num = sum(a.==i)
    wpi = sum(X[:,a.==i],dims=2)/num
    Wp[:,i] = wpi
end
Mw = W0\Wp; Mh = Wp\W0
#Wp, Hp = balanced_WH(Wp, X); Mw = W0\Wp; Mh = Hp/H0
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method).png",Wp,imgsz,borderwidth=1)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method).png",W0*Mw,imgsz,borderwidth=1)
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)

# rank-1
# rank-1 NMF for the clustered disjoint set
init_method = "rank1"
using Clustering
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
Mw = W0\Wp; Mh = Wp\W0
#Wp, Hp = balanced_WH(Wp, X); Mw = W0\Wp; Mh = Hp/H0
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method).png",Wp,imgsz,borderwidth=1)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method).png",W0*Mw,imgsz,borderwidth=1)
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)

# SPA(Successive Projection Algorithm)
# initialize 𝑆=𝑋 and succesively project 𝑆 to the 𝑝_𝑗=max(𝑠_𝑗 )⁡‖𝑠_𝑗 ‖^2
# and update 𝑆 ← 𝑆−proj(𝑝_𝑗,𝑆) then 𝑊=[𝑝_1,𝑝_2,…,𝑝_𝑟]
init_method = "spa"
Wp=zeros(m,0)
S=copy(X)
for i = 1:ncells
    maxi = argmax(norm.(eachcol(S)))
    si = S[:,maxi]
    Wp = hcat(Wp,si)
    S -= si*(si'./norm(si)^2*S)
end
Mw = W0\Wp; Mh = Wp\W0
#Wp, Hp = balanced_WH(Wp, X); Mw = W0\Wp; Mh = Hp/H0
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method).png",Wp,imgsz,borderwidth=1)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method).png",W0*Mw,imgsz,borderwidth=1)
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter)
for β in [2.0, 3.0]
    β1 = β; β2 = 0
    convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)
end

# HC (Hierarchical Clustering)
# merge dependent rows of X until r cluster is left and set it as a H
# initialization
init_method = "hc"

# SC (Subtracting Clustering)
# calculate the potentials pi of being cluster center of every point by
# summing gaussian distances with neighbors. Then choose the maximum
# point as 1st cluster center and subtract a potential from each
# remaining neighbor with the center. Repeat until remaining pi < 0.15p1
init_method = "sc"

# nnICA
# W as the absolute or truncation of the ICA component of X.
init_method = "nnica"
ica = fit(ICA, X', ncells; do_whiten = true, maxiter = 1000, tol = 1e0, mean = nothing)
Wp = Array(transform(ica, X')'); Wp[Wp.<0].=0
Mw = W0\Wp; Mh = Wp\W0
#Wp, Hp = balanced_WH(Wp, X); Mw = W0\Wp; Mh = Hp/H0
Edata = norm(X-W0*H0)^2; Em = norm(I-Mw*Mh)^2; Ew = norm(Wp-W0*Mw)^2
@show Edata, Em, Ew
imsaveW("Winit_$(SNR)dB_init-$(init_method).png",Wp,imgsz)
imsaveW("W0Mw_$(SNR)dB_init-$(init_method).png",W0*Mw,imgsz)
convex!(Mw,Mh,W0,H0,SNR,init_method,sd_group,order,λ1,λ2,β1,β2,maxiter; poweradjust=:none)



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

