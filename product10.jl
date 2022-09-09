""" 
TODO List
- 4D data test
- Physical properties which the solution should have
    - non-negative
    - orthogonal
    - localization
    - Sparse
- constraint instead of regularization : constraint || M'M-I||^2 = β
- constraint norm.(eachcol(W)) .== ones(p)
- sca2(W)*sca2(H) doesn't reflect priority of component that has bigger power
  How about Σ_k(sca2(W[:,k]*sca2(H[k,:]))? 
  Or, Σ_k(sca2(kron(W[:,k], H[k,:]'))) # Kronecker product ⊗

  <06282022>
- Simple method
    - Check convergence
    - Check M sparsity
- SQP method
- Newton method
    - using Optim package with W*Mw .> 0 and Mh*H .> 0 constraints 
    - implement using Null space method
"""
using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
end

Pkg.activate(".")

using Images, Colors
using SymmetricComponentAnalysis
using StaticArrays, IntervalSets, LinearAlgebra, Optim, NMF, BenchmarkTools
using FakeCells, AxisArrays, MappedArrays, JLD, Printf, PositiveFactorizations
using ForwardDiff, Calculus, RandomizedLinAlg#, ImageAxes
using ImageCore, Dates, ImageView#, CairoMakie
#using ProfileView, Profile # ProfileView conflicts with ImageView 

save("dummy.png",rand(2,2)) # If we didn't call this line before `using PyPlot`,
                            # ImageMagick is used when saving image. That doesn't
                            # properly close the file after saving.
using PyPlot

include(joinpath(pkgdir(SymmetricComponentAnalysis),"test","testdata.jl"))
include(joinpath(pkgdir(SymmetricComponentAnalysis),"test","testutils.jl"))

SCA = SymmetricComponentAnalysis

# function loadfakecell0(fname; svd_method=:isvd, gt_ncells=7, ncells=0, lengthT=100, imgsz=(40,20),
#                         fovsz=imgsz, SNR=10, jitter=0, save=true)
#     if isfile(fname)
#         fakecells_dic = load(fname)
#         gt_ncells = fakecells_dic["gt_ncells"]
#         imgrs = fakecells_dic["imgrs"]
#         img_nl = fakecells_dic["img_nl"]
#         gtW = fakecells_dic["gtW"]
#         gtH = fakecells_dic["gtH"]
#         gtWimgc = fakecells_dic["gtWimgc"]
#         gtbg = fakecells_dic["gtbg"]
#         imgsz = fakecells_dic["imgsz"]
#     else
#         @warn "$fname not found. Generating fakecell data..."
#         sigma = 5
#         imgsz = imgsz
#         lengthT = lengthT
#         revent = 10
#         gt_ncells, imgrs, img_nl, gtW, gtH, gtWimgc, gtbg = gaussian2D(sigma, imgsz, lengthT, revent,
#                                                                 jitter=jitter, fovsz=fovsz, SNR=SNR, orthogonal=false)
#         if save
#             Images.save(fname, "gt_ncells", gt_ncells, "imgrs", imgrs, "img_nl", img_nl, "gtW",
#                 gtW, "gtH", gtH, "gtWimgc", Array(gtWimgc), "gtbg", gtbg, "imgsz", imgsz, "SNR", SNR)
#             fakecells_dic = load(fname)
#         else
#             fakecells_dic = Dict()
#             fakecells_dic["gt_ncells"] = gt_ncells
#             fakecells_dic["imgrs"] = imgrs
#             fakecells_dic["img_nl"] = img_nl
#             fakecells_dic["gtW"] = gtW
#             fakecells_dic["gtH"] = gtH
#             fakecells_dic["gtWimgc"] = Array(gtWimgc)
#             fakecells_dic["gtbg"] = gtbg
#             fakecells_dic["imgsz"] = imgsz
#             fakecells_dic["SNR"] = SNR
#         end
#     end
#     X = imgrs
#     ncells == 0 && (ncells = gt_ncells+5)

#     return X, imgsz, ncells, fakecells_dic, img_nl
# end

# for SNR = [60, 40 , 20, 0, -10]
#     #show SNR
#     imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; SNR=SNR; # SNR=10(noisey), SNR=40(less noisey)
#     X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
#                                                 fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
#     gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

#     ncells = 40; F = svd(X); W = F.U;  imsaveW("Wsvd_SNR$(SNR).png",W[:,1:ncells],imgsz,borderwidth=1)
#     W0, H0, Mw, Mh = initsemisca(X, ncells); Wp = W0*Mw; Hp = Mh*H0; normalizeWH!(Wp,Hp)
#     imsaveW("Wnndsvd_SNR$(SNR).png",Wp,imgsz,borderwidth=1)
# end

# Generate Synthetic Dataset
imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; SNR=-10; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
imsaveW("X_SNR$(SNR).png",maxSNR_X,imgsz,borderwidth=1,colors=(colorant"black", colorant"black", colorant"white"))

# SCA
tol = -1#1e-20
stparams = StepParams(β1=0., β2=0., μ0=100, reg=:WkHk, order=2, hfirst=true, processorder=:none, poweradjust=:balance,
        method=:cbyc_cd, rectify=:pinv, objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=2000, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
Mw0, Mh0 = copy(Mw), copy(Mh);
#imshowW(W0,imgsz,borderwidth=1)
# imsaveW("W0_SNR$(SNR).png",W0,imgsz,borderwidth=1)
# imsaveW("Wnndsvd_rt1$(rt1)_SNR$(SNR).png",W0*Mw,imgsz,borderwidth=1)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(W0*Mw0,1)]; Mhspars = [norm(Mh0*H0,1)]
for i in 1:length(trs)
    Mw = trs[i].Mw; Mh = trs[i].Mh
    push!(pens, norm(I-Mw*Mh)^2)
    push!(Mwspars, norm(W0*Mw,1))
    push!(Mhspars, norm(Mh*H0,1))
end
ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],"iter_vs_log10penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log10.(penalty)",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=4, separate_win=false)
ax = plotW([pens Mwspars Mhspars],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=1, separate_win=false)
# pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(Mw0,1)]; Mhspars = [norm(Mh0,1)]
# for i in 1:length(trs)
#     Mw = trs[i].Mw; Mh = trs[i].Mh
#     push!(pens, norm(I-Mw*Mh)^2)
#     push!(Mwspars, norm(Mw,1))
#     push!(Mhspars, norm(Mh,1))
# end
# ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],"iter_vs_log10penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log10.(penalty)",
#         legendstrs = ["Invertibility penalty", "Mw sparsity", "Mh sparsity"], legendloc=5, separate_win=false)
# ax = plotW([pens Mwspars Mhspars],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
#         legendstrs = ["Invertibility penalty", "Mw sparsity", "Mh sparsity"], legendloc=1, separate_win=false)

# CD
tol = -1
stparams = StepParams(β1=0.1, β2=0.1, hfirst=true, processorder=:none, poweradjust=:none,
        rectify=:truncate) 
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=400, store_trace=true, show_trace=true)
rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells);
rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
Wcd2,Hcd2 = copy(Wcd), copy(Hcd); normalizeWH!(Wcd2,Hcd2); imshowW(sortWHslices(Wcd,Hcd)[1],imgsz, borderwidth=1)
imsaveW("Wcd_SNR$(SNR)_rt1$(rt1)_rt2$(rt2).png",sortWHslices(Wcd,Hcd)[1],imgsz,borderwidth=1)

plotW(Wcd,"Wcd.png"; title="W (CD)", legendloc=1)
x_abs, f_x = getdata(trs)
plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)

#================== CbyC UC line search ==================#
tol = -1#1e-20
stparams = StepParams(β1=0.1, β2=0.1, μ0=0.1, reg=:WkHk, order=2, hfirst=true, processorder=:none, poweradjust=:balance,
        method=:cbyc_uc, rectify=:none, objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=0.01, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=300, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
Mw0, Mh0 = copy(Mw), copy(Mh);
#imshowW(W0,imgsz,borderwidth=1)
# imsaveW("W0_SNR$(SNR).png",W0,imgsz,borderwidth=1)
# imsaveW("Wnndsvd_rt1$(rt1)_SNR$(SNR).png",W0*Mw,imgsz,borderwidth=1)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(W0*Mw0,1)]; Mhspars = [norm(Mh0*H0,1)]
for i in 1:length(trs)
    Mw = trs[i].Mw; Mh = trs[i].Mh
    push!(pens, norm(I-Mw*Mh)^2)
    push!(Mwspars, norm(W0*Mw,1))
    push!(Mhspars, norm(Mh*H0,1))
end
ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],"iter_vs_log10penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log10.(penalty)",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=4, separate_win=false)
ax = plotW([pens Mwspars Mhspars],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=1, separate_win=false)

penfunc(Mw,Mh, W0, H0, stparams) = ((W, H) = (W0*Mw, Mh*H0); β1 = stparams.β1/norm(W); β2 = stparams.β2/norm(H); od = stparams.order;
            norm(I-Mw*Mh)^2 + stparams.μ0*(sca2(W) + sca2(H)) + β1*norm(W,od)^od  + β2*norm(H,od)^od)
f_x0 = penfunc(Mw0, Mh0, W0, H0, stparams)
Mwpre = Mw0; Mhpre = Mh0
f_xs = []; x_abss = []
for iter = 1:1000
    Mw = trs[iter].Mw; Mh = trs[iter].Mh
    push!(x_abss, norm(Mwpre-Mw)*norm(Mhpre-Mh))
    push!(f_xs,penfunc(Mw,Mh, W0, H0, stparams)/f_x0)
    Mwpre = Mw; Mhpre = Mh
end
plot([log10.(f_xs) log10.(x_abss)])


#================== Convex.jl : Semi SCA ==================#

using Convex, SCS

penaltyL1(Mw,Mh,W0,H0,λ,β1,β2) = norm(I-Mw*Mh)^2 + λ*(sca2(W0*Mw)+sca2(Mh*H0)) + β1*norm(W0*Mw,1) + β2*norm(Mh*H0,1)
penaltyL2(Mw,Mh,W0,H0,λ,β1,β2) = norm(I-Mw*Mh)^2 + λ*(sca2(W0*Mw)+sca2(Mh*H0)) + β1*norm(W0*Mw)^2 + β2*norm(Mh*H0)^2

function mincol(Mw,Mh,W,H,k,λ,β,order)
    p = size(W,2)
    Eprev = I-Mw*Mh
    E = Eprev + Mw[:,k]*Mh[k,:]'
    x = Variable(p)
    mhk = Mh[k,:]'
    sparsity = order == 1 ? norm(W*x, 1) : sumsquares(W*x)
    problem = minimize(sumsquares(E-x*mhk) + λ*sumsquares(min(0,W*x)) + β*sparsity)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    xsol = x.value
    errprev = norm(E-Mw[:,k]*mhk)^2; err = norm(E-xsol*mhk)^2
    sparsprev = order == 1 ? β*norm(W0*Mw[:,k],1) : β*norm(W0*Mw[:,k])^2
    spars = order == 1 ? β*norm(W0*xsol,1) : β*norm(W0*xsol)^2 
    errprev+sparsprev < err+spars && @show problem.status
    xsol
end

function minMw_cbyc!(Mw,Mh,W,H,λ,β,order)
    p = size(Mw,2)
    for k in 1:p
        Mw[:,k] = mincol(Mw,Mh,W,H,k,λ,β,order)
    end
end

function minMw_ac!(Mw,Mh,W,H,λ,β,order)
    m, p = size(W)
    x = Variable(p^2)
    Ivec = vec(Matrix(1.0I,p,p))
    A = zeros(p^2,p^2)
    SCA.directMw!(A, Mh) # vec(I-reshape(x,p,p)*Mh) == Ivec-A*x
    Aw = zeros(m*p,p^2); bw = zeros(m*p)
    SCA.direct!(Aw, bw, W; allcomp = false) # vec(W*reshape(x,p,p)) == (Aw*x)
    spars = order == 1 ? norm(Aw*x, 1) : sumsquares(Aw*x)
    problem = minimize(sumsquares(Ivec-A*x)+ λ*sumsquares(min(0,A*x)) + β*spars)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    Mw[:,:] .= reshape(x.value,p,p)
end

function minMwMh(Mw,Mh,W0,H0,λ,β1,β2,maxiter,order; cbyc=true,imgsz=(40,20))
    f_xs=[]; x_abss=[]
    for iter in 1:maxiter
        Mwprev, Mhprev = copy(Mw), copy(Mh)
        if cbyc
            minMw_cbyc!(Mw,Mh,W0,H0,λ,β1,order)
            minMw_cbyc!(Mh',Mw',H0',W0',λ,β2,order)
        else
            minMw_ac!(Mw,Mh,W0,H0,λ,β1,order)
            minMw_ac!(Mh',Mw',H0',W0',λ,β2,order)
        end            
        pensum = order == 1 ? penaltyL1(Mw,Mh,W0,H0,λ,β1,β2) : penaltyL2(Mw,Mh,W0,H0,λ,β1,β2)
        push!(f_xs, pensum)
        x_abs = norm(Mwprev-Mw)^2*norm(Mhprev-Mh)^2
        push!(x_abss, x_abs)
        @show iter, x_abs, pensum
        W2,H2 = copy(W0*Mw), copy(Mh*H0)
        normalizeWH!(W2,H2)
        if iter%10 == 0
            if cbyc
                imsaveW("W2_SNR-10_Convex_cbyc_L$(order)_bw$(β1)_bh$(β2)_iter$(iter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            else
                imsaveW("W2_SNR-10_Convex_ac_L$(order)_bw$(β1)_bh$(β2)_iter$(iter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            end
        end
    end
    Mw, Mh, f_xs, x_abss
end

SNRs =  [60]

for SNR in SNRs
    imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; # SNR=10(noisey), SNR=40(less noisey)
    X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                                fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
    λ=0.1; cbyc = false
    order = 1
    for (β1,β2) = [(0.1, 0.1), (0.1, 0.0)]
        for maxiter in [80] 
            W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
            Mw0, Mh0 = copy(Mw), copy(Mh);
            rt2 = @elapsed Mw, Mh, f_xs, x_abss = minMwMh(Mw,Mh,W0,H0,λ,β1,β2,maxiter,order,cbyc=cbyc)
            if cbyc
                fname = "W2_SNR$(SNR)_Convex_cbyc_L$(order)_bw$(β1)_bh$(β2)_iter$(maxiter)_rt$(rt2).jld"
            else
                fname = "W2_SNR$(SNR)_Convex_ac_L$(order)_bw$(β1)_bh$(β2)_iter$(maxiter)_rt$(rt2).jld"
            end               
            save(fname, "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
            # W2,H2 = copy(W0*Mw), copy(Mh*H0)
            # normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
            # imsaveW("W2_SNR$(SNR)_Convex_L$(order)_b1$(β1)_b2$(β2)_iter$(maxiter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
        end
    end
    # order = 2
    # for β2 = [0,β1]
    #     for maxiter in [1,2,3,4]
    #         W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
    #         Mw0, Mh0 = copy(Mw), copy(Mh);
    #         Mw, Mh, pensums = minMwMh(Mw,Mh,W0,H0,λ,β1,β2,maxiter,order)
    #         W2,H2 = copy(W0*Mw), copy(Mh*H0)
    #         normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
    #         imsaveW("W2_SNR$(SNR)_Convex_L$(order)_b1$(β1)_b2$(β2)_iter$(maxiter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
    #     end
    # end
end

fname = "W2_SNR60_Convex_cbyc_L1_bw0.1_bh0.0_iter80_rt14643.9911256.jld"
dd = load(fname)
f_x = dd["f_xs"]
x_abs = dd["x_abss"]
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)


maxiter=50
SNRs =  [20] #[60, 20, -10]
βs = [0.1, 0.3]
for SNR in SNRs
imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
imsaveW("X_SNR$(SNR).png",maxSNR_X,imgsz,borderwidth=1,colors=(colorant"black", colorant"black", colorant"white"))

rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
Mw0, Mh0 = copy(Mw), copy(Mh);
λ=0.1; β1=0.1; β2=0.1
pensumss = Matrix(undef,maxiter,0) 
for β in βs
    @show SNR, "β2 zero", β
    Mw, Mh = copy(Mw0), copy(Mh0);
Mw, Mh, pensums = minMwMh(Mw,Mh,W0,H0,λ,β,β,maxiter,1)
W2,H2 = copy(W0*Mw), copy(Mh*H0)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_SNR$(SNR)_Convex_L1_b$(β)_iter$(maxiter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
end
fname = "W2_SNR$(SNR)_Convex_L1_b2nonzero_$(today())_$(hour(now()))_$(minute(now())).jld"
save(fname, "pensumss", pensumss, "SNR", SNR, "βs", βs)

pensumss = Matrix(undef,maxiter,0) 
for β in βs
    @show SNR, "β2 zero", β
    Mw, Mh = copy(Mw0), copy(Mh0);
Mw, Mh, pensums = minMwMh(Mw,Mh,W0,H0,λ,β,0,maxiter,1)
W2,H2 = copy(W0*Mw), copy(Mh*H0)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_SNR$(SNR)_Convex_L1_b1$(β)_b20.0_iter$(maxiter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
end
fname = "W2_SNR$(SNR)_Convex_L1_b2zero_$(today())_$(hour(now()))_$(minute(now())).jld"
save(fname, "pensumss", pensumss, "SNR", SNR, "βs", βs)
end

for SNR in SNRs
    imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; # SNR=10(noisey), SNR=40(less noisey)
    X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                                fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
    imsaveW("X_SNR$(SNR).png",maxSNR_X,imgsz,borderwidth=1,colors=(colorant"black", colorant"black", colorant"white"))
    
    rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
    Mw0, Mh0 = copy(Mw), copy(Mh);
    λ=0.1; β1=0.1; β2=0.1
    pensumss = Matrix(undef,maxiter,0) 
    for β in βs
        @show SNR, "β2 zero", β
        Mw, Mh = copy(Mw0), copy(Mh0);
    Mw, Mh, pensums = minMwMh(Mw,Mh,W0,H0,λ,β,β,maxiter,2)
    W2,H2 = copy(W0*Mw), copy(Mh*H0)
    normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
    imsaveW("W2_SNR$(SNR)_Convex_L2_b$(β)_iter$(maxiter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
    end
    fname = "W2_SNR$(SNR)_Convex_L2_b2nonzero_$(today())_$(hour(now()))_$(minute(now())).jld"
    save(fname, "pensumss", pensumss, "SNR", SNR, "βs", βs)
    
    pensumss = Matrix(undef,maxiter,0) 
    for β in βs
        @show SNR, "β2 zero", β
        Mw, Mh = copy(Mw0), copy(Mh0);
    Mw, Mh, pensums = minMwMh(Mw,Mh,W0,H0,λ,β,0,maxiter,2)
    W2,H2 = copy(W0*Mw), copy(Mh*H0)
    normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
    imsaveW("W2_SNR$(SNR)_Convex_L2_b1$(β)_b20.0_iter$(maxiter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
    end
    fname = "W2_SNR$(SNR)_Convex_L2_b2zero_$(today())_$(hour(now()))_$(minute(now())).jld"
    save(fname, "pensumss", pensumss, "SNR", SNR, "βs", βs)
end

#================== Convex.jl ==================#
function plotP(penP1, penP2, penP3, penP4, extent, vmaxP)
    # ticks = [-π, -π/2, 0, π/2, π]
    # ticklabels = ["-π", "-π/2", "0", "π/2", "π"]

    fig, axs = plt.subplots(1, 5, figsize=(13, 4), gridspec_kw=Dict("width_ratios"=>[1,0.1,1,1,1]))

    ax = axs[1]
    himg1 = ax.imshow(reverse(penP1; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("total")

    ax = axs[2]
    plt.colorbar(himg1, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)

    ax = axs[3]
    himg2 = ax.imshow(reverse(penP2; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("sumsquares(A*x - b)")

    ax = axs[4]
    himg3 = ax.imshow(reverse(penP3; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("sumsquares(min.(A*x))")

    ax = axs[5]
    himg3 = ax.imshow(reverse(penP4; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("norm(C*x, 1)")

    axs
end

function penP1P2P3(penfn1, penfn2, penfn3, x, x1s, x2s)
    [penfn1([x1+x[1], x2+x[2]])+penfn2([x1+x[1], x2+x[2]])+penfn3([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s],
    [penfn1([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s],
    [penfn2([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s],
    [penfn3([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s]
end

using Convex, SCS
for i = 1:100
A = randn(2, 2).-0.5; b = randn(2) .- 0.5; C = randn(2, 2) .- 0.5;
x = Variable(2)
pen_data(x) = norm(A*x - b,2)^2
pen_nn(x) = norm(min.(0,A*x))^2
pen_spars(x) = norm(C*x, 1)
problem = minimize(sumsquares(A*x - b) + sumsquares(min(0,A*x)) + sumsquares(x))
solve!(problem, SCS.Optimizer; silent_solver = true)
xsol = x.value
end

n = 10; vmax = 6
pensum, penP1, penP2, penP3 = penP1P2P3(pen_data, pen_nn, pen_spars, xsol, -n:0.1:n , -n:0.1:n)
extent = (xsol[1]-n,xsol[1]+n,xsol[2]-n,xsol[2]+n)
axs = plotP(pensum, penP1, penP2, penP3, extent, vmax)
ax = axs[1]
ax.plot([xsol[1]], [xsol[2]], color="red", marker="x", linewidth = 3)

plt.savefig("convex2dplot.png")

#=============================== cbyc_uc ========================================#
tol = -1#1e-20
stparams = StepParams(β1=0.1, β2=0.1, μ0=0.1, reg=:WkHk, order=2, hfirst=true, processorder=:none, poweradjust=:balance,
        method=:cbyc_uc, rectify=:none, objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=1000, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
Mw0, Mh0 = copy(Mw), copy(Mh);
#imshowW(W0,imgsz,borderwidth=1)
# imsaveW("W0_SNR$(SNR).png",W0,imgsz,borderwidth=1)
# imsaveW("Wnndsvd_rt1$(rt1)_SNR$(SNR).png",W0*Mw,imgsz,borderwidth=1)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(W0*Mw0,1)]; Mhspars = [norm(Mh0*H0,1)]
for i in 1:length(trs)
    Mw = trs[i].Mw; Mh = trs[i].Mh
    push!(pens, norm(I-Mw*Mh)^2)
    push!(Mwspars, norm(W0*Mw,1))
    push!(Mhspars, norm(Mh*H0,1))
end
ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],"iter_vs_log10penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log10.(penalty)",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=4, separate_win=false)
ax = plotW([pens Mwspars Mhspars],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=1, separate_win=false)

#=============================== NLopt ========================================#
using NLopt

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end
    return sqrt(x[2])
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_SLSQP, 2)
opt.lower_bounds = [-Inf, 0.]
opt.xtol_rel = 1e-4

opt.min_objective = myfunc
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

(minf,minx,ret) = NLopt.optimize(opt, [1.234, 5.678])
numevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $numevals iterations (returned $ret)")

#============ CbyC SQP L2 sparsity test =========================================#
p = ncells

# SCA
tol = -1#1e-20

for SNR = [60, 20,-10]
    @show SNR
    imgsz=(40,20); ncells=15; lengthT=1000; jitter=0# SNR=10(noisey), SNR=40(less noisey)
    X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                                fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

    method=:cbyc_uc; β=0.0; reg=:WkHk; order=2
    maxiter = 500
    x_abss = Matrix(undef,maxiter,0)
    f_xs = Matrix(undef,maxiter,0)
    f_rels = Matrix(undef,maxiter,0)
    semisympens = Matrix(undef,maxiter,0)
    regWs = Matrix(undef,maxiter,0)
    regHs = Matrix(undef,maxiter,0) 
    for μ in [0.05, 0.1, 0.5]
        @show μ
        stparams = StepParams(β1=β, β2=β, μ0=μ, reg=reg, order=order, hfirst=true, processorder=:none, poweradjust=:balance,
                method=method, rectify=:none#=pinv=#, objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
        lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=[1])
        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
        rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, use_nndsvd=true, balance=true)
        #Mw0, Mh0 = copy(Mw), copy(Mh);
        p = size(W0,2)
        Mw, Mh = Matrix(1.0I,p,p), Matrix(1.0I,p,p)
        rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
        W2,H2 = copy(W1), copy(H1)
        normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
        imsaveW("W2_wo_init_SNR$(SNR)_$(method)_reg$(reg)_od$(order)_mu$(μ)_iter$(length(trs))_rt1$(rt1)_rt2$(rt2)_.png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
        x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
        x_abss = hcat(x_abss, x_abs)
        f_xs = hcat(f_xs, f_x)
        f_rels = hcat(f_rels, f_rel)
        semisympens = hcat(semisympens, semisympen)
        regWs = hcat(regWs, regW)
        regHs = hcat(regHs, regH)
        for iter in []
            W2, H2 = W0*trs[iter].Mw, trs[iter].Mh*H0; normalizeWH!(W2,H2)
            method = stparams.method; reg = stparams.reg; order = stparams.order
            imsaveW("W2_SNR$(SNR)_$(method)_mu$(μ)_$(reg)_od$(order)_b$(β)_iter$(iter)_rt1$(rt1)_rt2$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
        end
    end
    fname = "mu_$(method)_SNR$(SNR)_$(today())_$(hour(now()))_$(minute(now())).jld"
    save(fname, "x_abss", x_abss, "f_xs", f_xs, "f_rels", f_rels,
            "semisympens",semisympens, "regWs", regWs,"regHs", regHs,
            "SNR", SNR, "β", β, "reg", reg, "order", order
        )
end


fname = "mu_cbyc_uc_SNR-10_2022-07-25_10_24.jld"
dd = load(fname)
x_abss = dd["x_abss"]
f_xs = dd["f_xs"]
ax = plotW(log10.(x_abss),"mu_iter_vs_penalty_x_abs_SNR-10.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs =[0, 0.1, 0.4, 0.7, 1.0, 10.0, 100.0] , legendloc=1, separate_win=false)

ax = plotW(log10.(f_xs),"mu_iter_vs_penalty_f_x_SNR-10.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs =[0, 0.1, 0.4, 0.7, 1.0, 10.0, 100.0] , legendloc=5, separate_win=false)


x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW(log10.(x_abss),"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["λ=0.05", "λ=0.1", "λ=0.5"], legendloc=5, separate_win=false)
pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(W0*Mw0,1)]; Mhspars = [norm(Mh0*H0,1)]
for i in 1:length(trs)
    Mw = trs[i].Mw; Mh = trs[i].Mh
    push!(pens, norm(I-Mw*Mh)^2)
    push!(Mwspars, norm(W0*Mw,1))
    push!(Mhspars, norm(Mh*H0,1))
end
ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],"iter_vs_log10penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log10.(penalty)",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=5, separate_win=false)
ax = plotW([pens Mwspars Mhspars],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=1, separate_win=false)

#============ AC SQP L1 sparsity test =========================================#
p = size(W0,2)

# sparsity M
Erw(x) = sum(abs,W0*Mw*(I+reshape(x,p,p)))

fdgradw = ForwardDiff.gradient(Erw,zeros(p^2))
fdhessw = ForwardDiff.hessian(Erw,zeros(p^2))

bw = zeros(Float64, p^2)
SCA.acL1grad!(bw,W0*Mw)

norm(fdgradw-bw)
norm(fdhessw)

# SCA
tol = -1#1e-20
for β in [3., 4.] #[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    @show β
stparams = StepParams(β1=β, β2=β, μ0=100, reg=:WkHk, order=1, hfirst=true, processorder=:none, poweradjust=:balance, method=:cbyc_sqp,
        rectify=:none#=pinv=#, objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=3000, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
Mw0, Mh0 = copy(Mw), copy(Mh);
#imshowW(W0,imgsz,borderwidth=1)
# imsaveW("W0_SNR$(SNR).png",W0,imgsz,borderwidth=1)
# imsaveW("Wnndsvd_rt1$(rt1)_SNR$(SNR).png",W0*Mw,imgsz,borderwidth=1)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
for iter in [200,500,1000,2000,3000]
    W2, H2 = W0*trs[iter].Mw, trs[iter].Mh*H0; normalizeWH!(W2,H2)
    imsaveW("W2_$(stparams.reg)_SNR$(SNR)_b$(β)_iter$(iter)_rt1$(rt1)_rt2$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
end
end

x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=5, separate_win=false)
pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(W0*Mw0,1)]; Mhspars = [norm(Mh0*H0,1)]
for i in 1:length(trs)
    Mw = trs[i].Mw; Mh = trs[i].Mh
    push!(pens, norm(I-Mw*Mh)^2)
    push!(Mwspars, norm(W0*Mw,1))
    push!(Mhspars, norm(Mh*H0,1))
end
ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],"iter_vs_log10penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log10.(penalty)",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=5, separate_win=false)
ax = plotW([pens Mwspars Mhspars],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
        legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=1, separate_win=false)

#============ test exact rectification =========================================#
k = 7
Mw = copy(Mw0)
imshowW(reshape(W0*Mw[:,k],*(imgsz...),1),imgsz,borderwidth=1)
p = size(W0,2)
W0inv = pinv(W0)
W0f = Matrix{eltype(W0)}(1.0I,p,p)
sca2(W0*Mw[:,k]) # 0.369

# cbyc_sqp
Mw = copy(Mw0)
@time SCA.rectifyWkexact!(W0,W0f,view(Mw,:,k),true,stparams.objective, :cbyc_sqp, cparams)
imshowW(reshape(W0*Mw[:,k],*(imgsz...),1),imgsz,borderwidth=1)
sca2(W0*Mw[:,k]) # 0.103

# cbyc_ipnewton
Mw = copy(Mw0)
cparams.show_inner_trace = true
SCA.rectifyWkexact!(W0,W0f,view(Mw,:,k),true,stparams.objective, :cbyc_ipnewton, cparams)
imshowW(reshape(W0*Mw[:,k],*(imgsz...),1),imgsz,borderwidth=1)
sca2(W0*Mw[:,k]) # 0.369

# pinv
Mw = copy(Mw0)
@time SCA.rectifyWk!(W0,W0inv,W0f,view(Mw,:,k),1,true,stparams.objective)
imshowW(reshape(W0*Mw[:,k],*(imgsz...),1),imgsz,borderwidth=1)
sca2(W0*Mw[:,k]) # 0.086(repeat=1), 0.0012(repeat=10) 

# Direct

#============ L1 sparsity regularization =========================================#
#JuMP.jl
using JuMP
using HiGHS
model = Model(HiGHS.Optimizer)
@variable(model, x >= 0)
@variable(model, 0 <= y <= 3)
@objective(model, Min, 12x + 20y)
@constraint(model, c1, 6x + 8y >= 100)
@constraint(model, c2, 7x + 12y >= 120)
print(model)
optimize!(model)
@show termination_status(model)
@show primal_status(model)
@show dual_status(model)
@show objective_value(model)
@show value(x)
@show value(y)
@show shadow_price(c1)
@show shadow_price(c2)

# FirstOrderSolvers.jl
using Convex, FirstOrderSolvers
m = 40;  n = 50
A = randn(m, n); b = randn(m, 1)
x = Variable(n)
problem = minimize(sumsquares(A * x - b), [x >= 0])
solve!(problem, GAP(0.5, 2.0, 2.0, max_iters=2000)) # error!!

# Lasso.jl
using DataFrames, Lasso
data = DataFrame(X=[1,2,3], Y=[2,4,7])
m = fit(LassoModel, @formula(Y ~ X), data) # error!!

# Convex.jl
using Convex, SCS

#=
    LassoEN(Y,X,γ,λ)

Do Lasso (set γ>0,λ=0), ridge (set γ=0,λ>0) or elastic net regression (set γ>0,λ>0).


# Input
- `Y::Vector`:     T-vector with the response (dependent) variable
- `X::VecOrMat`:   TxK matrix of covariates (regressors)
- `γ::Number`:     penalty on sum(abs.(b))
- `λ::Number`:     penalty on sum(b.^2)
=#
function LassoEN(Y, X, γ, λ = 0)
    (T, K) = (size(X, 1), size(X, 2))

    Y
    b_ls = X \ Y                    #LS estimate of weights, no restrictions

    Q = X'X / T
    c = X'Y / T                      #c'b = Y'X*b

    b = Variable(K)              #define variables to optimize over
    L1 = quadform(b, Q)            #b'Q*b
    L2 = dot(c, b)                 #c'b
    L3 = norm(b, 1)                #sum(|b|)
    L4 = sumsquares(b)            #sum(b^2)

    if λ > 0
        Sol = minimize(L1 - 2 * L2 + γ * L3 + λ * L4)      #u'u/T + γ*sum(|b|) + λ*sum(b^2), where u = Y-Xb
    else
        Sol = minimize(L1 - 2 * L2 + γ * L3)               #u'u/T + γ*sum(|b|) where u = Y-Xb
    end
    solve!(Sol, SCS.Optimizer; silent_solver = true)
    Sol.status == Convex.MOI.OPTIMAL ? b_i = vec(Convex.evaluate(b)) : b_i = NaN

    return b_i, b_ls
end

p = size(Mw,2)
y = vec(Matrix(1.0I,p,p))

A = zeros(p^2,p^2)
SCA.directMw!(A, Mh)
x = vec(Mw)
A*x ≈ vec(Mw*Mh)

A = zeros(p^2,p^2)
SCA.directMw!(A, Mw')
x = vec(Mh')
A*x ≈ vec(Mh'*Mw')

p = size(Mw,2)
y = vec(Matrix(1.0I,p,p))
Mw, Mh =  copy(Mw0), copy(Mh0)
pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(Mw0,1)]; Mhspars = [norm(Mh0,1)]
λ = 0
for i in 1:50
    @show i
    Aw = zeros(p^2,p^2)
    SCA.directMw!(Aw, Mh)
    bi, _ = LassoEN(y,Aw,λ)
    Mw = reshape(bi, p, p)

    Ah = zeros(p^2,p^2)
    SCA.directMw!(Ah, Mw')
    bi, _ = LassoEN(y,Ah,λ)
    Mh = reshape(bi, p, p)'

    push!(pens, norm(I-Mw*Mh)^2)
    push!(Mwspars, norm(Mw,1))
    push!(Mhspars, norm(Mh,1))
end

ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],"iter_vs_log10penalty.png"; title="convergence (Lasso)", xlbl = "iteration", ylbl = "log10.(penalty)",
        legendstrs = ["Invertibility penalty", "Mw sparsity", "Mh sparsity"], legendloc=5, separate_win=false)
ax = plotW([pens Mwspars Mhspars],"iter_vs_penalty.png"; title="convergence (Lasso)", xlbl = "iteration", ylbl = "penalty",
        legendstrs = ["Invertibility penalty", "Mw sparsity", "Mh sparsity"], legendloc=1, separate_win=false)
imshowW(W0*Mw,imgsz,borderwidth=1)
imsaveW("Wlasso_SNR$(SNR)_lambda$(λ).png",W0*Mw,imgsz,borderwidth=1)

#====Maya kato JuliaHub/Documentation/search, Bregman operator, FirstOrderOptimization.jl?, Stephen Boyd, ====#
using Convex, FirstOrderSolvers
m = 40;  n = 50
A = randn(m, n); b = randn(m, 1)
x = Variable(n)
problem = minimize(sumsquares(A * x - b), [x >= 0])

solve!(problem, FISTA())

#============ SQP =========================================#
# cbyc_sqp
stparams = StepParams(β1=αW, β2=αH, μ0=100, reg=:Mk, order=2, hfirst=true, processorder=:none, poweradjust=:balance, method=:cbyc_sqp,
        objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=200, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
#imsaveW("W0_SNR$(SNR).png",W0,imgsz,borderwidth=1)
imsaveW("Wnndsvd_rt1$(rt1)_SNR$(SNR).png",W0*Mw,imgsz,borderwidth=1)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
sca2Ws = []; sca2Hs = []
for i in 1:length(trs)
    push!(sca2Ws, sca2(W0*trs[i].Mw))
    push!(sca2Hs, sca2(trs[i].Mh*H0))
end
ax = plotW([log10.(x_abs) log10.(f_x) log10.(sca2Ws) log10.(sca2Hs)],"iter_vs_penalty.png"; title="convergence (SCA)",
        xlbl = "iteration", ylbl = "log(penalty)", legendstrs = ["log(x_abs)", "log(f_x)", "log(sca2(W))", "log(sca2(H))"],
        legendloc=1, separate_win=false)

# ac_sqp
stparams = StepParams(β1=0.0, β2=0.0, μ0=0.1, reg=:WkHk, order=2, hfirst=true, processorder=:none, poweradjust=:balance, method=:ac_sqp,
        objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=200, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
sca2Ws = []; sca2Hs = []
for i in 1:length(trs)
    push!(sca2Ws, sca2(W0*trs[i].Mw))
    push!(sca2Hs, sca2(trs[i].Mh*H0))
end
ax = plotW([log10.(x_abs) log10.(f_x) log10.(sca2Ws) log10.(sca2Hs)],"iter_vs_penalty.png"; title="convergence (SCA)",
        xlbl = "iteration", ylbl = "log(penalty)", legendstrs = ["log(x_abs)", "log(f_x)", "log(sca2(W))", "log(sca2(H))"],
        legendloc=1, separate_win=false)

#============ Weighted Semi-SCA =========================================#
imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; SNR=10; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

using Interpolations

function decimate(W,d)
    m,p = size(W)
    Wetp = zeros(d,p)
    for (w,wetp) in zip(eachcol(W),eachcol(Wetp))
        itp = Interpolations.interpolate(w, BSpline(Linear()))
        etp = extrapolate(itp, Interpolations.Flat())
        for i in 1:d
            wetp[i] = etp(i*m/d)
        end
    end
    return Wetp
end

α = 0.000; l₁ratio = 1.0; regularization = :both 
αH = 0; αW = 0
if (regularization == :both) || (regularization == :components)
    αH = α
end
if (regularization == :both) || (regularization == :transformation)
    αW = α
end

αW *= l₁ratio; αH *= l₁ratio; tol = 1e-20

# CD
stparams = StepParams(β1=αW, β2=αH, reg=:Mk, order=2, hfirst=true, processorder=:none, poweradjust=:balance, method=:cbyc_cd,
                            objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                            iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=5, maxiter=400, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1); imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)

# IPNewton
stparams = StepParams(β1=αW, β2=αH, μ0=0.1, reg=:Mk, order=2, hfirst=true, processorder=:none, poweradjust=:balance, method=:cbyc_ipnewton,
        objective=:normal, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=5, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)

W0, Mw, k, initial_x = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
function feasible_initial(W0::Matrix{T},mwk) where T
    if W0[1,1] < 0
        a = minimum(-(W0*mwk)./W0[:,1])
        x = zeros(length(mwk)); x[1] = a > 0 ? 0 : a - eps(T)
    else
        a = maximum(-(W0*mwk)./W0[:,1])
        x = zeros(length(mwk)); x[1] = a < 0 ? 0 : a + eps(T)
    end
    x
end

for i = 1:size(Mw,2)
    mwk = W0*(Mw[:,i]+feasible_initial(W0,Mw[:,i]))
    @show any(mwk.<0)
    mhk = H0'*(Mh'[:,i]+feasible_initial(H0',Mh'[:,i]))
    @show any(mhk.<0)
end

#============ factor =========================================#
fovsz=(20,20); lengthT0 = 100; factor = 8; SNR = 20
imgsz = (fovsz[1]*factor,fovsz[2]); lengthT = lengthT0*factor
X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT)_SNR$(SNR).jld";
        fovsz=imgsz, lengthT=lengthT, imgsz=imgsz, SNR=SNR, save=true)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
ncells = gt_ncells + 5 # 20

#============ Test Msparsity for the whole Mw and Mh  =================#

p = 15
Mw = rand(p,p)
Mh = rand(p,p)

# sparsity M
Esw(x) = sum(abs2,Mw*(I+reshape(x,p,p)))
Esh(x) = sum(abs2,(I+reshape(x,p,p))*Mh)

fdgradw = ForwardDiff.gradient(Esw,zeros(p^2))
fdgradh = ForwardDiff.gradient(Esh,zeros(p^2))
fdhessw = ForwardDiff.hessian(Esw,zeros(p^2))
fdhessh = ForwardDiff.hessian(Esh,zeros(p^2))

bsw = zeros(Float64, p^2); Hsw = zeros(Float64, p^2, p^2)
nbsh = zeros(Float64, p^2); Hsh = zeros(Float64, p^2, p^2)
SCA.add_direct!(Hsw, bsw, Mw, allcomp = true)
SCA.add_transpose!(Hsh, nbsh, Mh, allcomp = true)
bsh = -nbsh

norm(fdgradw-2*bsw)
norm(fdgradh-2*bsh)
norm(fdhessw-2*Hsw)
norm(fdhessh-2*Hsh)

imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; SNR=60; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

# :semisymmetric_full =>
#    stparams = StepParams(γ=0., β=50, order=0, optim_method=:constrained, useprecond=true, penaltytype=:semisymmetric_full)

stparams = StepParams(β1=0.2, β2=0.2, reg=:WkHk, order=0, hfirst=true, processorder=:none, poweradjust=:balance, method=:ac_simple_fast,
                objective=:weighted, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                            iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, x_abstol = 1e-12, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                maxiter=280, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
#W2,H2 = copy(W0*Mw), copy(Mh*H0); normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1)
rt2 = @elapsed W1, H1, objvals, trs = SCA.scasolve2(W0, H0, X, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1)#; imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x) log10.(regW)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)", "log(sca2(W))"], legendloc=5, separate_win=false)

#============ Semi symmetric  =========================================#

imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; SNR=10; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

#  :semisymmetric_cbyc =>
            :updateWHfunc => updateWHMwMh_linear,
            :updateMfunc => ()->(),             # not used
            :powerfunc => powerMwMh,
            :penfunc => (Mw,Mh)->norm(I-Mw*Mh)^2,
            :lspenfuncs => (()->(), ()->()),    # not used
            :constfuncs => (()->(), ()->()),    # not used
            :fghfuncs => (()->(), ()->()),      # not used
            :fgfuncs => (()->(), ()->())        # not used

#  :semisymmetric => 
#    stparams = StepParams(γ=0., β=50, order=0, optim_method=:ipnewton, useprecond=true, penaltytype=:semisymmetric)

func_dic = Dict(
    :updateWHfunc => SCA.updateWHMwMh_linear,
    :updateMfunc => (dMw,dMh) -> (I+dMw, I+dMh),
    :powerfunc => SCA.powerMwMh,
    :penfunc => SCA.semiscaMpair,
    :lspenfuncs => (SCA.semiscaMpair, SCA.semiscaMpair),
    :constfuncs => (SCA.semiscapair_WMw, SCA.semiscapair_MhH), # only for calculating Φ
    :fghfuncs => (SCA.prepare_fgh_semisymmetricW, SCA.prepare_fgh_semisymmetricH),
    :fgfuncs => (SCA.prepare_fg_penMw_fixW0, SCA.prepare_fg_penMh_fixH0),
)
# :semisymmetric_full =>
#    stparams = StepParams(γ=0., β=50, order=0, optim_method=:constrained, useprecond=true, penaltytype=:semisymmetric_full)
func_dic = Dict(
            :updateWHfunc => SCA.updateWHMwMh_linear,
            :updateMfunc => (dMw,dMh) -> (I+dMw, I+dMh),
            :powerfunc => SCA.powerMwMh,
            :penfunc => SCA.semiscaMpair,
            :lspenfuncs => (SCA.semiscaMpair, SCA.semiscaMpair),
            :constfuncs => (SCA.semiscapair_WMw, SCA.semiscapair_MhH), # only for calculating Φ
            :fghfuncs => (SCA.prepare_fgh_semisymmetricW_full, SCA.prepare_fgh_semisymmetricH_full),
            :fgfuncs => (SCA.prepare_fg_penMw_fixW0_full, SCA.prepare_fg_penMh_fixH0_full),
        )

imgsz=(40,20); ncells=15; lengthT=1000; jitter=0; SNR=60; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

stparams = StepParams(β1=0, β2=0, reg=:WkHk, order=0, hfirst=true, processorder=:none, poweradjust=:balance, method=:constrained,
                objective=:weighted, reductfn = (A)-> (ATA = (A'A); SCA.rectify!(ATA); sqrt.(ATA))) # Matrix(1.0I,p,p)) # decimate(A,200)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                            iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                maxiter=500, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
rt2 = @elapsed W1, H1, objvals, trs = SCA.scasolve2(W0, H0, X; func_dic=func_dic, stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1); imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)

#============ SQP test =========================================#
# equality simple first

# JuMP+Ipopt
using JuMP, Ipopt

function Ipopt_QP(whichtopen, whichtofix, W::Matrix{T}, H::Matrix{T}, Mw::Matrix{T}, Mh::Matrix{T}) where T
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:p^2])
    set_start_value.(x,0)
    # start_value.(x)

    A, b = SCA.add_directM(Mw, Mh)
    pen(x) = x'*A*x+2b'x+norm(I-Mw*Mh)^2
    @objective(model, Min, pen(x))

    bc = zeros(T, p^2); Ac = zeros(T, p^2, p^2)
    if whichtofix == :constW
        SCA.add_direct!(Ac, bc, W, allcomp = false)
        @constraint(model, x'*Ac*x+2bc'x+sca2(W)==0)
        #@constraint(model, W*reshape(x,p,p) .>= 0)
    elseif whichtofix == :constH
        SCA.add_transpose!(Ac, bc, H, allcomp = false)
        @constraint(model, x'*Ac*x+2bc'x+sca2(H)==0)
        #@constraint(model, reshape(x,p,p)*H .>= 0)
    end

    optimize!(model)
#    minobj = objective_value(model)
    opt_x = JuMP.value.(x)
    @show norm(opt_x)

    IpdM = I+reshape(opt_x,p,p)
    if whichtopen == :penW
        WH = W*IpdM
        M = Mw*IpdM
        @show sca2(WH), norm(I-M*Mh)^2
    else
        WH = IpdM*H
        M = IpdM*Mh
        @show sca2(WH), norm(I-Mw*M)^2
    end

    return WH, M
end

@show sca2(W), sca2(H), norm(I-Mw*Mh)^2
for i in 1:10
    @show i
    W, Mw = Ipopt_QP(:penW, :constW, W, H, Mw, Mh);
    H, Mh = Ipopt_QP(:penH, :constH, W, H, Mw, Mh);
    W, H, Mw, Mh = init_nn_MwMh(Wn, Hn, Mw, Mh, 5)
end

W2, H2 = copy(W), copy(H);
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1);
imsaveW("W2_QP.png",W2,imgsz,borderwidth=1)

#============ rectifysca2Wk! test =========================================#
SCA.rectifysca2Wk!(W0,W0inv,view(Mw,:,k),1,eninvert)

#============ Semi-SCA =========================================#
#stparams = StepParams(β1=0.00001, β2=0.0001, reg=:Mk, order=1, hfirst=true, poweradjust=:none) # 50dB
α = 0.0000; l₁ratio = 1.0; regularization = :both 
αH = 0; αW = 0
if (regularization == :both) || (regularization == :components)
    αH = α
end
if (regularization == :both) || (regularization == :transformation)
    αW = α
end
αW *= l₁ratio; αH *= l₁ratio

stparams = StepParams(β1=αW, β2=αH, reg=:orthog, order=2, option=1, hfirst=false, poweradjust=:balance, rectify = :pinv) 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                            iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = false, f_abstol = -1e-20, f_reltol=-1e-20, f_inctol=1e2,
        x_abstol=1e-10, successive_f_converge=5, maxiter=1000, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1)
imsaveW("W2_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(iter_num).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
legendloc = 4
ax = plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="SCA (β=$αW)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=legendloc, separate_win=false)
# plotW([f_x semisympen regW regH],"iter_vs_penalty2.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
#         legendstrs = ["f_x", "semi symmetric penalty", "regularization Mw", "regularizatioin Mh"],
#         legendloc=1, separate_win=false)

stparams = StepParams(β1=αW, β2=αH, reg=:orthog, order=2, option=1, hfirst=false, poweradjust=:balance, rectify = :none) 
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
x_abs0, f_x0, _ = getdata(trs)
legendstrs = ["log(x_abs)", "log(f_x)","log(x_abs) w/o rect.", "log(f_x) w/o rect."];
ax.plot([log10.(x_abs0) log10.(f_x0)]); ax.legend(legendstrs, fontsize = 12, loc=legendloc)
savefig("SCA_convergence_SNR$SNR.png")
        
#============ 1D case test =========================================#

# generate data
gtncells=2; lengthT=100; nevents=10; bias=0.1; SNR = 00
X, gtW, gtH, gtbg = gaussian1D(gtncells, lengthT, nevents, bias, SNR) 
# plotW(gtW,"gtW.png"; title="GT W", legendloc=1, separate_win=false)

ncells = 6; p = ncells

# Semi-SCA
objective = :normal
stparams = StepParams(β1=0, β2=0, reg=:WkHk, order=1, hfirst=false, poweradjust=:none,
                            objective=objective, reductfn = (A)-> decimate(A,200)) # A'A) # Matrix(1.0I,p,p)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                            iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = false, f_abstol = 1e-10, f_reltol=1e-10, f_inctol=1e2,
        x_abstol=1e-8, successive_f_converge=5, maxiter=1000, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
W2,H2 = copy(W0), copy(H0); normalizeWH!(W2,H2)
#plotW(W2,"Winit.png"; title="W init (SCA)", legendloc=1)

rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2)
#plotW(W2,"Wsca.png"; title="W (SCA)", legendloc=1)
x_abs, f_x = getdata(trs)
plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)

# CD
rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells);
rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
Wcd2,Hcd2 = copy(Wcd), copy(Hcd); normalizeWH!(Wcd2,Hcd2)

plotW(Wcd,"Wcd.png"; title="W (CD)", legendloc=1)
x_abs, f_x = getdata(trs)
plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)

# column by column
objective = :normal; maxiter = 10; option=1
stparams = StepParams(β1=0, β2=0, reg=:WkHk, order=1, hfirst=false, poweradjust=:none,
                            objective=objective, reductfn = (A)-> decimate(A,200)) # A'A) # Matrix(1.0I,p,p)) # 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                            iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = false, f_abstol = 1e-10, f_reltol=1e-10, f_inctol=1e2,
        x_abstol=1e-4, successive_f_converge=5, maxiter=maxiter, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1); normalizeWH!(W2,H2)
plotW(W2,"Wsca.png"; title="W (SCA)", legendloc=1)

β1 = stparams.β1; regularization = stparams.reg; order = stparams.order;
m, p = size(W0)
W0inv, H0inv = pinv(W0), pinv(H0')'
W0TW0 = W0'*W0; H0H0T = Matrix(H0*H0')
normfn(A) = norm(A,stparams.order)^stparams.order
(β1, β2) = stparams.reg == :WkHk ?  (stparams.β1/normfn(W0*Mw), stparams.β2/normfn(Mh*H0)) :
                                    (stparams.β1, stparams.β2)

Wim1 = W0*Mw; Him1 = Mh*H0 

# at column k
k = 1
hess, grad = SCA.hessgradpenMwk(Mw, Mh, k)
if β1 != 0
    if regularization == :Mk # Mwk sparsity
        hessr, gradr = SCA.hessgradsparsity(W0, Wim1, Mw, k, regularization, order)
    elseif regularization == :WkHk # Wk sparsity
        if order == 1 # L1 norm
            hessr, gradr = SCA.hessgradsparsity(W0, Wim1, Mw, k, regularization, order)
        elseif order == 2 # L2 norm
            hessr, gradr = SCA.hessgradregWkHko2(W0TW0, Mw, k)
        end
    end
    x = -Vector(SCA.invhess(hess,hessr,β1)*(grad+β1*gradr))
else
    x = -Vector(SCA.invhess(hess)*grad)
end
dMwk = option == 1 ? x : Mw[:,k]-x
normx2 = norm(dMwk)^2 # norm(W0*dMwk)^2
SCA.updatemwk!(Mw, x, k; option=option)

Wk = W0*Mw[:,k]; Wkn = Wk/norm(Wk)^2
plotW(reshape(Wkn,m,1),"Wcd_c$(k).png"; title="W (after minimize)", legendloc=1)

invert = SCA.rectifyWk!(W0,W0inv,view(Mw,:,k),1)

Wk = W0*Mw[:,k]; Wkn = Wk/norm(Wk)^2
plotW(reshape(Wkn,m,1),"Wcd_c$(k).png"; title="W (after rectification)", legendloc=1)
#============ Fast HALS =========================================#

# HALS
# for t in 1:p
#     for i in 1:size(W,1)
#          # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
#         grad = -XHt[i, t] # pt = P[i,t] (P = YB = XH')

#         for r in 1:n_components # -> Aqt = A*Q[:,t] (Q = B'B = HH')
#             grad += HHt[t, r] * W[i, r]
#         end

#         # projected gradient
#         pg = W[i, t] == 0 ? min(zero(grad), grad) : grad
#         violation += abs(pg)

#         # Hessian
#         hess = HHt[t, t]
#         if hess != 0
#             W[i, t] = max(W[i, t] - grad / hess, zero(grad)) # W[:,t] -= (-pt+Aqt)/qtt
#         end
#     end
# end

#stparams = StepParams(β1=0.00001, β2=0.0001, reg=:Mk, order=1, hfirst=true, poweradjust=:none) # 50dB
α = 0.0; l₁ratio = 0.5; regularization = :both 
if (regularization == :both) || (regularization == :components)
    αH = α
end
if (regularization == :both) || (regularization == :transformation)
    αW = α
end
αW *= l₁ratio; αH *= l₁ratio; tol=-1e-20
stparams = StepParams(β1=αW, β2=αH, reg=:WkHk, order=1, hfirst=false, poweradjust=:none, rectify = :truncate) 
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=5, maxiter=400, store_trace=true, show_trace=true)
rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells);
rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
W2,H2 = copy(Wcd), copy(Hcd)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1)
imsaveW("Wcd_rt1$(rt1)_rt2$(rt2)_SNR$(SNR)_iter$(length(trs)).png",W2,imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
plotW([f_x semisympen regW regH],"iter_vs_penalty2.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty",
        legendstrs = ["f_x", "semi symmetric penalty", "regularization Mw", "regularizatioin Mh"],
        legendloc=1, separate_win=false)

stparams = StepParams(β1=αW, β2=αH, reg=:WkHk, order=1, hfirst=false, poweradjust=:none, rectify = :none) 
rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells);
rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
x_abs0, f_x0, _ = getdata(trs)
legendstrs = ["log(x_abs)", "log(f_x)","log(x_abs) w/o rect.", "log(f_x) w/o rect."]
plotW([log10.(x_abs) log10.(f_x) log10.(x_abs0) log10.(f_x0)],"iter_vs_penalty.png"; title="CD (α=$αW)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = legendstrs, legendloc=5, separate_win=false)


# Original CD from NMF.jl
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz, borderwidth=1)
imsaveW("Wcd_rt1$(rtcd1)_rt2$(rtcd2)_SNR$(SNR).png",Wcd,imgsz,borderwidth=1)


x_abs0, f_x0, f_rel0 = copy(x_abs), copy(f_x), copy(f_rel)
legend(["log(|dW|²) 60dB", "log(|dH|²) 60dB", "log(|dW|²) 0dB", "log(|dH|²) 0dB"],fontsize = 12,loc=4)

#============ Reduced size HALS =========================================#

α = 0.0; l₁ratio = 0.5; regularization = :both 
if (regularization == :both) || (regularization == :components)
    αH = α
end
if (regularization == :both) || (regularization == :transformation)
    αW = α
end
αW *= l₁ratio; αH *= l₁ratio
stparams = StepParams(β1=αW, β2=αH, reg=:WkHk, order=1, hfirst=false, poweradjust=:none) 
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-6, f_inctol=1e2,
            x_abstol=1e-8, successive_f_converge=5, maxiter=1000, store_trace=true, show_trace=true)
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells);
d = 2
rtcd2 = @elapsed begin
    Wd = decimate(Wcd,Int(size(Wcd,1)/d)); Hd = Array(decimate(Hcd',Int(size(Hcd,2)/d))')
    Xd = Array(decimate(decimate(X,Int(size(Wcd,1)/d))',Int(size(Hcd,2)/d))')
    Wd0, Hd0 = copy(Wd), copy(Hd)
    # NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=0, l₁ratio=0.5), Xd, Wd, Hd)
    rt2 = @elapsed objvals, trs = SCA.halssolve!(Xd, Wd, Hd; stparams=stparams, cparams=cparams);
    Mw = Wd0\Wd; Mh = Hd/Hd0; Wcd = Wcd*Mw; Hcd = Mh*Hcd
end
W2,H2 = copy(Wcd), copy(Hcd)
normalizeWH!(W2,H2); imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1)
imsaveW("Wrcd_rt1$(rtcd1)_rt2$(rtcd2)_SNR$(SNR).png",W2,imgsz,borderwidth=1)
x_abs, f_x, f_rel, semisympen, regW, regH = getdata(trs)
plotW([log10.(x_abs) log10.(f_x)],"iter_vs_penalty.png"; title="convergence (CD)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)

rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz, borderwidth=1)

#============ jitter test =============#
for beta = [0, 0.01, 0.1, 1, 10, 100]
    stparams = StepParams(β1=beta, β2=beta)
    lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-6, f_inctol=1e-5,
        maxiter=500, store_trace=false, show_trace=false)
    rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells)
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2)#; imshowW(W2,imgsz, borderwidth=1)
    imsaveW("Wssca_J$(jitter)_SNR$(SNR)_b1$(stparams.β1)_b2$(stparams.β2)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)
end

for α = [0, 0.1]
    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz, borderwidth=1)
    imsaveW("Wcd_J$(jitter)_SNR$(SNR)_a$(α)_rt1$(rtcd1)_rt2$(rtcd2).png",Wcd,imgsz,borderwidth=1)
end

#============ noc vs runtime (:symmetric_orthogonality) =============#
ncellsrng = 6:2:100; xlabelstr = "number of components"

imgsz=(40,20); ncells=15; lengthT=1000; SNR=40; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sz$(imgsz)lengthT$(lengthT)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=true);
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-6, f_inctol=1e-5,
    maxiter=500, store_trace=false, show_trace=true)

rtcd1s = [];
rtcd2s = [];mssdcds=[];
rtcd2s0p1 = []; mssdcds0p1=[]
rtssca1s = []
rtssca2s = []; mssdsscas = []
rtssca2s0p1 = []; mssdsscas0p1 = []
for ncells in ncellsrng
    @show ncells
    @show "SSCA"
    stparams = StepParams(β1=0.0, β2=0.0, reg=:none)
    rt1 = @elapsed W0, H0, Mwinit, Mhinit = initsemisca(X, ncells)
    Mw, Mh = copy(Mwinit), copy(Mhinit)
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
    mssdssca, ml, ssds = matchedfiterr(gtW,W2);
    push!(rtssca1s, rt1); push!(rtssca2s, rt2); push!(mssdsscas,mssdssca)
    imsaveW("SSCA_SNR$(SNR)_n$(ncells)_mssd$(mssdssca)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)

    @show "SSCA with Mk reg"
    stparams = StepParams(β1=0.2, β2=0.2, reg=:WH)
    Mw, Mh = copy(Mwinit), copy(Mhinit)
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
    mssdssca, ml, ssds = matchedfiterr(gtW,W2);
    push!(rtssca2s0p1, rt2); push!(mssdsscas0p1,mssdssca)
    imsaveW("SSCA_SNR$(SNR)_n$(ncells)_β$(stparams.β1)_mssd$(mssdssca)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)
    # @show "SSCA+Rect"
    # W2,H2 = copy(W1), copy(H1)
    # W2[W2.<0].=0; H2[H2.<0].=0;
    # normalizeWH!(W2,H2); # imshowW(W2,imgsz, borderwidth=1)
    # mssdsscarect, ml, ssds = matchedfiterr(gtW,W2);
    # push!(mssdsscarects,mssdsscarect)
    # imsaveW("SSCA_rect_SNR$(SNR)_n$(ncells)_mssd$(mssdsscarect)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)

    # @show "SSCA+CD"
    # W2,H2 = copy(W1), copy(H1)
    # rtsscacd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=0.1, l₁ratio=0.5), X, W2, H2)
    # normalizeWH!(W2,H2); # imshowW(W2,imgsz, borderwidth=1)
    # mssdsscacd, ml, ssds = matchedfiterr(gtW,W2);
    # push!(rtsscacd2s, rtsscacd2); ; push!(mssdsscacds,mssdsscacd)
    # imsaveW("SSCA_CD_SNR$(SNR)_n$(ncells)_mssd$(mssdsscacd)_rt1$(rt1)_rt2$(rt2)_rt3$(rtsscacd2).png",W2,imgsz,borderwidth=1)

    @show "CD α = 0"
    α = 0.1
    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz, borderwidth=1)
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
    imsaveW("cd_SNR$(SNR)_n$(ncells)_a$(α)_mssd$(mssdcd)_rt1$(rtcd1)_rt2$(rtcd2).png",Wcd,imgsz)

    @show "CD α = 0.1"
    α = 3
    Wcd, Hcd = copy(Wcd0), copy(Hcd0)
    rtcd20p1 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz, borderwidth=1)
    mssdcd0p1, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd2s0p1, rtcd20p1); push!(mssdcds0p1,mssdcd0p1)
    imsaveW("cd_SNR$(SNR)_n$(ncells)_a$(α)_mssd$(mssdcd)_rt1$(rtcd1)_rt2$(rtcd2).png",Wcd,imgsz)
end

fname = "noc_SNR$(SNR)_$(today())_$(hour(now()))_$(minute(now())).jld"
save(fname, "ncellsrng", ncellsrng, "imgsz", imgsz, "lengthT", lengthT, "SNR", SNR,
        "β1", stparams.β1, "β2", stparams.β2,
        # semi-sca
        "rtssca1s",rtssca1s,"rtssca2s",rtssca2s, "mssdsscas", mssdsscas,
        # semi-sca with reg
        "rtssca2s0p1",rtssca2s0p1, "mssdsscas0p1", mssdsscas0p1,
        # CD α = 0
        "rtcd1s", rtcd1s,"rtcd2s",rtcd2s, "mssdcds", mssdcds,
        # CD α = 0.1
        "rtcd2s0p1",rtcd2s0p1, "mssdcds0p1", mssdcds0p1
        )

# dd = load("noc_SNR-15_2022-04-15_14_58.jld")
# β1 = dd["β1"]; β2 = dd["β2"]; ncellsrng = dd["ncellsrng"]
# imgsz = dd["imgsz"]; lengthT = dd["lengthT"];
# rtssca1s = dd["rtssca1s"]; rtssca2s = dd["rtssca2s"]; mssdsscas = dd["mssdsscas"];
# mssdsscarects = dd["mssdsscarects"];
# rtsscacd2s = dd["rtsscacd2s"]; mssdsscacds = dd["mssdsscacds"];
# rtcd1s = dd["rtcd1s"]; rtcd2s = dd["rtcd2s"]; mssdcds = dd["mssdcds"];
# rtcd2s0p1 = dd["rtcd2s0p1"]; mssdcds0p1 = dd["mssdcds0p1"];

rtcds = rtcd1s + rtcd2s
rtcds0p1 = rtcd1s + rtcd2s0p1
rtsscas = rtssca1s + rtssca2s
rtsscas0p1 = rtssca1s + rtssca2s0p1

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtsscas rtsscas0p1 rtcds rtcds0p1])
ax1.legend(["Semi-SCA", "Semi-SCA w reg.", "CD", "CD w reg."],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtssca1s rtcd1s])
ax1.legend(["Semi-SCA", "CD"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_runtime1.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtssca2s rtssca2s0p1 rtcd2s rtcd2s0p1])
ax1.legend(["Semi-SCA", "Semi-SCA w reg.", "CD", "CD w reg."],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_runtime2.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [mssdsscas mssdsscas0p1 mssdcds mssdcds0p1])
ax1.legend(["Semi-SCA", "Semi-SCA w reg.", "CD", "CD w reg."],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("MSSD",fontsize = 12)
savefig("noc_vs_mssd.png")



#==== factor vs different svd runtime (rsvd test) =============#
factorrng = 1:20; cncells = 60; SNR = 10
fovsz=(20,20); lengthT0 = 100

stparams = StepParams(β1=0.0, β2=0.0)
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-6, f_inctol=1e-5,
    maxiter=500, store_trace=false, show_trace=false)

# isvd
rtssca1s = []; rtssca2s = []; mssdsscas = []
rtcd1s = []; rtcd2s0 = []; mssdcds0 = []
rtcd2s0p1 = []; mssdcds0p1 = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_SNR$(SNR)_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, SNR=SNR, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

    ncells = cncells

    @show "SSCA"
    rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells)
    rt2 = @elapsed W1, H1, objval, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); # imshowW(W2,imgsz, borderwidth=1)
    mssdssca, ml, ssds = matchedfiterr(gtW,W2);
    push!(rtssca1s, rt1); push!(rtssca2s, rt2); push!(mssdsscas,mssdssca)
    imsaveW("W2_SNR$(SNR)_f$(factor)_mssd$(mssdssca)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)

    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    push!(rtcd1s, rtcd1)

    @show "CD α = 0.0"
    Wcd0, Hcd0 = copy(Wcd),copy(Hcd)
    α = 0.0
    rtcd20 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd0, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd2s0, rtcd20); push!(mssdcds0,mssdcd0)
    imsaveW("cd_SNR$(SNR)_f$(factor)_a$(α)_mssd$(mssdcd0)_rt1$(rtcd1)_rt2$(rtcd20).png",Wcd,imgsz)

    @show "CD α = 0.1"
    Wcd, Hcd = copy(Wcd0),copy(Hcd0)
    α = 0.1
    rtcd20p1 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd0p1, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd2s0p1, rtcd20p1); push!(mssdcds0p1,mssdcd0p1)
    imsaveW("cd_SNR$(SNR)_f$(factor)_a$(α)_mssd$(mssdcd0p1)_rt1$(rtcd1)_rt2$(rtcd20p1).png",Wcd,imgsz)
end

currenttime = now()
fname = "factor_SNR$(SNR)_$(today())_$(hour(currenttime))-$(minute(currenttime)).jld"
save(fname, "β1", β1, "β2", β2, "factorrng", factorrng,
        "SNR", SNR, "fovsz", fovsz, "lengthT0", lengthT0,
        "rtcd1s", rtcd1s, "rtcd2s0", rtcd2s0, "mssdcds0", mssdcds0, # cd α = 0
        "rtcd2s0p1", rtcd2s0p1, "mssdcds0p1", mssdcds0p1, # cd α = 0.1
        "rtssca1s", rtssca1s,"rtssca2s",rtssca2s, "mssdsscas", mssdsscas) # ssca

# dd = load("factor_2022-04-14_21-50.jld")
# β1 = dd["β1"]; β2 = dd["β2"]; factorrng = dd["factorrng"]; fovsz = dd["fovsz"]; lengthT0 = dd["lengthT0"];
# rtcd1s = dd["rtcd1s"]; rtcd2s0 = dd["rtcd2s0"]; mssdcds0 = dd["mssdcds0"];
# rtcd2s0p1 = dd["rtcd2s0p1"]; mssdcds0p1 = dd["mssdcds0p1"];
# rtssca1s = dd["rtssca1s"]; rtssca2s = dd["rtssca2s"]; mssdsscas = dd["mssdsscas"];
rtsscas = rtssca1s + rtssca2s
rtcds0 = rtcd1s + rtcd2s0
rtcds0p1 = rtcd1s + rtcd2s0p1

xlabelstr = "factor"
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsscas rtcds0 rtcds0p1])
ax1.legend(["Semi-SCA", "CD (α=0)", "CD (α=0.1)"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtssca1s rtcd1s])
ax1.legend(["Semi-SCA", "CD"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_svdruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtssca2s rtcd2s0 rtcd2s0p1])
ax1.legend(["Semi-SCA", "CD (α=0)", "CD (α=0.1)"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_coreruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [mssdsscas mssdcds0 mssdcds0p1])
ax1.set_yscale(:log) # :linear
ax1.legend(["Semi-SCA", "CD (α=0)", "CD (α=0.1)"],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("MSSD",fontsize = 12)
savefig("factor_vs_mssd.png")



#============ gradient and hessian test =========================================#
p = 15
mk = rand(p).-0.5

# sparsity M
Es(x) = sum(abs,mk+x)
fdgrad = ForwardDiff.gradient(Es,zeros(p))
grad = sign.(mk)
norm(fdgrad-grad)

# sparsity W
Es(x) = sum(abs,W0*(mk+x))
fdgrad = ForwardDiff.gradient(Es,zeros(p))
grad = W0'*sign.(W0*mk)
norm(fdgrad-grad)

# orthogonality
MwMwT = zeros(p,p); MhTMh = zeros(p,p)
if stparams.reg == :Orthog
    for k in 1:p
        MwMwT .+= Mw[:,k]*Mw[:,k]'; MhTMh .+= Mh[k,:]*Mh[k,:]'
    end
end

Eok(x,Mw,k) = (Mw0 = copy(Mw); Mw0[:,k] .= x; norm(Mw0'Mw0-I)^2)

for k=1:p
    @show k
    Eo(x) = Eok(x,Mw,k)
    fdgrad = Calculus.gradient(Eo,Mw[:,k])
    fdhess = Calculus.hessian(Eo,Mw[:,k])
    MwMwTk = MwMwT-Mw[:,k]*Mw[:,k]'
    Hess, grad = SCA.hessgradorthogWk(MwMwT, Mw, k)
    @show norm(fdgrad-4*grad)
    @show norm(fdhess-4*Hess)
end

# |f(W0)*g(H0)-f(W0)*Mw*Mh*g(H0)|^2
f(A) = A'A; g(A) = f(A')
option = 1
p = size(Mw,1)
W0f = f(W0); H0f = g(H0)
Y = W0f*H0f
# iteration
Wf = W0f*Mw; Hf = Mh*H0f
indices = option == 3 ? append!(collect(1:k-1),collect(k+1:p)) : collect(1:p)
E = Y - Wf[:,indices]*Hf[indices,:]
# column iteration
k = 9
pen(x) = norm(E-W0f*x*Hf[k,:]')^2
initial_x = option == 3 ? Mw[:,k] : zeros(p)
fdgrad = ForwardDiff.gradient(pen,initial_x)
fdhess = ForwardDiff.hessian(pen,initial_x)
Hess, grad = SCA.hessgradfMwk(E,W0f,Hf,Mw,k; option=option)
@show norm(fdgrad-2*grad)
@show norm(fdhess-2*Hess)

#============ noc vs. |X-WH|² =========================================#
ncellsrng = 2:2:200
W800, H800 = initWH(X, 800; svd_method=:svd)
err1=[]; err2=[]; err3=[]
for ncells in ncellsrng
    @show ncells
    W = W800[:,1:ncells]; H = W\X
    Wi, Hi = initWH(X, ncells; svd_method=:isvd)
    W0, H0, _ = initsemisca(X, ncells)
    push!(err1,norm(X-W*H)^2)
    push!(err2,norm(X-Wi*Hi)^2)
    push!(err3,norm(X-W0*H0)^2)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [err1 err2 err3])
ax1.legend(["|X-Wsvd*Hsvd|^2", "|X-Wisvd*Hisvd|^2", "|X-Wmp*Hmp|^2"],fontsize = 12,loc=1)
xlabel("Number of cells",fontsize = 12)
ylabel("Error",fontsize = 12)
savefig("noc_vs_errors.png")

#============ :semisymmetric(column by column) =========================================#
# -10dB(β=5-30), 0dB(50),  40dB(β=50), 
stparams = StepParams(r=0.1, β1=0.0, β2=0.0, option=1, recttype=:rectcolumnonly, optim_method=:unconstrained, penaltytype=:semisymmetric_cbyc)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=0.1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-6, f_inctol=1e-5,
    maxiter=100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = semiscasolve!3(Wn, Hn, X; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_$(stparams.optim_method)_SNR$(SNR).png",W2,imgsz,borderwidth=1)

#==================================== CD =====================================#
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 5 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH2!(Wcd,Hcd); imshowW(Wcd,imgsz,borderwidth=1)
imsaveW("Wcd_SNR$(SNR)_rt$(rtcd1+rtcd2).png",Wcd,imgsz,borderwidth=1)
# factor 1 ncells=47 0dB (β=50), rtcd1 0.025sec, 1 cell error rtcd2 8.16sec(α=0), 1 cell error rtcd2 8.37sec(α=0.1)
# factor 8 10dB , rtcd1 0.025sec, 1 cell error rtcd2 9.07sec(α=0), best, rtcd2 9.06sec(α=0.1)
# factor 8 -15dB , rtcd1 0.025sec, 1 cell error rtcd2 9.07sec(α=0), 22cells, rtcd2 9.25sec(α=0.1)

#============ :semisymmetric(column by column) =========================================#
# -10dB(β=5-30), 0dB(50),  40dB(β=50), 
stparams = StepParams(r=0.1, option=3, recttype=:rectcolumnonly, optim_method=:unconstrained, penaltytype=:semisymmetric_cbyc)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=200, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = semiscasolve!3(Wn, Hn, X; stparams=stparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_$(stparams.optim_method)_$(lsparams.method)_inneriter$(cparams.inner_maxiter)_SNR$(SNR).png",W2,imgsz,borderwidth=1)

#============ :semisymmetric(constrained) =========================================#
# -10dB(β=5-30), 0dB(50),  40dB(β=50), 
stparams = StepParams(γ=0., β=50, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:semisymmetric_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=5e-1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:2))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=100, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, Mw0, Mh0 = semiscasolve!2(Wn, Hn, X; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_$(stparams.optim_method)_$(lsparams.method)_inneriter$(cparams.inner_maxiter)_SNR$(SNR).png",W2,imgsz,borderwidth=1)

#============ :semisymmetric(IPNewton) =========================================#
# -10dB(β=5-30), 0dB(50),  40dB(β=50), 
stparams = StepParams(γ=0., β=50, order=0, optim_method=:ipnewton, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:semisymmetric)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=5e-1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:2))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=10, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, Mw0, Mh0 = semiscasolve!2(Wn, Hn, X; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_$(stparams.optim_method)_$(lsparams.method)_inneriter$(cparams.inner_maxiter)_SNR$(SNR).png",W2,imgsz,borderwidth=1)


#============ :semisymmetric(debugging) =========================================#

# IPNewton
T = eltype(W)
initial_x = zeros(T,p^2)
fcs = SCA.prepare_fgh_semisymmetricW(W, H, Mw, Mh; βi=0, allcompW = stparams.allcompW, allcompH = stparams.allcompH)

df = TwiceDifferentiable(
    x->fcs[1](x),
    (g, x) -> fcs[2](g,x),
    (h, x) -> fcs[3](h,x),
    initial_x)
constraints = TwiceDifferentiableConstraints(
    (c,x)->(c[1]=fcs[4](x)),
    (J,x)->(J[:,:] = fcs[5](x)),
    (h,x,λ)->(h[:,:] = λ[1]*fcs[6](x)),
    [], [], [0.], [0.]) # [0], [0] => c(x) = 0; [0], [Inf] => c(x) >= 0

method = Optim.IPNewton(μ0=stparams.μ)
options = Optim.Options(; iterations = cparams.inner_maxiter, show_trace = cparams.show_inner_trace,
                        Optim.default_options(method)...)
rst = optimize(df, constraints, initial_x, method, options)

x = rst.minimizer

b = zeros(length(rst.minimizer))
fcs[2](b, b); b ./= 2.

normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)


#============ :semisymmetric (columnbycolumn) debugging ======================================#
gradpenMwk(Mw, Mh, k; option=1) = (P = I-Mw*Mh; option == 1 ? -2*(P*Mh[k,:]) : -2*(Mw'*P*Mh[k,:]))
gradpenMhk(Mw, Mh, k; option=1) = gradpenMwk(Mh', Mw', k; option=option)
gradconWk(W0, W, k; option=1) = (Wkn = (W[:,k].<0).*W[:,k]; option == 1 ? 2*(W0'*Wkn) : 2*(W'*Wkn))
gradconHk(H0, H, k; option=1) = gradconWk(H0',H',k; option=option)

p = 15; k = 14
Mw = rand(p,p); Mh = rand(p,p)

#==== option 1 ====#
penMwk(x,Mw,Mh,k) = (M = copy(Mw); M[:,k] += x; norm(I-M*Mh)^2)
penMhk(x,Mw,Mh,k) = (M = copy(Mh); M[k,:] += x; norm(I-Mw*M)^2)
conWk(x,W0,W,k) = norm((W[:,k].<0).*(W[:,k]+W0*x))^2
conHk(x,H0,H,k) = norm((H[k,:].<0).*(H[k,:]+H0'*x))^2
# gradpenMwk
penMw(x) = penMwk(x, Mw, Mh, k)
cgradMw = Calculus.gradient(penMw, zeros(p))
g = gradpenMwk(Mw, Mh, k)
norm(cgradMw-g)
# gradpenMhk
penMh(x) = penMhk(x, Mw, Mh, k)
cgradMh = Calculus.gradient(penMh, zeros(p))
g = gradpenMhk(Mw, Mh, k)
norm(cgradMh-g)
# gradconWk
constW(x) = conWk(x, W0, W, k)
cgradW = Calculus.gradient(constW, zeros(p))
g = gradconWk(W0, W, k)
norm(cgradW-g)
# gradconHk
constH(x) = conHk(x, H0, H, k)
cgradH = Calculus.gradient(constH, zeros(p))
g = gradconHk(H0, H, k)
norm(cgradH-g)

#==== option 2 ====#
penMwk(x,Mw,Mh,k) = (M = copy(Mw); dM = Matrix(1.0I,p,p); dM[:,k] += x; M *= dM; norm(I-M*Mh)^2)
penMhk(x,Mw,Mh,k) = (M = copy(Mh); dM = Matrix(1.0I,p,p); dM[k,:] += x; M = dM*M; norm(I-Mw*M)^2)
conWk(x,W0,W,k) = norm((W[:,k].<0).*(W[:,k]+W*x))^2
conHk(x,H0,H,k) = norm((H[k,:].<0).*(H[k,:]+H'*x))^2
# gradpenMwk
penMw(x) = penMwk(x, Mw, Mh, k)
cgradMw = Calculus.gradient(penMw, zeros(p))
g = gradpenMwk(Mw, Mh, k; option=2)
norm(cgradMw-g)
# gradpenMhk
penMh(x) = penMhk(x, Mw, Mh, k)
cgradMh = Calculus.gradient(penMh, zeros(p))
g = gradpenMhk(Mw, Mh, k; option=2)
norm(cgradMh-g)
# gradconWk
constW(x) = conWk(x, W0, W, k)
cgradW = Calculus.gradient(constW, zeros(p))
g = gradconWk(W0, W, k; option=2)
norm(cgradW-g)
# gradconHk
constH(x) = conHk(x, H0, H, k)
cgradH = Calculus.gradient(constH, zeros(p))
g = gradconHk(H0, H, k; option=2)
norm(cgradH-g)

#==== updatate W H Mw Mh ===# 
function updateWMwcbyc!(W0, W, Mw, x, k; option=1)
    Mw[:,k] += option == 1 ? x : MW*x
    W[:,k] += option == 1 ? W0*x : W*x
end
updateHMhcbyc!(H0, h, Mh, x,k; option=1) = updateWMwcbyc!(H0', H', Mh', x, k; option=1)

# option 1
x = rand(p)
Mw = rand(p,p); Mh = rand(p,p)
W = W0*Mw; H = Mh*H0
Mwnext = copy(Mw); Mwnext[:,k] += x
Wprev = copy(W); Wnext = W0*Mwnext 
updateWMwcbyc!(W0, W, Mw, x, k; option=1)
norm(W-Wnext)
norm(Mw-Mwnext)
Mhnext = copy(Mh); Mhnext[k,:] += x
Hprev = copy(H); Hnext = Mhnext*H0 
updateHMhcbyc!(H0, H, Mh, x, k; option=1)
norm(H-Hnext)
norm(Mh-Mhnext)

#============ :semisymmetric_full(IPNewton) =========================================#
# -10dB(β=5-30), 0dB(50),  40dB(β=50), 
stparams = StepParams(γ=0., β=50, order=0, optim_method=:ipnewton, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:semisymmetric_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=5e-1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:2))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=2, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, Mw0, Mh0 = semiscasolve!2(Wn, Hn, X; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_$(stparams.optim_method)_$(lsparams.method)_inneriter$(cparams.inner_maxiter)_SNR$(SNR).png",W2,imgsz,borderwidth=1)
# factor 1 ncells=47 0dB (β=50), best, iter 50, rt 13.8sec
# factor 8 ncells=28 10dB (β=50), best, iter 12, rt 1.65sec
stparams = StepParams(γ=0., β=50, order=0, optim_method=:ipnewton, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:semisymmetric_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=5e-1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:2))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = semiscasolve!2(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_$(stparams.optim_method)_$(lsparams.method)_inneriter$(cparams.inner_maxiter)_SNR$(SNR).png",W2,imgsz,borderwidth=1)

function pen(Mw0, Mh0, dMwi, dMhi, x, y)
    norm(I-Mw0*(I+x*dMwi)*(I+y*dMhi)*Mh0)^2
end

#============ :semisymmetric_full(IPNewton) ==========#
using JuMP
import Ipopt
import Test

function example_qcp(; verbose = true)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, xw)
    @variable(model, xh)
    @objective(model, Min, norm(I-reshape(xw,p,p)*reshape(xw,p,p))^2)
    @constraint(model, x + y + z == 1)
    @constraint(model, x * x + y * y - z * z <= 0)
    @constraint(model, x * x - y * z <= 0)
    optimize!(model)
    if verbose
        print(model)
        println("Objective value: ", objective_value(model))
        println("x = ", value(x))
        println("y = ", value(y))
    end
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test primal_status(model) == FEASIBLE_POINT
    Test.@test objective_value(model) ≈ 0.32699 atol = 1e-5
    Test.@test value(x) ≈ 0.32699 atol = 1e-5
    Test.@test value(y) ≈ 0.25707 atol = 1e-5
    return
end

#============ :semisymmetric_full =========================================#
# -10dB(β=5-30), 0dB(50),  40dB(β=50), 
stparams = StepParams(γ=0., β=50, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:semisymmetric_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=5e-1, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:2))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = semiscasolve!2(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_om$(optim_method)_SNR$(SNR).png",W2,imgsz,borderwidth=1)
# factor 1 ncells=47 0dB (β=50), best, iter 50, rt 13.8sec
# factor 8 ncells=28 10dB (β=50), best, iter 12, rt 1.65sec

#============ :symmetric_orthogonality =========================================#
# -10dB(β=5-30), 0dB(50),  40dB(β=50), 
stparams = StepParams(γ=0., β=50, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz,borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)
# factor 1 ncells=47 0dB (β=50), best, iter 50, rt 13.8sec
# factor 8 ncells=28 10dB (β=50), best, iter 12, rt 1.65sec

#============ :symmetric_orthogonality (:newton, skew) =========================================#
# -10dB(β=5-30), 40dB(β=50), 
stparams = StepParams(γ=0., β=0, order=0, optim_method=:newton, useprecond=true, skew=true, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz,borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)
# factor 8 ncells=28 10dB (β=0,skew), not bad, iter 25, rt 1.5sec
# factor 8 ncells=28 10dB (β=50), best, iter 11, rt 0.6sec
# factor 8 ncells=28 10dB (β=0), fail

#============ :ac_symmetric (:both_constrained) =========================================#
stparams = StepParams(β=0.0, r=0.1, optim_method=:both_constrained, penaltytype=:ac_symmetric) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_bc_SNR$(SNR)_r$(stparams.r)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 1 ncells=47 0dB (β=50), good, iter 615, rt 59.6sec
# factor 8 10dB (r=0.1), good, iter 247, rt 7.4sec
# factor 8 10dB (r=0.05), good, iter 249, rt 6.7sec

#============ :ac_symmetric_sparsity_sum (:both_constrained) =========================================#
stparams = StepParams(β=0.1, r=0.5, optim_method=:both_constrained, penaltytype=:ac_symmetric_sparsity_sum) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 1 ncells=47 0dB (β=50), best, iter 39, rt 4.6sec
# factor 8 10dB (β=0.1, r=0.5), best, iter 74, rt 1.8sec

#============ :ac_symmetric =========================================#
stparams = StepParams(β=0.0, r=0.1, order=0, optim_method=:constrained, penaltytype=:ac_symmetric) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_r$(stparams.r)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 10dB (r=0.05), good, iter 333, rt 5.8sec

#============ :ac_symmetric (ipnewton) =========================================#
stparams = StepParams(γ=0., β=0.0, μ=0.0002, order=0, optim_method=:ipnewton, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:ac_symmetric)
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_ipnewton_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 10dB (μ=0.0002), good, iter 249, rt 38sec

#============ :ac_symmetric_sparsity_sum =========================================#
# SNR(β) : -15(0.05), 10(0.007~0.392) 
stparams = StepParams(γ=0., β=0.05, r=1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric_sparsity_sum);
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=0.5, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18));
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true);
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 40dB (β=0.1, α0=0.5), 1 cell fail, iter 66, rt 1.7sec
# factor 8 10dB (β=0.1, α0=0.5), best, iter 58, rt 1.2sec
# factor 8 -10dB ncells=40 (β=0.05, α0=0.05), best, iter 193, rt 11sec
# factor 8 -15dB (β=0.1, α0=0.5), poor, iter 11, rt 0.48sec
# factor 8 -15dB (β=0.05, α0=0.1), 18cells detected, iter 55, rt 1.65sec

#============ :ac_symmetric_sparsity_product =========================================#
# SNR(β) : 0dB(β=0.5)
stparams = StepParams(γ=0., β=0.5, r=0.5, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric_sparsity_product);
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18));
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false);
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)

#============ :ac_symmetric_sparsity_M =========================================#
# 0dB(β=0.05)
stparams = StepParams(γ=0., β=0.005, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric_sparsity_M) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# Doesn't work well

#============ :ac_symmetric_skewsparsity =========================================#
# 0dB(β=5)
stparams = StepParams(γ=0., β=5, r=0.5, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric_skewsparsity) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)

#============ :symmetric_sparsity_sum =========================================#
# 0dB(β=0.5) 
stparams = StepParams(γ=0., β=0.5, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_sparsity_sum)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
@time W1, H1, objvals, trs = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz,borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)

#============ :symmetric_sparsity_product =========================================#
# 0dB(β=5) 
stparams = StepParams(γ=0., β=5, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_sparsity_product)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
@time W1, H1, objvals, trs = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz,borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)

#============ :symmetric_sparsity_M =========================================#
# 0dB(β=0.01)
stparams = StepParams(γ=0., β=0.01, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_sparsity_M)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
@time W1, H1, objvals, trs = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz,borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)

#============ :penW_orthogonality =========================================#
# 0dB(β=30)
stparams = StepParams(γ=0., β=30#=30.=#, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:penW_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=100, inner_maxiter = 100, store_trace=true, show_trace=false)
@time W1, H1, objvals, trs = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)

#============ :penH_orthogonality =========================================#
# Fail
stparams = StepParams(γ=0., β=30#=30.=#, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:penH_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=100, inner_maxiter = 100, store_trace=true, show_trace=false)
@time W1, H1, objvals, trs = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)

#============ :symmetric_full =========================================#
# 10dB(β=1)
stparams = StepParams(γ=0., β=1, r=0.0, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_full) 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1, inner_maxiter = 200, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 (10dB), β=1, success except 1 component, inner_iter 200, rt 343sec

#============ :W_full =========================================#
stparams = StepParams(γ=0., β=10, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:W_full) 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=10, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 (10dB), β=10, success, iter 4, rt 491sec

#============ :W_true_orthog_full : ||(WM)'WM-I||^2 true orthogonality ==============#
stparams = StepParams(γ=0., β=5, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:W_true_orthog_full) 
lsparams = LineSearchParams(method=:none, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=10, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz,borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 (10dB), β=500, success, iter 4, rt 491sec

#============= :ac_apen =================================================#
stparams = StepParams(γ=0., β=0, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_apen) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=500, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)

#============ :ac_exp_full =========================================#
stparams = StepParams(γ=0., β=0, r=0.05, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_exp_full) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 (10dB), β=0, r=0.05, success, iter 335, rt 1560sec

#============ :ac_appexp_full =========================================#
stparams = StepParams(γ=0., β=0, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_appexp_full) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 (10dB), β=0, success, iter 232, rt 532sec

#============ :ac_symmetric_full =========================================#
stparams = StepParams(γ=0., β=0, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric_full) 
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
# factor 8 (10dB), β=0, not bad, iter 232, rt 687sec

#============ :ac_symmetric_full (ipnewton) =========================================#
stparams = StepParams(γ=0., β=0.0#=30.=#, μ=0.00001, order=0, optim_method=:ipnewton, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:ac_symmetric_full
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH2!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_SNR$(SNR)_b$(stparams.β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)

#============ :apen =========================================#
β = 0
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=1, inner_maxiter = 1, store_trace=true, show_trace=false)
W1, H1 = copy(Wn), copy(Hn)
for i in 1:10
    stparams = StepParams(γ=0., β=β#=30.=#, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                    allcompW=false, allcompH=false, penaltytype=:penW_orthogonality)
    @time W1, H1, objvals, trs = semiscasolve!(W1, H1; stparams=stparams, lsparams=lsparams, cparams=cparams);
    stparams = StepParams(γ=0., β=β#=30.=#, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
                    allcompW=false, allcompH=false, penaltytype=:penH_orthogonality)
    @time W1, H1, objvals, trs = semiscasolve!(W1, H1; stparams=stparams, lsparams=lsparams, cparams=cparams);
end
W2,H2 = copy(W1), copy(H1)
normalizeWH2!(W2,H2); imshowW(W2,imgsz,borderwidth=1)
imsaveW("W2_$(stparams.penaltytype).png",W2,imgsz,borderwidth=1)


#============ β vs. MSSD =========================================#
for SNR in [40, 30, 10, 0, -10, -15]
@show SNR
fovsz=(20,20); lengthT0 = 100; factor = 8; svd_method = :svd
imgsz = (fovsz[1]*factor,fovsz[2]); lengthT = lengthT0*factor
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT)_SNR$(SNR).jld";
        svd_method=svd_method, fovsz=imgsz, lengthT=lengthT, imgsz=imgsz, SNR=SNR, save=true)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
ncells = gt_ncells + 5
#ncells = gt_ncells + 20

rt01 = @elapsed W, H = initWH(X, ncells; svd_method=:svd) # 0.98sec(ncells=28)
Wn, Hn = balanced_WH(W,X)

rtcd1 = @elapsed Wcdsvd, Hcdsvd = NMF.nndsvd(X, ncells)

βrng = 0:0.01:1.0
mssds_acs = []; mssds_so = []; mssds_bcs = []; mssds_cd = [];
rt2_acs = []; rt2_so = []; rt2_bcs = []; rt2_cd = [];
Macs = []; Mso = []; Mbcs = []; 
for β in βrng
    @show β

    #============ :ac_symmetric_sparsity_sum =======================================#
    @show "acs"
    stparams = StepParams(β=β, r=0.1, optim_method=:constrained, penaltytype=:ac_symmetric_sparsity_sum);
    lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100);
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                                maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false);
    rtacs2 = @elapsed W1, H1, objvals, trs, _ = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
    normalizeWH!(W1,H1)
    mssdacs, mlacs, ssdacs = matchedfiterr(gtW,W1)
    # imsaveW("W_acs_SNR$(SNR)_β$(β)_rt$(rt01+rtacs2)_mssd$(mssdacs).png", W1, imgsz, borderwidth=1)
    push!(mssds_acs, mssdacs)
    push!(rt2_acs, rtacs2)
    push!(Macs,cal_M(trs))

    #============ :symmetric_orthogonality =========================================#
    @show "so"
    stparams = StepParams(β=β*100+0.01, optim_method=:cg, penaltytype=:symmetric_orthogonality);
    lsparams = LineSearchParams(method=:none, c=1e-4, α0=1e-2, ρ=0.5, maxiter=100);
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                                maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=false);
    rtso2 = @elapsed W1, H1, objvals, trs, _ = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
    normalizeWH!(W1,H1)
    mssdso, mlso, ssdso = matchedfiterr(gtW,W1)
    # imsaveW("W_so_SNR$(SNR)_β$(β*100)_rt$(rt01+rtso2)_mssd$(mssdso).png", W1, imgsz, borderwidth=1)
    push!(mssds_so, mssdso)
    push!(rt2_so, rtso2)
    push!(Mso,cal_M(trs))

    #============ :ac_symmetric_sparsity_sum (:both_constrained) ===================#
    @show "bcs"
    stparams = StepParams(β=β, r=0.1, optim_method=:both_constrained, penaltytype=:ac_symmetric_sparsity_sum);
    lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100);
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                                maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false);
    rtbcs2 = @elapsed W1, H1, objvals, trs, _ = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
    normalizeWH!(W1,H1)#; imshowW(W1,imgsz,borderwidth=1)
    mssdbcs, mlbcs, ssdbcs = matchedfiterr(gtW,W1)
    # imsaveW("W_bcs_SNR$(SNR)_β$(β)_rt$(rt01+rtbcs2)_mssd$(mssdbcs).png", W1, imgsz, borderwidth=1)
    push!(mssds_bcs, mssdbcs)
    push!(rt2_bcs, rtbcs2)
    push!(Mbcs,cal_M(trs))

    #================================= CD  =========================================#
    @show "cd"
    Wcd, Hcd = copy(Wcdsvd), copy(Hcdsvd);
    α = β # best α = 0.1
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd);
    normalizeWH!(Wcd,Hcd);
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd);
    # imsaveW("W_cd_SNR$(SNR)_α$(α)_rt$(rtcd1+rtcd2)_mssd$(mssdcd).png", Wcd, imgsz, borderwidth=1)
    push!(mssds_cd, mssdcd)
    push!(rt2_cd, rtcd2)
end

fname = "beta_vs_mssd_factor$(factor)_SNR$(SNR)_noc$(ncells)_$(today())_$(hour(now()))_$(minute(now())).jld"
save(fname, "βrng", βrng, "svd_method", svd_method, "X", X, "W", W, "imgsz", imgsz, "lengthT", lengthT, "SNR", SNR,
        "rt01", rt01, "rtcd1", rtcd1, "Macs", Macs, "Mso", Mso, "Mbcs", Mbcs,
        "mssds_acs", mssds_acs ,"mssds_so", mssds_so, "mssds_bcs", mssds_bcs, "mssds_cd", mssds_cd,
        "rt2_acs", rt2_acs, "rt2_so", rt2_so, "rt2_bcs", rt2_bcs, "rt2_cd", rt2_cd)
dd = load(fname)
βrng = dd["βrng"]
X = dd["X"]; W = dd["W"]; imgsz = dd["imgsz"]; lengthT = dd["lengthT"]; SNR = dd["SNR"]
rt01 = dd["rt01"]; rtcd1 = dd["rtcd1"]
Macs = dd["Macs"]; Mso = dd["Mso"]; Mbcs = dd["Mbcs"]
mssds_acs = dd["mssds_acs"]; mssds_so = dd["mssds_so"]; mssds_bcs = dd["mssds_bcs"]; mssds_cd = dd["mssds_cd"]
rt2_acs = dd["rt2_acs"]; rt2_so = dd["rt2_so"]; rt2_bcs = dd["rt2_bcs"]; rt2_cd = dd["rt2_cd"]
rt_acs = rt01 .+ rt2_acs; rt_so = rt01 .+ rt2_so; rt_bcs = rt01 .+ rt2_bcs; rt_cd = rtcd1 .+ rt2_cd

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(βrng, [mssds_so mssds_acs mssds_bcs mssds_cd])
ax1.set_yscale(:linear) # :log, :linear
ax1.legend(["Symmetric Orthog.", "AC sparsity","BC sparsity", "CD"],fontsize = 12,loc=5, title="β vs. MSSD ($(SNR)dB)")
xlabel("β",fontsize = 12)
ylabel("Mean of SSD",fontsize = 12)
savefig("β_vs_mssd_SNR($(SNR)).png")


fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(βrng, [rt2_so rt2_acs rt2_bcs rt2_cd])
ax1.set_yscale(:linear) # :log, :linear
ax1.legend(["Symmetric Orthog.", "AC sparsity","BC sparsity", "CD"],fontsize = 12,loc=1, title="β vs. runtime ($(SNR)dB)")
xlabel("β",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("β_vs_runtime_SNR($(SNR)).png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(βrng, [rt_so rt_acs rt_bcs rt_cd])
ax1.set_yscale(:linear) # :log, :linear
ax1.legend(["Symmetric Orthog.", "AC sparsity","BC sparsity", "CD"],fontsize = 12,loc=1, title="β vs. runtime ($(SNR)dB)")
xlabel("β",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("β_vs_runtime_SNR($(SNR)).png")
end

#============ ideal solution =========================================#

orthog = false;
orthogstr = orthog ? "orthog" : "nonorthog";
imgsz=(40,20); ncells=15; lengthT=1000; SNR=-15; # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_sz$(imgsz)lengthT$(lengthT)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=true);
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
W0 = W1 = W2 = W3 = copy(W);

rt01 = @elapsed W, H = initWH(X, ncells; svd_method=:svd) # 0.68sec(ncells=47)
Wn, Hn = balanced_WH(W,X);


p = size(W,2); gtW0 = [gtW zeros(800,8)]; gtH0 = [gtH';zeros(8,1000)]
function prepare_fg(W,H,M,col)
    function fg!(F,G,x)
        # f(x) = (M=reshape(x,p,p); norm(gtW0-Wn*M)^2*norm(gtH0-M\Hn)^2)
        f(x) = (M[:,col]=x; norm(min.(0,W*M))^2*norm(min.(0,M\H))^2)
        if G !== nothing
            g = Calculus.gradient(f,x)
            copyto!(G,g)
        end
        if F !== nothing
            f(x)
        end
    end
    fg!
end
M = Matrix(1.0I,p,p)
for iter = 1:3
    @show iter, Wn[1:3]
for col = 1:15
    fgh! = prepare_fg(Wn, Hn, M, col)
    opt = Optim.Options(f_tol=1e-5, outer_iterations=100, iterations = 3000, show_trace=false, show_every=100);
    initial_x = zeros(p); initial_x[col] = 1.
    rst = optimize(Optim.only_fg!(fgh!), initial_x, ConjugateGradient(), opt);
    M[:,col] = rst.minimizer
    Wn = Wn*M; Hn = M\Hn
end
end
W1, H1 = copy(Wn), copy(Hn)
normalizeWH!(W1,H1); imshowW(W1,imgsz)

stparams = StepParams(β=0.02, r=0.1, optim_method=:constrained, penaltytype=:ac_symmetric_sparsity_sum);
lsparams = LineSearchParams(method=:sca_full, c=1e-4, α0=1e-0, ρ=0.5, maxiter=100);
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                            maxiter=1000, inner_maxiter = 1, store_trace=true, show_trace=false);
rtacs2 = @elapsed W1, H1, objvals, trs, _ = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); imshowW(W1,imgsz)


rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd);
normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz) 


#============ semi-symmetric debugging =========================================#
function add_directM(Mw, Mh)
    p = size(Mw,2)
    M = zeros(p^2, p^2); b = zeros(p^2)
    for j = 1:p
        offsetj = (j-1)*p
        rowrng = (offsetj+1):(offsetj+p)
        for k = 1:p
            offsetk = (k-1)*p
            colrng = (offsetk+1):(offsetk+p)
            M[rowrng,colrng] = Mh[k,j]*Mw
            b[offsetj+k] = (k == j ? 1 : 0) - Mw[k,:]'*Mh[:,j] 
        end
    end
    M'M, -M'b 
end

p = 15
Mw = rand(p,p); Mh = rand(p,p)
pen(x) = (dM = reshape(x,p,p); norm(I-Mw*(I+dM)*Mh)^2)
A, b = add_directM(Mw, Mh)

calg = Calculus.gradient(pen, zeros(p^2));
calh = Calculus.hessian(pen, zeros(p^2));
norm(calh-2A)
norm(calg-2b)

fdg = ForwardDiff.gradient(pen, zeros(p^2));
fdh = ForwardDiff.hessian(pen, zeros(p^2));
norm(fdh-2A)
norm(fdg-2b)
#============ semi-symmetric debugging =========================================#
T = eltype(Wn); p = size(Wn,2)
W, H = (Wn, Hn); p = size(W, 2)
penfunc = stparams.func_dic[:penfunc]; powerfunc = stparams.func_dic[:powerfunc]
updateWHfunc = stparams.func_dic[:updateWHfunc]; updateMfunc = stparams.func_dic[:updateMfunc]

# Wp, Hp = NMF.nndsvd(X, p); Mw, Mh = (W\Wp), (Hp/H); normalizeWH2!(Wp,Hp); imshowW(Wp,imgsz, borderwidth=1)
# Wp, Hp = NMF.nndsvd(X, p); Mw, Mh = (W\Wp), (Hp/H); normalizeWH2!(Wp,Hp); imshowW(Wp,imgsz, borderwidth=1)
(sca2(Wn), sca2(Hn))
Wp, Hp = SCA.nndsvd2(Wn*Hn, p); Mw, Mh = (Wn\Wp), (Hp/Hn)#; normalizeWH2!(Wp,Hp); imshowW(Wp,imgsz, borderwidth=1)
Wp, Hp = updateWHfunc(Mw, Mh, Wn, Hn); (sca2(Wp), sca2(Hp))
Wp[Wp.<0].=0; Hp[Hp.<0].=0; Mw, Mh = (Wn\Wp), (Hp/Hn)
Wp, Hp = updateWHfunc(Mw, Mh, Wn, Hn); (sca2(Wp), sca2(Hp))
normalizeWH2!(Wp,Hp); imshowW(Wp,imgsz, borderwidth=1)

#============ CG W only (full) =========================================#
orthog = false
orthogstr = orthog ? "orthog" : "nonorthog"
imgsz=(40,20); ncells=20; lengthT=1000; SNR=-20 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld";
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=false)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

stparams = StepParams(γ=0., β=34.2#=30.=#, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_sparsity_sum) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 100, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz, borderwidth=1)
imsaveW("W_only.png",W1,imgsz,borderwidth=1)

stparams = StepParams(γ=0., β=200#=30.=#, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-4, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=300, inner_maxiter = 500, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("W1.png",W1,imgsz,borderwidth=1)

stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))


orthog = false; orthogstr = orthog ? "orthog" : "nonorthog"
imgsz=(40,20); ncells=20; lengthT=1000; SNR=-15
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld";
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=false)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

stparams = StepParams(γ=0., β=50#=30.=#, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
    allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
    iterations_to_show=collect(1:18))

iterrng = 10:10:500
mssdscas = []
for iter in iterrng
    @show iter
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-9, f_reltol=1e-8, f_inctol=1e-5,
            maxiter=iter, inner_maxiter = 500, store_trace=false, show_trace=false)
    rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
    pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
    normalizeWH!(W1,H1)
    mssdsca, mlsca, ssdsca = matchedfiterr(gtW,W1)
    imsaveW("noc_sca_iter$(iter)_mssd$(mssdsca).png",W1,imgsz,borderwidth=1)
    push!(mssdscas,mssdsca)
end

rtcd1 = @elapsed Wcd_init, Hcd_init = NMF.nndsvd(X, ncells)
imsaveW("noc_cd_n$(ncells).png",Wcd,imgsz,borderwidth=1)
α = 0.1 # best α = 0.1
mssdcds = []
for iter in iterrng
    @show iter
    Wcd, Hcd = copy(Wcd_init), copy(Hcd_init)
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=iter, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz,borderwidth=1)
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    imsaveW("noc_cd_iter$(iter)_mssd$(mssdcd).png",Wcd,imgsz,borderwidth=1)
    push!(mssdcds,mssdcd)
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(iterrng, [mssdscas mssdcds])
ax1.set_yscale(:linear) # :log, :linear
ax1.legend(["SCA","CD"],fontsize = 12,loc=5, title="Iteration vs. MSSD (-15dB)")
xlabel("iteration",fontsize = 12)
ylabel("Mean of SSD",fontsize = 12)
savefig("iter_vs_mssd_SNR($(SNR)).png")


#==================== LPFed SCA =================================#
imsaveW("W_X.png", W,imgsz,borderwidth=1)

# initialization
# X LPFed
#ImageView.imshow(reshape(X,imgsz...,size(X,2)))
Xlpf = lpf(X, imgsz)
#ImageView.imshow(reshape(Xlpf,imgsz...,size(X,2)))
Wxlpf, Hxlpf = initWH(Xlpf, ncells; svd_method=:isvd)
imshowW(Wxlpf,imgsz,borderwidth=1)
imsaveW("W_Xlpf.png", Wxlpf,imgsz,borderwidth=1)
# W, H LPFed
Wlpf, Hlpf = lpf(W, H, imgsz)
imsaveW("Wlpf_X.png", Wlpf,imgsz,borderwidth=1)

# Setup parameters
optim_method = :cg; skew = true; lsmethod=:none; β=10
if optim_method ==  :newton
    if skew == true
        optim_mtd_str = "newton_skew_$(lsmethod)"
    else
        optim_mtd_str = "newton_b$(β)_$(lsmethod)"
    end
else
    optim_mtd_str = "$(optim_method)_b$(β)_$(lsmethod)"
end
stparams = StepParams(γ=0., β=β#=30.=#, r=0.1, order=0, optim_method=optim_method, useprecond=true, skew=skew, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=lsmethod, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=500, inner_maxiter = 500, store_trace=true, show_trace=false)

        
# Without LPF
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); #imshowW(W1,imgsz,borderwidth=1,title="W_"*optim_mtd_str)
imsaveW("W_$(optim_mtd_str).png", W1,imgsz,borderwidth=1)

# X LPFed
name = "Wxlpf_"*optim_mtd_str
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wxlpf, Hxlpf; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); #imshowW(W1,imgsz,borderwidth=1,title=name)
imsaveW("$(name).png", W1,imgsz,borderwidth=1)
M=cal_M(trs); W2, H2 = W*M, M\H
name = "W_Mxlpf_"*optim_mtd_str
normalizeWH!(W2,H2); #imshowW(W2,imgsz,borderwidth=1,title=name)
imsaveW("$(name).png", W2,imgsz,borderwidth=1)

# W, H LPFed
name = "Wlpf_"*optim_mtd_str
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wlpf, Hlpf; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); #imshowW(W1,imgsz,borderwidth=1,title=name)
imsaveW("$(name).png", W1,imgsz,borderwidth=1)
M=cal_M(trs); W2, H2 = W*M, M\H
name = "W_Mlpf_"*optim_mtd_str
normalizeWH!(W2,H2); #imshowW(W2,imgsz,borderwidth=1,title=name)
imsaveW("$(name).png", W2,imgsz,borderwidth=1)

# CD
α = 0.1
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=iter, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd); #imshowW(Wcd,imgsz,borderwidth=1,title=name)
imsaveW("W_CD.png", Wcd,imgsz,borderwidth=1)

name = "Wxlpf_CD"
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(Xlpf, ncells)
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=iter, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd); #imshowW(Wcd,imgsz,borderwidth=1,title=name)
imsaveW("$(name).png", Wcd,imgsz,borderwidth=1)

#============ IPNewton (full) =========================================#
p = size(W,2)
pen(x) = (dM=reshape(x,(p,p)); scapair(I+dM, W, H; βi=0, γ=0, allcompW=false, allcompH=false))
gradpen(g, x) = copy!(g, Calculus.gradient(pen, x))
hesspen(h, x) = copy!(h, Calculus.hessian(pen, x))
n = length(initial_x) # number of variables

initial_x = zeros(p^2)
df = TwiceDifferentiable(x->pen(x), (g, x) -> gradpen(g,x), (h, x) -> hesspen(h,x), initial_x)

μ = 1e-20
method = Optim.IPNewton(μ0=μ)

cnst(x) = (dM=reshape(x,(p,p)); sca2(W)-sca2(W*(I+dM)))
gradcnst(x) = Calculus.gradient(cnst, x)
hesscnst(x) = Calculus.hessian(cnst, x)

constraints = TwiceDifferentiableConstraints(
    (c,x)->(c[1]=cnst(x)),
    (J,x)->(J[:,:] = gradcnst(x)),
    (h,x,λ)->(h[:,:] = λ[1]*hesscnst(x)),
    [], [], [0.], [0.]) # [0], [0] => c(x) = 0; [0], [Inf] => c(x) >= 0 
mc = Optim.nconstraints(constraints); mc == 1 # number of constraints

Optim.isfeasible(constraints, initial_x)

options = Optim.Options(; iterations = 2, show_trace = true, Optim.default_options(method)...)

results = optimize(df, constraints, initial_x, method, options)
isa(Optim.summary(results), String)
Optim.converged(results)
x = results.x

#============ IPNewton (approx.) =========================================#

stparams = StepParams(γ=0., β=0.0#=30.=#, μ=0.00001, order=0, optim_method=:ipnewton, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:ac_symmetric_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-2, ρ=0.5, maxiter=100, show_figure=false,
                iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=100, inner_maxiter = 1, store_trace=true, show_trace=true, show_inner_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
imshowW(W1,imgsz,borderwidth=1); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz,borderwidth=1)
imsaveW("W2_$(stparams.penaltytype)_iter$(cparams.maxiter)_rt$(rt).png",W2,imgsz,borderwidth=1)


objvals_n = objvals./objvals[1]
penW_n = penW./penW[1]; penH_n = penH./penH[1]
iterrng = 1:length(objvals_n)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(iterrng, [penW_n[iterrng] penH_n[iterrng]])
ax1.set_yscale(:linear) # :log, :linear
ax1.legend(["penW./penW0","penH./penH0"],fontsize = 12,loc=1, title="")
xlabel("iteration",fontsize = 12)
ylabel("normalized penalty values",fontsize = 12)
savefig("penW_and_penH(100).png")

#======== Suppressed ==========#
fovsz=(20,20); lengthT0 = 100; factor = 8; SNR = 10
imgsz = (fovsz[1]*factor,fovsz[2]); lengthT = lengthT0*factor
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT)_SNR$(SNR).jld";
        fovsz=imgsz, lengthT=lengthT, imgsz=imgsz, SNR=SNR)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
ncells = gt_ncells + 5

rt01 = @elapsed W, H = initWH(X, ncells; svd_method=:svd)
Wn, Hn = balanced_WH(W,X)

W[:,1] = -1 .* W[:,1]; H[1,:] = -1 .* H[1,:]

stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=300, inner_maxiter = 10, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("W_symmetric_orthogonality_SNR$(SNR)_r$(stparams.r)_b$(stparams.β)_iter$(length(objvals)).png",W1,imgsz,borderwidth=1)

M = Matrix(1.0I,ncells,ncells)
for i in 1:length(trs)
    M *= trs[i].M
end

p = size(Wn,2)
Wn[:,1] .*= -1; Hn[1,:] .*= -1
(fg!, P) = SCA.prepare_fg_suppressed(M, Wn, Hn);
opt = Optim.Options(f_tol=1e-5, outer_iterations=100, iterations = 3000, show_trace=false, show_every=100);
initial_x = zeros(p^2)
rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
M = reshape(rst.minimizer,(p,p)); for i = 1:p M[i,i] = 1 end
W1 = Wn*M; H1 = M\Hn
normalizeWH!(W1,H1); imshowW(W1,imgsz)


stparams = StepParams(γ=0., β=30#=30.=#, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=300, inner_maxiter = 10, store_trace=true, show_trace=true)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz)

pens = []; penWs = []; penHs = []
M = Matrix(1.0I,ncells,ncells)
for tr in trs
    M *= tr.M
    penW = sca2(W*M); penH = sca2(M\H); pen = penW*penH
    push!(pens, pen)
    push!(penWs, penW)
    push!(penHs, penH)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(1:length(pens), [pens penWs penHs])
ax1.set_yscale(:linear) # :log, :linear
ax1.legend(["pens", "penWs","penHs"],fontsize = 12,loc=1, title="")
xlabel("iteration",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("pen_penWs_penHs.png")

#======== Column by column minimization ==========#

p = size(W,2)

M = Matrix(1.0I,p,p)
for col_idx = 1:p
    (fg!, P) = SCA.prepare_fg_onecol(M, Wn, Hn, col_idx=col_idx);
    opt = Optim.Options(f_tol=1e-5, outer_iterations=100, iterations = 3000, show_trace=true, show_every=100);
    if col_idx == 1
        initial_x = zeros(p); initial_x[1] = -1
    else
        m = -sum(M[:,1:col_idx-1],dims=2)
        M[1:col_idx-1,col_idx] = m[1:col_idx-1]
        initial_x = zeros(p-col_idx+1); initial_x[1] = 1
    end
    rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
    x = rst.minimizer
end
W1,H1 = Wn*M, Mn\H
normalizeWH!(W1,H1); imshowW(W1,imgsz)

M = Matrix(1.0I,p,p)
for col_idx = 1:p
    (fg!, P) = SCA.prepare_fg_onecol2(M, Wn, Hn, col_idx=col_idx);
    opt = Optim.Options(f_tol=1e-5, outer_iterations=100, iterations = 3000, show_trace=true, show_every=100);
    if col_idx == 1
        initial_x = zeros(p); initial_x[1] = -1
    else
        # m = -vec(sum(M[:,1:col_idx-1],dims=2))
        # initial_x = m; initial_x[col_idx] += 1
        initial_x = zeros(p-col_idx+1); initial_x[1] = 1
    end
    rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
    x = rst.minimizer
end
W1,H1 = W*M, M\H
normalizeWH!(W1,H1); imshowW(W1,imgsz)

M0 = Matrix(1.0I,p,p)
W0, H0 = copy(Wn), copy(Hn)
M = Matrix(1.0I,p,p); M[1,1] = -1
for col_idx = 1:p-1
    if Bool(col_idx%2)
        (fg!, P) = SCA.prepare_fg_onecolW(M, W0, H0, col_idx=col_idx);
    else
        (fg!, P) = SCA.prepare_fg_onecolH(M, W0, H0, col_idx=col_idx);
    end
    opt = Optim.Options(f_tol=1e-5, outer_iterations=100, iterations = 3000, show_trace=false, show_every=100);
    initial_x = zeros(p-col_idx)
    rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
    W0 = W0*M; H0 = M\H0
    M0 = M0*M
    M = Matrix(1.0I,p,p)
    @show rst.minimizer
end
normalizeWH!(W0,H0); imshowW(W0,imgsz)

M0 = Matrix(1.0I,p,p)
W0, H0 = copy(Wn), copy(Hn)
for col_idx = 1:p
    (fg!, P) = SCA.prepare_fg_onecolnorm(M, W0, H0, col_idx=col_idx);
    opt = Optim.Options(f_tol=1e-5, outer_iterations=100, iterations = 3000, show_trace=false, show_every=100);
    if col_idx == 1
        initial_x = zeros(p); initial_x[1] = -1
    else
        initial_x = zeros(p); initial_x[col_idx] = 1
    end
    rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
    x = rst.minimizer
    @show x
    W0 = W0*M; H0 = M\H0
    M0 = M0*M
    M = Matrix(1.0I,p,p)
end
normalizeWH!(W0,H0); imshowW(W0,imgsz)

#======== Orthogonality constraint + alterative constraint ==========#
fovsz=(20,20); lengthT0 = 100; factor = 8; SNR = 10; ncells = 40
imgsz = (fovsz[1]*factor,fovsz[2]); lengthT = lengthT0*factor
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT)_SNR$(SNR).jld";
        fovsz=imgsz, lengthT=lengthT, imgsz=imgsz, SNR=SNR)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
ncells = gt_ncells + 5

rt01 = @elapsed W, H = initWH(X, ncells; svd_method=:svd)
Wn, Hn = balanced_WH(W,X)

stparams = StepParams(γ=0., β=50#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=300, inner_maxiter = 10, store_trace=true, show_trace=false)
rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
#imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("W_ac_orthogonality_SNR$(SNR)_r$(stparams.r)_b$(stparams.β)_iter$(length(objvals)).png",W1,imgsz,borderwidth=1)



#====== Do we lost important information when we truncate SVD =======#
orthog = false
orthogstr = orthog ? "orthog" : "nonorthog"
imgsz=(40,20); ncells=14; lengthT=100; SNR=-13 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld";
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=false)
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.1 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd)
mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
imsaveW("noc_cd_X_n$(ncells)_mssd$(mssdcd).png",Wcd,imgsz,borderwidth=1)

X_tr = W*H
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X_tr, ncells)
α = 0.1 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X_tr, Wcd, Hcd)
normalizeWH!(Wcd,Hcd)
mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
imsaveW("noc_cd_Xtr_n$(ncells)_mssd$(mssdcd).png",Wcd,imgsz,borderwidth=1)


#============ why exp(M) also has degeneration =======================#
imsaveW("W.png",W,imgsz)
prng = 1:size(Wn,2)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(prng, M[:,9:end])
ax1.set_xticks(Int.(round.(collect(LinRange(first(prng), last(prng), (last(prng)-first(prng)+1)÷2)))))
#ax1.legend(["penW./penW0","penH./penH0"],fontsize = 12,loc=1, title="")
xlabel("row index",fontsize = 12)
ylabel("M[i,j]",fontsize = 12)
savefig("M[i,j].png")
grid("on", which="major", linestyle="-") # axis="both",color="b", linestyle="-", linewidth=1

#============ Several tests =========================================#
fovsz=(20,20); lengthT0 = 100; factor = 8; SNR = 10
imgsz = (fovsz[1]*factor,fovsz[2]); lengthT = lengthT0*factor
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT)_SNR$(SNR).jld";
        fovsz=imgsz, lengthT=lengthT, imgsz=imgsz, SNR=SNR)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
ncells = gt_ncells + 5

rt01 = @elapsed W, H = initWH(X, ncells; svd_method=:svd)
Wn, Hn = balanced_WH(W,X)

# Pure Linear with orthogonality

stparams = StepParams(γ=0., β=30#=30.=#, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

rt02 = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("W_b$(stparams.β)_iter$(length(objvals)).png",W1,imgsz,borderwidth=1)


# Pure AC Linear
stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

rt02 = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("W_aclin_r$(stparams.r)_iter$(length(objvals)).png",W1,imgsz,borderwidth=1)

# Skew symmetric + AC Linear
stparams = StepParams(γ=0., β=30#=30.=#, r=0.1, order=0, optim_method=:newton, useprecond=true, skew=true, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

rt02 = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("W_skew_b$(stparams.β)_iter$(length(objvals)).png",W1,imgsz,borderwidth=1)

Wn, Hn = balanced_WH(W1,X)

stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

rt02 = @elapsed W2, H2, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W2,H2); imshowW(W2,imgsz)
imsaveW("W_skew+aclin_r$(stparams.r)_iter$(length(objvals)).png",W2,imgsz,borderwidth=1)

# LPFed W + SCA = M, W*M
using ImageFiltering

function LPF_WH!(W,H,imgsz,filterlength=3)
    isodd(filterlength) || error("filterlength should be odd number")
    ker1D = centered(fill!(zeros(filterlength),1/filterlength))
    ker2D = centered(fill!(zeros(filterlength,filterlength),1/filterlength^2))
    imfilter(img, ker)
    for idx in 1:size(W,2)
        eachcol_W = W[:,idx]
        eachrow_H = H[idx,:]
        img = reshape(eachcol_W,imgsz)
        W[:,idx] = vec(imfilter(img, ker2D))
        H[idx,:] = imfilter(H[idx,:], ker1D)
    end
end

WnLPF, HnLPF = copy(Wn), copy(Hn)
LPF_WH!(WnLPF,HnLPF, imgsz)

stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

rt02 = @elapsed W2, H2, objvals, trs, penW, penH = semiscasolve!(WnLPF, HnLPF; stparams=stparams, lsparams=lsparams, cparams=cparams);
p = size(Wn,2); M = Matrix(1.0I,p,p)
for tr in trs
    M *= tr.M
end
W2 = Wn*M; H2 = M\Hn

normalizeWH!(W2,H2); imshowW(W2,imgsz)
imsaveW("W_filtered_r$(stparams.r)_iter$(length(objvals)).png",W2,imgsz,borderwidth=1)


#==== AC test by changing SNR =============#

# exponential alternative constraint
for SNR in [-10, 10, 30]
    @show SNR
    imgsz=(40,20); ncells=20; lengthT=1000;  # SNR=10(noisey), SNR=40(less noisey)
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld";
                                                fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=false)
    #X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    W0 = W1 = W2 = W3 = copy(W)

    Wn, Hn = balanced_WH(W,X)
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]
        @show r
        stparams = StepParams(γ=0., β=0#=30.=#, r=r, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
                        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
                                # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
        lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
                        iterations_to_show=collect(1:18))
        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                        maxiter=300, inner_maxiter = 10, store_trace=true, show_trace=false)
        rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
        #imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
        pW,pH,pWH=sca2(W1), sca2(H1), scapair(W1,H1)
        normalizeWH!(W1,H1); imshowW(W1,imgsz)
        imsaveW("W_Hess_SNR$(SNR)_r$(r)_b$(β)_iter$(length(objvals))_rt$(rt)_pen$((pW,pH,pWH)).png",W1,imgsz,borderwidth=1)
    end
end

objvals_n = objvals./objvals[1]
penW_n = penW./penW[1]; penH_n = penH./penH[1]
iterrng = 1:41
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(iterrng, [penW_n[iterrng] penH_n[iterrng]])
ax1.set_yscale(:linear) # :log, :linear
ax1.legend(["penW./penW0","penH./penH0"],fontsize = 12,loc=1, title="")
xlabel("iteration",fontsize = 12)
ylabel("normalized penalty values",fontsize = 12)
savefig("penW_and_penH(8).png")

#==== incremental optimization =============#
ncellsrng = 6:2:100; xlabelstr = "number of components"
imgsz=(40,20); lengthT = 100

X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz).jld"; lengthT=lengthT, imgsz=imgsz)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
# Fail : 26(r=0.05), 34, 36, 42, *46, (48), 50, 56(r=0.1), 58, 60, 64, 68-82, 86-100

stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

ncells = 30
rt11 = @elapsed W, H = initWH(X, ncells; svd_method=:isvd)
Wn, Hn = balanced_WH(W,X)
Winc, Hinc = copy(Wn), copy(Hn)
for n = 2:ncells
    @show n
    Win = Winc[:,1:n]; Hin = Hinc[1:n,:]
    rt12 = @elapsed Wout, Hout, objvals, trs, penW, penH = semiscasolve!(Win, Hin; stparams=stparams, lsparams=lsparams, cparams=cparams);
    Winc[:,1:n] = Wout; Hinc[1:n,:] = Hout
end
normalizeWH!(Winc,Hinc); imshowW(Winc,imgsz)
    

#==== noc vs runtime (:ac_symmetric) =============#
ncellsrng = 6:2:100; xlabelstr = "number of components"
imgsz=(40,20); lengthT = 100

X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz).jld"; lengthT=lengthT, imgsz=imgsz)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
# Fail : 26(r=0.05), 34, 36, 42, *46, (48), 50, 56(r=0.1), 58, 60, 64, 68-82, 86-100


stparams = StepParams(γ=0., β=50#=30.=#, r=0.1, order=0, optim_method=:cg, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
lsparams = LineSearchParams(method=:none, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_exp_full) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

# isvd
rtisvd1s = []; rtisvd2s = []; mssdisvds = []
for ncells in ncellsrng
    @show ncells
    rt11 = @elapsed Wisvd, Hisvd = initWH(X, ncells; svd_method=:svd)
    Wn, Hn = balanced_WH(Wisvd,X)
    rt12 = @elapsed Wisvd1, Hisvd1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
    normalizeWH!(Wisvd1,Hisvd1); imshowW(Wisvd1,imgsz)
    mssdisvd, ml, ssds = matchedfiterr(gtW,Wisvd1);
    push!(rtisvd1s, rt11); push!(rtisvd2s, rt12); push!(mssdisvds,mssdisvd)
    imsaveW("noc_svd_W_n$(ncells).png",Wisvd,imgsz)
    imsaveW("noc_svd_Wp_n$(ncells)_iter$(length(objvals))_mssd$(mssdisvd).png",Wisvd1,imgsz)
end
rtisvds = rtisvd1s + rtisvd2s

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
rtcd1s = []; rtcd2s = []; mssdcds = []
for ncells in ncellsrng
    @show ncells
    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    imsaveW("noc_cd_n$(ncells).png",Wcd,imgsz)
    α = 0.0 # best α = 0.1
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
    imsaveW("noc_cd_n$(ncells)_mssd$(mssdcd).png",Wcd,imgsz)
end

fname = "noc_vs_ac_symmetric_r$(stparams.r)_$(today())_$(hour(now())).jld"
save(fname, "r", stparams.r, "ncellsrng", ncellsrng, "imgsz", imgsz, "lengthT0", lengthT0,
        "rtcd1s", rtcd1s, "rtcd21s", rtcd2s, "mssdcds", mssdcds, # cd
        "rtisvd1s",rtisvd1s,"rtisvd2s",rtisvd2s, "mssdisvds", mssdisvds) # isvd

dd = load("noc_vs_ac_symmetric_r2021-11-11_14.jld")
β = dd["beta"]; ncellsrng = dd["ncellsrng"]; imgsz = dd["imgsz"]; lengthT0 = dd["lengthT0"];
rtcd1s = dd["rtcd1s"]; rtcd2s = dd["rtcd21s"]; mssdcds = dd["mssdcds"];
rtisvd1s = dd["rtisvd1s"]; rtisvd2s = dd["rtisvd2s"]; mssdisvds = dd["mssdisvds"];
rtcds = rtcd1s + rtcd2s
rtisvds = rtisvd1s + rtisvd2s

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtisvds rtcds])
ax1.legend(["SCA(ISVD)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtsvd1s rtisvd1s rtrsvd101s rtrsvd501s rtrsvd1001s rtcd1s])
# ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
#         "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_svdruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtsvd2s rtisvd2s rtrsvd102s rtrsvd502s rtrsvd1002s rtcd2s])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_coreruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [mssdisvds mssdcds])
ax1.set_yscale(:linear) # :log
ax1.legend(["Alternative", "Coordinate Descent"],fontsize = 12,loc=1)
ax1.set_yscale(:log) # :linear
xlabel(xlabelstr,fontsize = 12)
ylabel("MSSD",fontsize = 12)
savefig("noc_vs_mssd.png")


#==== factor vs different svd runtime (rsvd test) =============#
factorrng = 1:40;
fovsz=(20,20); lengthT0 = 100
stparams = StepParams(γ=0., β=0#=30.=#, r=0.1, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=1000, inner_maxiter = 10, store_trace=true, show_trace=false)

# svd
rtsvd1s = []; rtsvd2s = []; mssdsvds = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    rt01 = @elapsed Wsvd, Hsvd = initWH(X, ncells; svd_method=:svd)

    Wn, Hn = balanced_WH(Wsvd,X)
    rt02 = @elapsed Wsvd1, Hsvd1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
    normalizeWH!(Wsvd1,Hsvd1);
    mssdsvd, ml, ssds = matchedfiterr(gtW,Wsvd1);
    push!(rtsvd1s, rt01); push!(rtsvd2s, rt02); push!(mssdsvds,mssdsvd)
    imsaveW("svd_f$(factor).png",Wsvd,imgsz)
    imsaveW("svd_f$(factor)_mssd$(mssdsvd).png",Wsvd1,imgsz)
end
rtsvds = rtsvd1s + rtsvd2s

# isvd
rtisvd1s = []; rtisvd2s = []; mssdisvds = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    rt11 = @elapsed Wisvd, Hisvd = initWH(X, ncells; svd_method=:isvd)
    rt12 = @elapsed Wisvd1, Hisvd1, objval, trs = semiscasolve!(Wisvd, Hisvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wisvd1,Hisvd1);
    mssdisvd, ml, ssds = matchedfiterr(gtW,Wisvd1);
    push!(rtisvd1s, rt11); push!(rtisvd2s, rt12); push!(mssdisvds,mssdisvd)
    imsaveW("isvd_f$(factor).png",Wisvd,imgsz)
    imsaveW("isvd_f$(factor)_mssd$(mssdisvd).png",Wisvd1,imgsz)
end
rtisvds = rtisvd1s + rtisvd2s

# rsvd
rtrsvd101s = []; rtrsvd102s = []; mssdrsvd10s = []
rtrsvd501s = []; rtrsvd502s = []; mssdrsvd50s = []
rtrsvd1001s = []; rtrsvd1002s = []; mssdrsvd100s = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    extra_n = 10; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd101s, rt21); push!(rtrsvd102s, rt22); push!(mssdrsvd10s,mssdrsvd)
    imsaveW("rsvd_f$(factor)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("rsvd_f$(factor)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)

    extra_n = 50; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd501s, rt21); push!(rtrsvd502s, rt22); push!(mssdrsvd50s,mssdrsvd)
    imsaveW("rsvd_f$(factor)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("rsvd_f$(factor)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)

    extra_n = 100; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd1001s, rt21); push!(rtrsvd1002s, rt22); push!(mssdrsvd100s,mssdrsvd)
    imsaveW("rsvd_f$(factor)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("rsvd_f$(factor)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)
end
rtrsvd10s = rtrsvd101s+rtrsvd102s
rtrsvd50s = rtrsvd501s+rtrsvd502s
rtrsvd100s = rtrsvd1001s+rtrsvd1002s

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
rtcd1s = []; rtcd2s = []; mssdcds = []
for factor in factorrng
    @show factor
    fovsz=(20,20); lengthT0 = 100
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    imsaveW("cd_f$(factor).png",Wcd,imgsz)
    α = 0.0 # best α = 0.1
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
    imsaveW("cd_f$(factor)_mssd$(mssdcd).png",Wcd,imgsz)
end
rtcds = rtcd1s + rtcd2s

currenttime = now()
fname = "isvd_vs_svd_vs_rsvd_b$(β)_$(today())_$(hour(currenttime))-$(minute(currenttime)).jld"
save(fname, "beta", β, "factorrng", factorrng, "fovsz", fovsz, "lengthT0", lengthT0,
        "rtcd1s", rtcd1s, "rtcd21s", rtcd2s, "mssdcds", mssdcds, # cd
        "rtsvd1s",rtsvd1s,"rtsvd2s",rtsvd2s, "mssdsvds", mssdsvds, # svd
        "rtisvd1s",rtisvd1s,"rtisvd2s",rtisvd2s, "mssdisvds", mssdisvds, # isvd
        "rtrsvd101s",rtrsvd101s,"rtrsvd102s",rtrsvd102s, "mssdrsvd10s", mssdrsvd10s,  # rsvd
        "rtrsvd501s",rtrsvd501s,"rtrsvd502s",rtrsvd502s, "mssdrsvd50s", mssdrsvd50s,
        "rtrsvd1001s",rtrsvd1001s,"rtrsvd1002s",rtrsvd1002s, "mssdrsvd100s", mssdrsvd100s)

dd = load("isvd_vs_svd_vs_rsvd_b50_2021-11-11_18-50.jld")
β = dd["beta"]; factorrng = dd["factorrng"]; fovsz = dd["fovsz"]; lengthT0 = dd["lengthT0"];
rtcd1s = dd["rtcd1s"]; rtcd2s = dd["rtcd21s"]; mssdcds = dd["mssdcds"];
rtsvd1s = dd["rtsvd1s"]; rtsvd2s = dd["rtsvd2s"]; mssdsvds = dd["mssdsvds"];
rtisvd1s = dd["rtisvd1s"]; rtisvd2s = dd["rtisvd2s"]; mssdisvds = dd["mssdisvds"];
rtrsvd101s = dd["rtrsvd101s"]; rtrsvd102s = dd["rtrsvd102s"]; mssdrsvd10s = dd["mssdrsvd10s"];
rtrsvd501s = dd["rtrsvd501s"]; rtrsvd502s = dd["rtrsvd502s"];  mssdrsvd50s = dd["mssdrsvd50s"];
rtrsvd1001s = dd["rtrsvd1001s"]; rtrsvd1002s = dd["rtrsvd1002s"]; mssdrsvd100s = dd["mssdrsvd100s"];
rtcds = rtcd1s + rtcd2s
rtsvds = rtsvd1s + rtsvd2s
rtisvds = rtisvd1s + rtisvd2s
rtrsvd10s = rtrsvd101s+rtrsvd102s
rtrsvd50s = rtrsvd501s+rtrsvd502s
rtrsvd100s = rtrsvd1001s+rtrsvd1002s

xlabelstr = "factor"
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsvds rtisvds rtrsvd10s rtrsvd50s rtrsvd100s rtcds])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsvd1s rtisvd1s rtrsvd101s rtrsvd501s rtrsvd1001s rtcd1s])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_svdruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsvd2s rtisvd2s rtrsvd102s rtrsvd502s rtrsvd1002s rtcd2s])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_coreruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [mssdsvds mssdisvds mssdrsvd10s mssdrsvd50s mssdrsvd100s mssdcds])
ax1.set_yscale(:log) # :linear
# ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
#         "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_mssd.png")


#==== noc vs runtime (rsvd test) =============#
ncellsrng = 6:2:100; xlabelstr = "number of components"
imgsz=(40,20); lengthT = 100

X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz).jld"; lengthT=lengthT, imgsz=imgsz)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

# svd
rtsvd1s = []; rtsvd2s = []; mssdsvds = []
for ncells in ncellsrng
    @show ncells
    rt01 = @elapsed Wsvd, Hsvd = initWH(X, ncells; svd_method=:svd)
    rt02 = @elapsed Wsvd1, Hsvd1, objval, trs = semiscasolve!(Wsvd, Hsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wsvd1,Hsvd1);
    mssdsvd, ml, ssds = matchedfiterr(gtW,Wsvd1);
    push!(rtsvd1s, rt01); push!(rtsvd2s, rt02); push!(mssdsvds,mssdsvd)
    imsaveW("noc_svd_n$(ncells).png",Wsvd,imgsz)
    imsaveW("noc_svd_n$(ncells)_mssd$(mssdsvd1).png",Wsvd1,imgsz)
end
rtsvds = rtsvd1s + rtsvd2s

# isvd
rtisvd1s = []; rtisvd2s = []; mssdisvds = []
for ncells in ncellsrng
    @show ncells
    rt11 = @elapsed Wisvd, Hisvd = initWH(X, ncells; svd_method=:isvd)
    rt12 = @elapsed Wisvd1, Hisvd1, objval, trs = semiscasolve!(Wisvd, Hisvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wisvd1,Hisvd1);
    mssdisvd, ml, ssds = matchedfiterr(gtW,Wisvd1);
    push!(rtisvd1s, rt11); push!(rtisvd2s, rt12); push!(mssdisvds,mssdisvd)
    imsaveW("noc_isvd_n$(ncells).png",Wisvd,imgsz)
    imsaveW("noc_isvd_n$(ncells)_mssd$(mssdisvd).png",Wisvd1,imgsz)
end
rtisvds = rtisvd1s + rtisvd2s

# rsvd
rtrsvd101s = []; rtrsvd102s = []; mssdrsvd10s = []
rtrsvd501s = []; rtrsvd502s = []; mssdrsvd50s = []
rtrsvd1001s = []; rtrsvd1002s = []; mssdrsvd100s = []
for ncells in ncellsrng
    @show ncells
    
    extra_n = 10; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd101s, rt21); push!(rtrsvd102s, rt22); push!(mssdrsvd10s,mssdrsvd)
    imsaveW("noc_rsvd_n$(ncells)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("noc_rsvd_n$(ncells)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)

    extra_n = 50; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd501s, rt21); push!(rtrsvd502s, rt22); push!(mssdrsvd50s,mssdrsvd)
    imsaveW("noc_rsvd_n$(ncells)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("noc_rsvd_n$(ncells)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)

    extra_n = 100; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd1001s, rt21); push!(rtrsvd1002s, rt22); push!(mssdrsvd100s,mssdrsvd)
    imsaveW("noc_rsvd_n$(ncells)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("noc_rsvd_n$(ncells)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)
end
rtrsvd10s = rtrsvd101s+rtrsvd102s
rtrsvd50s = rtrsvd501s+rtrsvd502s
rtrsvd100s = rtrsvd1001s+rtrsvd1002s

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
rtcd1s = []; rtcd2s = []; mssdcds = []
for ncells in ncellsrng
    @show ncells
    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    imsaveW("noc_cd_n$(ncells).png",Wcd,imgsz)
    α = 0.0 # best α = 0.1
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
    imsaveW("noc_cd_n$(ncells)_mssd$(mssdcd).png",Wcd,imgsz)
end

fname = "noc_vs_different_svds_b$(β)_$(today())_$(hour(now())).jld"
save(fname, "beta", β, "ncellsrng", ncellsrng, "imgsz", imgsz, "lengthT0", lengthT0,
        "rtcd1s", rtcd1s, "rtcd21s", rtcd2s, "mssdcds", mssdcds, # cd
        "rtsvd1s",rtsvd1s,"rtsvd2s",rtsvd2s, "mssdsvds", mssdsvds, # svd
        "rtisvd1s",rtisvd1s,"rtisvd2s",rtisvd2s, "mssdisvds", mssdisvds, # isvd
        "rtrsvd101s",rtrsvd101s,"rtrsvd102s",rtrsvd102s, "mssdrsvd10s", mssdrsvd10s,  # rsvd
        "rtrsvd501s",rtrsvd501s,"rtrsvd502s",rtrsvd502s, "mssdrsvd50s", mssdrsvd50s,
        "rtrsvd1001s",rtrsvd1001s,"rtrsvd1002s",rtrsvd1002s, "mssdrsvd100s", mssdrsvd100s)

dd = load("noc_vs_different_svds_b50_2021-11-11_14.jld")
β = dd["beta"]; ncellsrng = dd["ncellsrng"]; imgsz = dd["imgsz"]; lengthT0 = dd["lengthT0"];
rtcd1s = dd["rtcd1s"]; rtcd2s = dd["rtcd21s"]; mssdcds = dd["mssdcds"];
rtsvd1s = dd["rtsvd1s"]; rtsvd2s = dd["rtsvd2s"]; mssdsvds = dd["mssdsvds"];
rtisvd1s = dd["rtisvd1s"]; rtisvd2s = dd["rtisvd2s"]; mssdisvds = dd["mssdisvds"];
rtrsvd101s = dd["rtrsvd101s"]; rtrsvd102s = dd["rtrsvd102s"]; mssdrsvd10s = dd["mssdrsvd10s"];
rtrsvd501s = dd["rtrsvd501s"]; rtrsvd502s = dd["rtrsvd502s"];  mssdrsvd50s = dd["mssdrsvd50s"];
rtrsvd1001s = dd["rtrsvd1001s"]; rtrsvd1002s = dd["rtrsvd1002s"]; mssdrsvd100s = dd["mssdrsvd100s"];
rtcds = rtcd1s + rtcd2s
rtsvds = rtsvd1s + rtsvd2s
rtisvds = rtisvd1s + rtisvd2s
rtrsvd10s = rtrsvd101s+rtrsvd102s
rtrsvd50s = rtrsvd501s+rtrsvd502s
rtrsvd100s = rtrsvd1001s+rtrsvd1002s

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtsvds rtisvds rtrsvd10s rtrsvd50s rtrsvd100s rtcds])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtsvd1s rtisvd1s rtrsvd101s rtrsvd501s rtrsvd1001s rtcd1s])
# ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
#         "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_svdruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtsvd2s rtisvd2s rtrsvd102s rtrsvd502s rtrsvd1002s rtcd2s])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_coreruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [mssdsvds mssdisvds mssdrsvd10s mssdrsvd50s mssdrsvd100s mssdcds])
ax1.set_yscale(:log) # :linear
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=1)
ax1.set_yscale(:log) # :linear
xlabel(xlabelstr,fontsize = 12)
ylabel("MSSD",fontsize = 12)
savefig("noc_vs_mssd.png")


#==== factor vs different svd runtime (rsvd test) =============#
factorrng = 1:40;
fovsz=(20,20); lengthT0 = 100

cparams = Params(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5, maxiter=100, show_trace=false)

# svd
rtsvd1s = []; rtsvd2s = []; mssdsvds = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    rt01 = @elapsed Wsvd, Hsvd = initWH(X, ncells; svd_method=:svd)
    rt02 = @elapsed Wsvd1, Hsvd1, objval, trs = semiscasolve!(Wsvd, Hsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wsvd1,Hsvd1);
    mssdsvd, ml, ssds = matchedfiterr(gtW,Wsvd1);
    push!(rtsvd1s, rt01); push!(rtsvd2s, rt02); push!(mssdsvds,mssdsvd)
    imsaveW("svd_f$(factor).png",Wsvd,imgsz)
    imsaveW("svd_f$(factor)_mssd$(mssdsvd).png",Wsvd1,imgsz)
end
rtsvds = rtsvd1s + rtsvd2s

# isvd
rtisvd1s = []; rtisvd2s = []; mssdisvds = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    rt11 = @elapsed Wisvd, Hisvd = initWH(X, ncells; svd_method=:isvd)
    rt12 = @elapsed Wisvd1, Hisvd1, objval, trs = semiscasolve!(Wisvd, Hisvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wisvd1,Hisvd1);
    mssdisvd, ml, ssds = matchedfiterr(gtW,Wisvd1);
    push!(rtisvd1s, rt11); push!(rtisvd2s, rt12); push!(mssdisvds,mssdisvd)
    imsaveW("isvd_f$(factor).png",Wisvd,imgsz)
    imsaveW("isvd_f$(factor)_mssd$(mssdisvd).png",Wisvd1,imgsz)
end
rtisvds = rtisvd1s + rtisvd2s

# rsvd
rtrsvd101s = []; rtrsvd102s = []; mssdrsvd10s = []
rtrsvd501s = []; rtrsvd502s = []; mssdrsvd50s = []
rtrsvd1001s = []; rtrsvd1002s = []; mssdrsvd100s = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    extra_n = 10; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd101s, rt21); push!(rtrsvd102s, rt22); push!(mssdrsvd10s,mssdrsvd)
    imsaveW("rsvd_f$(factor)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("rsvd_f$(factor)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)

    extra_n = 50; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd501s, rt21); push!(rtrsvd502s, rt22); push!(mssdrsvd50s,mssdrsvd)
    imsaveW("rsvd_f$(factor)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("rsvd_f$(factor)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)

    extra_n = 100; @show extra_n
    rt21 = @elapsed (F = rsvd(X,ncells,extra_n); Wrsvd = F.U; Hrsvd = Wrsvd\X)
    rt22 = @elapsed Wrsvd1, Hrsvd1, objval, trs = semiscasolve!(Wrsvd, Hrsvd; β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(Wrsvd1,Hrsvd1);
    mssdrsvd, mlrsvd, ssdsrsvd = matchedfiterr(gtW,Wrsvd1);
    push!(rtrsvd1001s, rt21); push!(rtrsvd1002s, rt22); push!(mssdrsvd100s,mssdrsvd)
    imsaveW("rsvd_f$(factor)_en$(extra_n).png",Wrsvd,imgsz)
    imsaveW("rsvd_f$(factor)_en$(extra_n)_mssd$(mssdrsvd).png",Wrsvd1,imgsz)
end
rtrsvd10s = rtrsvd101s+rtrsvd102s
rtrsvd50s = rtrsvd501s+rtrsvd502s
rtrsvd100s = rtrsvd1001s+rtrsvd1002s

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
rtcd1s = []; rtcd2s = []; mssdcds = []
for factor in factorrng
    @show factor
    fovsz=(20,20); lengthT0 = 100
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    ncells = gt_ncells + 5

    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    imsaveW("cd_f$(factor).png",Wcd,imgsz)
    α = 0.0 # best α = 0.1
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
    imsaveW("cd_f$(factor)_mssd$(mssdcd).png",Wcd,imgsz)
end
rtcds = rtcd1s + rtcd2s

currenttime = now()
fname = "isvd_vs_svd_vs_rsvd_b$(β)_$(today())_$(hour(currenttime))-$(minute(currenttime)).jld"
save(fname, "beta", β, "factorrng", factorrng, "fovsz", fovsz, "lengthT0", lengthT0,
        "rtcd1s", rtcd1s, "rtcd21s", rtcd2s, "mssdcds", mssdcds, # cd
        "rtsvd1s",rtsvd1s,"rtsvd2s",rtsvd2s, "mssdsvds", mssdsvds, # svd
        "rtisvd1s",rtisvd1s,"rtisvd2s",rtisvd2s, "mssdisvds", mssdisvds, # isvd
        "rtrsvd101s",rtrsvd101s,"rtrsvd102s",rtrsvd102s, "mssdrsvd10s", mssdrsvd10s,  # rsvd
        "rtrsvd501s",rtrsvd501s,"rtrsvd502s",rtrsvd502s, "mssdrsvd50s", mssdrsvd50s,
        "rtrsvd1001s",rtrsvd1001s,"rtrsvd1002s",rtrsvd1002s, "mssdrsvd100s", mssdrsvd100s)

dd = load("isvd_vs_svd_vs_rsvd_b50_2021-11-11_18-50.jld")
β = dd["beta"]; factorrng = dd["factorrng"]; fovsz = dd["fovsz"]; lengthT0 = dd["lengthT0"];
rtcd1s = dd["rtcd1s"]; rtcd2s = dd["rtcd21s"]; mssdcds = dd["mssdcds"];
rtsvd1s = dd["rtsvd1s"]; rtsvd2s = dd["rtsvd2s"]; mssdsvds = dd["mssdsvds"];
rtisvd1s = dd["rtisvd1s"]; rtisvd2s = dd["rtisvd2s"]; mssdisvds = dd["mssdisvds"];
rtrsvd101s = dd["rtrsvd101s"]; rtrsvd102s = dd["rtrsvd102s"]; mssdrsvd10s = dd["mssdrsvd10s"];
rtrsvd501s = dd["rtrsvd501s"]; rtrsvd502s = dd["rtrsvd502s"];  mssdrsvd50s = dd["mssdrsvd50s"];
rtrsvd1001s = dd["rtrsvd1001s"]; rtrsvd1002s = dd["rtrsvd1002s"]; mssdrsvd100s = dd["mssdrsvd100s"];
rtcds = rtcd1s + rtcd2s
rtsvds = rtsvd1s + rtsvd2s
rtisvds = rtisvd1s + rtisvd2s
rtrsvd10s = rtrsvd101s+rtrsvd102s
rtrsvd50s = rtrsvd501s+rtrsvd502s
rtrsvd100s = rtrsvd1001s+rtrsvd1002s

xlabelstr = "factor"
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsvds rtisvds rtrsvd10s rtrsvd50s rtrsvd100s rtcds])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsvd1s rtisvd1s rtrsvd101s rtrsvd501s rtrsvd1001s rtcd1s])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_svdruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsvd2s rtisvd2s rtrsvd102s rtrsvd502s rtrsvd1002s rtcd2s])
ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
        "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_coreruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [mssdsvds mssdisvds mssdrsvd10s mssdrsvd50s mssdrsvd100s mssdcds])
ax1.set_yscale(:log) # :linear
# ax1.legend(["SCA(SVD)", "SCA(ISVD)", "SCA(RSVD,extra_n=10)", "SCA(RSVD,extra_n=50)",
#         "SCA(RSVD,extra_n=100)", "Coordinate Descent"],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_mssd.png")

#==== Add noise test =====#
imgsz=(40,20); ncells=20; lengthT=1000; SNR=40 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_nonorthog_lengthT$(lengthT)_SNR$(SNR).jld";
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=false)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
imshowW(W,imgsz)

Wn, Hn = balanced_WH(W,X)

signalpwr = sum(img_nl.^2)/length(img_nl)
SNR2 = 10; noise = randn(size(img_nl)).*sqrt(signalpwr/10^(SNR2/10))
img_nl = img_nl + noise
img_nla = AxisArray(img_nl, :x, :y, :time)
X0 = Matrix(reshape(img_nla, prod(imgsz), nimages(img_nla)))
W0, H0 = initWH(X0, ncells; svd_method=:isvd)
imshowW(W0,imgsz)

Wn0, Hn0 = balanced_WH(W0,X)


stparams = StepParams(γ=0., β=0#=30.=#, r=0.3, order=0, optim_method=:constrained, useprecond=true, skew=false, fixposdef=false,
        allcompW=false, allcompH=false, penaltytype=:ac_symmetric) # (optim_method=:cg, penaltytype=:symmetric_exp_full)
        # (optim_method=:constrained, penaltytype=:ac_exp_full), (optim_method=:constrained, penaltytype=:ac_appexp_full)
lsparams = LineSearchParams(method=:sca_full, c=1e-4#=0.5=#, α0=1e-0, ρ=0.5, maxiter=100, show_figure=false,
        iterations_to_show=collect(1:18))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
        maxiter=300, inner_maxiter = 10, store_trace=true, show_trace=false)

rt = @elapsed W1, H1, objvals, trs, penW, penH = semiscasolve!(Wn, Hn; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("Wsvd_SNR40.png",W,imgsz,borderwidth=1)
imsaveW("W_SNR40.png",W1,imgsz,borderwidth=1)

rt = @elapsed W10, H10, objvals, trs, penW, penH = semiscasolve!(Wn0, Hn0; stparams=stparams, lsparams=lsparams, cparams=cparams);
normalizeWH!(W10,H10); imshowW(W10,imgsz)
imsaveW("W0svd_SNR0.png",W0,imgsz,borderwidth=1)
imsaveW("W_SNR0.png",W10,imgsz,borderwidth=1)

M = Matrix(1.0I,ncells,ncells)
for i in 1:length(trs)
    M *= trs[i].M
end
W20 = Wn0*M; H20 = M\Hn0
normalizeWH!(W20,H20); imshowW(W20,imgsz)

W*R = W0
R = pinv(W)*W0
W2 = Wn*R*M; H2 = (R*M)\Hn
normalizeWH!(W2,H2); imshowW(W2,imgsz)
imsaveW("Wc.png",W2,imgsz,borderwidth=1)

X = Wn*Hn
X0 = X+noise
X0 = Wn0*Hn0
W10 = Wn0*M
Wp = X/Hn0*M
Hp = Wp\X
normalizeWH!(Wp,Hp); imshowW(Wp,imgsz)
imsaveW("Wcp.png",Wp,imgsz,borderwidth=1)

#==== imshow error =====#
cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\temp")
using Pkg
Pkg.activate(".")
using ImageView, ProfileView
imshow(rand(3,3))


#==== contour plot =====#

w(x,y) = 2(x-0.1)^2+(y-0.2)^2
h(x,y) = (x-0.15)^2+2(y-0.15)^2
e(x,y) = w(x,y)*h(x,y)

n = 2
objW = zeros(100*n+1,100*n+1)
objH = zeros(100*n+1,100*n+1)
objE = zeros(100*n+1,100*n+1)
as = bs = 0:0.01:n
for (aidx,a) = enumerate(as), (bidx,b) = enumerate(bs)
    objW[aidx,bidx] = w(a,b)
    objH[aidx,bidx] = h(a,b)
    objE[aidx,bidx] = e(a,b)
end

agrid = repeat(as',length(as),1)
bgrid = repeat(bs,1,length(bs))

contour(agrid, bgrid, objW, colors="green", linewidth=1.0)
contour(agrid, bgrid, objH, colors="cyan", linewidth=1.0)
contour(agrid, bgrid, objE, colors="black", linestyle="--", linewidth=1.0)
#label(cp, inline=1, fontsize=10)

#======================= gradient of exp(A) test ============================#
using Calculus
f(m) = (M=reshape(m,(10,10)); norm(exp(M))^2) # 
g(M) = 2*exp(M)
A = rand(10,10)
gA = Calculus.gradient(f,vec(A))
g(A) == reshape(gA,(10,10))
#===== Test if we lost information when we reduce component number (using CD) =======#
# generate low SNR data with only 20 number of components
imgsz=(40,20); ncells=20; lengthT=100; SNR=-15 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld";
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=false)
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

# Reconstruct data
Xt = W*H

noc=20
# CD with original X
α = 0.1 # best α = 0.1
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, noc)
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd);
imshowW(Wcd,imgsz,borderwidth=1)
imsaveW("Wcd$(ncells)_$(SNR)dB.png",Wcd,imgsz)

# CD with Xt
rtcd1 = @elapsed Wtcd, Htcd = NMF.nndsvd(Xt, noc)
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=1000, α=α, l₁ratio=0.5), Xt, Wtcd, Htcd)
normalizeWH!(Wtcd,Htcd);
imshowW(Wtcd,imgsz,borderwidth=1)
imsaveW("Wtcd$(ncells)_$(SNR)dB.png",Wtcd,imgsz)

#================== Do we loose low SNR component during truncation? ================#
orthog = false
orthogstr = orthog ? "orthog" : "nonorthog"
imgsz=(40,20); ncells=20; lengthT=1000; bias=0.1; SNR=-10 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld";
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=false)
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
bgW = ones(*(imgsz...)); normbgW = norm(bgW); bgW ./= normbgW; bgH = fill(normbgW*bias,lengthT)

gtW = [bgW, gtW]; gtH

#================== Old master branch ================#
stparams = StepParams(γ=0., β=10#=30.=#, order=0, usecg=false, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penalty=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=100, inner_maxiter = 100, store_trace=true, show_trace=true)
@time W1, H1, objvals, trs = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2,imgsz)


#================== Alternative minimizing ================#
using Calculus, ForwardDiff, Zygote, FiniteDifferences, ReverseDiff
using Nabla, AutoGrad, Yota


p = size(W,2); βi = 0
penfunc(x) = (M = reshape(x,p,p); SCA.scapair_exp(M,W,H,βi=βi))
penWfunc(x) = (M = reshape(x,p,p); SCA.scapair_exp_W(M,W,H,βi=βi))
constfuncH(x) = (M = reshape(x,p,p); sca2(exp(-M)*H)-sca2(H))
constfuncW(x) = (M = reshape(x,p,p); sca2(W*exp(M))-sca2(W))

pen_fixedH(x) = penfunc(x[1:end-1])+x[end]*constfuncH(x[1:end-1])
pen_fixedW(x) = penfunc(x[1:end-1])+x[end]*constfuncW(x[1:end-1])
penW_fixedH(x) = penWfunc(x[1:end-1])+x[end]*constfuncH(x[1:end-1])

x = zeros(p^2)

penWfunc(x) = (M = reshape(x,p,p); sca2(W*exp(M)))


# W, H and orthogonality components
@time gexpfn_cal = Calculus.gradient(penfunc,x);
@time gexpfn_zg = Zygote.gradient(penfunc,x)[1]; # error
@time gexpfn_fin = FiniteDifferences.grad(central_fdm(5, 1),penfunc,x)[1];
norm(gexpfn_cal-gexpfn_zg) #
norm(gexpfn_cal-gexpfn_fin) # 0.14
norm(gexpfn_zg-gexpfn_fin) #

# fixed W
@time gexpfWfn_cal = Calculus.gradient(constfuncW,x);
@time gexpfWfn_zg = Zygote.gradient(constfuncW,x)[1]; # error
@time gexpfWfn_fin = FiniteDifferences.grad(central_fdm(5, 1),constfuncW,x)[1];
norm(gexpfWfn_cal-gexpfWfn_zg) #
norm(gexpfWfn_cal-gexpfWfn_fin) # 1.4e-8
norm(gexpfWfn_zg-gexpfWfn_fin) #

# fixed H
@time gexpfHfn_cal = Calculus.gradient(constfuncH,x);
@time gexpfHfn_zg = Zygote.gradient(constfuncH,x)[1]; # error
@time gexpfHfn_fin = FiniteDifferences.grad(central_fdm(5, 1),constfuncH,x)[1];
norm(gexpfHfn_cal-gexpfHfn_zg) #
norm(gexpfHfn_cal-gexpfHfn_fin) # 0.00127
norm(gexpfHfn_zg-gexpfHfn_fin) #

x = ones(p^2+1)

# W, H fixed and orthogonality components
@time gexpfHfn_cal = Calculus.gradient(pen_fixedH,x);
@time gexpfHfn_zg = Zygote.gradient(pen_fixedH,x)[1];
@time gexpfHfn_fin = FiniteDifferences.grad(central_fdm(5, 1),pen_fixedH,x)[1];
norm(gexpfHfn_cal-gexpfHfn_zg) # 0.01182369909397127
norm(gexpfHfn_cal-gexpfHfn_fin) # 0.14490307919976675
norm(gexpfHfn_zg-gexpfHfn_fin) # 0.1494902815386609

# W fixed, H and orthogonality components
@time gexpfWfn_cal = Calculus.gradient(pen_fixedW,x);
@time gexpfWfn_zg = Zygote.gradient(pen_fixedW,x)[1]; # error
@time gexpfWfn_fin = FiniteDifferences.grad(central_fdm(5, 1),pen_fixedW,x)[1];
norm(gexpfWfn_cal-gexpfWfn_zg) #
norm(gexpfWfn_cal-gexpfWfn_fin) # 0.14
norm(gexpfWfn_zg-gexpfWfn_fin) #

#================== Alternative minimizing ================#
γ = 0.; β = 30.; order = 0; usecg = true; useprecond = true;
skew = false; fixposdef = false; allcompW = false; allcompH = false;
penaltytype = :symmetric_orthogonality;
penfunc = SCA.scapair_exp;
fgfunc = SCA.prepare_fg_exp;
updateWHfunc = SCA.updateWH_exp;
buildproblemfunc = SCA.buildproblem_symmetric;
powerfunc = (W,H)->norm(W)^2*norm(H)^2;
powerfuncstr = "|W₀|²|H₀|²";
stparams = StepParams(promote(γ, β)..., Int(order), usecg, useprecond, skew, fixposdef, allcompW,
    allcompH, penfunc, fgfunc, buildproblemfunc, updateWHfunc, powerfunc, powerfuncstr, penaltytype)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-6, f_inctol=1e-5,
    maxiter=1, inner_maxiter=100, store_trace=true, show_trace=true)
dM2, dM, b, g_converged, rst =  SCA.altanative_cg(W, H, OT=OT; βi=50, stparams=stparams, cparams=cparams)

imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz)
OT=zeros(eltype(W),size(W,2),size(W,2))


#================== why cg with Calculus and fast method are different ================#
p = size(W,2); T = eltype(W)
β=50; βi = β/p*scapair(W,H)
pen(x) = (dM = reshape(x,p,p); SCA.scapair_W_full(I+dM,W,H;βi=βi))

# W full
fg!, P = SCA.prepare_fg_W_full(W, H; βi=βi)
value = fg!(1,nothing,x)
grad_full = zeros(p^2); fg!(nothing,grad_full,x)

x = zeros(p^2)
grad_cal = Calculus.gradient(pen,x)

fg!, P = SCA.prepare_fg_penW_orthogonality(W, H; βi=βi)
value = fg!(1,nothing,x)
grad_cg = zeros(p^2); fg!(nothing,grad_cg,x)
norm(grad_cal-grad_cg)  # for x = rand(p^2), this isn't hold


#===================== gradient of matrix exponential function ========================#
using Calculus, ForwardDiff, Zygote, FiniteDifferences, ReverseDiff
p = 20; β=0
x = zeros(p^2)

# W component
expfnW(x) = (p=round(Int, sqrt(length(x))); M=reshape(x,p,p); sca2(real.(W*exp(M))))
@time gexpfnW_cal = Calculus.gradient(expfnW,x); # 0.406sec
# gexpfnW_fd = ForwardDiff.gradient(expfn,x) # no method matching exp(::Matrix{ForwardDiff.Dual)
# gexpfnW_rd = ReverseDiff.gradient(expfn,x) # no method matching exp(::ReverseDiff.TrackedArray)
@time gexpfnW_zg = Zygote.gradient(expfnW,x)[1]; # 1.57sec
@time gexpfnW_fin = FiniteDifferences.grad(central_fdm(5, 1),expfnW,x)[1]; # 2.2sec, 5th order central method
norm(gexpfnW_cal-gexpfnW_zg)
norm(gexpfnW_cal-gexpfnW_fin)

# H component
expfnH(x) = (p=round(Int, sqrt(length(x))); M=reshape(x,p,p); sca2(real.(exp(-M)*H)))
@time gexpfnH_cal = Calculus.gradient(expfnH,x); # 0.96sec
@time gexpfnH_zg = Zygote.gradient(expfnH,x)[1]; # 1.79sec
@time gexpfnH_fin = FiniteDifferences.grad(central_fdm(5, 1),expfnH,x)[1]; # 2.44sec
norm(gexpfnH_cal-gexpfnH_zg)
norm(gexpfnH_cal-gexpfnH_fin)

# W,H components
expfnWH(x) = (p=round(Int, sqrt(length(x))); M=reshape(x,p,p); scapair(real.(W*exp(M)), real.(exp(-M)*H)))
@time gexpfnWH_cal = Calculus.gradient(expfnWH,x); # 0.36sec
@time gexpfnWH_zg = Zygote.gradient(expfnWH,x)[1]; # 6.11sec
@time gexpfnWH_fin = FiniteDifferences.grad(central_fdm(5, 1),expfnWH,x)[1]; # 4.36sec
norm(gexpfnWH_cal-gexpfnWH_zg) # 0.0008003235815365343
norm(gexpfnWH_cal-gexpfnWH_fin) # 0.030087271017104976
norm(gexpfnWH_zg-gexpfnWH_fin) # 0.030117104505136473

# W,H and orthogonality components
expfn(x) = (p=round(Int, sqrt(length(x))); M=reshape(x,p,p); scapair(real.(W*exp(M)), real.(exp(-M)*H))+β*norm(M+M')^2)
@time gexpfn_cal = Calculus.gradient(expfn,x); # 2.02sec
@time gexpfn_zg = Zygote.gradient(expfn,x)[1]; # 6.52sec
@time gexpfn_fin = FiniteDifferences.grad(central_fdm(5, 1),expfn,x)[1]; # 6.36sec
norm(gexpfn_cal-gexpfn_zg) # 0.0008003235815365343
norm(gexpfn_cal-gexpfn_fin) # 0.030090059575770104
norm(gexpfn_zg-gexpfn_fin) # 0.030119862352059004

SNR=-15; ncells=14 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld"; ncells=ncells, lengthT=lengthT, SNR=SNR)
stparams = StepParams(γ=0., β=20, order=0, usecg=true, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penaltytype=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100, show_figure=false, iterations_to_show=collect(1:15))
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                maxiter=100, inner_maxiter=100, store_trace=true, show_trace=true, show_inner_trace=false)
@time Wexp, Hexp, objvals, trs = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
@show sca2(Wexp), sca2(Hexp), scapair(Wexp,Hexp)
normalizeWH!(Wexp,Hexp); imshowW(Wexp,imgsz,borderwidth=1)
imsaveW("W_$(stparams.penaltytype)_b$(stparams.β)_i$(cparams.inner_maxiter).png", Wexp, imgsz,borderwidth=1)

# CD
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.1 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd);
imsaveW("Wcd_SNR$(SNR).png", Wcd, imgsz,borderwidth=1)

#============== SCA penalty change from initial W to ideal W  =============#
whitenoise=randn(imgsz..., lengthT)
rthog = false
orthogstr = orthog ? "orthog" : "nonorthog"
ncells=14; lengthT=100;
SNRs = [5,10,20,30,40,50]

# ideal W
SNR=50 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld"; ncells=ncells, lengthT=lengthT, SNR=SNR)
stparams = StepParams(γ=0., β=50., order=0, usecg=true, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penalty=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5, maxiter=100, store_trace=true, show_trace=true)
@time Wideal, Hideal, objvals, trs = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
imshowW(Wideal,imgsz)

wrng = -1:0.001:2; penWHss = []; penWss = []; penHss = []
for SNR in SNRs
    # initila W for the noise data
    img_nl = fakecells_dic["img_nl"]
    signalpwr = sum(img_nl.^2)/length(img_nl)
    bg = whitenoise.*sqrt(signalpwr/10^(SNR/10)) .+ 0.1
    img = img_nl + bg# add noise and bg
    imga = AxisArray(img, :x, :y, :time)
    X = Matrix(reshape(imga, prod(imgsz), nimages(imga)))
    W,H = initWH(X,ncells)

    # ideal direction
    #  M = W\Wideal; dM = M-I
    M = log(W\Wideal)

    # random direction
    # dM = 2.0*(rand(ncells,ncells).-0.5)
    # dM -= Diagonal(dM)

    # skew symmetric direction
    # v = 2*(rand((ncells^2-ncells)÷2).-0.5)
    # dM = SCA.reshapeskew(v)

    penWHs=[]; penWs=[]; penHs=[]
    for w in wrng
        # wM = I+w*dM
        # penWH = scapair(W*wM, wM\H)
        # penW = SCA.sca2(W*wM)
        # penH = SCA.sca2(wM\H)
        wM = exp(w*M); invwM = exp(-w*M)
        penWH = scapair(real.(W*wM), real.(invwM*H))
        penW = SCA.sca2(real.(W*wM))
        penH = SCA.sca2(real.(invwM*H))
        push!(penWHs,penWH)
        push!(penWs,penW)
        push!(penHs,penH)
    end
    push!(penWHss, penWHs)
    push!(penWss, penWs)
    push!(penHss, penHs)
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(wrng, hcat(penHss...))
ax1.set_yscale(:log) # :linear
ax1.legend(SNRs,fontsize = 12,loc=3)
xlabel("α",fontsize = 12)
ylabel("penalty",fontsize = 12)
ax1.minorticks_on() # minortics_off()
# grid(true, which="both") # grid("on")
grid("on", which="major", linestyle="-") # axis="both",color="b", linestyle="-", linewidth=1
grid("on", which="minor", linestyle=":") # axis="both",color="b", linestyle="--", linewidth=1
savefig("penalties.png") 

SNR = 20
img_nl = fakecells_dic["img_nl"]
signalpwr = sum(img_nl.^2)/length(img_nl)
bg = whitenoise.*sqrt(signalpwr/10^(SNR/10)) .+ 0.1
img = img_nl + bg# add noise and bg
imga = AxisArray(img, :x, :y, :time)
X = Matrix(reshape(imga, prod(imgsz), nimages(imga)))
W,H = initWH(X,ncells)
M = W\Wideal; dM = M-I

pens = []
for w in wrng
    wM = I+w*dM
    pen = sca2(W*wM)
    push!(pens,pen)
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(wrng, pens)
ax1.set_yscale(:log) # :linear
ax1.legend(SNRs,fontsize = 12,loc=3)
xlabel("α",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("penalty20dB.png") 


for w in [0.28, 0.29, 0.3, 0.35, 0.43, 0.45, 0.5, 0.75, 0.84, 0.86, 0.95, 1.0, 1.03, 1.5, 10]
    wM = I+w*dM
    pen = scapair(W*wM, wM\H)
    imsaveW("WM_w$(w)_pen$(pen).png", W*wM, imgsz)
end

#============== SCA penalty change from initial W to ideal W (real dataset) =============#
roi = (71:170, 31:130)
dataname = "neurofinder.02.00.cut100250_sqrt"
fullorgimg = load(dataname*".tif")
orgimg = fullorgimg[roi...,:]
imgsz = size(orgimg)[1:end-1]; lengthT = size(orgimg)[end]; ncells = 60
T=Float32
X = Matrix{T}(reshape(orgimg, prod(imgsz), lengthT))

# initila W
W,H = initWH(X,ncells)

# CD : ideal W
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.1 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)

# M = W_init\W_ideal
M = W\Wcd; dM = M-I

wrng = -1:0.001:2; penWs=[]; penHs=[]; pens=[]
for w in wrng
    wM = I+w*dM
    penW = sca2(W*wM)
    penH = sca2(wM\H)
    pen = penW*penH
    push!(penWs,penW)
    push!(penHs,penH)
    push!(pens,pen)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(wrng, penHs)
ax1.set_yscale(:log) # :linear
xlabel("α",fontsize = 12)
ylabel("penalty",fontsize = 12)
ax1.minorticks_on() # minortics_off()
# grid(true, which="both") # grid("on")
grid("on", which="major", linestyle="-") # axis="both",color="b", linestyle="-", linewidth=1
grid("on", which="minor", linestyle=":") # axis="both",color="b", linestyle="--", linewidth=1
savefig("penalties.png") 

Wcdn = copy(Wcd);
Hcdn = copy(Hcd);
normalizeWH!(Wcdn, Hcdn);
imsaveW("Wcd.png",Wcdn[:, 21:30],imgsz,borderwidth=1);

Wm = copy(W*(I+0.55*dM));
Hm = copy((I+0.55*dM)\H);
normalizeWH!(Wm, Hm);
imsaveW("Wm_w0.55.png",Wm[:, 21:30],imgsz,borderwidth=1);

Wm = copy(W*(I+dM));
Hm = copy((I+dM)\H);
normalizeWH!(Wm, Hm);
imsaveW("Wm_w1.0.png",Wm[:, 21:30],imgsz,borderwidth=1);

tol = 1e-7
cparams = Params(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e-5,
            maxiter=50, store_trace=false, show_trace=true)
W1, H1, objvals, trs = semiscasolve!(W, H; usecg=true, useprecond=true, penalty = :orthogonality,
    order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);

normalizeWH!(W1, H1);
imsaveW("cg.png",W1[:, 21:30],imgsz);

####
M = log(W\Wcd)

penWHs=[]; penWs=[]; penHs=[]
for w in wrng
    wM = exp(w*M); invwM = exp(-w*M)
    penW = SCA.sca2(real.(W*wM))
    penH = SCA.sca2(real.(invwM*H))
    penWH = penW*penH
    push!(penWHs,penWH)
    push!(penWs,penW)
    push!(penHs,penH)
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(wrng, penHs)
ax1.set_yscale(:log) # :linear
xlabel("α",fontsize = 12)
ylabel("penalty",fontsize = 12)
ax1.minorticks_on() # minortics_off()
# grid(true, which="both") # grid("on")
grid("on", which="major", linestyle="-") # axis="both",color="b", linestyle="-", linewidth=1
grid("on", which="minor", linestyle=":") # axis="both",color="b", linestyle="--", linewidth=1
savefig("penalties.png") 

#=================================== Real dataset =======================================#
dataname = "neurofinder.02.00.cut100250_sqrt"
orgimg = load(dataname*".tif")
ImageView.imshow(orgimg)
fullimgsz = size(orgimg); imgsz = fullimgsz[1:end-1]; lengthT=fullimgsz[end]
gt_ncells = 95; ncells = 100
T=Float32
X = Matrix{T}(reshape(orgimg, prod(imgsz), lengthT))

#== small size ==#
roi = (71:170, 31:130)
dataname = "neurofinder.02.00.cut100250_sqrt"
fullorgimg = load(dataname*".tif")
orgimg = fullorgimg[roi...,:]
imgsz = size(orgimg)[1:end-1]; lengthT = size(orgimg)[end]; ncells = 30
T=Float32
X = Matrix{T}(reshape(orgimg, prod(imgsz), lengthT))

# SCA
rt1 = @elapsed W,H = initWH(X,ncells)
stparams = StepParams(γ=0., β=50., order=0, usecg=true, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penalty=:symmetric_orthogonality)
lsparams = LineSearchParams(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
    maxiter=100, store_trace=true, show_trace=true)
rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
imshowW(W1,imgsz); @show sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); W1[W1.<0].= 0
ncols = 5
for i = 1:ncells÷ncols
    imsaveW("W_mi$(maxiter)_$(i)_rt2$(rt2).png",W1[:,((i-1)*ncols+1):(i*ncols)],imgsz,draw_borderwidth=1)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(1:length(objvals),objvals)
ax1.set_yscale(:log) # :linear
xlabel("iterations",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("true_dataset.png") 

# CD
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.1 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
normalizeWH!(Wcd,Hcd)
ncols = 5
for i = 1:ncells÷ncols
    imsaveW("Wcd_a0.1_$i.png",Wcd[:,((i-1)*ncols+1):(i*ncols)],imgsz,draw_borderwidth=1)
end

# SCA two steps : Symmetric + penW
ncells = 30
W,H = initWH(X,ncells)

maxiter = 9
stparams = StepParams(γ=0., β=30., order=0, usecg=true, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penalty=:symmetric_orthogonality)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-7, f_reltol=1e-7, f_inctol=1e-4,
    maxiter=maxiter, store_trace=true, show_trace=true)
@time W1, H1, objvals1, trs1 = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2, H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); imshowW(W2[:,21:30],imgsz,draw_borderwidth=1)
plot(1:length(objvals1), objvals1); yscale("log")

# - E(W) + linesearch
stparams = StepParams(γ=0., β=50., order=0, usecg=true, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penalty=:penW_orthogonality)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-5, f_reltol=1e-5, f_inctol=1e-2,
    maxiter=100, store_trace=true, show_trace=true)
@time W3, H3, objvals3, trs3 = semiscasolve!(W1, H1; stparams=stparams, lsparams=lsparams, cparams=cparams);
H3 = W3\X; W4, H4 = copy(W3), copy(H3)
normalizeWH!(W4,H4); imshowW(W4[:,21:30],imgsz,draw_borderwidth=1)
plot(maxiter+1:length(objvals3)+maxiter, objvals3)

ncols = 5
for i = 1:ncells÷ncols
    imsaveW("W_hybrid_$(i).png",W4[:,((i-1)*ncols+1):(i*ncols)],imgsz,draw_borderwidth=1)
end

# - E(W) + linesearch
stparams = StepParams(γ=0., β=50., order=0, usecg=true, useprecond=true, skew=false, fixposdef=false,
                allcompW=false, allcompH=false, penalty=:penW_orthogonality)
lsparams = LineSearchParams(method=:full, c=0.5, α0=1.0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-5, f_reltol=1e-5, f_inctol=1e-2,
    maxiter=100, store_trace=true, show_trace=true)
@time W3, H3, objvals3, trs3 = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
H3 = W3\X; W4, H4 = copy(W3), copy(H3)
normalizeWH!(W4,H4); imshowW(W4[:,21:30],imgsz,draw_borderwidth=1)
plot(maxiter+1:length(objvals3)+maxiter, objvals3)

#============== Low SNR test  =============#
SNR=-15 # SNR=10(noisey), SNR=40(less noisey)
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld"; ncells=ncells, lengthT=lengthT, SNR=SNR)
lsparams = LineSearchParams(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-5, f_reltol=1e-5, f_inctol=1e-4, maxiter=100, store_trace=true, show_trace=false)
for beta in [10, 100,500,1000]
    @show beta
    stparams = StepParams(γ=0., β=beta, order=0, usecg=true, useprecond=true, skew=false, fixposdef=false,
                    allcompW=false, allcompH=false, penalty=:symmetric_orthogonality)
    rt = @elapsed W1, H1, objvals, trs = semiscasolve!(W, H; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W1[W1.<0].=0
    imsaveW("W1_b$(beta)_rt$(rt).png",W1,imgsz)
end

# CD
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
α = 0.0 # best α = 0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
imshowW(Wcd,imgsz)
imsaveW("Wcd_a$(α).png",Wcd,imgsz)

# check eigen values
evs = []; SNRs = [-15, -10, 0, 20, 40]
for SNR in SNRs
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld"; ncells=ncells, lengthT=lengthT, SNR=SNR)
    F = svd(X)
    push!(evs, F.S)
end

componentrng=1:50
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(componentrng, hcat(map(v->v[1:100]/v[1], evs)...)[componentrng,:])
ax1.set_xticks(Int.(round.(collect(LinRange(first(componentrng), last(componentrng), (last(componentrng)-first(componentrng)+1)÷1)))))
ax1.legend(SNRs,fontsize = 12,loc=1, title="SNR(dB)")
xlabel("components",fontsize = 12)
ylabel("Eigen Values",fontsize = 12)
savefig("noiselevel_vs_eigenvalues.png")

fovsz = (40,20); lengthT0=100; SNR=20
ev0s = []; ev1s = []; SNR0s=[]; factors = 1:5:40
for factor in factors
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    ncells, X, gtW, gtH, gtWimgc, gtbg, SNR0 = gaussian2D_noiselevel(5, imgsz, lengthT, 10; fovsz=fovsz, bias=0.1, noiselevel = 0.1, orthogonal=false, overlaplevel=1) 
    push!(SNR0s,SNR0)
    F = svd(X)
    push!(ev0s, F.S)
    X, W, H, imgsz0, ncells0, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_lengthT$(lengthT)_SNR$(SNR).jld"; fovsz=imgsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz, SNR=SNR, save=false)
    F = svd(X)
    push!(ev1s, F.S)
end

components = 1:12
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(components, hcat(map(v->v[1:100]/v[1], ev0s)...)[components,:])
ax1.set_xticks(collect(StepRange(components[1], components[end]÷10, components[end])))
ax1.legend(factors,fontsize = 12,loc=1, title="factor")
xlabel("components",fontsize = 12)
ylabel("Eigen Values",fontsize = 12)
savefig("factor_vs_eigenvalues0.png")

components = 1:12
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(components, hcat(map(v->v[1:100], ev0s)...)[components,8])
ax1.set_xticks(collect(StepRange(components[1], components[end]÷10, components[end])))
ax1.legend(["36"],fontsize = 12,loc=1, title="factor")
xlabel("components",fontsize = 12)
ylabel("Eigen Values",fontsize = 12)
savefig("factor_vs_eigenvalues0_36.png")

components = 1:100
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(components, hcat(map(v->v[1:100]/v[1], ev1s)...)[components,:])
ax1.set_xticks(collect(StepRange(components[1], components[end]÷10, components[end])))
ax1.legend(factors,fontsize = 12,loc=1, title="factor")
xlabel("components",fontsize = 12)
ylabel("Eigen Values",fontsize = 12)
savefig("factor_vs_eigenvalues1.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factors,SNR0s)
xlabel("factor",fontsize = 12)
ylabel("SNR",fontsize = 12)
savefig("factor_vs_SNR.png")

#============= plot figure ========================#
function plotfig(Wlast::Matrix{T}, Hlast::Matrix{T}, OT, dM, αs, f_x0, cbx, βi, γ, penalty, skew, fn=nothing;
        αrng = -0.5:0.01:1.5, showallα=false, titlestr="", legendstrs=[], xlblstr = "α", ylblstr = "penalty") where T
    if penalty==:orthogonality
        penfunc = SCA.scapair_orthogonality
    elseif penalty==:sparsity_product
        penfunc = SCA.scapair_WHsparsity_product
    elseif penalty==:sparsity_sum
        penfunc = SCA.scapair_WHsparsity_sum
    elseif penalty==:sparsity_M
        penfunc = SCA.scapair_Msparsity
    else
        error("$(penalty) is unknown penalty")
    end
    p = size(Wlast,2)
    A, b, _ = SCA.buildproblem(Wlast, Hlast, OT; order=order, βi=βi, γ=γ, skew=skew)
    x = -A\b

    approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x
    truefn(α, Wlast, Hlast) = penfunc(I + α*dM, Wlast, Hlast, βi=βi; γ=γ)
    approxfn(α, Wlast, Hlast) = approxobj(α*vec(dM), scapair(Wlast,Hlast; γ=γ), A, b)
    linefn(α) = f_x0 + α*cbx

    fig, ax = plt.subplots(1,1, figsize=(5,4))
    println("calculating obj...")
    objs = [truefn(α, Wlast, Hlast) for α in αrng]
    println("calculating objsapprox...")
    objsapprox = [approxfn(α, Wlast, Hlast) for α in αrng]
    if length(legendstrs) > 2
        println("calculating objsline...")
        objsline = [linefn(α) for α in αrng]
    end
    length(legendstrs) > 2 ? ax.plot(αrng, [objs objsapprox objsline]) : ax.plot(αrng, [objs objsapprox])
    maxy = max(maximum(objs),maximum(objsapprox)); miny = min(minimum(objs),minimum(objsapprox));
    if showallα
    else
        ax.plot([αs[end],αs[end]],[miny, maxy],color="red")
        push!(legendstrs, "αs[end]=$(αs[end])")
    end
    ax.legend(legendstrs,fontsize = 12) #,loc=2
    ax.set_title(titlestr)
    xlabel(xlblsrt,fontsize = 12)
    ylabel(ylblstr, fontsize = 12)
    fn != nothing && savefig(fn)
end

inititer=12; enditer = 16
flipsign=false; usecg=true; useprecond=true; penalty=:orthogonality; order=0; fixposdef=false; skew=false;
β=50; γ=0.; c=0.5; α0=1.0; ρ=0.5;
fnprefix = "alpha_vs_penalties"

linesearch = LineSearch(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100, store_trace=true)
cparams = Params(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5,
                    maxiter=inititer, store_trace=true, show_trace=true)

W1 = copy(W); H1 = copy(H); T=eltype(W1); p = size(W1,2); power2 = (norm(W1)*norm(H1))^2
if inititer > 0
    W2, H2, objvals, trs = semiscasolve!(W1, H1; usecg=usecg, useprecond=useprecond, penalty=penalty,
            order=order, fixposdef=fixposdef, skew=skew, β=β, γ=γ, linesearch=linesearch, params=cparams)
else
    objvals = []
end
β=β/p;
OT = β != 0 ? (SCA.add_orthogonality(T, p)) : zeros(p^2,p^2)
W3, H3, objvals3 = copy(W2), copy(H2), copy(objvals)
for outer_iter in inititer+1:enditer#inititer+3
    # x, b, _ = pocastep(W1, H1; β=β, useprecond=useprecond, linesearch_method=linesearch_method);
    # #x, b = inverse(W1, H1; β=β);
    # αs, dM, cbx, perform_ls, iter_ls, f_x0, ls, wols = linesearch(W1,H1,x,b; linesearch_method=linesearch_method,
    #                                                 α0=α0, ρ=ρ, c=c, β=β, params=Params());

    dM2, dM, b, βi, inner_f_x0, inner_f_x, inner_iter, f_calls, g_calls, g_converged = SCA.scastep(W3, H3, OT;
        usecg=usecg, useprecond=useprecond, order=order, fixposdef=fixposdef, skew=skew, β=β, γ=γ,
        penalty=penalty, params=cparams)
    # line search
    if (linesearch.method == :full) || (linesearch.method == :conditional && inner_f_x > inner_f_x0)
        α, inner_f_x, iter_ls, αs = SCA.scalinesearch(W3, H3, dM, b, inner_f_x0; βi=βi, γ=γ, skew=skew,
            penalty=penalty, linesearch=linesearch)
    else
        if outer_iter == 16
            alpha = 0.05
            α, iter_ls, αs = alpha, 0, [alpha]
        else
            α, iter_ls, αs = 1.0, 0, [1.0]
        end
    end
    cbx = linesearch.c*b'*vec(dM)

    if linesearch.method != :none
        legendstrs = ["true penalty", "approximation", "linesearch line"]
    else
        legendstrs = ["true penalty", "approximation"]
    end
    fn = fnprefix*"_$(linesearch.method)_$(outer_iter).png"
    plotfig(W3, H3, OT, dM, αs, inner_f_x0, cbx, βi, γ, penalty, skew, fn;
        αrng = -0.5:0.01:1.5, showallα=false, titlestr="$outer_iter", legendstrs=legendstrs)

    f_increment = !cparams.allow_f_increases ? (inner_f_x - inner_f_x0)/inner_f_x0/power2 : zero(inner_f_x)
    M = f_increment > cparams.f_inctol ? Matrix{T}(1.0I,p,p) : T.(I + α*dM)
    W3=W3*M; H3=M\H3
    f_x = scapair(W3,H3); push!(objvals3, f_x)
    cparams.show_trace && @show outer_iter, α, dM2, objvals3[end]
end
normalizeWH!(W3,H3)
imshowW(W3,imgsz)

Wcdrs = load("neuro_cut_cd.png"); ImageView.imshow(Wcdrs)

#==== Noise estimation =============#
using DSP

function estimate_noise(X, qpos)
    ssignals=[]; snoises = []
    for t in 1:size(X,2)
        autocon = conv(X[:,t], X[:,t])
        middle = length(autocon)÷2; quarter = middle-length(autocon)÷qpos
        push!(snoises, mean(autocon[1:quarter]))
        push!(ssignals, mean(autocon[quarter:middle]))
    end
    tsignals=[]; tnoises = []
    for s in 1:size(X,1)
        autocon = conv(X[s,:], X[s,:])
        middle = length(autocon)÷2; quarter = middle-length(autocon)÷qpos
        push!(tnoises, mean(autocon[1:quarter]))
        push!(tsignals, mean(autocon[quarter:middle]))
    end
    ssignals, snoises, tsignals, tnoises
end

nlrng = 0:0.05:0.8
sSNRs = []; tSNRs = []
for nl in nlrng
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("notexisting.jld"; noiselevel=nl, save=false)
    ssignals, snoises, tsignals, tnoises = estimate_noise(X, 4)
    push!(sSNRs,norm(ssignals)/norm(snoises))
    push!(tSNRs,norm(tsignals)/norm(tnoises))
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(nlrng, [sSNRs tSNRs])
ax1.legend(["Spatial SNR", "Temporal SNR"],fontsize = 12,loc=2)
xlabel("noise level",fontsize = 12)
ylabel("sSNRs",fontsize = 12)
savefig("noiselevel_vs_SNR.png")

#==== PSNR(Peak signal-to-noise ratio), SSIM(Structural similarity) =============#
# we can use PSNR and SSIM instead of MSSD
using ImageQualityIndexes, TestImages

img = testimage("lena_gray_256") .|> float64
noisy_img = img .+ 0.1 .* randn(size(img))
ssim(noisy_img, img) # 0.3577
psnr(noisy_img, img) # 19.9941

function awgn(X,SNR) # SNR (dB) = 20 log10(S/N). Doubling S/N corresponds to increasing SNR (dB) by 6.02 dB
    #Assumes X to be a matrix and SNR a signal-to-noise ratio specified in decibel (dB)
    #Implented by author, inspired by https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
    N=length(X) #Number of elements in X
    signalPower = sum(X[:].^2)/N
    linearSNR = 10^(SNR/10)
    a,b=size(X)
    noiseMat = randn((a,b)).*√(signalPower/linearSNR) #Random gaussian noise, scaled according to the signal power of the entire matrix (!) and specified SNR

    return solution = X + noiseMat
end

# PSNR or SSIM
function matchedimg(W, matchlist)
    Wmimg = zeros(size(W,1),length(matchlist))
    for mp in matchlist
        Wmimg[:,mp[1]] = W[:,mp[2]]
    end
    Wmimg
end

ImageQualityIndexes.ssim(W, gtW, ml) = ssim(matchedimg(W, ml), gtW)
ImageQualityIndexes.psnr(W, gtW, ml) = psnr(matchedimg(W, ml), gtW)

#==== compare constraints =============#
imgsz = (40,20)
orthog = false; lengthT=100 ; gt_ncells=7; factor=1
orthogstr = orthog ? "orthog" : "nonorthog"
fname = "fakecells__$(orthogstr)_sfactor$(factor)_T$(lengthT)_nl$(noiselevel).jld"
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(fname;
        imgsz=imgsz, ncells=14, gt_ncells=gt_ncells, lengthT=lengthT, noiselevel=noiselevel)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

cparams = Params(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5, maxiter=100, show_trace=false)

βrng = 5:5:1000; xlabelstr = "β"
rtsums = []; rtprods = []; rtorthogs = [];
mssdsums = []; mssdprods = []; mssdorthogs = []
for β in βrng
    @show β
    rtorthog = @elapsed W1, H1, objval, trs = semiscasolve!(W, H; usecg=true, useprecond=true,
        order=0, fixposdef=false, skew=false, β=β, penalty=:orthogonality, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssdorthog, mlorthog, ssdsorthog = matchedfiterr(gtW,W1)
    push!(rtorthogs, rtorthog); push!(mssdorthogs, mssdorthog)
    imsaveW("const_orthog_beta$(β)_mssd$(mssdorthog).png",W1,imgsz)
end
for β in βrng
    @show β
    rtsum = @elapsed W1, H1, objval, trs = semiscasolve!(W, H; usecg=true, useprecond=true,
        order=0, fixposdef=false, skew=false, β=β, penalty=:sparsity_sum, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssdsum, mlsum, ssdssum = matchedfiterr(gtW,W1)
    push!(rtsums, rtsum); push!(mssdsums, mssdsum)
    imsaveW("const_sum_beta$(β)_mssd$(mssdsum).png",W1,imgsz)
end
for β in βrng
    @show β
    rtprod = @elapsed W1, H1, objval, trs = semiscasolve!(W, H; usecg=true, useprecond=true,
        order=0, fixposdef=false, skew=false, β=β, penalty=:sparsity_product, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssdprod, mlprod, ssdsprod = matchedfiterr(gtW,W1)
    push!(rtprods, rtprod); push!(mssdprods, mssdprod)
    imsaveW("const_prod_beta$(β)_mssd$(mssdprod).png",W1,imgsz)
end

fname = "const_vs_beta_$(today())_$(hour(now())).jld"
save(fname, "betarng", βrng, "imgsz", imgsz, "lengthT", lengthT,
        "rtorthogs", rtorthogs, "mssdsorthogs", mssdsorthogs, # :orthogonality
        "rtsums",rtsums,"mssdsums", mssdsums, # :sparsity_sum
        "rtprods",rtprods,"mssdprods", mssdprods) # :sparsity_prod
dd = load("isvd_vs_svd_b50_2021-10-21_9.jld")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(βrng, [rtorthogs rtsums rtprods])
ax1.legend(["Orthogonality", "Sparsity sum", "Sparsity prod"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("const_vs_beta.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(βrng, [mssdorthogs mssdsums mssdprods])
ax1.legend(["Orthogonality", "Sparsity sum", "Sparsity prod"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("MSSD",fontsize = 12)
savefig("const_vs_mssd.png")

#==== rsvd =============#
imgsz = (40,20)
orthog = false; lengthT=2000 ; gt_ncells=7; factor=1; ncells=14
orthogstr = orthog ? "orthog" : "nonorthog"

for noiselevel in [0, 1e-13, 0.1, 0.2, 0.4, 0.6]
    @show noiselevel
    fname = "fc_$(orthogstr)_sfactor$(factor)_T$(lengthT)_nl$(noiselevel).jld"
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(fname; svd_method = :svd,
            imgsz=imgsz, ncells=ncells, gt_ncells=gt_ncells, lengthT=lengthT, noiselevel=noiselevel )
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    imsaveW("Wsvd_nl$(noiselevel).png",W,imgsz)
    W1, H1, objval, trs = semiscasolve!(W, H; usecg=true, useprecond=true,
        order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
    @show "svd", mssd1
    imsaveW("Wsvd_nl$(noiselevel)_mssd$(mssd1).png",W1,imgsz)
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(fname; svd_method = :isvd,
            imgsz=imgsz, ncells=ncells, gt_ncells=gt_ncells, lengthT=lengthT, noiselevel=noiselevel )
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    imsaveW("Wisvd_nl$(noiselevel).png",W,imgsz)
    W1, H1, objval, trs = semiscasolve!(W, H; usecg=true, useprecond=true,
        order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
    @show "isvd", mssd1
    imsaveW("Wisvd_nl$(noiselevel)_mssd$(mssd1).png",W1,imgsz)
    for extra_n in [5, 10, 30, 50]
        @show extra_n
        F = rsvd(X,ncells,extra_n);
        Wr = F.U; Hr = Wr\X
        imsaveW("Wrsvd_nl$(noiselevel)_p$(extra_n).png",Wr,imgsz)
        W1, H1, objval, trs = semiscasolve!(Wr, Hr; usecg=true, useprecond=true,
            order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
        @show extra_n, mssd1
        imsaveW("Wrsvd_nl$(noiselevel)_p$(extra_n)_mssd$(mssd1).png",W1,imgsz)
    end
end

rts=[]; rng=5:1:93
for extra_n in rng
    @show extra_n
    rt = @elapsed F = rsvd(X,ncells,extra_n);
    push!(rts,rt)
end
rtsvd = @elapsed F = svd(X)
rtisvd = @elapsed SCA.isvd(X,14)

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(rng, rts)
ax1.plot(rng, rtisvd*ones(length(rts)))
ax1.plot(rng, rtsvd*ones(length(rts)))
ax1.legend(["RSVD", "ISVD", "SVD"],fontsize = 12,loc=2)
#ax1.legend(["CG S-POCA (ISVD)", "CG S-POCA (SVD)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel("extra n",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("svdruntimes.png")

#==== scalability for data size =============#
factorrng = 1:40; β = 50
ncells = 14; 
extra_n = 10; # number of extra vector
fovsz=(40,20); lengthT0 = 100
C = 1:1:80; plotxrng = factorrng; plotxstr = "factor"; xlabelstr="factor" # imgszyrng = 30:10:100, 40

datart1sca0=[]; datart2sca0=[]; datart1sca1=[]; datart2sca1=[]; datart1sca2=[]; datart2sca2=[];
mssd0sca=[]; mssd1sca=[]; mssd2sca=[];
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    linesearch = LineSearch(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100)
    cparams = Params(allow_f_increases=true, f_abstol=1e-8, f_reltol = 1e-8, maxiter=100, successive_f_converge=3)
    # isvd
    runtime1 = @elapsed U,s = SCA.isvd(X,ncells);
    Wi = U; Hi = Wi\X
    runtime2 = @elapsed W1, H1, objval, trs = semiscasolve!(Wi, Hi; usecg=true, useprecond=true,
                        order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
    push!(datart1sca0, runtime1)
    push!(datart2sca0, runtime2)
    push!(mssd0sca, mssd1)
    imsaveW("isvdfc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG__b$(β).png",W1,imgsz)
    # svd
    runtime1 = @elapsed F = svd(X);
    W = F.U[:,1:ncells]; H = W\X;
    runtime2 = @elapsed W1, H1, objval, trs = semiscasolve!(W, H; usecg=true, useprecond=true,
                        order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
    push!(datart1sca1, runtime1)
    push!(datart2sca1, runtime2)
    push!(mssd1sca, mssd1)
    imsaveW("svdfc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG__b$(β).png",W1,imgsz)
    # rsvd
    runtime1 = @elapsed F = rsvd(X,ncells,extra_n);
    Wr = F.U; Hr = Wr\X
    runtime2 = @elapsed W1, H1, objval, trs = semiscasolve!(Wr, Hr; usecg=true, useprecond=true,
                        order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
    push!(datart1sca2, runtime1)
    push!(datart2sca2, runtime2)
    push!(mssd2sca, mssd1)
    imsaveW("rsvdfc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG__b$(β).png",W1,imgsz)
end
datartsca0 = datart1sca0 + datart2sca0
datartsca1 = datart1sca1 + datart2sca1
datartsca2 = datart1sca2 + datart2sca2

for extra_n in [100]
    @show extra_n
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    linesearch = LineSearch(method=:none, c=0.5, α0=1.0, ρ=0.5, maxiter=100)
    cparams = Params(allow_f_increases=true, f_abstol=1e-8, f_reltol = 1e-8, maxiter=100, successive_f_converge=3)
    # rsvd
    runtime1 = @elapsed F = rsvd(X,ncells,extra_n);
    Wr = F.U; Hr = Wr\X
    runtime2 = @elapsed W1, H1, objval, trs = semiscasolve!(Wr, Hr; usecg=true, useprecond=true,
                        order=0, fixposdef=false, skew=false, β=β, linesearch=linesearch, params=cparams);
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
    @show runtime1, runtime2, mssd1
    imsaveW("Wr_sz$(imgsz)_lT$(lengthT)_en$(extra_n).png",Wr,imgsz)
    imsaveW("rsvdfc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG_b$(β)_en$(extra_n).png",W1,imgsz)
end

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
datart1scd0 = []; datart2scd0 = []; mssdscd0 = []
for factor in factorrng
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells)
    datart1cd=[]; datart2cd=[]; mssdcd=[]
    for α in [0.0] # best α = 0.1
        println("α=$(α)")
        runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
        normalizeWH!(W88,H88)
        mssd88, ml88, ssds = matchedfiterr(gtW,W88)
        mssdH88 = ssdH(ml88,gtH,H88')
        push!(datart1cd, runtime1)
        push!(datart2cd, runtime2)
        push!(mssdcd, mssd88)
        #imshowW(W88,imgsz)
        imsaveW("fc_sz$(imgsz)_lT$(lengthT)_a$(α).png", W88, imgsz)
    end
    push!(datart1scd0, datart1cd)
    push!(datart2scd0, datart2cd)
    push!(mssdscd0, mssdcd)
end
datart1cd0 = getindex.(datart1scd0,1)
datart2cd0 = getindex.(datart2scd0,1)
mssdcd0 = getindex.(mssdscd0,1)
datartcd0 = datart1cd0+datart2cd0

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datart1sca0 datart1sca1 datart1sca2 datart1cd0])
ax1.legend(["ISVD", "SVD", "RSVD", "Init(CD)"],fontsize = 12,loc=2)
#ax1.legend(["CG S-POCA (ISVD)", "CG S-POCA (SVD)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_vs_svdruntime1.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datart1sca0 datart2sca0 datartsca0])
ax1.legend(["ISVD", "S-POCA", "Total"],fontsize = 12,loc=2)
#ax1.legend(["CG S-POCA (ISVD)", "CG S-POCA (SVD)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_vs_runtime1.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datart1sca1 datart2sca1 datartsca1])
ax1.legend(["SVD", "S-POCA", "Total"],fontsize = 12,loc=2)
#ax1.legend(["CG S-POCA (ISVD)", "CG S-POCA (SVD)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_vs_runtime2.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datart1sca2 datart2sca2 datartsca2])
ax1.legend(["RSVD", "S-POCA", "Total"],fontsize = 12,loc=2)
#ax1.legend(["CG S-POCA (ISVD)", "CG S-POCA (SVD)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_vs_runtime3.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datartsca0 datartsca1 datartsca2 datartcd0])
ax1.legend(["CG S-POCA (ISVD)", "CG S-POCA (SVD)", "CG S-POCA (RSVD)", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_vs_totalruntime.png")

fname = "isvd_vs_svd_vs_rsvd_b$(β)_$(today())_$(hour(now())).jld"
save(fname, "beta", β, "factorrng", factorrng,
        "datart1cd0", datart1cd0, "datart2cd0", datart2cd0, "mssdcd0", mssdcd0,
        "datart1sca0",datart1sca0,"datart2sca0",datart2sca0, "mssd0sca", mssd0sca, # isvd
        "datart1sca1",datart1sca1,"datart2sca1",datart2sca1, "mssd1sca", mssd1sca, # svd
        "datart1sca2",datart1sca2,"datart2sca2",datart2sca2, "mssd2sca", mssd2sca) # rsvd

dd = load("isvd_vs_svd_b50_2021-10-21_9.jld")
datart1cd0 = dd["datart1cd0"]
datart2cd0 = dd["datart2cd0"] 
mssdcd0 = dd["mssdcd0"]
datartcd0 = datart1cd0 + datart2cd0


#============= inferrence error ========================#
function ttt(W::Matrix{T},H::Matrix{T}; useprecond=true, βi=10, allcompW=false, allcompH=false) where T
    initial_x = zeros(T,p^2)
    fg!, P = SCA.prepare_fg_orthogonality(W, H; useprecond=useprecond, βi=βi, allcompW=allcompW, allcompH=allcompH)
    opt = Optim.Options(f_tol=1e-5, show_trace=false, show_every=10, allow_f_increases=false,
        store_trace=true, allow_outer_f_increases=false)
    rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(P=P), opt)
    return rst
end
@inferred(SCA.prepare_fg(W, H; useprecond=true, β=10, allcompW=false, allcompH=false))
@inferred(ttt(W,H,useprecond=false))
@code_warntype(ttt(W,H,useprecond=false))

initial_x = zeros(T,p^2)
fg!, P = SCA.prepare_fg(W, H; useprecond=true, β=10, allcompW=false, allcompH=false)
opt = Optim.Options(f_tol=1e-5, show_trace=false, show_every=10, allow_f_increases=false,
                    allow_outer_f_increases=false)
@inferred(optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(P=P), opt))


function ttt1(W::Matrix{T},H::Matrix{T}; useprecond=true, βi=10, allcompW=false, allcompH=false) where T
    initial_x = zeros(T,p^2)
    fg!, P = SCA.prepare_fg(W, H; useprecond=useprecond, β=βi, allcompW=allcompW, allcompH=allcompH)
    return fg!, P
end
@inferred(ttt1(W,H,useprecond=false))
@inferred(ttt1(W,H,useprecond=false, βi=10, allcompW=false, allcompH=false))
@inferred(SCA.prepare_fg(W, H; useprecond=true, β=10, allcompW=false, allcompH=false))


#============= noiseless dataset ========================#
imgsz = (40,20)
orthog = false; lengthT=100 ; gt_ncells=7; noiselevel=0.1
orthogstr = orthog ? "orthog" : "nonorthog"
fname = "fakecells__$(orthogstr)_sfactor$(factor)_T$(lengthT)_nl$(noiselevel).jld"
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(fname;
        imgsz=imgsz, ncells=14, gt_ncells=gt_ncells, lengthT=lengthT, noiselevel=noiselevel)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
imsaveW("W0_nl$(noiselevel).png",W,imgsz)

ncells = 14
W, s = POCA.isvd(X,ncells); H = W\X
Wn, Hn = balanced_WH(W,X)

β=50; 
for β in 1:50
    @show β
    params = Params(allow_f_increases = true, f_abstol = 1e-12, f_reltol=1e-12, f_inctol=1e-5, maxiter=1000, store_trace=true, show_trace=true)
    @time W1, H1, objval, iters, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params);
    imshowW(W1,imgsz); @show iters[1], sca2(W1), sca2(H1), scapair(W1,H1)
    W2 = copy(W1); Wn, Hn = balanced_WH(W2,X)
    imsaveW("Wn_nl$(noiselevel)_beta$(β).png", Wn,imgsz)
    normalizeWH!(W1,H1); imsaveW("W1_nl$(noiselevel)_beta$(β).png", W1,imgsz)
end

runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells, variant=:ar)
α =0.0;
runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
normalizeWH!(W88,H88)
imshowW(W88, imgsz)
imsaveW("W88.png",W88,imgsz)
#============= ISVD ========================#
using ISVD

ncells = 14; r = ncells
@time begin
    U = Array{Float64}(undef, size(X,1), 0)
    s = Array{Float64}(undef, 0)
    for j = 1:r:size(X,2)-r+1
        U, s = ISVD.update_U_s!(U, s, X[:,j:j+r-1])
    end
end # 0.013067 seconds (454 allocations: 3.943 MiB)
@time svd(X); # 0.039051 seconds (11 allocations: 1.615 MiB)

ncells = 100; r = ncells
@time begin
    U = Array{Float64}(undef, size(X,1), 0)
    s = Array{Float64}(undef, 0)
    for j = 1:r:size(X,2)-r+1
        U, s = ISVD.update_U_s!(U, s, X[:,j:j+r-1])
    end
end # 0.038553 seconds (29 allocations: 2.240 MiB)
@time svd(X); # 0.040710 seconds (11 allocations: 1.615 MiB)

function isvd(X,r)
    U = Array{Float64}(undef, size(X,1), 0)
    s = Array{Float64}(undef, 0)
    for j = 1:r:size(X,2)-r+1
        U, s = update_U_s!(U, s, X[:,j:j+r-1])
    end
    U, s
end

rs = 1:100
isvdrts = []; svdrts = []
rt2 = @elapsed F2 = svd(X)
U2 = F2.U; (m,n) = size(U2)
for r in rs
    @show r
    rt1 = @elapsed U,s = isvd(X,r)
    push!(isvdrts,rt1)
    push!(svdrts,rt2)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(rs, [isvdrts svdrts])
ax1.legend(["ISVD", "SVD"], fontsize = 12, loc=4)
xlabel("number of components",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_svdruntime.png")

#======== noc =========#
fovsz = (40,20); factor=20
imgsz = (fovsz[1]*factor,fovsz[2])
lengthT = lengthT0
orthog = false; lengthT=1000 ; gt_ncells=7;
orthogstr = orthog ? "orthog" : "nonorthog"
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sfactor$(factor)_$(orthogstr)_T$(lengthT).jld"; imgsz=imgsz, ncells=14, gt_ncells=gt_ncells, lengthT=lengthT)
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
W0 = W1 = W2 = W3 = copy(W)

# CG S-POCA with preconditioning
ncellss = 6:2:100
datart1sca=[]; datart2sca=[]; mssdsca=[]; iters=[]; inner_iters=[];
for ncells in ncellss # svd
    println("ncells=$ncells")
    runtime1 = @elapsed F = POCA.svd(X);
    W = F.U[:,1:ncells]; H = W\X
    #Wn, Hn = balanced_WH(W,X);
    for β in [50.]
        println("beta=$β")
        params = Params(allow_f_increases=true, f_abstol=1e-8, f_reltol = 1e-8, maxiter=34, successive_f_converge=3, store_trace=true, show_trace=false)
        runtime2 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
            order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params)
        rt = runtime1+runtime2
        W2 = copy(W1)
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchednssd(gtW,W1)
        push!(datart1sca, runtime1)
        push!(datart2sca, runtime2)
        push!(mssdsca, mssd1)
        push!(iters,iter[1])
        inner_iter = [trs[1][d].inner_iters for d in 1:length(trs[1])]
        push!(inner_iters, mean(inner_iter))
        imsaveW("size_SPOCA_svd_nc$(ncells)_b$(β).png",W1,imgsz)
    end
    datartsca = datart1sca+datart2sca
    total_iters = iters.*inner_iters
end

datart1sca0=[]; datart2sca0=[]; mssdsca0=[]; iters0=[]; inner_iters0=[];
for ncells in ncellss # isvd
    println("ncells=$ncells")
    runtime1 = @elapsed W,s = POCA.isvd(X,ncells);
    H = W\X
    #Wn, Hn = balanced_WH(W,X);
    for β in [50.]
        println("beta=$β")
        params = Params(allow_f_increases=true, f_abstol=1e-8, f_reltol = 1e-8, maxiter=34, successive_f_converge=3, store_trace=true, show_trace=false)
        runtime2 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
            order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params)
        rt = runtime1+runtime2
        W2 = copy(W1)
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchednssd(gtW,W1)
        push!(datart1sca0, runtime1)
        push!(datart2sca0, runtime2)
        push!(mssdsca0, mssd1)
        push!(iters0,iter[1])
        inner_iter0 = [trs[1][d].inner_iters for d in 1:length(trs[1])]
        push!(inner_iters0, mean(inner_iter0))
        imsaveW("size_SPOCA_isvd_nc$(ncells)_b$(β).png",W1,imgsz)
    end
    datartsca0 = datart1sca0+datart2sca0
    total_iters0 = iters0.*inner_iters0
end

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
if true
    αs = 0:0.1:10
    datart1cd0=[]; datart2cd0=[]; mssdcd0=[]
    for ncells in ncellss    
        println("ncells=$ncells")
        runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells, variant=:ar)
        for α in [0.0] # best α = 0.1
            println("α=$(α)")
            runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
            normalizeWH!(W88,H88)
            mssd88, ml88, ssds = matchednssd(gtW,W88)
            mssdH88 = ssdH(ml88,gtH,H88')
            push!(datart1cd0, runtime1)
            push!(datart2cd0, runtime2)
            push!(mssdcd0, mssd88)
            #imshowW(W88,imgsz)
            imsaveW("size_CD_nosparsity_nc$(ncells)_a$(α).png", W88, imgsz)
        end
    end
    datartcd0 = datart1cd0+datart2cd0
end

fname = "noc_vs_isvd_b$(β)_$(today())_$(hour(now())).jld"
save(fname, "beta", β, "ncellss", ncellss,
        "datart1cd0", datart1cd0, "datart2cd0", datart2cd0, "mssdcd0", mssdcd0,
        "datart1sca0",datart1sca0,"datart2sca0",datart2sca0, "mssdsca0",
            mssdsca0, "iters0", iters0, "inner_iters0", inner_iters0, # isvd
        "datart1sca",datart1sca,"datart2sca",datart2sca, "mssd1sca",
            mssd1sca, "iters", iters, "inner_iters", inner_iters) # svd

dataname = "noc_vs_isvd_b50_2021-10-25_11.jld"
dd = load(dataname)
datart1cd0 = dd["datart1cd0"];
datart2cd0 = dd["datart2cd0"];
# datart1sca = dd["datart1sca2"];
# datart2sca = dd["datart2sca2"];
datartcd0 = datart1cd0+datart2cd0;
datartsca0 = datart1sca0+datart2sca0;
datartsca = datart1sca+datart2sca;
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca0 datartsca datartcd0])
ax1.legend(["S-POCA(ISVD)", "S-POCA(SVD)", "CD"], fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_scaruntime.png")

#============= Temporally longer dataset ========================#
orthog = false; lengthT=1000 ; gt_ncells=7;
orthogstr = orthog ? "orthog" : "nonorthog"
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr)_T$(lengthT).jld"; ncells=14, gt_ncells=gt_ncells, lengthT=lengthT)
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
W0 = W1 = W2 = W3 = copy(W)

β=50; 
params = Params(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5, maxiter=100, store_trace=true, show_trace=true)
@time W1, H1, objval, iters, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params);
imshowW(W1,imgsz); @show iters[1], sca2(W1), sca2(H1), scapair(W1,H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz)
imsaveW("W1.png",W1,imgsz)

#============= Line Search 2 ========================#
function cg(W, H;β=50, useprecond=true, allcompW=false, allcompH=false, linesearch_method=:full)
    @show β
    m, p = size(W)
    initial_x = zeros(p^2)
    fg!, P = POCA.prepare_fg(W, H; useprecond=useprecond, β=β, allcompW=allcompW, allcompH=allcompH)
    opt = Optim.Options(f_tol=1e-5, show_trace=false, show_every=10, allow_f_increases=false,
                        allow_outer_f_increases=false)
    if useprecond
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(P=P), opt)
    else
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt)
    end
    x = rst.minimizer
    g_converged = (rst.iterations==0 && rst.g_converged) # don't need to do outer_iteration more
    b = zeros(p^2)
    fg!(nothing, b, b)
    b ./= 2.
    x, b, rst.iterations, rst.f_calls, rst.g_calls, g_converged
end

function inverse(W::Matrix{T}, H::Matrix{T}; β=50) where T
    @show β
    p = size(W,2)
    OT = POCA.add_orthogonality(T,p)
    A, b, _ = POCA. buildproblem(W, H, OT; order=0, β=β)
    -A\b, b
end

function linesearch(W,H,x,b; linesearch_method=:full, α0=1, ρ=0.9, c=0.5, γ=0.0, β=50, allcompW=false, allcompH=false, params=Params())
    @show β
    α = 1.0; αs = [α0]
    dM = POCA.reshapex(x, (p, p))
    f_x0 = scapair(W, H; γ=γ, allcompW = allcompW, allcompH = allcompH) # when dM = zeros(p,p), we don't care β
    f_x = scapair(I + α*dM, W, H, β=β*f_x0, γ=γ, allcompW = allcompW, allcompH = allcompH)
    if b'*x > 0
        x *= -1
        dM *= -1
    end
    cbx = c*b'*x
    perform_ls = (linesearch_method == :full) || (linesearch_method == :conditional && f_x > f_x0)
    iter_ls = 0
    if perform_ls
        α = α0
        while (((f_x = scapair(I + α*dM, W, H, β=β*f_x0, γ=γ, allcompW = allcompW, allcompH = allcompH)) > f_x0 + α*cbx)
                && (iter_ls < params.linesearch_maxiter))
            @show f_x, f_x0 + α*cbx
            iter_ls += 1
            α = ρ*α
            push!(αs,α)
        end
    end
    ls = scapair(I+α*dM,W1,H1; β=β*f_x0); wols = scapair(I+dM,W1,H1; β=β*f_x0)
    αs, dM, cbx, perform_ls, iter_ls, f_x0, ls, wols
end

function plotfig(Wlast::Matrix{T}, Hlast::Matrix{T}, dM, αs, β, cbx, titlestr, legendstrs, xlblsrt, ylblstr,
        fn=nothing; αrng = -0.5:0.01:1.5, showallα=false) where T
    @show β
    p = size(Wlast,2)
    OT = POCA.add_orthogonality(T,p)
    f_x0 = scapair(Wlast, Hlast; γ=0, allcompW = false, allcompH = false)
    A, b, _ = POCA.buildproblem(Wlast, Hlast, OT; order=0, β=β)
    x, b = inverse(Wlast, Hlast; β=β);

    approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x
    truefn(α, Wlast, Hlast) = scapair(I + α*dM, Wlast, Hlast, β=β*f_x0; γ=γ)
    approxfn(α, Wlast, Hlast) = approxobj(α*vec(dM), scapair(Wlast,Hlast; γ=γ), A, b)
    linefn(α) = f_x0 + α*cbx

    fig, ax = plt.subplots(1,1, figsize=(5,4))
    println("calculating obj...")
    objs = [truefn(α, Wlast, Hlast) for α in αrng]
    println("calculating objsapprox...")
    objsapprox = [approxfn(α, Wlast, Hlast) for α in αrng]
    if length(legendstrs) > 2
        println("calculating objsline...")
        objsline = [linefn(α) for α in αrng]
    end
    length(legendstrs) > 2 ? ax.plot(αrng, [objs objsapprox objsline]) : ax.plot(αrng, [objs objsapprox])
    maxy = max(maximum(objs),maximum(objsapprox)); miny = min(minimum(objs),minimum(objsapprox));
    if showallα
    else
        ax.plot([αs[end],αs[end]],[miny, maxy],color="red")
        push!(legendstrs, "α[end]=$(α[end])")
    end
    ax.legend(legendstrs,fontsize = 12) #,loc=2
    ax.set_title(titlestr)
    xlabel(xlblsrt,fontsize = 12)
    ylabel(ylblstr, fontsize = 12)
    fn != nothing && savefig(fn)
end

inititer=15
flipsign=false; usecg=true; useprecond=true; linesearch_method=:conditional; order=0; fixposdef=false; skew=false;
β=50; c=0.5; α0=1.0; ρ=0.5;
xlblsrt = "α"; ylblstr = "penalty"; fnprefix = "alpha_vs_penalties"
W1 = copy(W); H1 = copy(H); p = size(W1,2);
if inititer > 0
    params = Params(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5, maxiter=inititer, store_trace=true, show_trace=true)
    W1, H1, objval, iters, trs = semiscasolve!(W1, H1; flipsign=flipsign, usecg=usecg, useprecond=useprecond, linesearch_method=linesearch_method, 
        order=0, fixposdef=fixposdef, skew=skew, β=β, c=c, α0=α0, ρ=ρ, params=params);
end
β=β/p;
for iterstep in inititer+1:20#inititer+3
    x, b, _ = cg(W1, H1; β=β, useprecond=useprecond, linesearch_method=linesearch_method);
    #x, b = inverse(W1, H1; β=β);
    αs, dM, cbx, perform_ls, iter_ls, f_x0, ls, wols = linesearch(W1,H1,x,b; linesearch_method=linesearch_method,
                                                    α0=α0, ρ=ρ, c=c, β=β, params=Params());
    if perform_ls
        legendstrs = ["true penalty", "approximation", "linesearch line"]
    else
        legendstrs = ["true penalty", "approximation"]
    end
    fn = fnprefix*"_$(linesearch_method)_$(iterstep).png"
    plotfig(W1, H1, dM, αs, β, cbx, iterstep, legendstrs, xlblsrt, ylblstr, fn; αrng = -0.5:0.01:1.5, showallα=false)
    M = I+αs[end]*dM; W1 = W1*M; H1 = M\H1
end

#============= Line Search 2 ========================#
function cg(W, H;β=50, useprecond=true, allcompW=false, allcompH=false, linesearch_method=:full)
    @show β
    m, p = size(W)
    initial_x = zeros(p^2)
    fg!, P = POCA.prepare_fg(W, H; useprecond=useprecond, β=β, allcompW=allcompW, allcompH=allcompH)
    opt = Optim.Options(f_tol=1e-5, show_trace=false, show_every=10, allow_f_increases=false,
                        allow_outer_f_increases=false)
    if useprecond
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(P=P), opt)
    else
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt)
    end
    x = rst.minimizer
    g_converged = (rst.iterations==0 && rst.g_converged) # don't need to do outer_iteration more
    b = zeros(p^2)
    fg!(nothing, b, b)
    b ./= 2.
    x, b, rst.iterations, rst.f_calls, rst.g_calls, g_converged
end

function inverse(W::Matrix{T}, H::Matrix{T}; β=50) where T
    @show β
    p = size(W,2)
    OT = POCA.add_orthogonality(T,p)
    A, b, _ = POCA. buildproblem(W, H, OT; order=0, β=β)
    -A\b, b
end

function linesearch(W,H,x,b; linesearch_method=:full, α0=1, ρ=0.9, c=0.5, γ=0.0, β=50, allcompW=false, allcompH=false, params=Params())
    @show β
    α = 1.0; αs = [α0]
    dM = POCA.reshapex(x, (p, p))
    f_x0 = scapair(W, H; γ=γ, allcompW = allcompW, allcompH = allcompH) # when dM = zeros(p,p), we don't care β
    f_x = scapair(I + α*dM, W, H, β=β*f_x0, γ=γ, allcompW = allcompW, allcompH = allcompH)
    if b'*x > 0
        x *= -1
        dM *= -1
    end
    cbx = c*b'*x
    perform_ls = (linesearch_method == :full) || (linesearch_method == :conditional && f_x > f_x0)
    iter_ls = 0
    if perform_ls
        α = α0
        while (((f_x = scapair(I + α*dM, W, H, β=β*f_x0, γ=γ, allcompW = allcompW, allcompH = allcompH)) > f_x0 + α*cbx)
                && (iter_ls < params.linesearch_maxiter))
            @show f_x, f_x0 + α*cbx
            iter_ls += 1
            α = ρ*α
            push!(αs,α)
        end
    end
    ls = scapair(I+α*dM,W1,H1; β=β*f_x0); wols = scapair(I+dM,W1,H1; β=β*f_x0)
    αs, dM, cbx, perform_ls, iter_ls, f_x0, ls, wols
end

function plotfig(Wlast::Matrix{T}, Hlast::Matrix{T}, dM, αs, β, cbx, titlestr, legendstrs, xlblsrt, ylblstr,
        fn=nothing; αrng = -0.5:0.01:1.5, showallα=false) where T
    @show β
    p = size(Wlast,2)
    OT = POCA.add_orthogonality(T,p)
    f_x0 = scapair(Wlast, Hlast; γ=0, allcompW = false, allcompH = false)
    A, b, _ = POCA.buildproblem(Wlast, Hlast, OT; order=0, β=β)
    x, b = inverse(Wlast, Hlast; β=β);

    approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x
    truefn(α, Wlast, Hlast) = scapair(I + α*dM, Wlast, Hlast, β=β*f_x0; γ=γ)
    approxfn(α, Wlast, Hlast) = approxobj(α*vec(dM), scapair(Wlast,Hlast; γ=γ), A, b)
    linefn(α) = f_x0 + α*cbx

    fig, ax = plt.subplots(1,1, figsize=(5,4))
    println("calculating obj...")
    objs = [truefn(α, Wlast, Hlast) for α in αrng]
    println("calculating objsapprox...")
    objsapprox = [approxfn(α, Wlast, Hlast) for α in αrng]
    if length(legendstrs) > 2
        println("calculating objsline...")
        objsline = [linefn(α) for α in αrng]
    end
    length(legendstrs) > 2 ? ax.plot(αrng, [objs objsapprox objsline]) : ax.plot(αrng, [objs objsapprox])
    maxy = max(maximum(objs),maximum(objsapprox)); miny = min(minimum(objs),minimum(objsapprox));
    if showallα
    else
        ax.plot([αs[end],αs[end]],[miny, maxy],color="red")
        push!(legendstrs, "α[end]=$(α[end])")
    end
    ax.legend(legendstrs,fontsize = 12) #,loc=2
    ax.set_title(titlestr)
    xlabel(xlblsrt,fontsize = 12)
    ylabel(ylblstr, fontsize = 12)
    fn != nothing && savefig(fn)
end

inititer=15
flipsign=false; usecg=true; useprecond=true; linesearch_method=:conditional; order=0; fixposdef=false; skew=false;
β=50; c=0.5; α0=1.0; ρ=0.5;
xlblsrt = "α"; ylblstr = "penalty"; fnprefix = "alpha_vs_penalties"
W1 = copy(W); H1 = copy(H); p = size(W1,2);
if inititer > 0
    params = Params(allow_f_increases = true, f_abstol = 1e-8, f_reltol=1e-8, f_inctol=1e-5, maxiter=inititer, store_trace=true, show_trace=true)
    W1, H1, objval, iters, trs = semiscasolve!(W1, H1; flipsign=flipsign, usecg=usecg, useprecond=useprecond, linesearch_method=linesearch_method, 
        order=0, fixposdef=fixposdef, skew=skew, β=β, c=c, α0=α0, ρ=ρ, params=params);
end
β=β/p;
for iterstep in inititer+1:20#inititer+3
    x, b, _ = cg(W1, H1; β=β, useprecond=useprecond, linesearch_method=linesearch_method);
    #x, b = inverse(W1, H1; β=β);
    αs, dM, cbx, perform_ls, iter_ls, f_x0, ls, wols = linesearch(W1,H1,x,b; linesearch_method=linesearch_method,
                                                    α0=α0, ρ=ρ, c=c, β=β, params=Params());
    if perform_ls
        legendstrs = ["true penalty", "approximation", "linesearch line"]
    else
        legendstrs = ["true penalty", "approximation"]
    end
    fn = fnprefix*"_$(linesearch_method)_$(iterstep).png"
    plotfig(W1, H1, dM, αs, β, cbx, iterstep, legendstrs, xlblsrt, ylblstr, fn; αrng = -0.5:0.01:1.5, showallα=false)
    M = I+αs[end]*dM; W1 = W1*M; H1 = M\H1
end

#================== Line search =================#
function prepare_fg(W, H; useprecond=true, β=0.7, allcompW=false, allcompH=false) # _best
    m, p = size(W)
    p, n = size(H)
    bW = vec((W.<0).*W); bH = vec((H.<0).*H)
    AWTbW = POCA.cal_AWTb(W, bW); AHTbH = POCA.cal_AHTb(H, bH)
    pW, pH = sca2(W, allcomp = allcompW), sca2(H, allcomp = allcompH)
    pWH = pW*pH
    gradient0 = 2*(pH*AWTbW+pW*AHTbH)

    function fg!(F, G, x)
        AWx, AHx = POCA.cal_AWx(W,x; allcomp=allcompW), POCA.cal_AHx(H,x; allcomp=allcompH)
        if G != nothing
            AWTAWx = POCA.cal_AWTb(W,AWx); AHTAHx = POCA.cal_AHTb(H,AHx); OTOx = POCA.cal_OTOx(x,β)
            g = 2(pH*AWTAWx+pW*AHTAHx+OTOx*pWH)+gradient0
            copyto!(G,g)
        end
        if F != nothing
            Ox = POCA.cal_Ox(x,β)
            f = pH*(AWx'AWx)+pW*(AHx'AHx)+(Ox'Ox)*pWH+2(pH*(AWTbW'*x)+pW*(AHTbH'*x))+pWH
            return f
        end
    end

    if useprecond
        diagA = 2*(pH*POCA.cal_diagAWTAW(W)+pW*POCA.cal_diagAHTAH(H)+pWH*β*POCA.cal_diagOTO(eltype(W),p))
        if all(diagA .>= 0)
            P = Optim.Diagonal(diagA) # Jacobi
        else
#            println("Some components of diapgA are negative")
            P = Optim.Diagonal(ones(p^2)) # no preconditioning
        end
    else
        P = Optim.Diagonal(ones(p^2))
    end
    fg!, P
end

function prepare_fg_updateWH(W, H; useprecond=true, β=0.7, allcompW=false, allcompH=false) # _updataWH
    m, p = size(W)
    p, n = size(H)
    Wlast = copy(W); Hlast = copy(H)
    xlast = zeros(p^2)
    function fg!(F, G, x)
        xi = x-xlast
        copyto!(Wlast,Wlast*(I+POCA.reshape(xi,p,p)))
        copyto!(Hlast,(I+POCA.reshape(xi,p,p))\Hlast)
        pW, pH = sca2(Wlast, allcomp = allcompW), sca2(Hlast, allcomp = allcompH)
        pWH = pW*pH
        if G != nothing
            # OTOxlast = all(xi .== 0) ? OTOxlast : POCA.cal_OTOx(xi,β)
            bW = vec((Wlast.<0).*Wlast); bH = vec((Hlast.<0).*Hlast)
            AWTbW = POCA.cal_AWTb(Wlast, bW); AHTbH = POCA.cal_AHTb(Hlast, bH); OTOx = POCA.cal_OTOx(xi,β)
            g = 2*(pH*AWTbW+pW*AHTbH+OTOx*pWH)
            copyto!(G,g)
        end
        copyto!(xlast, x)
        if F != nothing
            # Oxlast = all(xi .== 0) ? Oxlast : POCA.cal_Ox(xi,β)
            Ox = POCA.cal_Ox(xi,β)
            f = pWH*(1+(Ox'Ox))
            print("$f ")
            return f
        end
    end

    if useprecond
        pW, pH = sca2(Wlast, allcomp = allcompW), sca2(Hlast, allcomp = allcompH)
        diagA = 2*(pH*POCA.cal_diagAWTAW(W)+pW*POCA.cal_diagAHTAH(H)+pWH*β*POCA.cal_diagOTO(eltype(W),p))
        if all(diagA .>= 0)
            P = Optim.Diagonal(diagA) # Jacobi
        else
#            println("Some components of diagA are negative")
            P = Optim.Diagonal(ones(p^2)) # no preconditioning
        end
    else
        P = Optim.Diagonal(ones(p^2))
    end
    fg!, P
end

X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)

objtol = 1e-7; β = 0.7
rt0 = @elapsed W0, H0, objval, iter, inner_iters = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none,
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=9, objtol=objtol, showdbg=true)
rt1 = @elapsed W1, H1, objval, iter, inner_iters = semiscasolve!(W0, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none,
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1, objtol=objtol, showdbg=true)
rt2 = @elapsed W2, H2, objval_p, iter_p, inner_iters_p = semiscasolve!(W0, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:full,
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1, objtol=objtol, showdbg=true)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
iter = min(iter[1], iter_p[1])
ax1.plot(1:iter, [inner_iters[1][1:iter] inner_iters_p[1][1:iter]])
ax1.legend(["w/o line search", "with line search"],fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("penalty", fontsize = 12)
savefig("iter_vs_penalty(linesearch).png")

function optimize_once(W,H; β=0.7, useprecond = true)
    p = size(W,2);
    initial_x = zeros(p^2);
    fg!, P = prepare_fg(W, H; useprecond=useprecond, β=β);
    opt = Optim.Options(f_tol=1e-5, show_trace=false, show_every=10);
    if useprecond
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(P=P), opt);
    else
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
    end
    rst
end

rst1 = optimize_once(W1,H1,β=0.7,useprecond=true);
x = rst1[1].minimizer;
rst1[1].iterations
(f1,x1,fx1,gx1) = rst1[2]
rst2 = optimize_once(W2,H2,β=0.7,useprecond=true);
x = rst2[1].minimizer;
rst2[1].iterations
(f2,x2,fx2,gx2) = rst2[2]

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(1:length(f1), f1)
ax1.plot(1:length(f2), f2)
ax1.legend(["w/o line search", "with line search"],fontsize = 12,loc=1)
xlabel("CG iteration",fontsize = 12)
ylabel("penalty", fontsize = 12)
savefig("CGiter_vs_penalty(linesearch).png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
iter = min(iter[1], iter_p[1])
ax1.plot(1:56, tt0)
ax1.plot(1:260, tt)
ax1.legend(["w/o line search", "with line search"],fontsize = 12,loc=1)
xlabel("CG iteration",fontsize = 12)
ylabel("|x – x’|/|x’| ", fontsize = 12)
savefig("CGiter_vs_x(linesearch).png")

dM = POCA.reshapex(x, (p, p))
legendstrs = ["true fn.", "approx. fn." #=, "line search" =#]
xlblsrt = "α"
ylblstr = "penalty"
fn = "withlinesearch.png"
plotfig(W1, H1, dM, legendstrs, xlblsrt, ylblstr, fn)

#============= conditional line search ========================#

# CG S-POCA with preconditioning
ncellss = 6:2:100; β=50
datartsvd=[];
datartsca1=[]; mssdsca1=[]; iters1=[]; inner_iters1=[]
datartsca2=[]; mssdsca2=[]; iters2=[]; inner_iters2=[]
datartsca3=[]; mssdsca3=[]; iters3=[]; inner_iters3=[]
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtimesvd = @elapsed F = svd(X);
    W = F.U[:,1:ncells]; H=W\X
    push!(datartsvd, runtimesvd)
    # without linesearch
    params = Params(allow_f_increases=true, f_abstol=1e-6, f_reltol = 1e-6, maxiter=34, successive_f_converge=3, store_trace=true, show_trace=true)
    runtime1 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params)
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchednssd(gtW,W1)
    push!(datartsca1, runtime1)
    push!(mssdsca1, mssd1)
    push!(iters1,iter[1])
    inner_iter1 = [trs[1][d].inner_iters for d in 1:length(trs[1])]
    push!(inner_iters1, mean(inner_iter1))
    imsaveW("wols_nc$(ncells)_b$(β)_W1.png",W1,imgsz)
    # with line search
    runtime2 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:full, 
        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params)
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchednssd(gtW,W1)
    push!(datartsca2, runtime2)
    push!(mssdsca2, mssd1)
    push!(iters2,iter[1])
    inner_iter2 = [trs[1][d].inner_iters for d in 1:length(trs[1])]
    push!(inner_iters2, mean(inner_iter2))
    imsaveW("wls_nc$(ncells)_b$(β)_W1.png",W1,imgsz)
    # with conditional line search
    runtime3 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:conditional, 
        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params)
    normalizeWH!(W1,H1)
    mssd1, ml1, ssds1 = matchednssd(gtW,W1)
    push!(datartsca3, runtime3)
    push!(mssdsca3, mssd1)
    push!(iters3,iter[1])
    inner_iter3 = [trs[1][d].inner_iters for d in 1:length(trs[1])]
    push!(inner_iters3, mean(inner_iter3))
    imsaveW("wcls_nc$(ncells)_b$(β)_W1.png",W1,imgsz)
end

fname = "linesearch_methods_b$(β)_$(today())_$(hour(now())).jld"
save(fname, "beta", β, "ncellss", ncellss, "datartsvd", datartsvd,
        "datartsca1",datart1sca1, "mssdsca1", mssdsca1, "iters1", iters1, "inner_iters1", inner_iters1, # wols
        "datartsca2",datart1sca2, "mssdsca2", mssdsca2, "iters2", iters2, "inner_iters2", inner_iters2, # wls
        "datartsca3",datart1sca3, "mssdsca3", mssdsca3, "iters3", iters3, "inner_iters3", inner_iters3) # wcls

dd = load("isvd_vs_svd_2021-10-21_0.jld")
datartsvd = dd["datartsvd"]
datartsca1 = dd["datartsca1"]; datartsca2 = dd["datartsca2"]; datartsca3 = dd["datartsca3"];
iters1 = dd["iters1"]; iters2 = dd["iters2"]; iters3 = dd["iters3"];
inner_iters1 = dd["inner_iters1"]; inner_iters2 = dd["inner_iters2"]; inner_iters3 = dd["inner_iters3"];
mssdsca1 = dd["mssdsca1"]; mssdsca2 = dd["mssdsca2"]; mssdsca3 = dd["mssdsca3"];

datart1 = datartsvd + datartsca1
datart2 = datartsvd + datartsca2
datart3 = datartsvd + datartsca3
total_iters1 = iters1.*inner_iters1
total_iters2 = iters2.*inner_iters2
total_iters3 = iters3.*inner_iters3

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca1 datartsca2 datartsca3])
ax1.legend(["w/o line search", "full line search", "conditional line search"],fontsize = 12,loc=2)
xlabel("number of components",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_runtime3.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [mssdsca1 mssdsca2 mssdsca3])
ax1.legend(["w/o line search", "full line search", "conditional line search"],fontsize = 12,loc=2)
xlabel("number of components",fontsize = 12)
ylabel("mssd",fontsize = 12)
savefig("noc_vs_mssd.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [total_iters1 total_iters2 total_iters3])
ax1.legend(["w/o line search", "full line search", "conditional line search"],fontsize = 12,loc=2)
xlabel("number of components",fontsize = 12)
ylabel("number of iterations",fontsize = 12)
savefig("noc_vs_iterations.png")

#============= Case 1 with ForwardDiff ========================#
β=50000; 
@time W1, H1, objvals, iters, inner_iterss, Ms, f_callsss, g_callsss = POCA.semiscasolve!_full(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
    order=0, fixposdef=false, objtol=1e-5, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=50, showdbg=true);
normalizeWH!(W1,H1); imshowW(W1,imgsz)

#============= H = W\X vs H = SV' ========================#
ncellss = 6:2:100
errs1=[]; errs2=[]
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X);
    W = F.U[:,1:ncells];
    Hw = W\X; Hf = (F.V*Diagonal(F.S))'[1:ncells,:]
    push!(errs1,norm(X-W*Hw))
    push!(errs2,norm(X-W*Hf))
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [errs1 errs2])
ax1.legend(["H=W\\X", "H=SV'"], fontsize = 12, loc=1)
xlabel("number of components",fontsize = 12)
ylabel("|X-WH|",fontsize = 12)
savefig("noc_vs_errs.png")

factorrng = 1:40; ncells = 14; fovsz=(40,20); lengthT0 = 100

errs1=[]; errs2=[]
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    F = svd(X);
    W = F.U[:,1:ncells];
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells]);
    Hw = W\X; Hf = (F.V*Diagonal(F.S))'[1:ncells,:]
    push!(errs1,norm(X-W*Hw))
    push!(errs2,norm(X-W*Hf))
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [errs1 errs2])
ax1.legend(["H=W\\X", "H=SV'"], fontsize = 12, loc=2)
xlabel("factor",fontsize = 12)
ylabel("|X-WH|",fontsize = 12)
savefig("factor_vs_errs.png")

#============= Convergence criterion ========================#
# X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_nonorthog200.jld"; ncells=14, lengthT=200)
# gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_noiseless_nonorthog.jld"; ncells=14, lengthT=200, noiselevel=0.)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

ncellss2 = 10:10:100
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
objvals = []
for ncells in ncellss2
    println("ncells=$ncells")
    F = svd(X); W = F.U[:,1:ncells]; H = W\X;
    Wn, Hn = balanced_WH(W,X)
    params = Params(allow_f_increases=true, f_abstol=1e-9, f_reltol = 1e-9, maxiter=100, successive_f_converge=3, store_trace=true, show_trace=true)
    runtime2 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
        order=0, fixposdef=false, skew=false, β=10, c=0.5, α0=1.0, ρ=0.5, params=params);
    W2, H2 = balanced_WH(W1,X)
    imshowW(W2,imgsz)
    normalizeWH!(W1,H1)
    imshowW(W1,imgsz)
    @show norm.(eachcol(W1))[1] norm.(eachrow(H1))[1]
    imsaveW("W2_$(ncells).png",W2,imgsz)
    push!(objvals,objval[1])
    ax1.plot(1:length(objval[1]), objval[1])
end
ax1.legend(collect(ncellss2),fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("iter_vs_pens2.png")
save("objvals.jld", "objvals", objvals)
save("objvals_ls.jld", "objvals", objvals)

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
for objval in objvals
    ax1.plot(1:length(objval), objval)
end


# abs(f_x - f_x_previous)/power2
F = svd(X); β = 0.7
reltol = 1e-7; finctol = 1e-5; allow_f_increment=false
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
for objval in objvals
    @show length(objval)
    W = F.U[:,1:ncells]; H = W\X; power2 = (norm(W)*norm(H))^2
    dobj = abs.(objval[2:end]-objval[1:end-1])./power2
    obj5avg = (objval[5:end]+objval[4:end-1]+objval[3:end-2]+objval[2:end-3]+objval[1:end-4])./5
    dobj5avg = abs.(obj5avg[5:end]-obj5avg[4:end-1])./power2
    dobjcon = copy(objval[1:4]); idx = 5; successive_cnt = 0
    for d in dobj5avg
        if d < reltol
            if successive_cnt == 3
                push!(dobjcon, objval[idx])
                break
            else
                successive_cnt += 1
            end
        else
            successive_cnt = 0
        end
        if !allow_f_increment && ((objval[idx]-objval[idx-1])/objval[idx-1]/power2 > finctol)
            break
        end
        push!(dobjcon, objval[idx])
        idx += 1
    end
    @show idx
    ax1.plot(1:length(dobjcon), dobjcon)
end
ax1.legend(collect(ncellss2),fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("iter_vs_dpens2.png")

f_tol = params.f_abstol*power2
abs(f_x - f_x_previous)/power2 < params.f_reltol

#============= Compare Newton S-POCA, CG S-POCA, CD ========================#
# using Colors, FixedPointNumbers

# imgsz = (40,20)
imgsz = (80,40)
sigma = 5
nevents = 140
lengthT = 200
gt_ncells, imgrs, gtW, gtH, gtWimgc, gtbg = gaussian2D(sigma, imgsz, lengthT, nevents, orthogonal=false)
datafilename = "fc_nonorth_size8.jld"
Images.save(datafilename, "gt_ncells", gt_ncells, "imgrs", imgrs, "gtW", gtW, "gtH", gtH, "gtWimgc",Array(gtWimgc), "gtbg", gtbg, "imgsz", imgsz)
# imshowW(gtW,imgsz)
# imsaveW("size8_gtW.png",gtW,imgsz)
# img = reshape(imgrs8,imgsz...,lengthT)
# ImageView.imshow(img)
# save("size8.gif", RGB{N0f8}.(img./maximum(img)))
# mp4(img, "size8.mp4", fps=30)

X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_nonorthog200.jld"; ncells=14, lengthT=200)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
# X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_noiseless_nonorthog.jld"; ncells=14, lengthT=200, noiselevel=0.)
# gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

ncellss = 6:2:100
if true
scapens=[]; norms=[]; scapensn=[]; normsn=[];
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X);
    W = F.U[:,1:ncells];
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells]);
    H = W\X;
    Wn, Hn = balanced_WH(W,X);
    push!(scapens,scapair(W,H))
    push!(norms,(norm(W)*norm(H))^2)
    push!(scapensn,scapair(Wn,Hn))
    push!(normsn,(norm(Wn)*norm(Hn))^2)
end
end
plot(ncellss, [scapens norms scapensn normsn])

# A\b S-POCA
if false
datartssca0=[]; mssdssca0=[]; iterss0=[]
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    datartsca0=[]; mssdsca0=[]; iters0=[]
    for β in [0.7] # best β = 5.0
        println("beta=$β")
        runtime2 = @elapsed  W1, H1, objval, iter = semiscasolve!(U, X; flipsign=false, usecg=false, useprecond=false, linesearch_method=:none, 
                        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1000, objtol=objtol, showdbg=false);
        rt = runtime1+runtime2
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchedssd(gtW,W1)
        push!(datartsca0, runtime1+runtime2)
        push!(mssdsca0, mssd1)
        push!(iters0, iter[1])
        #imshowW(W1,imgsz)
        imsaveW("size_SPOCA_inv_nc$(ncells)_b$(β)_ssd$(mssd1)_rt$(rt).png",W1,imgsz)
    end
    push!(datartssca0, datartsca0)
    push!(mssdssca0, mssdsca0)
    push!(iterss0, iters0)
end
end

# CG S-POCA
if false
datartssca1=[]; mssdssca1=[]; iterss1=[]; inner_iterss1=[]
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    datartsca0=[]; mssdsca0=[]; iters0=[]; inner_iters0=[]
    for β in [0.7] # best β = 5.0
        println("beta=$β")
        params = Params(f_abstol = 1e-5, maxiter=100, store_trace=true, show_trace=false)
        runtime2 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(F, X, ncells; flipsign=false, usecg=true, useprecond=false, linesearch_method=:none, 
            order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params);
        rt = runtime1+runtime2
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchednssd(gtW,W1)
        push!(datartsca0, runtime1+runtime2)
        push!(mssdsca0, mssd1)
        push!(iters0,iter[1])
        inner_iter = [trs[1][d].inner_iters for d in 1:length(trs[1])]
        push!(inner_iters0, mean(inner_iter))
        #imshowW(W1,imgsz)
        imsaveW("size_SPOCA_CG_nc$(ncells)_b$(β)_ssd$(mssd1)_rt$(rt).png",W1,imgsz)
    end
    push!(datartssca1, datartsca0)
    push!(mssdssca1, mssdsca0)
    push!(iterss1, iters0)
    push!(inner_iterss1, inner_iters0)
end
end

# CG S-POCA with preconditioning
if true
datart1ssca=[]; datart2ssca=[]; mssdssca=[]; iterss=[]; inner_iterss=[]
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X);
    W = F.U[:,1:ncells];
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells]);
    H = W\X;
    Wn, Hn = balanced_WH(W,X);
    datart1sca=[]; datart2sca=[]; mssdsca=[]; iters=[]; inner_iters=[]
    for β in [10.]
        # factor = scapair(W,H)/scapair(Wn,Hn)
        # β /= factor
        println("beta=$β")
        params = Params(allow_f_increases=true, f_abstol=1e-8, f_reltol = 1e-8, maxiter=34, successive_f_converge=3, store_trace=true, show_trace=true)
        runtime2 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(W, H; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
            order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params);
        rt = runtime1+runtime2
        W2 = copy(W1)
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchednssd(gtW,W1)
        push!(datart1sca, runtime1)
        push!(datart2sca, runtime2)
        push!(mssdsca, mssd1)
        push!(iters,iter[1])
        inner_iter = [trs[1][d].inner_iters for d in 1:length(trs[1])]
        push!(inner_iters, mean(inner_iter))
        #imshowW(W1,imgsz)
        imsaveW("size_SPOCA_CG_Precond_nc$(ncells)_b$(β)_W1.png",W1,imgsz)
        #imsaveW("size_SPOCA_CG_Precond_nc$(ncells)_b$(β)_W2.png",W2,imgsz) # save before normalization
    end
    push!(datart1ssca, datart1sca)
    push!(datart2ssca, datart2sca)
    push!(mssdssca, mssdsca)
    push!(iterss, iters)
    push!(inner_iterss, inner_iters)
end
datart1sca = getindex.(datart1ssca,1)
datart2sca = getindex.(datart2ssca,1)
datartsca = datart1sca+datart2sca
iters = getindex.(iterss,1)
inner_iters = getindex.(inner_iterss,1)
total_iters = iters.*inner_iters

fname = "sca_precond_vs_noc_normalizedWH_tol-6_$(today())_$(hour(now())).jld"
save(fname, "ncellss", ncellss,
        "datart1sca",datart1sca,"datart2sca",datart2sca,
        "iters", iters, "inner_iters", inner_iters)
dd = load(fname)
ncellss = dd["ncellss"]
datart1sca = dd["datart1sca"]
datart2sca = dd["datart2sca"]
iters = dd["iters"]
inner_iters = dd["inner_iters"]
datartsca = datart1sca+datart2sca
end

#dataname = "compare200_091721" # constant beta
dataname = "compare200_091721(beta_div_p)"
dd = load("$dataname.jld")
datart1cd0 = dd["datart1cd0"];
datart2cd0 = dd["datart2cd0"];
# datart1sca = dd["datart1sca2"];
# datart2sca = dd["datart2sca2"];
datartcd0 = datart1cd0+datart2cd0;
datartsca = datart1sca+datart2sca;
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca datartcd0])
ax1.legend(["S-POCA", "CD"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_runtime.png")



xabs = [maximum(abs, tr.M-I) for tr in trs[1]]
inner_fx = [tr.inner_fx for tr in trs[1]]
dinner_fx = POCA.f_relchange.(inner_fx[2:end],inner_fx[1:end-1],)
dinner_fx2 = abs.(inner_fx[1:end-1]-inner_fx[2:end])./inner_fx[1]
push!(dinner_fx,0)
push!(dinner_fx2,0)
inner_fx_lpf = (inner_fx[1:end-2]+inner_fx[2:end-1]+inner_fx[3:end])./3
push!(inner_fx_lpf,1); push!(inner_fx_lpf,1); 
dinner_fx_lpf = POCA.f_relchange.(inner_fx_lpf[2:end],inner_fx_lpf[1:end-1])
push!(dinner_fx_lpf,1)

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(1:50, inner_fx)
ax1.legend(["E(n)", "|E(n)-E(n-1)|/E(n)"],fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("iter_vs_dinner_fx.png")

Wn, Hn = balanced_WH(W,X);
pens = []; pens_W = []; pens_H = []
for tr in trs[1]
    M = tr.M
    Wn = Wn*M
    Hn = M\Hn
    push!(pens_W,sca2(Wn))
    push!(pens_H,sca2(Hn))
    push!(pens,scapair(Wn,Hn))
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot([pens pens_W pens_H])
ax1.legend(["scapair(W,H)", "sca2(W)", "sca2(H)"],fontsize = 12,loc=1)


β=5; linesearch_method=:none
Wn, Hn = copy(W), copy(H);
#Wn, Hn = balanced_WH(W,X);
println("beta=$β")
params = Params(allow_f_increases=true, f_abstol=1e-8, f_reltol = 1e-8, maxiter=40, successive_f_converge=3, store_trace=true, show_trace=true)
runtime2 = @elapsed W1, H1, objval, iter, trs = semiscasolve!(Wn, Hn; flipsign=false, usecg=true, useprecond=true, linesearch_method=linesearch_method, 
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, params=params);

p = size(W,2)
Ms = [Matrix(1.0I,p,p)]
for tr in trs[1]
    push!(Ms,tr.M)
end
pen = scapair(Wn,Hn); iter = 0
pens=[]; pens_W=[]; pens_H=[]; skews=[]; orthogs=[]; orthogs_W=[]; mssds=[]; norm_dM=[]; max_dM=[]
for M in Ms
    dM = M-I
    Wn = Wn*M
    Hn = M\Hn
    W1, H1 = copy(Wn), copy(Hn)
    normalizeWH!(W1,H1)
    imsaveW("$iter.png",W1,imgsz)
    push!(pens_W,sca2(Wn))
    push!(pens_H,sca2(Hn))
    push!(orthogs_W,norm(offdiag!(Wn'Wn)))
    orthogpen=norm(dM+dM')^2
    push!(skews,orthogpen)
    push!(orthogs, pen*β/ncells*orthogpen)
    pen = scapair(Wn,Hn)
    push!(pens,pen)
    push!(norm_dM, norm(dM) )
    push!(max_dM, maximum(abs.(dM)))
    mssd1, ml1, ssds1 = matchednssd(gtW,Wn)
    push!(mssds, mssd1)
    iter += 1
end
# fig, ax1 = plt.subplots(1,1, figsize=(5,4))
# ax1.plot([pens pens_W pens_H orthogs norm_dM./2 max_dM skews.*10])
# ax1.legend(["scapair(W,H)", "sca2(W)", "sca2(H)","E_orthog","norm_dM","max_dM"],fontsize = 12,loc=1)

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot([pens skews.*5 norm_dM orthogs_W./100])
ax1.set_ylim(-0.5,5); ax1.set_xlim(0,iter-1)
ax1.legend(["Es", "Eo(M)", "|dM|", "Eo(W)"],fontsize = 12,loc=1)
savefig("iter_vs_pen.png")

# CG S-POCA with preconditioning controlled
if true
datart1ssca2=[]; datart2ssca2=[]; mssdssca2=[]; iterss2=[]; inner_iterss2=[]
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    datart1sca2=[]; datart2sca2=[]; mssdsca2=[]; iters2=[]; inner_iters2=[]
    for β in [0.7] # best β = 5.0
        println("beta=$β")
        runtime2 = @elapsed  W1, H1, objval, iter, inner_iter = semiscasolve!(F, X, ncells; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
                        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=10000, objtol=objtol, showdbg=false);
        rt = runtime1+runtime2
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchednssd(gtW,W1)
        push!(datart1sca2, runtime1)
        push!(datart2sca2, runtime2)
        push!(mssdsca2, mssd1)
        push!(iters2,iter[1])
        push!(inner_iters2, mean(inner_iter[1]))
        #imshowW(W1,imgsz)
        imsaveW("size_SPOCA_CG_Precond_ls_nc$(ncells)_b$(β)_ssd$(mssd1)_rt$(rt).png",W1,imgsz)
    end
    push!(datart1ssca2, datart1sca2)
    push!(datart2ssca2, datart2sca2)
    push!(mssdssca2, mssdsca2)
    push!(iterss2, iters2)
    push!(inner_iterss2, inner_iters2)
end
datart1sca2 = getindex.(datart1ssca2,1)
datart2sca2 = getindex.(datart2ssca2,1)
datartsca2 = datart1sca2+datart2sca2
iters2 = getindex.(iterss2,1)
inner_iters2 = getindex.(inner_iterss2,1)
total_iters2 = iters2.*inner_iters2
end

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
if false
αs = 0:0.1:10
datart1scd = []; datart2scd = []; mssdscd = []
for ncells in ncellss    
    println("ncells=$ncells")
    runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells, variant=:ar)
    datart1cd=[]; datart2cd=[]; mssdcd=[]
    for α in [0.1] # best α = 0.1
        println("α=$(α)")
        runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
        normalizeWH!(W88,H88)
        mssd88, ml88, ssds = matchednssd(gtW,W88)
        mssdH88 = ssdH(ml88,gtH,H88')
        push!(datart1cd, runtime1)
        push!(datart2cd, runtime2)
        push!(mssdcd, mssd88)
        #imshowW(W88,imgsz)
        imsaveW("size_CD_nc$(ncells)_a$(α).png", W88, imgsz)
    end
    push!(datart1scd, datart1cd)
    push!(datart2scd, datart2cd)
    push!(mssdscd, mssdcd)
end
datart1cd = getindex.(datart1scd,1)
datart2cd = getindex.(datart2scd,1)
datartcd = datart1cd+datart2cd
end

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
if true
αs = 0:0.1:10
datart1scd0 = []; datart2scd0 = []; mssdscd0 = []
for ncells in ncellss    
    println("ncells=$ncells")
    runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells, variant=:ar)
    datart1cd0=[]; datart2cd0=[]; mssdcd0=[]
    for α in [0.0] # best α = 0.1
        println("α=$(α)")
        runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
        normalizeWH!(W88,H88)
        mssd88, ml88, ssds = matchednssd(gtW,W88)
        mssdH88 = ssdH(ml88,gtH,H88')
        push!(datart1cd0, runtime1)
        push!(datart2cd0, runtime2)
        push!(mssdcd0, mssd88)
        #imshowW(W88,imgsz)
        imsaveW("size_CD_nosparsity_nc$(ncells)_a$(α).png", W88, imgsz)
    end
    push!(datart1scd0, datart1cd0)
    push!(datart2scd0, datart2cd0)
    push!(mssdscd0, mssdcd0)
end
datart1cd0 = getindex.(datart1scd0,1)
datart2cd0 = getindex.(datart2scd0,1)
datartcd0 = datart1cd0+datart2cd0
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, datartsca)
ax1.legend(["CG S-POCA"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_sca_runtime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca datartsca2 datartcd0])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_noc_vs_runtime.png")

save("compare200_091721.jld", "datart1cd",datart1cd,"datart2cd",datart2cd,
        "datart1cd0",datart1cd0,"datart2cd0",datart2cd0,
        "datartcd",datartcd,"datartcd0",datartcd0,
        "datart1sca",datart1sca,"datart2sca",datart2sca,
        "datart1sca2",datart1sca2,"datart2sca2",datart2sca2)
save("compare200_091721(beta_div_p2).jld", "datart1cd",datart1cd,"datart2cd",datart2cd,
        "datart1cd0",datart1cd0,"datart2cd0",datart2cd0,
        "datartcd",datartcd,"datartcd0",datartcd0,
        "datart1sca",datart1sca,"datart2sca",datart2sca,
        "datart1sca2",datart1sca2,"datart2sca2",datart2sca2)
save("compare200_091721(beta_div_p).jld", "ncellss", ncellss,
        "datart1cd",datart1cd,"datart2cd",datart2cd,
        "datart1cd0",datart1cd0,"datart2cd0",datart2cd0,
        "datartcd",datartcd,"datartcd0",datartcd0,
        "datart1sca",datart1sca,"datart2sca",datart2sca,
        "datart1sca2",datart1sca2,"datart2sca2",datart2sca2,
        "iters", iters, "inner_iters", inner_iters,
        "iters2", iters2, "inner_iters2", inner_iters2)

dataname = "compare200_091721(beta_div_p)"
dd = load("$dataname.jld")
datart1sca = dd["datart1sca"]
datart2sca = dd["datart2sca"]
datart1sca2 = dd["datart1sca2"]
datart2sca2 = dd["datart2sca2"]
datart1cd0 = dd["datart1cd0"]
datart2cd0 = dd["datart2cd0"]
iters = dd["iters"]
inner_iters = dd["inner_iters"]
iters2 = dd["iters2"]
inner_iters2 = dd["inner_iters2"]
datartsca = datart1sca+datart2sca
datartsca2 = datart1sca2+datart2sca2
datartcd0 = datart1cd0+datart2cd0        
total_iters = iters.*inner_iters
total_iters2 = iters2.*inner_iters2

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca datartsca2 datartcd0])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("$(dataname)_toatlruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [iters iters2])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')"],fontsize = 12,loc=4)
xlabel("number of cells",fontsize = 12)
ylabel("iteration",fontsize = 12)
savefig("$(dataname)_outeriter.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [inner_iters inner_iters2])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')"],fontsize = 12,loc=4)
xlabel("number of cells",fontsize = 12)
ylabel("iteration",fontsize = 12)
savefig("$(dataname)_cgiter.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [total_iters total_iters2])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')"],fontsize = 12,loc=4)
xlabel("number of cells",fontsize = 12)
ylabel("iteration",fontsize = 12)
savefig("$(dataname)_totaliter.png")

#datartsca = getindex.(datartssca,1)
datartcd = getindex.(datartscd,1)
datartsca0 = getindex.(datartssca0,1)
datartsca1 = getindex.(datartssca1,1)
datartsca2 = getindex.(datartssca2,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca0 datartsca1 datartsca2 datartcd])
ax1.legend(["Newton S-POCA", "CG S-POCA", "CG S-POCA with Predond.", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_noc_vs_runtime.png")

#mssdsca = getindex.(mssdssca,1)
mssdcd0 = getindex.(mssdscd0,1)
mssdsca = getindex.(mssdssca,1)
mssdsca2 = getindex.(mssdssca2,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [mssdsca mssdsca2 mssdcd0])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')", "Coordinate Descent"],fontsize = 12,loc=1)
xlabel("number of cells",fontsize = 12)
ylabel("ssd(W)",fontsize = 12)
savefig("size_noc_vs_ssd(W).png")

#iters = getindex.(iterss,1)
iters0 = getindex.(iterss0,1)
iters1 = getindex.(iterss1,1)
iters = getindex.(iterss,1)
iters2 = getindex.(iterss2,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [iters iters2 ])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("iterations",fontsize = 12)
savefig("size_noc_vs_iter.png")

# fig, ax1 = plt.subplots(1,1, figsize=(5,4))
# ax1.scatter(iters, datartcd)
# xlabel("number of cells",fontsize = 12)
# ylabel("iterations",fontsize = 12)
# savefig("size_noc_vs_iter.png")

#====== Preconditioning =======#
datartcd0 = getindex.(datartscd0,1)
datartcd = getindex.(datartscd,1)
datartsca = getindex.(datartssca,1)
datartsca1 = getindex.(datartssca1,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca1 datartsca datartcd0 datartcd])
ax1.legend(["CG S-POCA", "CG S-POCA with Predond.", "CD", "CD  with sparsity"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_noc_vs_runtime(precond).png")

mssdsca = getindex.(mssdssca,1)
mssdsca1 = getindex.(mssdssca1,1)
mssdcd = getindex.(mssdscd,1)
mssdcd0 = getindex.(mssdscd0,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [mssdsca1 mssdsca mssdcd])
ax1.legend(["CG S-POCA", "CG S-POCA with Predond.", "Coordinate Descent"],fontsize = 12,loc=1)
xlabel("number of cells",fontsize = 12)
ylabel("ssd(W)",fontsize = 12)
savefig("size_noc_vs_ssd(W)(precond).png")

iters = getindex.(iterss,1)
iters1 = getindex.(iterss1,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [iters1 iters])
ax1.legend(["CG S-POCA", "CG S-POCA with Predond."],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("iterations",fontsize = 12)
savefig("size_noc_vs_iter(precond).png")

inner_iters = getindex.(inner_iterss,1)
inner_iters2 = getindex.(inner_iterss2,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [inner_iters inner_iters2])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("iterations",fontsize = 12)
savefig("size_noc_vs_inner_iter(H=SV').png")

save("data0901_objtolXobjval(precondfixed)tol1e-7.jld",
    "datartcd", datartcd, "datartcd0", datartcd0, "datartsca0", datartsca0, "datartsca1", datartsca1, "datartsca", datartsca,
    "mssdcd", mssdcd, "mssdcd0", mssdcd0, "mssdsca0", mssdsca0, "mssdsca1", mssdsca1, "mssdsca", mssdsca,
    "iters0", iters0, "iters1", iters1, "iters", iters,
    "inner_iters1", inner_iters1, "inner_iters", inner_iters, "ncellss", ncellss )

dd = load("data0901_objtolXobjval(precond)tol1e-7.jld")
datartsca0 = dd["datartsca0"]
datartsca1 = dd["datartsca1"]
datartsca = dd["datartsca"]
datartcd0 = dd["datartcd0"]
datartcd = dd["datartcd"]
mssdsca0 = dd["mssdsca0"]
inner_iters1 = dd["inner_iters1"]
inner_iters = dd["inner_iters"]
ncellss = dd["ncellss"]
iters1 = dd["iters1"]
iters0 = dd["iters0"]

#===== CD without sparsity =========#
datartcd = getindex.(datartscd,1)
datartcd0 = getindex.(datartscd0,1)
datartsca1 = getindex.(datartssca1,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [datartsca1 datartcd datartcd0])
ax1.legend(["CG S-POCA", "CD", "CD w/o sparsity"],fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_noc_vs_runtime(precond)CDwoSparsity.png")

mssdcd = getindex.(mssdscd,1)
mssdcd0 = getindex.(mssdscd0,1)
mssdsca1 = getindex.(mssdssca1,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, [mssdsca1 mssdcd mssdcd0])
ax1.legend(["CG S-POCA", "CD", "CD w/o sparsity"],fontsize = 12,loc=1)
xlabel("number of cells",fontsize = 12)
ylabel("ssd(W)",fontsize = 12)
savefig("size_noc_vs_ssd(W)(precond)CDwoSparsity.png")

save("data0901_objtolXobjval(precond)tol1e-7CDwoSparsity.jld",
    "datartcd", datartcd, "datartcd0", datartcd0, "datartsca0", datartsca0, "datartsca1", datartsca1, "datartsca", datartsca,
    "mssdcd", mssdcd, "mssdcd0", mssdcd0, "mssdsca0", mssdsca0, "mssdsca1", mssdsca1, "mssdsca", mssdsca,
    "iters0", iters0, "iters1", iters1, "iters", iters,
    "inner_iters1", inner_iters1, "inner_iters", inner_iters )

    
#============= scalability for datasize ========================#

ncells = 20; fovsz=(40,20)
imgszyrng = 30:10:300; plotxrng = imgszyrng; plotxstr = "imgszy"; xlabelstr="spatial y size" # imgszyrng = 30:10:100, 40
imgszxrng = 20:10:20#; plotxrng = imgszxrng; plotxstr = "imgszx"; xlabelstr="spatial x size"
lengthTrng = 200:10:200#; plotxrng = lengthTrng; plotxstr = "lengthT"; xlabelstr="number of frames" # lengthTrng = 100:10:400, 200
# CG S-POCA

datartssca=[]; mssdssca=[]; iterss=[]; inner_iterss=[]
for imgszy in imgszyrng, imgszx in imgszxrng, lengthT in lengthTrng
    println("imgsz=($imgszy,$imgszx), lengthT=$lengthT")
    imgsz = (imgszy,imgszx)
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    runtime1 = @elapsed F = svd(X)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    datartsca=[]; mssdsca=[]; iters=[]; inner_iters=[]
    for β in [0.7] # best β = 5.0
        println("beta=$β")
        runtime2 = @elapsed  W1, H1, objval, iter, inner_iter = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=false, linesearch_method=:none, 
                        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=10000, objtol=objtol, showdbg=false);
        rt = runtime1+runtime2
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
        push!(datartsca, runtime1+runtime2)
        push!(mssdsca, mssd1)
        push!(iters,iter[1])
        push!(inner_iters, mean(inner_iter[1]))
        #imshowW(W1,imgsz)
        imsaveW("fc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG__b$(β).png",W1,imgsz)
   end
    push!(datartssca, datartsca)
    push!(mssdssca, mssdsca)
    push!(iterss, iters)
    push!(inner_iterss, inner_iters)
end
# Coordinate Descent (α is a regularization parameter to enforce sparsity)
αs = 0:0.1:10
datartscd = []; mssdscd = []
for imgszy in imgszyrng, imgszx in imgszxrng, lengthT in lengthTrng
    println("imgsz=($imgszy,$imgszx), lengthT=$lengthT")
    imgsz = (imgszy,imgszx)
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells, variant=:ar)
    datartcd=[]; mssdcd=[]
    for α in [0.0] # best α = 0.1
        println("α=$(α)")
        runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
        normalizeWH!(W88,H88)
        mssd88, ml88, ssds = matchedfiterr(gtW,W88)
        mssdH88 = ssdH(ml88,gtH,H88')
        push!(datartcd, runtime1+runtime2)
        push!(mssdcd, mssd88)
        #imshowW(W88,imgsz)
        imsaveW("fc_sz$(imgsz)_lT$(lengthT)_a$(α).png", W88, imgsz)
    end
    push!(datartscd, datartcd)
    push!(mssdscd, mssdcd)
end

datartcd = getindex.(datartscd,1)
datartsca = getindex.(datartssca,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, [datartsca datartcd])
ax1.legend(["CG S-POCA", "CD", "CD w/o sparsity"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("fcvs_$(plotxstr)_vs_runtime.png")

mssdcd = getindex.(mssdscd,1)
mssdsca = getindex.(mssdssca,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, [mssdsca mssdcd])
ax1.legend(["CG S-POCA", "CD", "CD w/o sparsity"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("ssd(W)",fontsize = 12)
savefig("fcvs_$(plotxstr)_vs_ssd(W).png")

save("data0903_scalability_in_datasize_$(plotxstr).jld",
    "datartcd", datartcd, "datartsca", datartsca, "mssdcd", mssdcd, "mssdsca", mssdsca)

#=== increase spatial and temporal size at the same time ===#

factorrng = 1:40
ncells = 14; fovsz=(40,20); lengthT0 = 100
C = 1:1:80; plotxrng = factorrng; plotxstr = "factor"; xlabelstr="factor" # imgszyrng = 30:10:100, 40
maxiter = 1000

datart1ssca=[]; datart2ssca=[]; mssdssca=[]; iterss=[]; inner_iterss=[]; penaltyss=[]; f_callss=[]; g_callss=[]
for factor in factorrng
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    runtime1 = @elapsed F = svd(X)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    datart1sca=[]; datart2sca=[]; mssdsca=[]; iters=[]; inner_iters=[]; penaltys=[]
    for β in [0.7] # best β = 5.0
        println("beta=$β")
        runtime2 = @elapsed  W1, H1, objval, iter, inner_iter, f_calls, g_calls = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
                        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=maxiter, objtol=objtol, showdbg=false);
        rt = runtime1+runtime2
        normalizeWH!(W1,H1)
        #mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
        push!(datart1sca, runtime1)
        push!(datart2sca, runtime2)
        #push!(mssdsca, mssd1)
        push!(iters,iter[1])
        push!(inner_iters, mean(inner_iter[1]))
        push!(penaltys, objval[1][end])
        push!(f_callss, sum(f_calls[1]))
        push!(g_callss, sum(g_calls[1]))
        #imshowW(W1,imgsz)
        imsaveW("fc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG__b$(β).png",W1,imgsz)
    end
    push!(datart1ssca, datart1sca)
    push!(datart2ssca, datart2sca)
    #push!(mssdssca, mssdsca)
    push!(iterss, iters)
    push!(inner_iterss, inner_iters)
    push!(penaltyss, penaltys)
end

datart1ssca2=[]; datart2ssca2=[]; mssdssca2=[]; iterss2=[]; inner_iterss2=[]; penaltyss2=[]; f_callss2=[]; g_callss2=[]
for factor in factorrng
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    runtime1 = @elapsed F = svd(X)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    datart1sca=[]; datart2sca=[]; mssdsca=[]; iters=[]; inner_iters=[]; penaltys=[]
    for β in [0.7] # best β = 5.0
        println("beta=$β")
        runtime2 = @elapsed  W1, H1, objval, iter, inner_iter, f_calls, g_calls = semiscasolve!(F, X, ncells; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
                        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=maxiter, objtol=objtol, showdbg=false);
        rt = runtime1+runtime2
        normalizeWH!(W1,H1)
        #mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
        push!(datart1sca, runtime1)
        push!(datart2sca, runtime2)
        #push!(mssdsca, mssd1)
        push!(iters,iter[1])
        push!(inner_iters, mean(inner_iter[1]))
        push!(penaltys, objval[1][end])
        push!(f_callss2, sum(f_calls[1]))
        push!(g_callss2, sum(g_calls[1]))
        #imshowW(W1,imgsz)
        imsaveW("fc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG__b$(β).png",W1,imgsz)
    end
    push!(datart1ssca2, datart1sca)
    push!(datart2ssca2, datart2sca)
    #push!(mssdssca2, mssdsca)
    push!(iterss2, iters)
    push!(inner_iterss2, inner_iters)
    push!(penaltyss2, penaltys)
end

# Coordinate Descent (α is a regularization parameter to enforce sparsity)
αs = 0:0.1:10
datart1scd0 = []; datart2scd0 = []; mssdscd0 = []
for factor in factorrng
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells)
    datart1cd=[]; datart2cd=[]; mssdcd=[]
    for α in [0.0] # best α = 0.1
        println("α=$(α)")
        runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
        normalizeWH!(W88,H88)
        #mssd88, ml88, ssds = matchedfiterr(gtW,W88)
        mssdH88 = ssdH(ml88,gtH,H88')
        push!(datart1cd, runtime1)
        push!(datart2cd, runtime2)
        #push!(mssdcd, mssd88)
        #imshowW(W88,imgsz)
        imsaveW("fc_sz$(imgsz)_lT$(lengthT)_a$(α).png", W88, imgsz)
    end
    push!(datart1scd0, datart1cd)
    push!(datart2scd0, datart2cd)
    #push!(mssdscd0, mssdcd)
end

datart1cd0 = getindex.(datart1scd0,1)
datart2cd0 = getindex.(datart2scd0,1)
datartcd0 = datart1cd0+datart2cd0
datart1sca = getindex.(datart1ssca,1)
datart2sca = getindex.(datart2ssca,1)
datart1sca2 = getindex.(datart1ssca2,1)
datart2sca2 = getindex.(datart2ssca2,1)
datartsca = datart1sca+datart2sca
datartsca2 = datart1sca2+datart2sca2
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datartsca datartsca2 datartcd0])
#ax1.plot(factorrng, [datart2sca datart2sca2 datart2cd0])
ax1.legend(["CG S-POCA (H=W\\X)", "CG S-POCA (H=SV')", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_$(xlabelstr)_vs_runtime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datartsca datart1sca datart2sca])
ax1.legend(["total runtime", "svd(X)", "S-POCA"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_$(xlabelstr)_vs_scaruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datartsca2 datart1sca2 datart2sca2])
ax1.legend(["total runtime", "svd(X)", "S-POCA"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_$(xlabelstr)_vs_sca2runtime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datart1sca datart1cd0])
ax1.legend(["svd(X)", "rsvd(X)"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_$(xlabelstr)_vs_initruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [datart2sca datart2cd0])
ax1.legend(["S-POCA", "CD"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("size_$(xlabelstr)_vs_coreruntime.png")

save("factor_092021_40.jld", "datart1cd0",datart1cd0,"datart2cd0",datart2cd0, "datartcd0",datartcd0,
        "datart1sca",datart1sca,"datart2sca",datart2sca,"datartsca",datartsca,
        "datart1sca2",datart1sca2,"datart2sca2",datart2sca2,"datartsca2",datartsca2,
        "mssdscd0", mssdscd0, "mssdssca", mssdssca, "mssdssca2", mssdssca2,
        "iterss",iterss,"inner_iterss",inner_iterss,
        "iterss2",iterss2,"inner_iterss2",inner_iterss2,
        "penaltyss",penaltyss, "penaltyss2",penaltyss2,
        "f_callss",f_callss, "f_callss2",f_callss2,
        "g_callss",g_callss, "g_callss2",g_callss2,
        "factorrng",factorrng)

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, [f_callss g_callss])
ax1.legend(["f calls", "g! calls"],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("number of calls",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_fgcalls.png")

iters = getindex.(iterss,1)
inner_iters = getindex.(inner_iterss,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, [iters inner_iters])
ax1.legend(["outer iteration", "inner_iteration"],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("number of iterations",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_iters.png")

datartcd = getindex.(datartscd,1)
datartsca = getindex.(datartssca,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, [datartsca datartcd])
ax1.legend(["CG S-POCA", "CD", "CD w/o sparsity"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_runtime.png")

mssdcd = getindex.(mssdscd,1)
mssdsca = getindex.(mssdssca,1)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, [mssdsca mssdcd])
ax1.legend(["CG S-POCA", "CD", "CD w/o sparsity"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("ssd(W)",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_ssd(W).png")

save("data0903_scalability_in_datasize_$(plotxstr).jld",
    "datartcd", datartcd, "datartsca", datartsca, "mssdcd", mssdcd, "mssdsca", mssdsca)

β = 0.7; objtol = 1e-7
ncells = 14; fovsz=(40,20); lengthT0 = 100
factorrng = 1:1:40; plotxrng = factorrng; plotxstr = "factor"; xlabelstr="factor" # imgszyrng = 30:10:100, 40
maxiter = 1000
datart = zeros(length(factorrng),13)
for (i,factor) in enumerate(factorrng)
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz, save=false)
    datart[i,12] = sum(W.<0)
    datart[i,13] = sum(H.<0)
    p = ncells
    x = rand(p^2)
    st = zeros(p^2);
    fg!, P = POCA.prepare_fg(W,H;β=β)
    datart[i,10] = @belapsed (f3 = fg!(1,nothing,x))
    datart[i,11] = @belapsed (fg!(nothing,st,x))
    bW = rand(*(size(W)...))
    bH = rand(*(size(H)...))
    datart[i,1] = @belapsed (AWx = POCA.cal_AWx(W,x))
    datart[i,2] = @belapsed (AHx = POCA.cal_AHx(H,x))
    datart[i,3] = @belapsed (AWTb = POCA.cal_AWTb(W,bW))
    datart[i,4] = @belapsed (AHTb = POCA.cal_AHTb(H,bH))
    datart[i,5] = @belapsed (Ox = POCA.cal_Ox(x, β))
    datart[i,6] = @belapsed (OTOx = POCA.cal_OTOx(x, β))
    datart[i,7] = @belapsed (diagAWTAW = POCA.cal_diagAWTAW(W))
    datart[i,8] = @belapsed (diagAHTAH = POCA.cal_diagAHTAH(H))
    datart[i,9] = @belapsed (diagOTO = POCA.cal_diagOTO(Float64, p))
end
save("datart3.jld","datart",datart,"factorrng",factorrng)
dd = load("datart3.jld")
datart = dd["datart"]
factorrng = dd["factorrng"]

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, datart[:,10:11])
ax1.legend(["f", "g!"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_fgruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, datart[:,12:13])
ax1.legend(["W", "H"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("number of negative components",fontsize = 12) # very linear
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_nnc.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, datart[:,1:4])
ax1.legend(["AWx", "AHx", "AWTb", "AHTb"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_rts1-4.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, datart[:,5:6])
ax1.legend(["Ox", "OTOx"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_rts5-6.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(plotxrng, datart[:,7:9])
ax1.legend(["diagAWTAW", "diagAHTAH", "diagOTO"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("fcvs_$(plotxstr)$(plotxrng[end])_vs_rts7-9.png")

#================== Stopping criterion =================#
options.iterations = 100
options.x_abstol = 0.0
options.x_reltol = 0.0
options.f_abstol = 0.0
options.f_reltol = 1.0e-5
options.g_abstol = 1.0e-8
options.allow_f_increases = true
options.successive_f_tol = 1

stopped = false
x_converged, f_increased, counter_f_tol = false, false, 0
gradient!(d, initial_x)
g_converged = !isfinite(value(d)) || any(!isfinite, gradient(d))
f_converged = maximum(abs, gradient(d)) <= options.g_abstol
converged = f_converged || g_converged

iteration = 0

ls_success::Bool = true
while !converged && !stopped && iteration < options.iterations
    iteration += 1
    ls_success = !update_state!(d, state, method)
    if !ls_success
        break
    end
    x_converged, f_converged, g_converged, f_increased = 
        assess_convergence(x, x_previous, f_x, f_x_previous, gradient(x),
        options.x_abstol, options.x_reltol, options.f_abstol, options.f_reltol, options.g_abstol)

    # For some problems it may be useful to require `f_converged` to be hit multiple times
    counter_f_tol = f_converged ? counter_f_tol+1 : 0
    converged = x_converged || g_converged || (counter_f_tol > options.successive_f_tol)

    if (f_increased && !options.allow_f_increases)
        stopped = true
    end
end

#================== Case 1 skew symmetric test =================#

M = zeros(p,p)+I
initial_x = POCA.vecodskew(M)
fg!, P = POCA.prepare_fg_skew(W, H; useprecond=false, β=0, allcompW=false, allcompH=false)
opt = Optim.Options(f_tol=1e-5, show_trace=true, show_every=10)
rst, _ = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt)
x = rst.minimizer
M = POCA.reshapeodskew(x)
W1 = W*M; H1 = M\H
sca2(W1)
sca2(H1)
normalizeWH!(W1,H1); imshowW(W1,imgsz)
norm(offdiag!(W1'W1))^2
norm(offdiag!(H1*H1'))^2
imsaveW("test.png", W1,imgsz)

#================== Follow up first several iterations for case 1 =================#

using LineSearches, NaNMath

function plotfig(Wlast, Hlast, x_prev, d, alpha0, β; show_alpha=true, legend_loc=2,
    titlestr="", xlblstr="alpha", ylblstr = "penalty", legendstrs=["true","approx."], fn=nothing)
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    A, b = plotfig(Wlast, Hlast, x_prev, d, alpha0, β, ax; show_alpha=show_alpha,
            titlestr=titlestr, legend_loc=legend_loc, legendstrs=legendstrs, xlblstr=xlblstr, ylblstr=ylblstr)
    fn != nothing && savefig(fn)
    A, b
end

function plotfig(Wlast, Hlast, x_prev, d, alpha0, β, ax; show_alpha=true, legend_loc=2,
        titlestr="", xlblstr="alpha", ylblstr = "penalty", legendstrs=["true","approx."], alpharngfac=1.5)
    αs = range(-alpharngfac*alpha0, alpharngfac*alpha0, length=3001)
    p = size(Wlast,2)
    OT = β*POCA.add_orthogonality(p)
    
    M_prev = POCA.reshapex(x_prev,(p,p))
    Wnew = Wlast*M_prev; Hnew = M_prev\H; pWH = scapair(Wnew,Hnew)
    A, b, _ = POCA.buildproblem(Wnew, Hnew, OT; order=0, β=β)
    approxobj(x, A, b) = (dx = x - x_prev; M = POCA.reshapex(x_prev,(p,p)); pWH + 2*b'*dx + dx'*A*dx + β*pWH*norm(M'M-I)^2)
    truefn(α, Wlast, Hlast) = (x = x_prev+α*d; M = POCA.reshapex(x,(p,p)); scapair(M, Wlast, Hlast) + β*pWH*norm(M'M-I)^2)
    approxfn(α, Wlast, Hlast) = (x = x_prev+α*d; approxobj(x, A, b))
    c = 0.9; cbx = c*b'*d
    linefn(α) = pWH + α*cbx

    objs = [truefn(α, Wlast, Hlast) for α in αs]
    objsapprox = [approxfn(α, Wlast, Hlast) for α in αs]
    ax.plot(αs, [objs objsapprox])
    show_alpha && ax.plot([alpha0,alpha0],[min(NaNMath.minimum(objs),NaNMath.minimum(objsapprox)),max(NaNMath.maximum(objs),NaNMath.maximum(objsapprox))])
    ax.legend(legendstrs,fontsize = 12,loc=legend_loc)
    ax.set_title(titlestr)
    ax.set_xlabel(xlblstr,fontsize = 12)
    ax.set_ylabel(ylblstr, fontsize = 12)
    A, b
end

β = 0.7; alpha0 = 1
Wlast = copy(W); Hlast = copy(H); fx = scapair(W, H)

p = size(W,2)
legendstrs=["true","approx."]; xlblstr="alpha"; ylblstr="penalty"
initial_M = zeros(p,p)+I; initial_x = vec(initial_M)
A,b = plotfig(Wlast, Hlast, initial_x, rand(p^2), alpha0, β, show_alpha=false,
    titlestr="random dirction", legendstrs=legendstrs, xlblstr=xlblstr, ylblstr=ylblstr)
plotfig(Wlast, Hlast, initial_x, -b, 0.00001, β, show_alpha=false,
    titlestr="-gradient dirction", legendstrs=legendstrs, xlblstr=xlblstr, ylblstr=ylblstr)
plotfig(Wlast, Hlast, initial_x, -A\b, 1, β, show_alpha=false,
    titlestr="-A\\b dirction", legendstrs=legendstrs, xlblstr=xlblstr, ylblstr=ylblstr)


iters = 5
plotcols = 5
cal_plotrows(iter,plotcols) = (iter-1)÷plotcols +1
plotrows = cal_plotrows(iters,plotcols)
plotlayout = (plotrows, plotcols); plotnum = *(plotlayout...)
fig, axs = plt.subplots(plotlayout..., figsize=(5,4),gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))

opt = Optim.Options(iterations = iterations, f_tol=1e-5, show_trace=false, show_every=10);
fg!, P = POCA.prepare_fg_full(W, H; useprecond=true, β=β, allcompW=false, allcompH=false)
@test fg!(1,nothing,initial_x) == scapair(initial_M,W,H,β=β*fx)
f = Optim.only_fg!(fg!) # only_fg!(fg) = InplaceObjective(fdf=fg) NLSolversBase\src\incomplete.jl 26
alphaguess = LineSearches.InitialHagerZhang();
linesearch = LineSearches.HagerZhang();
eta = 0.4;
precondprep = (P, x) -> nothing;
manifold=Flat()
method0 = ConjugateGradient(eta, P, precondprep, Optim._alphaguess(alphaguess), linesearch, manifold)
options = opt; inplace = true; autodiff = :finite
d = Optim.promote_objtype(method0, initial_x, autodiff, inplace, f); # Optim\multivariate\optimize\interface.jl 64
# optimize(d, initial_x, method0, options, Optim.initial_state(method0, options, d, initial_x)) # Optim\multivariate\optimize\optimize.jl 32
    state = Optim.initial_state(method0, options, d, initial_x)
    stopped = false
    f_converged, g_converged = Optim.initial_convergence(d, state, method0, initial_x, options)
    converged = f_converged || g_converged
    iteration = 0

    ls_success = true
    while !converged && !stopped && iteration < options.iterations
        iteration += 1
        @show iteration, norm(Optim.gradient(d))
        # ls_success = !Optim.update_state!(d, state, method0)
            # Maintain a record of the previous gradient
            copyto!(state.g_previous, Optim.gradient(d))
            # Determine the distance of movement along the search line
            lssuccess = Optim.perform_linesearch!(state, method0, Optim.ManifoldObjective(method0.manifold, d)) # Optim\src\utilities\perform_linesearch.jl 41
            state.x = state.x + state.alpha * state.s
            Optim.value_gradient!(d, state.x)
            g_x = Optim.gradient(d)
            f_x = Optim.value(d)
            xlblstr = cal_plotrows(iteration, plotcols) == plotrows ? "alpha" : ""
            ylblstr = iteration%plotcols == 1 ? "penalty" : ""
            A, b = plotfig(Wlast, Hlast, state.x_previous,state.s, state.alpha, β, axs[iteration];
                alpharngfac=10, show_alpha=true, xlblstr=xlblstr, ylblstr=ylblstr, titlestr="iteration $iteration")
            M = POCA.reshapex(state.x,(p,p)); @show norm(M'M-I)^2

            dPd = real(dot(state.s, method0.P, state.s))
            etak = method0.eta * real(dot(state.s, state.g_previous)) / dPd # New in HZ2013
            state.y .= Optim.gradient(d) .- state.g_previous
            ydots = real(dot(state.y, state.s))
            copyto!(state.py, state.pg)        # below, store pg - pg_previous in py
            ldiv!(state.pg, method0.P, Optim.gradient(d))
            state.py .= state.pg .- state.py
            # ydots may be zero if f is not strongly convex or the line search does not satisfy Wolfe
            betak = (real(dot(state.y, state.pg)) - real(dot(state.y, state.py)) * real(dot(Optim.gradient(d), state.s)) / ydots) / ydots
            # betak may be undefined if ydots is zero (may due to f not strongly convex or non-Wolfe linesearch)
            beta = NaNMath.max(betak, etak) # TODO: Set to zero if betak is NaN?
            state.s .= beta.*state.s .- state.pg
            Optim.project_tangent!(method0.manifold, state.s, state.x)
            lssuccess == false # break on linesearch error
        Optim.update_g!(d, state, method0)
        g_converged, f_increased = Optim.assess_convergence(state, d, options)   
    end


#================== Follow up first several iterations =================#
# Case 1 CG :
β = 0.7
αs = -15:0.01:15
Wlast = copy(W); Hlast = copy(H)
p = size(Wlast,2)
fx = scapair(Wlast,Hlast)

fg!, P = POCA.prepare_fg_full(W, H; useprecond=false, β=β, allcompW=false, allcompH=false);
dM = zeros(p,p)+I; x = vec(dM)
G = zeros(p^2)
fg!(nothing, G, x); 
M = POCA.reshapex(-G,(p,p))
truefn(α, Wlast, Hlast) = (α==0 && return NaN; scapair(α*M, Wlast, Hlast, β=β*fx; γ=0))
fig, ax = plt.subplots(1,1, figsize=(5,4))
trueobjs = [truefn(α, Wlast, Hlast) for α in αs]
ax.plot(αs, trueobjs)

dM = zeros(p,p); x = vec(dM)
αs = -1.5:0.001:1.5
OT = β*POCA.add_orthogonality(p)
A, b, _ = POCA.buildproblem(Wlast, Hlast, OT; order=0, β=β)
#x = -A\b
#x = -b
dM = POCA.reshapex(x,(p,p))
approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x
truefn(α, Wlast, Hlast) = scapair(I + α*dM, Wlast, Hlast, β=β*fx; γ=0)
approxfn(α, Wlast, Hlast) = approxobj(α*vec(dM), scapair(Wlast,Hlast; γ=0), A, b)
linefn(α) = fx + α*cbx

fig, ax = plt.subplots(1,1, figsize=(5,4))
objs = [truefn(α, Wlast, Hlast) for α in αs]
objsapprox = [approxfn(α, Wlast, Hlast) for α in αs]
length(legendstrs) > 2 && (objsline = [linefn(α) for α in αs])
length(legendstrs) > 2 ? ax.plot(αs, [objs objsapprox objsline]) : ax.plot(αs, [objs objsapprox])
ax.legend(legendstrs,fontsize = 12,loc=4)
xlabel(xlblsrt,fontsize = 12)
ylabel(ylblstr, fontsize = 12)


#================== Why case 1 doesn't work  =================#

A, b, _ = POCA.buildproblem(W, H, OT; order=0, β=β)
dM = POCA.reshapex(-2b,(p,p))
function plotfig(Wlast, Hlast, dM, legendstrs, xlblsrt, ylblstr, fn=nothing)
    αs = -1.5:0.001:1.5
    p = size(Wlast,2)
    fx = scapair(Wlast,Hlast)
    OT = β*POCA.add_orthogonality(p)
    A, b, _ = POCA.buildproblem(Wlast, Hlast, OT; order=0, β=β)
    approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x
    truefn(α, Wlast, Hlast) = scapair(I + α*dM, Wlast, Hlast, β=β*fx; γ=0)
    approxfn(α, Wlast, Hlast) = approxobj(α*vec(dM), scapair(Wlast,Hlast; γ=0), A, b)
    linefn(α) = fx + α*cbx

    fig, ax = plt.subplots(1,1, figsize=(5,4))
    objs = [truefn(α, Wlast, Hlast) for α in αs]
    objsapprox = [approxfn(α, Wlast, Hlast) for α in αs]
    length(legendstrs) > 2 && (objsline = [linefn(α) for α in αs])
    length(legendstrs) > 2 ? ax.plot(αs, [objs objsapprox objsline]) : ax.plot(αs, [objs objsapprox])
    ax.legend(legendstrs,fontsize = 12,loc=4)
    xlabel(xlblsrt,fontsize = 12)
    ylabel(ylblstr, fontsize = 12)
    fn != nothing && savefig(fn)
end
legendstrs = ["approx. fn", "approx. fn"]
xlblsrt = "α"
ylblstr = "penalty"
fn = "true_vs_approx.png"
plotfig(W, H, dM, legendstrs, xlblsrt, ylblstr)


#================== Test case 1 : semiscasolve!_full  =================#
β=5; objtol = 1e-7; maxiter = 1


function sca_full(W,H,M,β0)
    Wn = W*M; Hn = svd(M)\H
    pWn = sca2(Wn); pHn = sca2(Hn)
    pOWHn = orthogW(Wn)*orthogH(Hn)
    pWn*pHn + β0*pOWHn
end

function sca_full_M(W,H,M,βpWH0)
    Wn = W*M; Hn = svd(M)\H
    pWn = sca2(Wn); pHn = sca2(Hn); pOn = orthogM(M)
    pWn*pHn + βpWH0*pOn
end

function FD_fg!(F, G, x)
    if G != nothing
        g = ForwardDiff.gradient(trueWHOf,x)
        copyto!(G,g)
    end
    if F != nothing
        f = trueWHOf(x) 
        return f
    end
end
trueOf(x) = (M = POCA.reshapex(x,(p,p)); norm(M'M-I)^2)
trueWHOf(x) = (M = POCA.reshapex(x,(p,p)); sca2(W*M)*sca2(M\H)*(1+β*trueOf(x)))

orthogW(W) = norm(offdiag!(W'W))^2
orthogH(H) = norm(offdiag!(H*H'))^2
orthogM(M) = norm(M'M-I)^2
orthogMoffdiag(M) = norm(offdiag!(M'M))^2

m, p = size(W)
x = rand(p^2); M = POCA.reshapex(x,(p,p))
G = zeros(p^2); FD_G = zeros(p^2) 
fg!, P = POCA.prepare_fg_full(W,H, β=β)
@btime fg!(1,nothing,x) # f Time 0.154ms
@btime fg!(nothing,G,x) # g Time 2.512ms

fg!, P = POCA.prepare_fg(W,H, β=β)
@btime fg!(1,nothing,x) # f Time 0.12ms
@btime fg!(nothing,G,x) # g Time 0.07ms

sca_full(W,H,Matrix(1.0I,p,p),β0) # 28069.4379 (β=10)
initial_x = vec(Matrix(1.0I,p,p));
fg!, P = POCA.prepare_fg_full(W, H; useprecond=false, β=β, allcompW=false, allcompH=false);
opt = Optim.Options(f_tol=1e-5, outer_iterations=1000, iterations=1000, show_trace=true, show_every=10);
rst, _ = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
x = rst.minimizer; M = POCA.reshapex(x, (p, p));
iter, f_calls, g_calls = rst.iterations, rst.f_calls, rst.g_calls
sca_full(W,H,M,β0) # 2.08
W1 = W*M; H1 = M\H;
sca2(W1) # 11.33
sca2(H1) # 0.18
scapair(W1,H1) # 2.07
βpH0*orthogW(W1) # 0.012

β = 0.0001; pWH0 = scapair(W,H)
βpWH0 = β*pWH0
sca_full_M(W,H,Matrix(1.0I,p,p),βpWH0) # 28069.4379 (β=10)
initial_x = vec(Matrix(1.0I,p,p));
# @btime (
(fg!, P) = POCA.prepare_fg_full(W, H; useprecond=false, βpH0=βpWH0, allcompW=false, allcompH=false);
opt = Optim.Options(f_tol=1e-5, outer_iterations=100, iterations = 3000, show_trace=true, show_every=100);
(rst, _) = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt);
x = rst.minimizer; M = POCA.reshapex(x, (p, p));
#)
iter, f_calls, g_calls = rst.iterations, rst.f_calls, rst.g_calls
sca_full_M(W,H,M,βpWH0) # 2.08
W1 = W*M; H1 = M\H;
sca2(W1) # 11.33
sca2(H1) # 0.18
scapair(W1,H1) # 2.07
βpWH0*orthogM(M) # 0.012

sca2(W) # 7.56
sca2(H) # 3713.21
norm(POCA.offdiag!(W'W)) # 2.6161485706734647e-15
norm(POCA.offdiag!(H*H')) # 3.978181811399219e-12
sca_full(W,H,Matrix(1.0I,p,p),β)
@time W1, H1, objval, iter, inner_iter, f_calls, g_calls = POCA.semiscasolve!_full(W, X; flipsign=false, usecg=true, useprecond=false, linesearch_method=:none, 
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=maxiter, objtol=objtol, showdbg=true);
sca2(W1) # 5.65
sca2(H1) # 0
norm(POCA.offdiag!(W1*W1')) # 3.42
norm(POCA.offdiag!(H1*H1')) # 2.46

# semiscasolve!_full : 1.95sec (10X slower)
# f_time : 0.154ms
# g_time : 2.512ms
# f_calls : 452
# g_calls : 446
# fcalls*f_time+g_calls*g_time : 1190msec

# semiscasolve! : 0.15sec
# f_time : 0.12ms
# g_time : 0.07ms
# f_calls : 477
# g_calls : 243
# fcalls*f_time+g_calls*g_time : 74msec


#================== Test if ∇E(M; W) ≈ ∇E(I+0;WM)  =================#
p = size(W,2)
truef(x) = (M = POCA.reshapex(x,(p,p)); scapair(M,W,H))

function updatedf(x; order=0, β=0)
    l = length(x)
    p = Int(sqrt(l))
    M = POCA.reshapex(x,(p,p))
    Wlast = W*M; Hlast = M\H
    scapair(Wlast, Hlast)
end

function gradapproxf(x; order=0, β=0)
    l = length(x)
    p = Int(sqrt(l))
    M = POCA.reshapex(x,(p,p))
    Wlast = W*M; Hlast = M\H
    fg!, P = POCA.prepare_fg(Wlast,Hlast;β=0)
    g = zeros(p^2)
    fg!(nothing,g,zeros(p^2))
    g
end

function approxf(dx; order=0, β=0)
    l = length(dx)
    p = Int(sqrt(l))
    dM = POCA.reshapex(dx,(p,p))
    Wlast = W*(I+dM); Hlast = (I-dM)*H
    scapair(Wlast,Hlast)
end

function my_gradient(f::Function,x,dx=1e-10)
    l = length(x)
    p = Int(sqrt(l))
    M = POCA.reshapex(x,(p,p))
    g = zeros(l)
    for idx = 1:l
        xpdx = copy(x)
        xpdx[idx] += dx
        MpdM = POCA.reshapex(xpdx,(p,p))
        g[idx] = (f(MpdM,W,H)-f(M,W,H))/dx
    end
    g
end

x = rand(p^2); M = POCA.reshapex(x,(p,p))
@btime gtruef = ForwardDiff.gradient(truef,x); # 17.8ms
@btime cgtruef = Calculus.gradient(truef,x); # 65.4ms
@btime gf = my_gradient(scapair,x); # 65.8ms
@btime gapproxf = gradapproxf(x); # 0.5ms 2.5X faster even this is full WH gradient but value is different 

gtruef[1:10]
cgtruef[1:10]
gf[1:10]
gapproxf[1:10]

trueW(x) = (M = POCA.reshapex(x,(p,p)); sca2(W*M))

function gradWH(x,Wlast,Hlast)
    m, p = size(Wlast)
    p, n = size(Hlast)
    M = POCA.reshapex(x,(p,p))
    W = Wlast*M; H = M\Hlast
    bW = vec((W.<0).*W); bH = vec((H.<0).*H)
    AWTbW = POCA.cal_AWTb(W, bW); AHTbH = POCA.cal_AHTb(H, bH)
    pW, pH = sca2(W, allcomp = false), sca2(H, allcomp = false)
    gW = 2*AWTbW
    gH = 2*AHTbH
    gWH = 2*(pH*AWTbW+pW*AHTbH)
    gW, gH, gWH
end

function gradientW0(x,W,dx=1e-10)
    m,p = size(W)
    M = reshapex(x,(p,p))
    WM = W*M; WM0 = copy(WM); WM0[WM0.>0].=0; WM0 .^= 2 # WM0 = map(a -> a < 0 ? a^2 : 0, WM) 
    g = zeros(p,p)
    for i = 1:p, j = 1:p
        oldcolj = copy(WM0[:,j])
        oldsumj = sum(oldcolj)
        for k = 1:m
            newWMkj = WM[k,j]+W[k,i]*dx
            WM0[k,j] = newWMkj < 0 ? newWMkj^2 : 0
        end
        newsumj = sum(WM0[:,j])
        g[i,j] = (newsumj-oldsumj)/dx
        WM0[:,j] = oldcolj
    end
    vec(g)
end

gtrueW = Calculus.gradient(trueW,x)
gaprxW, _ = gradWH(x,W,H)
M = POCA.reshapex(x,(p,p))
gaprxWdivM = vec(POCA.reshapex(gaprxW,(p,p))/M)

function detj(A,j)
    length(A) == 1 && return A[1]
    s = 0
    for i in 1:size(A,1)
#        detminorij = detj(A[[1:i-1...,i+1:end...], [1:j-1...,j+1:end...]],1)
        detminorij = det(A[[1:i-1...,i+1:end...], [1:j-1...,j+1:end...]])
        s += (i+j)%2 == 0 ? A[i,j]*detminorij : -A[i,j]*detminorij 
        @show i, s
    end
    s
end

using Test
deltaij(i,j,dx,p) = (d=zeros(p,p); d[i,j] = dx; d)
x = rand(p^2); M = POCA.reshapex(x,(p,p))
p = size(W,2); dx = 1e-7
invM = inv(M); 
for i = 1:p, j = 1:p
    invMdelta = invM*deltaij(i,j,dx,p)
    IpinvMdelta = I + invMdelta
    @test det(IpinvMdelta) ≈ IpinvMdelta[j,j]
    @test det(IpinvMdelta) ≈ 1+dx*invM[j,i]
    invIpinvMdelta = inv(IpinvMdelta)
    invIpinvMdelta_fast = copy(IpinvMdelta)
    invIpinvMdelta_fast[j,j] = -1
    invIpinvMdelta_fast[:,j] /= -IpinvMdelta[j,j]
    @test invIpinvMdelta ≈ invIpinvMdelta_fast
    @show invIpinvMdelta ≈ Matrix(1.0I,p,p)
end

dx = 1e-8; β = 0.7

trueWf(x) = (M = POCA.reshapex(x,(p,p)); sca2(W*M))
trueHf(x) = (M = POCA.reshapex(x,(p,p)); sca2(M\H))
trueOWf(x) = (M = POCA.reshapex(x,(p,p)); WM = W*M; WMTWM = WM'WM;
    foreach(i->WMTWM[i,i]=0, 1:p); norm(WMTWM)^2)
trueOHf(x) = (M = POCA.reshapex(x,(p,p)); MH = M\H; MHMHT = MH*MH';
    foreach(i->MHMHT[i,i]=0, 1:p); norm(MHMHT)^2)
trueOoffdiagf(x) = (M = POCA.reshapex(x,(p,p)); norm(offdiag!(M'M))^2)
trueOf(x) = (M = POCA.reshapex(x,(p,p)); norm(M'M-I)^2)
trueWHOWf(x) = (M = POCA.reshapex(x,(p,p)); sca2(W*M)*sca2(M\H)+β*trueOWf(x))
trueWHOWHf(x) = (M = POCA.reshapex(x,(p,p)); sca2(W*M)*sca2(M\H)+β*trueOWf(x)*trueOHf(x))
trueWHOf(x) = (M = POCA.reshapex(x,(p,p)); sca2(W*M)*sca2(M\H)*(1+β*trueOf(x)))

fgtrueWf = ForwardDiff.gradient(trueWf,x); # 14.4ms
cgtrueWf = Calculus.gradient(trueWf,x); # 25.4ms
gWf, _ = POCA.gradientW(x,W,dx); # 0.9ms
norm(fgtrueWf-cgtrueWf) # 3.494942216123217e-8
norm(fgtrueWf-gWf) # 9.339864075159185e-7
norm(cgtrueWf-gWf) # 9.380979701169044e-7

x = rand(p^2); M = POCA.reshapex(x,(p,p))
fgtrueHf = ForwardDiff.gradient(trueHf,x); # 2.1ms
cgtrueHf = Calculus.gradient(trueHf,x); # 30.4ms
gHf, _ = POCA.gradientH(x,H,dx); # 1.3ms
norm(fgtrueHf-cgtrueHf) # 0.0001815
norm(fgtrueHf-gHf) # 0.0118137
norm(cgtrueHf-gHf) # 0.0118409

fgtrueOWf = ForwardDiff.gradient(trueOWf,x); # 24.7ms
cgtrueOWf = Calculus.gradient(trueOWf,x); # 17.9ms
gOWf, _ = POCA.gradientOW(x,W,dx); # 0.06ms
norm(fgtrueOWf-cgtrueOWf) # 5.355982380769943e-7
norm(fgtrueOWf-gOWf) # 1.1802220594907603e-6
norm(cgtrueOWf-gOWf) # 1.3004798594880845e-6

fgtrueOHf = ForwardDiff.gradient(trueOHf,x); # 3.55ms
cgtrueOHf = Calculus.gradient(trueOHf,x); # 34.06ms
gOHf, _ = POCA.gradientOH(x,H,dx); # 1.26ms
norm(fgtrueOHf-cgtrueOHf) # 0.02058356596212234
norm(fgtrueOHf-gOHf) # 11.325378957838947
norm(cgtrueOHf-gOHf) # 11.329580898530292

fgtrueOoffdiagf = ForwardDiff.gradient(trueOoffdiagf,x); # 0.19ms
gtrueOoffdiagf = Calculus.gradient(trueOoffdiagf,x); # 0.52ms
gOoffdiagf, _ = POCA.gradientOoffdiag(x,W,dx); # 0.055ms
norm(fgtrueOoffdiagf-cgtrueOoffdiagf) # 3.4236625281577873e-12
norm(fgtrueOoffdiagf-gOoffdiagf) # 1.1508381949771286e-6
norm(cgtrueOoffdiagf-gOoffdiagf) # 1.150838664978935e-6

@btime fgtrueOf = ForwardDiff.gradient(trueOf,x); # 0.21ms
@btime gtrueOf = Calculus.gradient(trueOf,x); # 0.58ms
@btime gOf, _ = POCA.gradientO(x,W,dx); # 0.009ms
norm(fgtrueOf-cgtrueOf) # 1472.919958843174
norm(fgtrueOf-gOf) # 2.650055756413107e-6
norm(cgtrueOf-gOf) # 1472.9199614867803

fgtrueWHOWf = ForwardDiff.gradient(trueWHOWf,x); # 41.9ms
cgtrueWHOWf = Calculus.gradient(trueWHOWf,x); # 85.4ms
gWHOWf = ((gWf, pW) = POCA.gradientW(x,W,dx);
        (gHf, pH) = POCA.gradientH(x,H,dx);
        (gOWf, _) = POCA.gradientOW(x,W,dx);
        (pH*gWf+pW*gHf)+β*gOWf); # 2.3ms
norm(fgtrueWHOWf-cgtrueWHOWf) # 0.00655
norm(fgtrueWHOWf-gWHOWf) # 0.42316
norm(cgtrueWHOWf-gWHOWf) # 0.42406

fgtrueWHOWHf = ForwardDiff.gradient(trueWHOWHf,x); # 48.2ms
cgtrueWHOWHf = Calculus.gradient(trueWHOWHf,x); # 124.0.4ms
gWHOWHf = ((gWf, pW) = POCA.gradientW(x,W,dx);
        (gHf, pH) = POCA.gradientH(x,H,dx);
        (gOWf, pOW) = POCA.gradientOW(x,W,dx);
        (gOHf, pOH) = POCA.gradientOH(x,H,dx);
        (pH*gWf+pW*gHf)+β*(pOH*gOWf+pOW*gOHf)); # 3.8ms
norm(fgtrueWHOWHf-cgtrueWHOWHf) # 3.187603302955668e-5
norm(fgtrueWHOWHf-gWHOWHf) # 0.0163
norm(cgtrueWHOWHf-gWHOWHf) # 0.0163

fgtrueWHOf = ForwardDiff.gradient(trueWHOf,x); # 41.9ms
cgtrueWHOf = Calculus.gradient(trueWHOf,x); # 85.4ms
gWHOf = ((gWf, pW) = POCA.gradientW(x,W,dx);
        (gHf, pH) = POCA.gradientH(x,H,dx);
        (gOf, _) = POCA.gradientO(x,W,dx);
        (pH*gWf+pW*gHf)+β*pW*pH*gOf); # 2.3ms
norm(fgtrueWHOf-cgtrueWHOf) # 0.00655
norm(fgtrueWHOf-gWHOf) # 0.42316
norm(cgtrueWHOf-gWHOf) # 0.42406

ncellss = 6:2:80
rtWs = []; rtHs = []; rtOWs = []; rtAlls = []
for ncells in ncellss # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X)
    W = F.U[:,1:ncells]
    H = W\X
    x = rand(ncells^2)
    rtW = @belapsed gWf, _ = POCA.gradientW(x,W,dx)
    rtH = @belapsed gHf, _ = POCA.gradientH(x,H,dx)
    rtOW = @belapsed gOWf = POCA.gradientOW(x,W,dx)
    rtAll = @belapsed gWHOWf = ((gWf, pW) = POCA.gradientW(x,W,dx);
        (gHf, pH) = POCA.gradientH(x,H,dx);
        gOWf = POCA.gradientOW(x,W,dx);
        (pH*gWf+pW*gHf)+β*gOWf)
    push!(rtWs,rtW)
    push!(rtHs,rtH)
    push!(rtOWs,rtOW)
    push!(rtAlls,rtAll)
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss,[rtWs rtHs rtOWs rtAlls])
ax1.legend(["∇Ew", "∇Eh", "∇Eow","∇E"],fontsize = 12,loc=2)
xlabel("number of components",fontsize = 12)
ylabel("runtime", fontsize = 12)
savefig("noc_vs_case1_gradient.png")

# older prepare_fg using I+dM approximation takes 0.15ms
fg!, P = POCA.prepare_fg(W,H;β=β)
g = zeros(p^2)
@btime fg!(1,g,x) # 0.15ms


#================== Why not linear for factor =================#

factorrng = 1:40
ncells = 14; fovsz=(40,20); lengthT0 = 100
C = 1:1:80; plotxrng = factorrng; plotxstr = "factor"; xlabelstr="factor" # imgszyrng = 30:10:100, 40
maxiter = 1000

datart1sca=[]; datart2sca=[]; datart3sca=[]; mssdssca=[]
iterss=[]; inner_iterss=[]; penaltyss=[]; f_callss=[]; g_callss=[]
for factor in factorrng
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz, save=false)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]
    runtime1 = @elapsed F = svd(X)
    push!(datart1sca, runtime1)
    mssdsca=[]; iters=[]; inner_iters=[]; penaltys=[]
    for β in [0.7] # best β = 5.0
        println("beta=$β")
        runtime2 = @elapsed  W1, H1, objval, iter, inner_iter, f_calls, g_calls = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
                        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=maxiter, objtol=objtol, showdbg=false);
        runtime3 = @elapsed  W2, H2, objval2, iter2, inner_iter2, f_calls2, g_calls2 = semiscasolve!(F, X, ncells; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none, 
                        order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=maxiter, objtol=objtol, showdbg=false);
        rt = runtime1+runtime2
        normalizeWH!(W1,H1)
        mssd1, ml1, ssds1 = matchedfiterr(gtW,W1)
        push!(datart2sca, runtime2)
        push!(datart3sca, runtime3)
        push!(mssdsca, mssd1)
        push!(iters,iter[1])
        push!(inner_iters, mean(inner_iter[1]))
        push!(penaltys, objval[1][end])
        push!(f_callss, sum(f_calls[1]))
        push!(g_callss, sum(g_calls[1]))
        #imshowW(W1,imgsz)
        imsaveW("fc_sz$(imgsz)_lT$(lengthT)_SPOCA_CG__b$(β).png",W1,imgsz)
    end
    push!(mssdssca, mssdsca)
    push!(iterss, iters)
    push!(inner_iterss, inner_iters)
    push!(penaltyss, penaltys)
end
save("091621_CG_factorrng.jld","datart1sca",datart1sca,"datart2sca",datart2sca,"datart3sca", datart3sca,"mssdssca",mssdssca,"iterss",iterss,"inner_iterss",inner_iterss,
    "penaltyss",penaltyss,"f_callss",f_callss,"g_callss",g_callss,"factorrng",factorrng)


#================== ProfileView =================#

(buffersize, delay) = Profile.init()
m = 10; Profile.init(buffersize*m, delay/m)

β = 0.7; objtol = 1e-7
ncells = 14; fovsz=(40,20); lengthT0 = 100
factorrng = 1:1:40
# factor = 15
factor = 15
imgsz = (fovsz[1]*factor,fovsz[2])
lengthT = lengthT0*factor
println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
X, F, U, W, H, imgsz, nc, fakecells_dic, img_nl = loadfakecell("fc_sz$(imgsz)_lT$(lengthT).jld"; fovsz=fovsz, ncells=ncells, lengthT=lengthT, imgsz=imgsz, save=false)
@profview W1, H1, objval, iter, inner_iters = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none,
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1, objtol=objtol, showdbg=true)

# factor = 40



#============= Preconditioning ========================#
using Test

p = size(W,2)
Hw = zeros(p^2,p^2); gw = zeros(p^2)
Hh = zeros(p^2,p^2); gh = zeros(p^2)
POCA.add_direct!(Hw,gw,W;allcomp=false);
POCA.add_transpose!(Hh,gh,H;allcomp=false);
@test POCA.cal_diagAWTAW(W) ≈ diag(Hw)
@test POCA.cal_diagAHTAH(H) ≈ diag(Hh)
OTO = POCA.add_orthogonality(p);
@test POCA.cal_diagOTO(eltype(W),p) ≈ diag(OTO)

β = 0.7
m, p = size(W); p, n = size(H)
OT = β != 0 ? β*POCA.add_orthogonality(p) : zeros(p^2,p^2);
(A, b, AW, bW, AH, bH, pW, pH) = POCA.buildproblem(W, H, OT; order=0, β=β);
diagA = 2*(pH*POCA.cal_diagAWTAW(W)+pW*POCA.cal_diagAHTAH(H)+pW*pH*β*POCA.cal_diagOTO(eltype(W),p));
@test diagA ≈ 2*diag(A)

evs = eigen(A).values;
k_before = maximum(evs)/minimum(evs) # 492.46

evs_after = eigen(diagm(diagA)\A).values;
k_after = maximum(evs_after)/minimum(evs_after) # 377.12

ncellsrng = 5:50; ks = []; ks_after = []
for ncells in ncellsrng # true = 20 including bg
    println("ncells=$ncells")
    runtime1 = @elapsed F = svd(X)
    W = F.U[:,1:ncells]; H = W\X
    p = ncells
    OT = β != 0 ? β*POCA.add_orthogonality(p) : zeros(p^2,p^2);
    (A, b, AW, bW, AH, bH, pW, pH) = POCA.buildproblem(W, H, OT; order=0, β=β);
    evs = eigen(A).values;
    k_before = maximum(evs)/minimum(evs)
    push!(ks,k_before)
    evs_after = eigen(diagm(2*diag(A))\A).values;
    k_after = maximum(evs_after)/minimum(evs_after)
    push!(ks_after,k_after)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [ks ks_after])
ax1.legend(["w/o precond.", "with precond."],fontsize = 12,loc=2)
xlabel("number of components",fontsize = 12)
ylabel("condition number", fontsize = 12)
savefig("noc_vs_condition_number.png")

# == Iteration vs penalty
objtol = 1e-7
rt1 = @elapsed W1, H1, objval, iter, inner_iters = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none,
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1000, objtol=objtol, showdbg=true)
rt2 = @elapsed W1, H1, objval_p, iter_p, inner_iters_p = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:full,
    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1000, objtol=objtol, showdbg=true)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
iter = min(iter[1], iter_p[1])
ax1.plot(0:iter+1, [objval[1][1:iter+2] objval_p[1][1:iter+2]])
ax1.legend(["w/o line search", "with line search"],fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("penalty", fontsize = 12)
savefig("iter_vs_penalty(linesearch).png")


function itervspen(W,H,β,itermax,useprecond)
    initial_x = zeros(p^2)
    fg!, P = POCA.prepare_fg(W, H; useprecond=useprecond, β=β, allcompW=false, allcompH=false)
    opt = Optim.Options(iterations=itermax,f_tol=1e-5, show_trace=false, show_every=10)
    if useprecond
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(P=P), opt)
    else
        rst = optimize(Optim.only_fg!(fg!), initial_x, ConjugateGradient(), opt)
    end
    x = rst.minimizer
    dM = POCA.reshapex(x, (p, p))
    M = I + dM
    pen = scapair(M,W,H;β=β) # fg!(1, nothing, x)+scapair(W,H) # 
    pen, rst.iterations
end

pens = []; pen_ps = []
itermaxrng = 1:25
for itermax in itermaxrng
    pen, iter = itervspen(W,H,β,itermax,false)
    pen_p, iter_p = itervspen(W,H,β,itermax,true)
    push!(pens,pen)
    push!(pen_ps,pen_p)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(itermaxrng, [pens pen_ps])
ax1.legend(["w/o precond.", "with precond."],fontsize = 12,loc=1)
xlabel("CG iteration",fontsize = 12)
ylabel("E(M;W,H,β)", fontsize = 12)
savefig("iter_vs_truepenalty(precond).png")



# == check condition number and iteration number
fg!, P = POCA.prepare_fg(W, H; useprecond=false);
opt = Optim.Options(f_tol=1e-5, show_trace=false, show_every=10);
rst = optimize(Optim.only_fg!(fg!), zeros(p^2), ConjugateGradient(P=P), opt)
rst.iterations

function iternum_of_optim(A,b;useprecond=false)
    if useprecond
        P = inv(diagm(diag(2A)))
        GA = PA = P*A; Gb = Pb = P*b
    else
        PA = A; Pb = b
    end
    f(x) = (print("f"); x'PA*x+2Pb'*x)
    g!(storage, x) = (print("g"); g = 2 .*(PA*x+Pb); copyto!(storage,g))
    h!(storage, x) = (print("h"); copyto!(storage, 2 .*PA)) # h! is not used
    rtime = @elapsed rst = optimize(f,g!,h!, zeros(p^2), ConjugateGradient(), Optim.Options(f_tol=1e-5));
    rst.iterations, rst.f_calls, rst.g_calls
end

iternum_of_optim(A,b;useprecond=false) # 0.22sec, 26 iterations
iternum_of_optim(A,b;useprecond=true)  # 0.96sec, 17 iterations

#======== ncells vs sca2 and scapair ==================================#

X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_nonorthog.jld"; ncells=14, lengthT=100)
F = svd(X)

ncellss = 6:2:100
scapairs=[]; sca2Ws=[]; sca2Hs=[]
for ncells in ncellss # true = 20 including bg
    W = F.U[:,1:ncells]
    H = W\X    
    push!(scapairs, scapair(W, H))
    push!(sca2Ws, sca2(W))
    push!(sca2Hs, sca2(H))
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, scapairs)
ax1.legend(["E(W,H)"], fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("noc_vs_penaltyWH.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, sca2Ws)
ax1.legend(["E(W)"], fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("noc_vs_penaltyW.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, sca2Hs)
ax1.legend(["E(H)"], fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("noc_vs_penaltyH.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellss, sca2Ws.*sca2Hs)
ax1.legend(["E(W,H)"], fontsize = 12,loc=2)
xlabel("number of cells",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("noc_vs_penaltyWH2.png")

#======== Orthogonality vs norm(X-W[1,:]H[:,1]) ==================================#
ncompmax = 10
βs = 0.1:0.1:1
orthoggtWs = []
orthoggtHs = []
orthoggts = []
orthogs = []
mssdscas = []
scapairs = []
for i in 0:6
    println("overlap level = $i")
    sigma = 5
    imgsz = (40,20)
    nevents = 70
    lengthT = 100
    lambda = 0 # 0.01 Possion firing rate (if 0 some fixed points)
    gt_ncells, imgrs, Wgt, Hgt, gtWimgc, gtbg = 
        gaussian2D_calcium_transient(sigma, imgsz, lengthT, lambda=lambda, decayconst=0.76+0.02*i, bias = 0.1, noiselevel = 0.1, overlaplevel=i)
    ncells = 14
    F = svd(imgrs)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    X = imgrs
    W = copy(U)
    H = W\X
    orthoggtW = norm(Wgt'*Wgt-I)/norm(Wgt)^2
    orthoggtH = norm(Hgt*Hgt'-I)/norm(Hgt)^2
    orthoggt = orthoggtW*orthoggtH
    push!(orthoggtWs,orthoggtW)
    push!(orthoggtHs,orthoggtH)
    push!(orthoggts,orthoggt)
    push!(orthogs, norm(X-W*H)/norm(W)/norm(H))
    push!(scapairs, scapair(W,H))
    if false
        mssdsca = []
        for β in βs
            mssds = []
            for i in 1:10
                W1, H1, objval, iter = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=false, linesearch_method=:none, 
                    order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1000, objtol=objtol, showdbg=false);
                normalizeWH!(W1,H1)
                mssd1, ml1, ssds1 = matchedssd(Wgt,W1)
                push!(mssds,mssd1)
            end
            push!(mssdsca, mean(mssds))
        end
        maxidx = argmin(mssdsca)
        β = βs[maxidx]
        mssd = mssdsca[maxidx]
        W1, H1, objval, iter = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=false, linesearch_method=:none, 
            order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5, itermax=1000, objtol=objtol, showdbg=false);
        normalizeWH!(W1,H1)
        imsaveW("fc_ovlp$(i)_orthog$(orthogs[end])_b$(β)_ssd$(mssd).png", W1, imgsz)
        push!(mssdscas, mssdsca)
    end
    if false
        imsaveW("nonorthg$(i)Wgt.png",Wgt[:,1:7],imgsz)
        imsaveW("nonorthg$(i)Wsvd.png",W[:,1:7],imgsz)
        clf()
        fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
        ax.plot(Hgt)
        ax.set_xlabel("time")
        ax.set_ylabel("intensity")
        plt.savefig("nonorthg$(i)Hgt.png")
    end
end

fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
ax.plot(0:6, orthoggts)
xlabel("Overlap level",fontsize = 12)
ylabel("non-orthognality",fontsize = 12)
#ax.legend(["1", "2", "3", "4", "5", "8", "9"], fontsize = 12,loc=2)
savefig("overlap_vs_NO.png")

fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
ax.plot(0:6, orthogs)
xlabel("Overlap level",fontsize = 12)
ylabel("‖X-WH‖/(‖W‖‖H‖)",fontsize = 12)
#ax.legend(["1", "2", "3", "4", "5", "8", "9"], fontsize = 12,loc=2)
savefig("overlap_vs_X-WH.png")

fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
ax.plot(orthoggts, orthogs)
xlabel("Orthogonality",fontsize = 12)
ylabel("‖X-WH‖/(‖W‖‖H‖)",fontsize = 12)
#ax.legend(["1", "2", "3", "4", "5", "8", "9"], fontsize = 12,loc=2)
savefig("Orthogonality_vs_X-WH.png")

fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
ax.plot(0:6, scapairs)
xlabel("Overlap level",fontsize = 12)
ylabel("E(W,H)",fontsize = 12)
#ax.legend(["1", "2", "3", "4", "5", "8", "9"], fontsize = 12,loc=2)
savefig("overlap_vs_scapair.png")

fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
ax.plot(βs, hcat(mssdscas...))
xlabel("β",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
ax.legend(collect(0:6), fontsize = 12,loc=2)
savefig("beta_vs_mssd.png")

#======== Fast gradient calculation for CG method ==================================#
using SparseArrays, Test

# A\b with A,b = buildproblem(W, H) : O(mp^3+p^6)
function t0(W,H; β=0.7, maxiter=5)
    m, p = size(W)
    p, n = size(H)
    x = rand(p^2)
    OT = β != 0 ? β*POCA.add_orthogonality(p) : zeros(p^2,p^2)
    (A, b, _) = POCA.buildproblem(W, H, OT; order=0, β=β);
    A\b
end

# CG with A,b = buildproblem(W, H) : O(mp^3+np^4)
function prepare_fg_buildproblem(W,H; β=0.7)
    m, p = size(W)
    p, n = size(H)
    x = rand(p^2)
    OT = β != 0 ? β*POCA.add_orthogonality(p) : zeros(p^2,p^2)
    (A, b, AWold, bWold, AHold, bHold, pWold, pHold) = POCA.buildproblem(W, H, OT; order=0, β=β);
    fold(x) = x'A*x+2b'*x;
    fg!(storage, x) = (Ax=A*x; g=(Ax+b).*2; copy!(storage,g); x'Ax+2b'*x)
    fg!
end

function t1(W,H; β=0.7, maxiter=5)
    m, p = size(W)
    p, n = size(H)
    x = rand(p^2)
    fg! = prepare_fg_buildproblem(W,H; β=β)
    st = zeros(p^2);
    for i in 1:maxiter
        fg!(st,x)
    end
end

# CG with A'Ax : O(nmp^3)
function prepare_fg_slow(W,H; β=0.7)
    AxOxAtb(AW,AH,O,bW,bH,x) = (AW*x, AH*x, O*x, AW'bW, AH'bH)
    m, p = size(W)
    p, n = size(H)
    (bW, bH) = (zeros(m*p), zeros(n*p));
    (AW, AH) = (zeros(m*p, p^2), zeros(n*p, p^2));
    POCA.direct!(AW, bW, W, allcomp = false);
    POCA.transpose!(AH, bH, H, allcomp = false);
    O =  β != 0 ? sqrt(β)*POCA.orthogonality(p) : zeros(p^2,p^2)
    # AW = sparse(AW); bW = sparse(bW); AH = sparse(AH); bH = sparse(bH);
    (pW, pH) = (sca2(W, allcomp = false), sca2(H, allcomp = false));
    fg!(storage, x) = ((AWx, AHx, Ox, AWTbW, AHTbH) = AxOxAtb(AW,AH,O,bW,bH,x);
            g = 2(pH*(AW'AWx)+pW*(AH'AHx)+pH*AWTbW+pW*AHTbH+(O'Ox)*pW*pH);
            copy!(storage,g);
            pH*(AWx'AWx)+pW*(AHx'AHx)+(Ox'Ox)*pW*pH+2(pH*(AWTbW'*x)+pW*(AHTbH'*x)))
    fg!
end

function  t2(W,H; β=0.7, maxiter=5)
    m, p = size(W)
    p, n = size(H)
    x = rand(p^2)
    fg! = prepare_fg_slow(W,H; β=β)
    st = zeros(p^2);
    for i in 1:maxiter
        fg!(st,x)
    end
end 

# CG with A'Ax = vec(W'*WM) : O(nmp^2)
function  t3(W,H; β=0.7, maxiter=5)
    m, p = size(W)
    p, n = size(H)
    x = rand(p^2)
    fg! = POCA.prepare_fg(W,H;β=β)
    st = zeros(p^2);
    for i in 1:maxiter
        fg!(st,x)
    end
end 

β=0.7; maxiter = 100

# test if how f and g! are accurate
m, p = size(W)
p, n = size(H)

st = zeros(p^2);
fgbp! = prepare_fg_buildproblem(W,H; β=β)   # O(mp^3+np^4) CG with A,b
fgsl! = prepare_fg_slow(W,H; β=β)           # O(nmp^3) CG with Ax
fg! = POCA.prepare_fg(W,H;β=β)              # O(nmp^2) CG with vec(WM)
x = rand(p^2)
f1 = fgbp!(st,x); g1 = copy(st)
f2 = fgsl!(st,x); g2 = copy(st)
f3 = fg!(st,x);   g3 = copy(st)
f2 ≈ f1
g2 ≈ g1
f3 ≈ f1
g3 ≈ g1

# runtime test when ncells = 14
@btime t0(W, H; maxiter = maxiter); #   6.6msec # O(mp^3+p^6) A\b
@btime t1(W, H; maxiter = maxiter); #   8.0msec # O(mp^3+np^4) CG with A,b
@btime t2(W, H; maxiter = maxiter); # 238.3msec # O(nmp^3) CG with Ax
@btime t3(W, H; maxiter = maxiter); #  13.8msec # O(nmp^2) CG with vec(WM)
# runtime test when ncells = 100
@btime t0(W, H; maxiter = maxiter); #  13.43sec # O(mp^3+p^6) A\b
@btime t1(W, H; maxiter = maxiter); #  14.31sec # O(mp^3+np^4) CG with A,b
@btime t2(W, H; maxiter = maxiter); # 143.85sec # O(nmp^3) CG with Ax
@btime t3(W, H; maxiter = maxiter); #   0.18sec # O(nmp^2) CG with vec(WM)

x = rand(p^2)
(bW, bH) = (zeros(m*p), zeros(n*p));
(AW, AH) = (zeros(m*p, p^2), zeros(n*p, p^2));
POCA.direct!(AW, bW, W, allcomp = false);
POCA.transpose!(AH, bH, H, allcomp = false);

AWx = POCA.cal_AWx(W,x; allcomp=false)
AWTbW = zeros(Float64,p^2)
POCA.cal_AWTb!(AWTbW, AW', bW)
AWTbWf = POCA.cal_AWTb(W,bW)
AWTbW ≈ AWTbWf

AHx = POCA.cal_AHx(H,x; allcomp=false)
AHTbH = zeros(Float64,p^2)
POCA.cal_AHTb!(AHTbH, AH', bH)
AHTbHf = POCA.cal_AHTb(H,bH)
AHTbH ≈ AHTbHf

#======== Preconditioning ==================================#
β = 0.7
p = size(W, 2)
OT = β != 0 ? β*POCA.add_orthogonality(p) : zeros(p^2,p^2)
# Compute the gradient and Hessian
A, b = POCA.buildproblem(W, H, OT; β=0.7)
initial_x = zeros(p^2)
P = Optim.InverseDiagonal(diag(A))
f(x) = x'A*x+2b'*x
g!(storage, x) = (g = 2 .*(A*x+b); for (i,gi) in enumerate(g) storage[i] = gi end)
@time x = optimize(f,g!, initial_x, ConjugateGradient(P=P), Optim.Options(f_tol=1e-5));
@time x = optimize(f,g!, initial_x, ConjugateGradient(), Optim.Options(f_tol=1e-5));

# with preconditioning
rt1 = @elapsed W1, H1, objval, iters = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, linesearch_method=:none,
    order=0, fixposdef=false, skew=false, β=0.7, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_cg_precond_rt$(rt).png", W1,imgsz)

# without preconditioning
rt0 = @elapsed W1, H1, objval, iters = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=false, linesearch_method=:none,
    order=0, fixposdef=false, skew=false, β=0.7, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=true);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_cg_rt$(rt).png", W1,imgsz)

# ncells=50
# A\b : 17.15sec(37iter, ncells:50), 294.05sec(38iter, ncells:100)
# CG : 30.34sec(44iter, ncells:50), 175.39sec(49iter, ncells:100)
#====== compare scastep vs scastep_cg with many datasets ==================================#
beta=0.7
drng = 1:1
nocrng = 2:30
rts0=zeros(length(nocrng),length(drng))
rts1=zeros(length(nocrng),length(drng))
rts2=zeros(length(nocrng),length(drng))
iters0=zeros(length(nocrng),length(drng))
iters1=zeros(length(nocrng),length(drng))
iters2=zeros(length(nocrng),length(drng))
ssds0=zeros(length(nocrng),length(drng))
ssds1=zeros(length(nocrng),length(drng))
ssds2=zeros(length(nocrng),length(drng))
for i in drng
    println("dataset=$i")
    fname = "fc_nonorth_o0_PFfalse_skfalse_c0.5_ucfalse_r0_updrfalse_($i).jld"
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(fname)
    for (j,ncells) in enumerate(nocrng)
        print("$ncells ")
        F = svd(X)
        U = F.U[:,1:ncells]
        T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
        rt0 = @elapsed W1, H1, objval0, iter0 = semiscasolve!(U, X; flipsign=false, usecg=false, useprecond=false, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
            itermax=1000, objtol=1e-4, showdbg=false);
        normalizeWH!(W1,H1)
        mssdssca0, mlssca0, _ = matchedssd(gtW,W1)
        rt1 = @elapsed W1, H1, objval1, iter1 = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=false, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
            itermax=1000, objtol=1e-4, showdbg=false);
        normalizeWH!(W1,H1)
        mssdssca1, mlssca1, _ = matchedssd(gtW,W1)
        rt2 = @elapsed W1, H1, objval2, iter2 = semiscasolve!(U, X; flipsign=false, usecg=true, useprecond=true, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
            itermax=1000, objtol=1e-4, showdbg=false);
        normalizeWH!(W1,H1)
        mssdssca2, mlssca2, _ = matchedssd(gtW,W1)
        rts0[j,i] = rt0
        rts1[j,i] = rt1
        rts2[j,i] = rt2
        iters0[j,i] = iter0[1]
        iters1[j,i] = iter1[1]
        iters2[j,i] = iter2[1]
        ssds0[j,i] = mssdssca0
        ssds1[j,i] = mssdssca1
        ssds2[j,i] = mssdssca2
    end
    println("")
end
rt0 = mean(rts0,dims=2)
rt1 = mean(rts1,dims=2)
rt2 = mean(rts2,dims=2)
iter0 = mean(iters0,dims=2)
iter1 = mean(iters1,dims=2)
iter2 = mean(iters2,dims=2)
mssdssca0 = mean(ssds0,dims=2)
mssdssca1 = mean(ssds1,dims=2)
mssdssca2 = mean(ssds2,dims=2)

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(nocrng, [rt0 rt1 rt2])
ax0.legend(["S-POCA", "CG S-POCA", "CG S-POCA Procond."],fontsize = 12,loc=2)
xlabel("Number of components",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("cgssca_noc_vs_runtime.png")

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(nocrng, [iter0 iter1 iter2])
ax0.legend(["S-POCA", "CG S-POCA", "CG S-POCA Procond."],fontsize = 12,loc=2)
xlabel("Number of components",fontsize = 12)
ylabel("Iterations",fontsize = 12)
savefig("cgssca_noc_vs_iter.png")

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(nocrng, [mssdssca0 mssdssca1 mssdssca2])
ax0.legend(["S-POCA", "CG S-POCA", "CG S-POCA Procond."],fontsize = 12,loc=1)
xlabel("Number of components",fontsize = 12)
ylabel("ssd(W)",fontsize = 12)
savefig("cgssca_noc_vs_ssd(W).png")

(buffersize, delay) = Profile.init()
Profile.init(buffersize*10, delay/10)

rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usesvd=true, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
itermax=1000, objtol=1e-5, showdbg=true);

@profview W1, H1, objval = semiscasolve!(U, X; flipsign=false, usesvd=true, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
itermax=1000, objtol=1e-5, showdbg=false);


#====== Orthogonality test (1D) =====================================================#

function makedata(overlapW, overlapH, ncomp; sigmanoise=0)
    W1 = [zeros(overlapW); ones(20); zeros(20-overlapW)]
    W2 = [zeros(20-overlapW); ones(20); zeros(overlapW)]
    Wgt = [W1 W2]
    H1 = [zeros(overlapH); ones(20); zeros(20-overlapH)]
    H2 = [zeros(20-overlapH); ones(20); zeros(overlapH)]
    Hgt = [H1 H2]'
    X = Wgt*Hgt
    X .+= sigmanoise*rand(size(X)...).+sigmanoise
    F = svd(X)
    W = F.U[:,1:ncomp]
    H = W\X
    X, W, H, Wgt, Hgt
end
# Why Es doesn't reflect orthogonality of W and H components
X, W, H, Wgt, Hgt = makedata(0,0,14,sigmanoise=0.1) # Orthogonal data
title = "W component of GT"
xa = ("Pixel position")
ya = ("Intensity")
Plots.plot(Wgt[:,1:2], xaxis = xa, yaxis = ya, title = title, label=["W1" "W2"])
title = "H component of GT"
xa = ("Time")
ya = ("Intensity")
Plots.plot(Hgt[1:2,:]', xaxis = xa, yaxis = ya, title = title, label=["H1" "H2"])
title = "W component of SVD"
xa = ("Pixel position")
ya = ("Intensity")
Plots.plot(W[:,1:2], xaxis = xa, yaxis = ya, title = title, label=["W1" "W2"])
title = "H component of SVD"
xa = ("Time")
ya = ("Intensity")
Plots.plot(H[1:2,:]', xaxis = xa, yaxis = ya, title = title, label=["H1" "H2"])


# H Orhogonal
ncompmax = 10
orthoggtWs = []
orthoggtHs = []
orthoggts = []
errorss = []
orthogs = []
for i in 1:10
    X, W, H, Wgt, Hgt = makedata(i,0,ncompmax,sigmanoise=0.1)
    orthoggtW = norm(Wgt'*Wgt-I)/norm(Wgt)^2
    orthoggtH = norm(Hgt*Hgt'-I)/norm(Hgt)^2
    orthoggt = orthoggtW*orthoggtW
    push!(orthoggtWs,orthoggtW)
    push!(orthoggtHs,orthoggtH)
    push!(orthoggts,orthoggt)
    errors=[]
    for ncomp in 1:ncompmax
        error = norm(X-W[:,1:ncomp]*H[1:ncomp,:])/norm(W)^2/norm(H)^2
        push!(errors,error)
    end
    push!(errorss,errors)
    push!(orthogs, norm(X-W[:,1:1]*H[1:1,:]))
end

Plots.plot(errorss)
Plots.plot(orthogs)

clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(1:10, hcat(errorss...))
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Number of component(ncomp)")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("∥X-W[1:ncomp]*H1[1:ncomp]∥/(∥W∥∥H∥)²")
ax.set_title("Number of component vs. Error with input X")
plt.show(block=false)
plt.savefig("overlapvsnonorthogncomp.png")

clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(orthoggts)
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Overlap level")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("Non-orthogonality of GT")
ax.set_title("Overlap level vs. Non-orthogonality of GT")
plt.show(block=false)
plt.savefig("overlapvsnonorthog.png")

clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(orthogs)
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Overlap level")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("∥X-W1*H1∥/(∥W∥∥H∥)²")
ax.set_title("Overlap level vs. Orthogonality of SVD")
plt.show(block=false)
plt.savefig("overlapvsorthogsvd.png")

# H Non-orhogonal
ncompmax = 10
orthoggtWss = []
orthoggtHss = []
orthoggtss = []
orthogss = []
for j in 1:10
    orthoggtWs = []
    orthoggtHs = []
    orthoggts = []
    orthogs = []
    for i in 1:10
        X, W, H, Wgt, Hgt = makedata(i,j,ncompmax,sigmanoise=0.1)
        orthoggtW = norm(Wgt'*Wgt-I)/norm(Wgt)^2
        orthoggtH = norm(Hgt*Hgt'-I)/norm(Hgt)^2
        orthoggt = orthoggtW*orthoggtH
        push!(orthoggtWs,orthoggtW)
        push!(orthoggtHs,orthoggtH)
        push!(orthoggts,orthoggt)
        push!(orthogs, norm(X-W[:,1:1]*H[1:1,:])/norm(W)^2/norm(H)^2)
    end
    push!(orthoggtWss,orthoggtWs)
    push!(orthoggtHss,orthoggtHs)
    push!(orthoggtss,orthoggts)
    push!(orthogss, orthogs)
end
Plots.plot(orthogss)
Plots.plot(orthoggtss)

clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(orthoggtss)
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Overlap level Ow")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("Non-orthogonality of GT")
ax.set_title("Overlap level Ow vs. Non-orthogonality of GT")
legend(collect(1:10),fontsize = 12,loc=3)
plt.show(block=false)
plt.savefig("overlapvsnonorthogss.png")

clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(orthogss)
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Overlap level Ow")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("∥X-W1*H1∥/(∥W∥∥H∥)²")
ax.set_title("Overlap level Ow vs. Orthogonality of SVD")
legend(collect(1:10),fontsize = 12,loc=3)
plt.show(block=false)
plt.savefig("overlapvsorthogsvdss.png")

#====== Orthogonality test (FakeCells) =====================================================#

ncompmax = 10
orthoggtWs = []
orthoggtHs = []
orthoggts = []
orthogs = []
for i in 0:6
    sigma = 5
    imgsz = (40,20)
    nevents = 70
    lengthT = 100
    lambda = 0 # 0.01 Possion firing rate (if 0 some fixed points)
    gt_ncells, imgrs, Wgt, Hgt, gtWimgc, gtbg = 
        gaussian2D_calcium_transient(sigma, imgsz, lengthT, lambda=lambda, decayconst=0.76+0.02*i, bias = 0.1, noiselevel = 0.1, overlaplevel=i)
    ncells = 14
    F = svd(imgrs)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])
    X = imgrs
    W = copy(U)
    H = W\X
    orthoggtW = norm(Wgt'*Wgt-I)/norm(Wgt)^2
    orthoggtH = norm(Hgt*Hgt'-I)/norm(Hgt)^2
    orthoggt = orthoggtW*orthoggtH
    push!(orthoggtWs,orthoggtW)
    push!(orthoggtHs,orthoggtH)
    push!(orthoggts,orthoggt)
    push!(orthogs, norm(X-W[:,1:1]*H[1:1,:])/norm(W)^2/norm(H)^2)
    if true
        imsaveW("nonorthg$(i)Wgt.png",Wgt[:,1:7],imgsz)
        imsaveW("nonorthg$(i)Wsvd.png",W[:,1:7],imgsz)
        clf()
        fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
        ax.plot(Hgt)
        ax.set_xlabel("time")
        ax.set_ylabel("intensity")
        plt.savefig("nonorthg$(i)Hgt.png")
    end
end
fig, ax = plt.subplots(1,1, figsize=(5, 4), gridspec_kw=Dict("width_ratios"=>ones(1))) # figsize=(horsize, versize)
ax.plot(orthoggtHs)

Plots.plot(orthogs)
Plots.plot(orthoggts)
Plots.plot(orthoggtHs)
Plots.scatter(orthogs, orthoggts)


clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(orthoggts)
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Overlap level")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("Non-orthogonality of GT")
ax.set_title("Overlap level vs. Non-orthogonality of GT")
plt.show(block=false)
plt.savefig("overlapvsnonorthogfc.png")


clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(orthoggtHs)
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Overlap level")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("Non-orthogonality of Hgt")
ax.set_title("Overlap level vs. Non-orthogonality of Hgt")
plt.show(block=false)
plt.savefig("overlapvsnonorthogHfc.png")

clf()
plotlayout = (1,1); horsize, versize = 5, 4
fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
ax.plot(orthogs)
#ax.plot(θrng, Lijss, color="black", linewidth = 1)
#ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("Overlap level")
#ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("∥X-W1*H1∥/(∥W∥∥H∥)²")
ax.set_title("Overlap level vs. Orthogonality of SVD")
plt.show(block=false)
plt.savefig("overlapvsorthogsvdfc.png")

#======== varius S-POCA test ==================================#
beta0=0.7 # order 0
beta1=0.01 # order 1

# A\b, linesearch
rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=false, linesearch_method=:full, order=0,
    fixposdef=false, skew=false, β=beta0, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_withls_rt$(rt).png", W1,imgsz)

rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=false, linesearch_method=:full, order=1,
    fixposdef=false, skew=false, β=beta1, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_withls_rt$(rt)_order1.png", W1,imgsz)

# A\b
rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=false, linesearch_method=:none, order=0,
    fixposdef=false, skew=false, β=beta0, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_wols_rt$(rt).png", W1,imgsz)

rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=false, linesearch_method=:none, order=1,
    fixposdef=false, skew=false, β=beta1, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_wols_rt$(rt)_order1.png", W1,imgsz)

# cg, linesearch
rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=true, linesearch_method=:full, order=0,
    fixposdef=false, skew=false, β=beta0, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_cg_withls_rt$(rt).png", W1,imgsz)

rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=true, linesearch_method=:full, order=1,
    fixposdef=false, skew=false, β=beta1, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_cg_withls_rt$(rt)_order1.png", W1,imgsz)

# cg
rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=true, linesearch_method=:none, order=0,
    fixposdef=false, skew=false, β=beta0, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_cg_wols_rt$(rt).png", W1,imgsz)

rt = @elapsed W1, H1, objval = semiscasolve!(U, X; flipsign=false, usecg=true, linesearch_method=:none, order=1,
    fixposdef=false, skew=false, β=beta1, c=0.5, α0=1.0, ρ=0.5, itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imshowW(W1,imgsz)
imsaveW("scastep_cg_wols_rt$(rt)_order1.png", W1,imgsz)

#======== Skew symmetric matrix M Hessian check ==================================#
using Test

orthog = false
orthogstr = orthog ? "orthog" : "nonorthog"
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_$(orthogstr).jld")
m, p = size(W)
p, n = size(H)

#= Normal M case =#
x = rand(p^2)
M = POCA.reshapex(x,(p,p),false)

# For W
for j in 1:p, k in 1:m # check v before calculate cv and v⊗v
    b = POCA.vkjforW(W,k,j)
    @test b'*x ≈ W[k,:]'*M[:,j]
end
A = zeros(m*p,p^2)
b = zeros(m*p)
POCA.direct!(A, b, W; allcomp = true)
@test A*x+b ≈ vec(W*(I+M))
A = zeros(m*p,p^2)
b = zeros(m*p)
POCA.direct!(A, b, W; allcomp = false)
IM = I+M
WIM = zeros(m,p)
for i in 1:m
    for k in 1:p
        if W[i,k] < 0
            WIM[i,k] = W[i,:]'*IM[:,k]
        end
    end
end
@test A*x+b ≈ vec(WIM)

Hw = zeros(p^2,p^2); gw = zeros(p^2)
Aw = zeros(m*p,p^2); bw = zeros(m*p)
@time POCA.add_direct!(Hw,gw,W;allcomp=false);
@time (POCA.direct!(Aw,bw,W;allcomp=false); Aw'Aw; Aw'bw;)


# For H
for i in 1:p, k in 1:n # check v before calculate cv and v⊗v
    b = POCA.vikforH(H,i,k)
    @test b'*x ≈ M[i,:]'*H[:,k]
end
A = zeros(n*p,p^2)
b = zeros(n*p)
POCA.transpose!(A, b, H; allcomp = true)
@test A*x+b ≈ vec((I-M)H)
A = zeros(n*p,p^2)
b = zeros(n*p)
POCA.transpose!(A, b, H; allcomp = false)
IM = I-M
IMH = zeros(p,n)
for k in 1:p
    for j in 1:n
        if H[k,j] < 0
            IMH[k,j] = IM[k,:]'*H[:,j]
        end
    end
end
@test A*x+b ≈ vec(IMH)

# For M
A = POCA.orthogonality(p)
@test A*x ≈ vec(M+M')

#= off-diagonal skew constraint =#
l = Int(p*(p-1)/2) # size of x from dM with off-diagonal skew constraint
x = rand(l)
M = POCA.reshapeskew(x)

# For W
for j in 1:p, k in 1:m # check v before calculate cv and v⊗v
    b = POCA.vkjforW_skew(W,k,j)
    @test b'*x ≈ W[k,:]'*M[:,j]
end

truef(x) = (dM = POCA.reshapeskew(x); POCA.sca2(W*(I+dM)))
gradtruef = ForwardDiff.gradient(truef,zeros(l))
Hesstruef = ForwardDiff.hessian(truef,zeros(l))
AW, bW = zeros(l,l), zeros(l)
POCA.add_direct_skew!(AW, bW, W, allcomp = false)
norm(gradtruef-2bW) # 0.0
norm(Hesstruef-2AW) # 0.0

# For H
for i in 1:p, k in 1:n # check v before calculate cv and v⊗v
    b = POCA.vikforH_skew(H,i,k)
    @test b'*x ≈ M[i,:]'*H[:,k]
end

truef(x) = (dM = POCA.reshapeskew(x); POCA.sca2((I-dM)*H))
gradtruef = ForwardDiff.gradient(truef,zeros(l))
Hesstruef = ForwardDiff.hessian(truef,zeros(l))
AH, bH = zeros(l,l), zeros(l)
POCA.add_transpose_skew!(AH, bH, H, allcomp = false)
norm(gradtruef-2bH) # 5.995711664418482e-13
norm(Hesstruef-2AH) # 4.332274918715011e-12

# For both W and H
skew=true; allcompW=false; allcompH=false
A, b = POCA.buildproblem(W, H; order=1, skew=skew, allcompW=allcompW, allcompH=allcompH)
scapair_vec(x, W, H) = (dM = POCA.reshapex(x, (p,p), skew); POCA.scapair(W*(I+dM),(I-dM)*H; allcompW=allcompW, allcompH=allcompH))
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), zeros(l))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), zeros(l))
norm(grad-2b) # 3.86064596548465e-12
norm(Hess-2A) # 3.0222234014482127e-11

#======== Semi-skew matrix M Hessian check ==================================#
m, p = size(W)
p, n = size(H)
l = Int(p*(p+1)/2) # size of x from dM with od(off-diagonal)skew constraint
x = rand(l)
M = POCA.reshapeodskew(x)

# For W
for j in 1:p, k in 1:m # check v before calculate cv and v⊗v
    b = POCA.vkjforW_odskew(W,k,j)
    @test b'*x ≈ W[k,:]'*M[:,j]
end

truef(x) = (dM = POCA.reshapeodskew(x); POCA.sca2(W*(I+dM)))
gradtruef = ForwardDiff.gradient(truef,zeros(l))
Hesstruef = ForwardDiff.hessian(truef,zeros(l))
AW, bW = zeros(l,l), zeros(l)
POCA.add_direct_odskew!(AW, bW, W, allcomp = false)
norm(gradtruef-2bW) # 0.0
norm(Hesstruef-2AW) # 0.0

# For H
for i in 1:p, k in 1:n # check v before calculate cv and v⊗v
    b = POCA.Hkforj(H,i,k)
    @test b'*x ≈ M[i,:]'*H[:,k]
end

truef(x) = (dM = POCA.reshapeodskew(x); POCA.sca2((I-dM)*H))
gradtruef = ForwardDiff.gradient(truef,zeros(l))
Hesstruef = ForwardDiff.hessian(truef,zeros(l))
AH, bH = zeros(l,l), zeros(l)
POCA.add_transpose_odskew!(AH, bH, H, allcomp = false)
norm(gradtruef-2bH) # 5.995711664418482e-13
norm(Hesstruef-2AH) # 4.332274918715011e-12

# For both W and H
skew=true; allcompW=false; allcompH=false
A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H; order=1, skew=skew, allcompW=allcompW, allcompH=allcompH)
scapair_vec(x, W, H) = (dM = POCA.reshapex(x, (p,p), skew); POCA.scapair(W*(I+dM),(I-dM)*H; allcompW=allcompW, allcompH=allcompH))
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), zeros(l))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), zeros(l))
norm(grad-2b) # 3.86064596548465e-12
norm(Hess-2A) # 3.0222234014482127e-11

#============= semiscasolve!W ========================#
for β in 0.1:0.1:1 #[0.001,0.01,0.1,1,10,100,1000]
    # W-POCA (with orthogonal constraint)
    @show β
    runtime = @elapsed  W1, objval, _ = semiscasolve!W(U, X; flipsign=false, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-5, showdbg=false);
    H1 = W1\X
    normalizeWH!(W1,H1)
#    imshowW(W1,imgsz)
    imsaveW("WPOCA_b$(β)_r$(runtime).png", W1, imgsz)
end

β = 30#0.4

rts = []
for i in 1:10
    @show i
    fname = "fc_nonorth_o0_PFfalse_skfalse_c0.5_ucfalse_r0_updrfalse_($i).jld"
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(fname)

    runtime = @elapsed  W1, objval, _ = semiscasolve!W(W, H; flipsign=false, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-2, showdbg=false);
    H1 = W1\X
    normalizeWH!(W1,H1)
    imshowW(W1,imgsz)
    imsaveW("WPOCA_$(i)_r$(runtime).png", W1, imgsz)
    push!(rts,runtime)
end
mean(rts)

#============= number of components vs runtime ========================#
nocrng = 2:20
datart, dataW1, dataH1, dataH2 = experiment_noc(nocrng)
save("noc(2_20)2.jld", "datart", datart, "dataW1", dataW1 ,"dataH1", dataH1 ,"dataH2", dataH2)

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
# ax0.plot(nocrng, datart)
# ax0.legend(["S-POCA", "Multiplicative", "Naive ALS", "Projected  GD ALS", "Coordinate Descent", "Greedy CD"],fontsize = 12,loc=2)
ax0.plot(3:20, [rt1[2:19] datart[2:19,2] datart[2:19,3] datart[2:19,5]])
ax0.legend(["CG S-POCA", "Multiplicative", "Naive ALS", "Coordinate Descent"],fontsize = 12,loc=2)
xlabel("Number of components",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("P1_noc_vs_runtime2.png")

#====== semiscasolve! test with many datasets ==================================#
beta=0.7
rts=[]
for i in 1:50
    @show i
    fname = "fc_nonorth_o0_PFfalse_skfalse_c0.5_ucfalse_r0_updrfalse_($i).jld"
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(fname)
    rt = @elapsed W1, H1, objval = semiscasolve!(W, H; flipsign=false, usesvd=true, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-5, showdbg=false);
    push!(rts,rt)
    normalizeWH!(W1,H1)
    imshowW(W1,imgsz)
    imsaveW("CGwOTinE1e-5$i.png",W1,imgsz)
end
mean(rts)

#====== Test Non-linear Conjugate Gradient method  ==================================#
using Optim

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function g!(storage, x)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

optimize(f,g!, [0.0, 0.0], ConjugateGradient(), Optim.Options(show_trace=false, show_every=10))

#============= removing svd(A) ========================#
beta=0.7
@time W1, H1, objval, numsvds, rtsvds = semiscasolve!(U, X; flipsign=false, usesvd=true, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-5, showdbg=false);
@time W1, H1, objval, nums, rts = semiscasolve!(U, X; flipsign=false, usesvd=false, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-5, showdbg=false);

numsvdWs = []
numsvdHs = []
numWs = []
numHs = []
for i = 1:length(nums)
    push!(numsvdWs, numsvds[i][1])
    push!(numsvdHs, numsvds[i][2])
    push!(numWs, nums[i][1])
    push!(numHs, nums[i][2])
end

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(1:length(nums), [numsvdWs numWs])
ax0.legend(["with svd(A)", "without svd(A)"],fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("number of negative component",fontsize = 12)
savefig("nocW.png")

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(1:length(nums), [numsvdHs numHs])
ax0.legend(["with svd(A)", "without svd(A)"],fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("number of negative component",fontsize = 12)
savefig("nocH.png")

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(1:length(nums), [rtsvds rts])
ax0.legend(["with svd(A)", "without svd(A)"],fontsize = 12,loc=1)
xlabel("iteration",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("rtimes.png")

@time W1, H1, objval = semiscasolve!(U, X; flipsign=false, usesvd=true, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imsaveW("withsvd.png", W1, imgsz)
@time W1, H1, objval = semiscasolve!(U, X; flipsign=false, usesvd=false, order=0, fixposdef=false, skew=false, β=beta, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-5, showdbg=false);
normalizeWH!(W1,H1)
imsaveW("withoutsvd.png", W1, imgsz)

#============= Finding optimum β ========================#
order = 1
rng = 1:10
βrng = 0.0:0.002:0.2 # 0.0:0.01:2, 0.0:1:100, 0.0:10:1000(option4), 0.0:0.05:2(option3)
datart, dataW1, dataH1, dataH2 = experiment2(rng,βrng,order=order)
save("optimum_betaP1_order$(order)0p2.jld", "datart", datart, "dataW1", dataW1 ,"dataH1", dataH1 ,"dataH2", dataH2)

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(βrng, dataW1)
xlabel("β",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
savefig("order1_beta_vs_ssdWs.png")

fig, ax2 = plt.subplots(1,1, figsize=(5,4))
ax2.plot(βrng, sum(dataW1,dims=2)./length(rng))
xlabel("β",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
savefig("order1_beta_vs_ssdW.png")

fig, ax3 = plt.subplots(1,1, figsize=(5,4))
ax3.plot(βrng, datart)
xlabel("β",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("order1_beta_vs_runtimes.png")

fig, ax4 = plt.subplots(1,1, figsize=(5,4))
ax4.plot(βrng, sum(datart,dims=2)./length(rng))
xlabel("β",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("order1_beta_vs_runtime.png")

#============= add_direct!, add_transpose!========================#

nocrng = 2:20
rt0 = []; rt1 = []; rt2 = []
for ncells in nocrng
    println("noc = $ncells")

    i = ncells-first(nocrng)+1
    F = svd(X)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])

    runtime = @elapsed  W1, H1, objval = semiscasolve!(U, X; flipsign=false, order=0, fixposdef=false, skew=false, β=0.7, c=0.5, α0=1.0, ρ=0.5,
        itermax=100, objtol=1e-5, showdbg=false);
    normalizeWH!(W1,H1)
    imsaveW("noc$ncells.png", W1, imgsz)
    push!(rt1,runtime)
end

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(nocrng, [rt0 rt1])
ax0.legend(["before", "after", "w/o allocation"],fontsize = 12,loc=2)
xlabel("Number of components",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("add_direct.png")

normalizeWH!(W1,H1)
imsaveW("SPOCA2_b$(β)_r$(runtime).png", W1, imgsz)

#============= number of components vs runtime ========================#
nocrng = 2:20
datart, dataW1, dataH1, dataH2 = experiment_noc(nocrng)
save("noc(2_20)2.jld", "datart", datart, "dataW1", dataW1 ,"dataH1", dataH1 ,"dataH2", dataH2)

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(nocrng, datart)
ax0.legend(["S-POCA", "Multiplicative", "Naive ALS", "Projected  GD ALS", "Coordinate Descent", "Greedy CD"],fontsize = 12,loc=2)
xlabel("Number of components",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("P1_noc_vs_runtime2.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax1.plot(nocrng, dataW1)
ax1.legend(["S-POCA", "Multiplicative", "Naive ALS", "Projected  GD ALS", "Coordinate Descent", "Greedy CD"],fontsize = 12,loc=1)
xlabel("Number of components",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
savefig("P1_noc_vs_W32.png")


fig, ax2 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax2.plot(nocrng, dataW1[:,[1,3,4]])
ax2.legend(["S-POCA", "Naive ALS", "Projected  GD ALS"],fontsize = 12,loc=1)
xlabel("Number of components",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
savefig("P1_noc_vs_W32.png")


#============= Error cases ========================#
errors = [25, 52, 58, 68, 76, 88, 98]

# constant parameters
flipsign=true; order=0; fixposdef=false; skew=false; c=0.5; useconstraint=false;
γ=0; usepdratio=false; truehess=false; showfig=false; plotssign=1; plotiters=(1,30);
showdbg=false; allcompW=false; allcompH=false; α0=1.0; ρ=0.5;

# critical parameters
β = 0.7 # 1.0(option4), 0.7(option3)
itermax=100
objtol=1e-5

rt0=[]
for i in 76:76#errors
    println("dataset=$i")

    # Generate data
    paramstrsuffix = "$(order)_PF$(fixposdef)_sk$(skew)_c$(c)_b$(β)_uc$(useconstraint)_r$(γ)_updr$(usepdratio)_($i)"
    datafilename = "fc_nonorth_o$paramstrsuffix.jld"
    if isfile(datafilename)
        fakecells_dic = load(datafilename)
        gt_ncells = fakecells_dic["gt_ncells"]
        imgrs = fakecells_dic["imgrs"]
        gtW = fakecells_dic["gtW"]
        gtH = fakecells_dic["gtH"]
        gtWimgc = fakecells_dic["gtWimgc"]
        gtbg = fakecells_dic["gtbg"]
        imgsz = fakecells_dic["imgsz"]
    else
        sigma = 5
        imgsz = (40,20)
        nevents = 70
        lengthT = 100
        gt_ncells, imgrs, gtW, gtH, gtWimgc, gtbg = gaussian2D(sigma, imgsz, lengthT, nevents, orthogonal=false)
        datafilename = "fc_nonorth_o$paramstrsuffix.jld"
        Images.save(datafilename, "gt_ncells", gt_ncells, "imgrs", imgrs, "gtW", gtW, "gtH", gtH, "gtWimgc",Array(gtWimgc), "gtbg", gtbg, "imgsz", imgsz)
    end
    ncells = 14
    F = svd(imgrs)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])

    X = imgrs

    # S-POCA (with orthogonal constraint)
    runtime = @elapsed  W1, H1, objval = semiscasolve!(U, X; flipsign=false, order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5,
                            itermax=itermax, objtol=objtol, showdbg=false);
    normalizeWH!(W1,H1)
    push!(rt0,runtime)
    imsaveW("$i.png",W1,imgsz)
    # imshowW(W1,imgsz)
end

fig, ax0 = plt.subplots(1,1, figsize=(5,4))#,gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
ax0.plot(1:100, [rt0 rt1])
ax0.legend(["with svd(A)", "without svd(A)"],fontsize = 12,loc=2)
xlabel("Dataset",fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("svd(A)_runtime2.png")

#============= semiscasolve! : w*OT vs w^2*OT ========================#
betarng = 0.001:0.001:0.07

ssdw = []; ssdh = []; runtime
for β in betarng
    println("β=$β")
    # S-POCA (with orthogonal constraint)
    runtime = @elapsed  W1, H1, objval = semiscasolve!(U, X; flipsign=false, order=1, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5,
                            itermax=itermax, objtol=objtol, showdbg=false);
    normalizeWH!(W1,H1)
    mssdssca, mlssca, ssds = matchedssd(gtW,W1)
    mssdHssca = ssdH(mlssca,gtH,H1')
    push!(ssdw, mssdssca)
    push!(ssdh, mssdHssca)
#    imsaveW("SPOCA_order1_wCT_b$(β)_ssd$(mssdssca)_ssdW$(mssdHssca)_r$(runtime).png", W1, imgsz)
end

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(betarng, ssdw)
xlabel("β",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
savefig("order1_beta_vs_ssdW.png")


fig, ax2 = plt.subplots(1,1, figsize=(5,4))
ax2.plot(betarng, ssdw)
xlabel("β",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
savefig("order1_beta_vs_ssdW.png")


#============= Box plot ========================#

# Compare with other methods
rng = 1:100

# dd = load("compareP1.jld")
# datart = dd["datart"]
# dataW1 = dd["dataW1"]
# dataH1 = dd["dataH1"]
# dataH2 = dd["dataH2"]

datartc, dataW1c, dataH1c, dataH2c = experiment(rng)
datart, dataW1, dataH1, dataH2 = datartc, dataW1c, dataH1c, dataH2c
# mthod = 1
# datart[:,mthod] = datartc[:,mthod]
# dataW1[:,mthod] = dataW1c[:,mthod]
# dataH1[:,mthod] = dataH1c[:,mthod]
# dataH2[:,mthod] = dataH2c[:,mthod]
save("compareP1(beta0.7_tol-5_W1all0).jld", "datart", datart, "dataW1", dataW1 ,"dataH1", dataH1 ,"dataH2", dataH2 )

plt.boxplot(dataW1, # Each column/cell is one box
	notch=false, # Notched center
	whis=0.75, # Whisker length as a percent of inner quartile range
	widths=0.25, # Width of boxes
	vert=true, # Horizontal boxes
    sym="*") # Symbol color and shape (rs = red square)
plt.xticks(collect(1:6),["S-POCA", "Multiplicative", "Naive ALS", "Projected GD ALS", "CD", "Greedy CD"],rotation=13, fontsize=10)
#legend(["S-POCA", "Multiplicative", "Naive ALS", "Projected  GD ALS", "Coordinate Descent", "Greedy CD"],fontsize = 12,loc=9)
#xlabel("Time index",fontsize = 12)
ylabel("SSD(W)",fontsize = 12)
savefig("P1boxplot_W3.png")

plt.boxplot(dataH1, # Each column/cell is one box
	notch=false, # Notched center
	whis=0.75, # Whisker length as a percent of inner quartile range
	widths=0.25, # Width of boxes
	vert=true, # Horizontal boxes
	sym="*") # Symbol color and shape (rs = red square)
plt.xticks(collect(1:6),["S-POCA", "Multiplicative", "Naive ALS", "Projected GD ALS", "CD", "Greedy CD"],rotation=13, fontsize=10)
ylabel("SSD(H)",fontsize = 12)
savefig("P1boxplot_H3.png")

plt.boxplot(dataH2, # Each column/cell is one box
	notch=false, # Notched center
	whis=0.75, # Whisker length as a percent of inner quartile range
	widths=0.25, # Width of boxes
	vert=true, # Horizontal boxes
	sym="*") # Symbol color and shape (rs = red square)
plt.xticks(collect(1:6),["S-POCA", "Multiplicative", "Naive ALS", "Projected GD ALS", "CD", "Greedy CD"],rotation=13, fontsize=10)
ylabel("SSD(H)",fontsize = 12)
savefig("P1boxplot_Hinv3.png")

plt.boxplot(datart, # Each column/cell is one box
	notch=false, # Notched center
	whis=0.75, # Whisker length as a percent of inner quartile range
	widths=0.25, # Width of boxes
	vert=true, # Horizontal boxes
	sym="*") # Symbol color and shape (rs = red square)
plt.xticks(collect(1:6),["S-POCA", "Multiplicative", "Naive ALS", "Projected GD ALS", "CD", "Greedy CD"],rotation=13, fontsize=10)
#legend(["S-POCA", "Multiplicative", "Naive ALS", "Projected GD ALS", "CD", "Greedy CD"],fontsize = 12,loc=2)
ylabel("running time",fontsize = 12)
savefig("P1boxplot_rt33.png")

#========= Semi skew POCA  ========================#
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_sktrue_c0.5_b0false_ucfalse_r0_updrfalse_msttrue_(1).jld")

flipsign=true; order=0; fixposdef=false; skew=false; c=0.5; β=0.5; useconstraint=false;
γ=0; usepdratio=false; truehess=false; itermax=100; showfig=false; plotssign=1; plotiters=(1,30);
objtol=1e-5; showdbg=false; allcompW=false; allcompH=false; α0=1.0; ρ=0.5;

# H-POCA
objtolH=sqrt(objtol)
@time H2, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!H(U, X; flipsign=flipsign, skew=skew,
    c=c, α0=α0, ρ=ρ, useconstraint=useconstraint, truehess=truehess, itermax=itermax,
    showfig=showfig, plotssign=plotssign, plotiters=plotiters, objtol=objtol, showdbg=false);
W2 = X/H2
sca2(W2)
sca2(H2)
Wn, Hn = copy(W2), copy(H2);
normalizeWH!(Wn,Hn);
imshowW(Wn,imgsz);

# W-POCA
objtolW=sqrt(objtol)
@time W2, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!W(U, X; flipsign=flipsign, skew=skew,
    c=c, α0=α0, ρ=ρ, useconstraint=useconstraint, truehess=truehess, itermax=itermax,
    showfig=showfig, plotssign=plotssign, plotiters=plotiters, objtol=objtolW, showdbg=false);
H2 = W2\X
sca2(W2)
sca2(H2)
Wn, Hn = copy(W2), copy(H2);
normalizeWH!(Wn,Hn);
imshowW(Wn,imgsz);

# S-POCA
@time W1, H1, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W, X; flipsign=flipsign, order=order,
    fixposdef=fixposdef, skew=skew, c=c, β=β, α0=α0, ρ=ρ, useconstraint=useconstraint, γ=γ, 
    usepdratio=usepdratio, truehess=truehess, itermax=itermax, showfig=showfig, plotssign=plotssign,
    plotiters=plotiters, objtol=objtol, showdbg=showdbg);
sca2(W1)
sca2(H1)
Wn, Hn = copy(W1), copy(H1);
normalizeWH!(Wn,Hn);
imshowW(Wn,imgsz);

@time W2, H2, objvalsorder2, Ms, A, endval, iters, idx  = semiscasolve!0_2ndstep(W1, H1; order=order, fixposdef=fixposdef,
    β=0, allcompW = false, allcompH = true, useconstraint=useconstraint, γ=γ, itermax=itermax,
    showfig=showfig, plotiters=(1,30), objtol=1e-4, showdbg=false, c=1);
sca2(W2)
sca2(H2)
Wn, Hn = copy(W2), copy(H2);
normalizeWH!(Wn,Hn);
imshowW(Wn,imgsz);

@time W3, H3, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W2, X; flipsign=false, order=order,
    fixposdef=fixposdef, skew=false, c=c, β=β, α0=α0, ρ=ρ, useconstraint=useconstraint, γ=γ, 
    usepdratio=usepdratio, truehess=truehess, itermax=itermax, showfig=showfig, plotssign=plotssign,
    plotiters=plotiters, objtol=objtol, showdbg=true);
sca2(W3)
sca2(H3)
Wn, Hn = copy(W3), copy(H3);
normalizeWH!(Wn,Hn);
imshowW(Wn,imgsz);
imsaveW("tmp.png",Wn,imgsz);




rng=1:40

# constant parameters
flipsign=true; order=0; fixposdef=false; skew=false; c=0.5; useconstraint=false;
γ=0; usepdratio=false; truehess=false; showfig=false; plotssign=1; plotiters=(1,30);
showdbg=false; allcompW=false; allcompH=false; α0=1.0; ρ=0.5;

# critical parameters
β = 1 # 0.7(option3)
itermax=100
objtol=1e-6

@time for r in rng
    println("dataset=$r")

    # Generate data
    paramstrsuffix = "$(order)_PF$(fixposdef)_sk$(skew)_c$(c)_b$(β)_uc$(useconstraint)_r$(γ)_updr$(usepdratio)_($r)"
    datafilename = "fc_nonorth_o$paramstrsuffix.jld"
    if isfile(datafilename)
        fakecells_dic = load(datafilename)
        gt_ncells = fakecells_dic["gt_ncells"]
        imgrs = fakecells_dic["imgrs"]
        gtW = fakecells_dic["gtW"]
        gtH = fakecells_dic["gtH"]
        gtWimgc = fakecells_dic["gtWimgc"]
        gtbg = fakecells_dic["gtbg"]
        imgsz = fakecells_dic["imgsz"]
    else
        sigma = 5
        imgsz = (40,20)
        nevents = 70
        lengthT = 100
        gt_ncells, imgrs, gtW, gtH, gtWimgc, gtbg = gaussian2D(sigma, imgsz, lengthT, nevents, orthogonal=false)
        datafilename = "fc_nonorth_o$paramstrsuffix.jld"
        Images.save(datafilename, "gt_ncells", gt_ncells, "imgrs", imgrs, "gtW", gtW, "gtH", gtH, "gtWimgc",Array(gtWimgc), "gtbg", gtbg, "imgsz", imgsz)
    end
    ncells = 14
    F = svd(imgrs)
    U = F.U[:,1:ncells]
    T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])

    X = imgrs

    # S-POCA with orthogonal constraint
    # W1, H1, objvals, Ms, As, bs, endval, iters, idx  = semiscasolve!1(U, X; flipsign=false, order=0,
    #     fixposdef=false, skew=false, c=0.5, β=β, α0=1.0, ρ=0.5, useconstraint=false, γ=0., 
    #     truehess=false, itermax=itermax, showfig=false, plotssign=1, plotiters=(1,30), objtol=objtol, showdbg=false);
    W1, H1, objval = semiscasolve!(U, X; flipsign=false, order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5,
                             itermax=itermax, objtol=objtol, showdbg=false);
    normalizeWH!(W1,H1)
    imshowW(W1,imgsz)
end

for beta in [0.0000001, 0.000001, 10000, 100000]
    β = beta
    W1, H1, objval = semiscasolve!(U, X; flipsign=false, order=0, fixposdef=false, skew=false, β=β, c=0.5, α0=1.0, ρ=0.5,
        itermax=itermax, objtol=objtol, showdbg=false);
    normalizeWH!(W1,H1)
    imsaveW("P2b$(beta).png",W1,imgsz)
end

β = 0.00005 # ** On entry to DLASCLS parameter number  4 had an illegal value
datasets = [(0, false)]#, # order=0; fixposdef=false (Order 1 wo corssterm)
#             (1, true)]# # order=1; fixposdef=true (order 1 w PF)
# datasets = [(1, false), # order=1; fixposdef=false (order 1 wo PF)
#             (2, true)] # order=1; fixposdef=true (order 1 w PF)
# datasets = [(2, false)] # order=1; fixposdef=false (order 1 wo PF)
rng=1:40#[12,13,26,28,36,40]#1:40
minLssss = []; Lsss = []; rt = []; dfnamess=[]; ofnamess=[]
@time for (order, fixposdef) in datasets
    # Original w constraint
    minLsss, Lss, rt0, rt1, rt2, sscaiter, dfnames, ofnames = runexp(rng; order = order, fixposdef = fixposdef, flipsign=flipsign, 
            skew=skew, c=c, β = β, α0=α0, useconstraint=useconstraint, γ=γ, usepdratio=usepdratio,
            itermax=itermax, objtol=objtol, multiplestep=false)
    push!(Lsss,Lss)
    push!(rt,rt0)
    push!(rt,rt1)
    push!(rt,rt2)
    push!(rt,sscaiter)
    push!(dfnamess,dfnames)
    push!(ofnamess,ofnames)
end

for i in 1:length(rng)
    r = rng[i]
    ddic = load(dfnamess[1][i])
    odic = load(ofnamess[1][i])
    X = ddic["imgrs"]
    imgsz = ddic["imgsz"]
    M = odic["M"]
    matchlist = odic["matchlist"]
    p = size(M,1)
    U = svd(X).U[:,1:p]
    W=U*M
    H = W\X
    normalizeWH!(W,H)
    imshowW(W,imgsz)

#     Msort = sortmatchlist(M,matchlist)
#     W=U*Msort
#     H = W\X
#     normalizeWH!(W,H)
#     imshowW(W,imgsz)
    imsaveW("Wimg_beta_r$(r).png",W,imgsz)

#     interp_num = 5
#     Mimg = interpolate(makeimg(Msort),interp_num)
#     #save("Mimg_r$(r).png",Mimg)
#     @show eigen(Msort).values
#     Msum += Msort
end


for r in rng
    ddic = load(dfnamess[1][r])
    imgrs = ddic["imgrs"]
    imgsz = ddic["imgsz"]
    ncells = 14
    F = svd(imgrs)
    U = F.U[:,1:ncells]
    # R = Matrix(1.0I,ncells,ncells)
    # scheduler = Sequential{typeof(zero(eltype(U))*zero(eltype(R)))}(size(U, 2)) # good(full(50sec))
    # Msca, sign_flip_idx, minL, minLss = mostly_positive_combination_plot(minimizeLij,Rθ,U; 
    #         scheduler=scheduler, tol=1e-4, maxiter=10^3, show_debug=false, enplot=false, finetune=true)
    # W = U*Msca; H = W\imgrs
    objtolW=2.0
    W2, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!W(U, X; flipsign=flipsign, skew=skew,
        c=c, α0=α0, ρ=ρ, useconstraint=useconstraint, truehess=truehess, itermax=itermax,
        showfig=showfig, plotssign=plotssign, plotiters=plotiters, objtol=objtolW, showdbg=false);
    H2 = W2\X
    normalizeWH!(W2,H2)
    imshowW(W2,imgsz)

#     Msort = sortmatchlist(M,matchlist)
#     W=U*Msort
#     H = W\X
#     normalizeWH!(W,H)
#     imshowW(W,imgsz)
    imsaveW("Wimg_wsca_r$(r).png",W2,imgsz)

#     interp_num = 5
#     Mimg = interpolate(makeimg(Msort),interp_num)
#     #save("Mimg_r$(r).png",Mimg)
#     @show eigen(Msort).values
#     Msum += Msort
end

# Mulitiplicative 0.17sec (errors: 12/40)
rt = @elapsed for r in rng # 14, 26
    @show r
    X, F, U, W0, H0, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_sktrue_c0.5_b0false_ucfalse_r0_updrfalse_msttrue_($r).jld")
    gtW = fakecells_dic["gtW"]
    gtH = fakecells_dic["gtH"]

    runtime1 = @elapsed W2, H2 = NMF.nndsvd(X, ncells, variant=:ar)
    # optimize
    runtime2 = @elapsed NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse,maxiter=200), X, W2, H2)
    normalizeWH!(W2,H2)
    mssd2, ml2, ssds = matchedssd(gtW,W2)
    mssdH2 = ssdH(ml2,gtH,H2') # 3424.19(nonorthog)
    rtstr = @sprintf("%1.4f",runtime2)
    mssdstr = @sprintf("%1.4f",mssd2)
    mssdHstr = @sprintf("%1.4f",mssdH2)
    imsaveW("W_Multiplicative_nndsvdinit_nonorthog_t$(rtstr)_ssd$(mssdstr)_ssdH$(mssdHstr).png",W2,imgsz)
end
rt /= length(rng)

# NaiveALS : 0.14sec (errors: 0/40)
rt = @elapsed for r in rng # 14, 26
    @show r
    X, F, U, W0, H0, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_sktrue_c0.5_b0false_ucfalse_r0_updrfalse_msttrue_($r).jld")
    gtW = fakecells_dic["gtW"]
    gtH = fakecells_dic["gtH"]

    # nndsvd initialize
    runtime1 = @elapsed W4, H4 = NMF.nndsvd(X, ncells, variant=:ar)
    # optimize
    runtime2 = @elapsed NMF.solve!(NMF.ProjectedALS{Float64}(maxiter=200), X, W4, H4)
    normalizeWH!(W4,H4)
    mssd4, ml4, ssds = matchedssd(gtW,W4)
    mssdH4 = ssdH(ml4,gtH,H4')
    rtstr = @sprintf("%1.4f",runtime2)
    mssdstr = @sprintf("%1.4f",mssd4)
    mssdHstr = @sprintf("%1.4f",mssdH4)
    imsaveW("W_NaiveALS_nndsvdinit_nonorthog_r($r)_t$(rtstr)_ssd$(mssdstr)_ssdH$(mssdHstr).png",W4,imgsz)
end
rt /= length(rng)

# ProjectedGDALS : 22,12sec (errors: 13/40) 
rt = @elapsed for r in rng # 14, 26
    @show r
    X, F, U, W0, H0, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_sktrue_c0.5_b0false_ucfalse_r0_updrfalse_msttrue_($r).jld")
    gtW = fakecells_dic["gtW"]
    gtH = fakecells_dic["gtH"]

    # initialize
    runtime1 = @elapsed W6, H6 = NMF.nndsvd(X, ncells, variant=:ar)
    # optimize
    runtime2 = @elapsed NMF.solve!(NMF.ALSPGrad{Float64}(maxiter=200, tolg=1.0e-6), X, W6, H6)
    normalizeWH!(W6,H6)
    mssd6, ml6, ssds = matchedssd(gtW,W6)
    mssdH6 = ssdH(ml6,gtH,H6')
    rtstr = @sprintf("%1.4f",runtime2)
    mssdstr = @sprintf("%1.4f",mssd6)
    mssdHstr = @sprintf("%1.4f",mssdH6)
    imsaveW("W_ProjectedGDALS_nndsvdinit_nonorthog_r($r)_t$(rtstr)_ssd$(mssdstr)_ssdH$(mssdHstr).png",W6,imgsz)
end
rt /= length(rng)

# CoordinateDescent : 2.53sec (errors: 5/40) 
rt = @elapsed for r in rng # 14, 26
    @show r
    X, F, U, W0, H0, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_sktrue_c0.5_b0false_ucfalse_r0_updrfalse_msttrue_($r).jld")
    gtW = fakecells_dic["gtW"]
    gtH = fakecells_dic["gtH"]

    # initialize
    runtime1 = @elapsed W8, H8 = NMF.nndsvd(X, ncells, variant=:ar)
    # optimize
    runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=3000), X, W8, H8) # , α=0.5, l₁ratio=0.5
    normalizeWH!(W8,H8)
    mssd8, ml8, ssds = matchedssd(gtW,W8)
    mssdH8 = ssdH(ml8,gtH,H8')
    rtstr = @sprintf("%1.4f",runtime2)
    mssdstr = @sprintf("%1.4f",mssd8)
    mssdHstr = @sprintf("%1.4f",mssdH8)
    imsaveW("W_CoordinateDescent_nndsvdinit_nonorthog_r($r)_t$(rtstr)_ssd$(mssdstr)_ssdH$(mssdHstr).png",W8,imgsz)
end
rt /= length(rng)


# CoordinateDescent : 0.95sec (errors: 12/40) 
rt = @elapsed for r in rng # 14, 26
    @show r
    X, F, U, W0, H0, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_sktrue_c0.5_b0false_ucfalse_r0_updrfalse_msttrue_($r).jld")
    gtW = fakecells_dic["gtW"]
    gtH = fakecells_dic["gtH"]

    # initialize
    runtime1 = @elapsed W10, H10 = NMF.nndsvd(X, ncells, variant=:ar)
    # optimize
    runtime2 = @elapsed NMF.solve!(NMF.GreedyCD{Float64}(maxiter=200), X, W10, H10) # maxiter=50
    normalizeWH!(W10,H10)
    mssd10, ml10, ssds = matchedssd(gtW,W10)
    mssdH10 = ssdH(ml10,gtH,H10')
    rtstr = @sprintf("%1.4f",runtime2)
    mssdstr = @sprintf("%1.4f",mssd10)
    mssdHstr = @sprintf("%1.4f",mssdH10)
    imsaveW("W_GreedyCD_nndsvdinit_nonorthog_r($r)_t$(rtstr)_ssd$(mssdstr).png",W10,imgsz)
end
rt /= length(rng)


#======== Semi-skew matrix vs orthogonality =====================#

rrng = 0.1:0.01:10
n1s = []; n2s = []; ndmss=[]
for r in rrng
    R = rand(14,14)
    dMss = r.*(R-R') # skew Symmetric
    Mss = I+dMss # semi skew
    n1 = norm(Mss'Mss-I)/norm(dMss'dMss)
    push!(n1s,n1)

    dMnss = r.*(2*R.-1)
    for i in 1:14 dMnss[i,i]=0.0 end
    Mnss = I+dMnss
    n2 = norm(Mnss'Mnss-I)/norm(dMnss'dMnss)
    push!(n2s,n2)

    push!(ndmss,norm(dMss'dMss))
    @show n1, n2
end

plot(rrng, [n1s n2s])
#plot(rrng, [log10.(n1s) log10.(n2s)])
legend(["skew", "non-skew"],fontsize = 12,loc=1)

#======== Order 2 orthogonality penalty (doesn't work) ==================================#
# generate [I ... I; ...; I ... I] matrix
function blockidentity(T::Type, n::Integer)
    n2 = n^2
    [ mod(i,n) == mod(j,n) ? one(T) : zero(T) for i=1:n2, j=1:n2 ]
end

dM = rand(2,2)
x = vec(dM)
BI = blockidentity(eltype(dM),size(dM,1))
norm(dM'dM) ≈ norm(x'*BI*x) # fail
sum(dM'dM) ≈ sum(x'*BI*x) # true

#========= Random initialization test ========================#
flipsign=true; order=0; fixposdef=false; skew=false; c=0.5; β=.5; useconstraint=false;
γ=0; usepdratio=false; truehess=false; itermax=1000; showfig=false; plotssign=1; plotiters=(1,30);
objtol=1e-12; showdbg=false; allcompW=false; allcompH=false; α0=1; ρ=0.5;

W = copy(W0)
M = ones(14,14)+10*I
colnormalize!(M)
W = W*M
# normalizeWH!(W,H)
# imshowW(W,imgsz)

Wt, Ht, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W, X; flipsign=flipsign, order=order,
    fixposdef=fixposdef, skew=skew, c=c, β=β, useconstraint=useconstraint, γ=γ, 
    usepdratio=usepdratio, truehess=truehess, itermax=itermax, showfig=showfig, plotssign=plotssign,
    plotiters=plotiters, objtol=objtol, showdbg=showdbg);

normalizeWH!(Wt,Ht)
imshowW(Wt,imgsz)

for M in Ms[3]
    colnormalize!(M)
    @show det(M)
end


W = copy(W0)
H = W\X

W, H, α, dM2, dM, A, b, M0 = scastep(W, H; order=order, fixposdef=fixposdef,
skew=skew, allcompW=allcompW, allcompH=allcompH, α0=α0, ρ=ρ, c=c, γ=γ, β=β,
useconstraint=useconstraint, truehess=truehess)

#========= independence vs eigenvalues ========================#
function colnormalize!(A) # column normalize
    for i in 1:size(A,2)
        A[:,i] ./= norm(A[:,i])
    end
    A
end

A = ones(14,14)+6*I
colnormalize!(A)
norm.(eigen(A).values)
reduce(*,norm.(eigen(A).values))
det(A)
WA = W*A
imshowW(WA,imgsz)

A = Matrix(1.0*I,14,14)
A[13,14] = 0.5
A[14,14] = 0.5
for i in 1:14
    A[:,i] ./= norm(A[:,i])
end
norm.(eigen(A).values)
det(A)
# orth = norm.(eigen(Ã).values) reflect how much each component is not orthogonal
# where Ã is a column normalized A
# reduce(*,orth) == det(Ã)
# If all orth are 1 then A is orthogonal

p = 14
W = Matrix(1.0*I,p,p)
H = Matrix(1.0*I,p,p)
#W, H = copy(W0), copy(H0)
Mx = rand(p,p)
Wx = W*Mx
all(Wx[:,rand(1:p)].>0) # if all((W*M)[:,rand(1:p)].>0) is always true (define as positive relation)
Hx = Mx\H
all(Hx[rand(1:p),:].>0) # then all((M\H)[rand(1:p),:].>0) is always false ((define as non positive relation))
Wx'*Wx
Hx*Hx'

Worg = Horg = [1 0.5; 0.5 1]
X = Worg*Horg

W = svd(X).U
H = W\X
M = W*Worg
M\H
colnormalize!(M)

for i in 1:10
#    M = rand(14,14).-0.5
    M = rand(14,14) + i*I .- 0.5
    colnormalize!(M)
    Minv = inv(M)
    Minv=colnormalize!(Minv')' # row normalize
    MM = Minv*Minv'
    @show det(M), det(Minv), sum(MM[MM.<0])
end

#========= How does the origin look like in gradient sense? ========================#
X, F, U, W0, H0, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_mstfalse_(15).jld") # 9, 15

tol = 1e-12
#scapair0(W,H) = (sca2(W)+tol)*(sca2(H)+tol)
scapair0(W,H) = scapair(W,H)

α = 0.125
dMs = []
for i in 1:100
    push!(dMs,rand(14,14).-0.5)
end

approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x

W = copy(W0)
H = copy(H0)
A, b = POCA.buildproblem(W, H; order=0, truehess=true)
x = svd(A)\b.*-1
dMstep = POCA.reshapex(b.*-1,(p,p),false)
dMs[1] = dMstep.*(3.7/norm(dMstep))

W = W*(I+dMs[1]*3)
H = W\X
scapair0(W,H)

# true obj
obj(dM,W,H) = scapair0(W*(I+dM), (I+dM)\H)
objname = "true obj"
order = 2

A, b = POCA.buildproblem(W, H; order=order, truehess=false)
#A = Matrix(cholesky(Positive, A, Val{false})) # positive PositiveFactorizations

true_vec(x, W, H, p) = (dM = POCA.reshapex(x, (p,p), false); obj(dM,W,H))

plotlayout = (4,6); plotnum = *(plotlayout...)
fig, axs = plt.subplots(plotlayout..., figsize=(5,4),gridspec_kw=Dict("width_ratios"=>ones(plotlayout[2])))
for i in 1:plotnum
    ax = axs[i]
    dM = dMs[i+48]
    objold = scapair0(W, H)
    objnew = 10^9#scapair0(W*(I+3*dM), (I+α*dM)\H)
    truefn(α, W, H) = obj(α*dM,W,H)
    approxfnPOCA(α, W, H) = approxobj(α*vec(dM), scapair0(W,H), A, b)
    plotfigures(ax,α,W,H,objold,objnew,truefn,approxfnPOCA,show_minimizer=false,fixrng=true,rng=-1:0.001:1)
end
legend([objname, "quad. approx.", "origin"],fontsize = 12,loc=4)
plt.show(block=false)
plt.savefig("approxPOCA_$(objname)_o$(order)_wo_pivot_loc2.png")

#============= manual optimization ================#
tol = 1e-12
scapair0(W,H) = (sca2(W)+tol)*(sca2(H)+tol)
# scapair0(W,H) = scapair(W,H)

function gradient_direction_manual_optimize(α,W,H; fixxrng=false, xrng=rng=-0.001:0.00001:0.001, fixyrng=false,objold=scapair(W, H),objnew=0)
    A, b = POCA.buildproblem(W, H; order=0, truehess=true)
    obj(dM,W,H) = scapair0(W*(I+dM), (I+dM)\H)
    approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x
    B = POCA.reshapex(b,(p,p),false).*-1
    if !fixyrng
        objold = scapair0(W, H)
        objnew = scapair0(W*(I+α*B), (I+α*B)\H)
    end
    truefn(α, W, H) = obj(α*B,W,H)
    approxfnPOCA(α, W, H) = approxobj(α*vec(B), scapair0(W,H), A, b)
    fig, ax = plt.subplots(1,1, figsize=(5,4),gridspec_kw=Dict("width_ratios"=>ones(1)))   
    plotfigures(ax,α,W,H,objold,objnew,truefn,approxfnPOCA,show_minimizer=true,fixrng=fixxrng,rng=xrng)
    plt.show(block=false)
    B
end

W,H = copy(W0), copy(H0)
α, objold, objnew, xrng = 0.0001,30000,0, -0.0001:0.000001:0.0002
B = gradient_direction_manual_optimize(α,W,H; fixxrng = true, xrng=xrng, fixyrng=true, objold=objold, objnew=objnew)
M0 = I + α*B
W = W*M0; H = M0\H
α, objold, objnew, xrng = 0.0022, 7000, 0, -0.001:0.000001:0.003
B = gradient_direction_manual_optimize(α,W,H; fixxrng = false, xrng=xrng, fixyrng=false, objold=objold, objnew=objnew)
M1 = I + α*B
W = W*M1; H = M1\H
normalizeWH!(W,H)
imshowW(W,imgsz)
M = M0*M1
colnormalize!(M)
det(M) # 1.7308803294794258e-9


function step_direction_manual_optimize(α,W,H,constraint=false; fixxrng=false, xrng=rng=-0.001:0.00001:0.001, fixyrng=false, objold=scapair(W, H), objnew=0)
    A, b = POCA.buildproblem(W, H; order=0, truehess=true)
    obj(dM,W,H) = scapair0(W*(I+dM), (I+dM)\H)
    approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x

    btr = zero(b)
    for i = 1:p btr[i+(i-1)*p] = 1; end
    Asolve = svd(A); x= Asolve \ b
    dM = POCA.reshapex(x, (p, p), false)
    if constraint
        xtr = Asolve \ btr; dMtr = POCA.reshapex(xtr, (p, p), false)
        λ = - tr(dM)/tr(dMtr)
        dM += λ*dMtr; x += λ*xtr
    end
    if b'*x > 0
        x *= -1
        dM *= -1
    end
    if !fixyrng
        objold = scapair0(W, H)
        objnew = scapair0(W*(I+α*dM), (I+α*dM)\H)
    end
    truefn(α, W, H) = obj(α*dM,W,H)
    approxfnPOCA(α, W, H) = approxobj(α*vec(dM), scapair0(W,H), A, b)
    fig, ax = plt.subplots(1,1, figsize=(5,4),gridspec_kw=Dict("width_ratios"=>ones(1)))
    plotfigures(ax,α,W,H,objold,objnew,truefn,approxfnPOCA,show_minimizer=true,fixrng=fixxrng,rng=xrng)
    plt.show(block=false)
    dM
end

constraint = false
W,H = copy(W0), copy(H0)
α, objold, objnew, xrng  = -1,25454,25450, -100:0.1:1.2
dM = step_direction_manual_optimize(α,W,H,constraint; fixxrng = true, xrng=xrng, fixyrng=true, objold=objold, objnew=objnew)
M = I + α*dM
W = W*M; H = M\H
α, objold, objnew, xrng = 0.002, 7000, 0, -0.001:0.00001:0.003
dM = step_direction_manual_optimize(α,W,H,constraint; fixxrng = false, xrng=xrng, fixyrng=false, objold=objold, objnew=objnew)
M = I + α*dM
W = W*M; H = M\H
normalizeWH!(W,H)
imshowW(W,imgsz)

#========= How does optimal M look like? =============================================================#
include(joinpath(pkgdir(PhaseOptimizedComponentAnalysis),"test","obsoletePOCA.jl"))
makeimg(dM) = (dMimg = dM.+(-minimum(dM)); dMimg./maximum(dMimg))
function interpolate(dM,n)
    dMinter = zeros(size(dM).*(n+1))
    offset = CartesianIndex(1,1)
    for i in CartesianIndices(dM)
        ii = (i-offset)*n+i
        for j in CartesianIndices((n+1,n+1))
            dMinter[ii+j-offset] = dM[i]
        end
    end
    dMinter
end
function colnormalize!(A) # column normalize
    for i in 1:size(A,2)
        A[:,i] ./= norm(A[:,i])
    end
    A
end

datasets = [(0, false)]#, # order=0; fixposdef=false (Order 1 wo corssterm)
#             (1, true)]# # order=1; fixposdef=true (order 1 w PF)
# datasets = [(1, false), # order=1; fixposdef=false (order 1 wo PF)
#             (2, true)] # order=1; fixposdef=true (order 1 w PF)
# datasets = [(2, false)] # order=1; fixposdef=false (order 1 wo PF)
rng=1:10
minLssss = []; Lsss = []; rt = []; dfnamess=[]; ofnamess=[]
@time for (order, fixposdef) in datasets
    # Original w constraint
    minLsss, Lss, rt0, rt1, rt2, sscaiter, dfnames, ofnames = runexp(rng; order = order, fixposdef = fixposdef, skew=false,
            c=0.5, β = 0, useconstraint=true, γ=0, usepdratio=false, itermax=100, 
            objtol=1e-4, multiplestep=true)
    push!(Lsss,Lss)
    push!(rt,rt0)
    push!(rt,rt1)
    push!(rt,rt2)
    push!(rt,sscaiter)
    push!(dfnamess,dfnames)
    push!(ofnamess,ofnames)
end

dfnamess = [[
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(1).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(2).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(3).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(4).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(5).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(6).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(7).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(8).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(9).jld",
 "fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(10).jld"
]]

ofnamess = [[
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(1).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(2).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(3).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(4).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(5).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(6).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(7).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(8).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(9).jld",
 "linesearch_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_msttrue_(10).jld"
 ]]

function sortmatchlist(M,matchlist)
    p = size(M,1)
    Msort = rand(p,p)
    for i in 1:length(matchlist)
        Msort[:,matchlist[i][1]] = M[:,matchlist[i][2]]
    end
    done = getindex.(matchlist,2)
    j=length(matchlist)+1
    for i in 1:p
        if i in done
            continue
        end
        Msort[:,j] = M[:,i]
        j += 1
    end
    Msort
end

p= 14; rng=8:10
Msum = zeros(p,p)
for r in rng
    ddic = load(dfnamess[1][r])
    odic = load(ofnamess[1][r])
    X = ddic["imgrs"]
    imgsz = ddic["imgsz"]
    M = odic["M"]
    matchlist = odic["matchlist"]
    p = size(M,1)
    U = svd(X).U[:,1:p]
    W=U*M
    H = W\X
    normalizeWH!(W,H)
    imshowW(W,imgsz)

    Msort = sortmatchlist(M,matchlist)
    W=U*Msort
    H = W\X
    normalizeWH!(W,H)
    imshowW(W,imgsz)
#    imsaveW("Wimg_r$(r).png",W,imgsz)

    interp_num = 5
    Mimg = interpolate(makeimg(Msort),interp_num)
    #save("Mimg_r$(r).png",Mimg)
    @show eigen(Msort).values
    Msum += Msort
end
Msum./=length(rng)
interp_num = 5
Mimg = interpolate(makeimg(Msum),interp_num)
save("Mavg.png",Mimg)

# Check the determinent of column normalized M
for r in rng
    odic = load(ofnamess[1][r])
    matchlist = odic["matchlist"]
    M = odic["M"]
    Msort = sortmatchlist(M,matchlist)
    M7 = Msort[1:7,1:7]
    for i in 1:14 # column normalize
        Msort[:,i] ./= norm(Msort[:,i])
    end
    for i in 1:7 # column normalize
        M7[:,i] ./= norm(M7[:,i])
    end
    @show det(Msort), det(M7)
#    @show norm.(eigen(Msort).values)
end

# by M = W\gtW

p=14; rng=1:10
Msum = zeros(p,p)
for r in rng
    ddic = load(dfnamess[1][r])
    X = ddic["imgrs"]
    imgsz = ddic["imgsz"]
    gtW = [ddic["gtW"] colnormalize!(reshape(ddic["gtbg"][:,:,1:7],(800,7)))]
    W = svd(X).U[:,1:14]
    M = W\gtW

    interp_num = 5
#   Mimg = interpolate(makeimg(M),interp_num)
#   ImageView.imshow(Mimg)
#   imsaveW("W.png",W,imgsz)
#   imsaveW("gtW.png",W*M,imgsz)
#   save("Mimg_r$(r).png",Mimg)
    Msum += M
end
Msum./=length(rng)
interp_num = 5
Mimg = interpolate(makeimg(Msum),interp_num)
save("Mavg.png",Mimg)

# Check the determinent of column normalized M
for r in rng
    ddic = load(dfnamess[1][r])
    X = ddic["imgrs"]
    imgsz = ddic["imgsz"]
    gtW = [ddic["gtW"] colnormalize!(reshape(ddic["gtbg"][:,:,1:7],(800,7)))]
    W = svd(X).U[:,1:14]
    M = W\gtW
    M7 = M[1:7,1:7]
    for i in 1:14 # column normalize
        M[:,i] ./= norm(M[:,i])
    end
    for i in 1:7 # column normalize
        M7[:,i] ./= norm(M7[:,i])
    end
    @show det(M), det(M7)
#    @show norm.(eigen(Msort).values)
end

# Lss = Lsss[1]
# Ls = zeros(maximum(length.(Lss)),length(Lss))
# display_rng= 1:10
# for (i,L) in enumerate(Lss)
#     if i in display_rng
#         Ls[1:length(L),i] = L
#     end
# end

# clf()
# plotlayout = (1,1); horsize, versize = 5, 4
# fig, ax = plt.subplots(plotlayout..., figsize=(horsize, versize), gridspec_kw=Dict("width_ratios"=>ones(1)))
# ax.plot(log10.(Ls))
# legend(collect(display_rng),fontsize = 12,loc=7)
# plt.show(block=false)
# plt.savefig("convergence.png")

    # # wo constraint
    # POCA.runexp(rng; order = order, fixposdef = fixposdef, β = 0,
    #             useconstraint=false, γ=0, usepdratio=false)
    # # Power difference ratio
    # POCA.runexp(rng; order = order, fixposdef = fixposdef, β = 0,
    #             useconstraint=true, γ=0, usepdratio=true)
    # # Add difference term
    # POCA.runexp(rng; order = order, fixposdef = fixposdef, β = 0,
    #            useconstraint=true, γ=1, usepdratio=false)
    # # Add independence term to penalty
    # POCA.runexp(rng; order = order, fixposdef = fixposdef, β = 0.5,
    #            useconstraint=true, γ=0, usepdratio=false)
    # # Add independence term only to step
    # POCA.runexp(rng; order = order, fixposdef = fixposdef, β = 0.5,
    #            useconstraint=true, γ=0, usepdratio=false)
    # # Add independence term only to step wo constraint
    # POCA.runexp(rng; order = order, fixposdef = fixposdef, β = 0.5,
    #             useconstraint=false, γ=0, usepdratio=false)

#====== Why true hessian converge too late to the gradient = 0 point ===========#

W = copy(U); H=W\X
αs = -1.0:0.00001:0; p = size(W,2)
scapair_vec(x, W, H, p) = (dM = POCA.reshapex(x, (p,p), false); scapair(W*(I+dM),(I+dM)\H; allcompW=false, allcompH=false))
b = ForwardDiff.gradient(x -> scapair_vec(x, W, H, p), zeros(p^2))./2
A = ForwardDiff.hessian(x -> scapair_vec(x, W, H, p), zeros(p^2))./2
Asolve = svd(A); x= Asolve \ b
dM = POCA.reshapex(x, (p, p))
B = POCA.reshapex(b, (p, p))

αs = -30000000.0:1000:0
objsx = [scapair(I + α*dM, W, H) for α in αs]
origin = fill(scapair(W,H),length(αs))
res = 1e-9 # 1e5
ya = ("Penalty", (origin[1]-res,origin[1]+res), origin[1]-res:res/3:origin[1]+res)
Plots.plot(αs, [objsx origin], yaxis=ya)

αs = -0.0001:0.0000001:0.0001;
objsb = [scapair(I + α*B, W, H) for α in αs]
origin = fill(scapair(W,H),length(αs))
ya = ("Penalty", (0,10^5), 0:10^4:10^5)
Plots.plot(αs, [objsb origin], yaxis=ya)

#====== Check hessian and gradient ==========================================#
α = 0.125
dM = rand(14,14)

approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x

# true obj
# obj(dM,W,H) = scapair(W*(I+dM), (I+dM)\H)
# objname = "true obj"
# order = 1

# order 1
obj(dM,W,H) = scapair(W*(I+dM), (I-dM)*H)
objname = "order 1 obj"
order = 1

# order 2
# obj(dM,W,H) = scapair(W*(I+dM), (I-dM+dM^2)*H)
# objname = "order 2 obj"
# order = 2

A, b = POCA.buildproblem(W, H; order=order, truehess=false)

true_vec(x, W, H, p) = (dM = POCA.reshapex(x, (p,p), false); obj(dM,W,H))
truefn(α, W, H) = obj(α*dM,W,H)

bf = ForwardDiff.gradient(x -> true_vec(x, W, H, p), zeros(p^2))./2
Af = ForwardDiff.hessian(x -> true_vec(x, W, H, p), zeros(p^2))./2
bc = Calculus.gradient(x -> true_vec(x, W, H, p), zeros(p^2))./2
Ac = Calculus.hessian(x -> true_vec(x, W, H, p), zeros(p^2))./2
norm(b-bf) # 4091.1595482744506
norm(b-bc) # 1.2524102805260326e-5
norm(bf-bc) # 4091.1595482557436

objold = 25000#scapair(W, H)
objnew = 30000#scapair(W*(I+α*dM), (I+α*dM)\H)
approxfnFD(α, W, H) = approxobj(α*vec(dM), scapair(W,H), Af, bf)
approxfnCL(α, W, H) = approxobj(α*vec(dM), scapair(W,H), Ac, bc)
approxfnPOCA(α, W, H) = approxobj(α*vec(dM), scapair(W,H), A, b)

fig, ax = plt.subplots(1,1, figsize=(5,4),gridspec_kw=Dict("width_ratios"=>ones(1)))
plotfigures(ax,α,W,H,objold,objnew,truefn,approxfnFD,fixrng=true,rng=-0.2:0.0001:0.2)
legend([objname, "quadratic approx. obj", "origin"],fontsize = 12,loc=4)
plt.show(block=false)
plt.savefig("approxFD_$(objname).png")

fig, ax = plt.subplots(1,1, figsize=(5,4),gridspec_kw=Dict("width_ratios"=>ones(1)))
plotfigures(ax,α,W,H,objold,objnew,truefn,approxfnCL,fixrng=true,rng=-0.2:0.0001:0.2)
legend([objname, "quadratic approx. obj", "origin"],fontsize = 12,loc=4)
plt.show(block=false)
plt.savefig("approxCL_$(objname).png")

fig, ax = plt.subplots(1,1, figsize=(5,4),gridspec_kw=Dict("width_ratios"=>ones(1)))
plotfigures(ax,α,W,H,objold,objnew,truefn,approxfnPOCA,fixrng=true,rng=-0.2:0.0001:0.2)
legend([objname, "quadratic approx. obj", "origin"],fontsize = 12,loc=4)
plt.show(block=false)
plt.savefig("approxPOCA_$(objname).png")
plt.savefig("approxPOCA_order0.png")

invapprox_vec(x, W, H, p) = (dM = POCA.reshapex(x, (p,p), false); scapair(W*(I+dM),(I-dM)*H))
bf = ForwardDiff.gradient(x -> invapprox_vec(x, W, H, p), zeros(p^2))./2
bc = Calculus.gradient(x -> invapprox_vec(x, W, H, p), zeros(p^2))./2
norm(b-bf) # 0
norm(b-bc) # 1.2218563832784719e-5
norm(bf-bc) # 1.2218563832784719e-5

#====== Why true objective and approximation objective using true hessian doesn't relevent ===========#
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_mstfalse_(15).jld") # 9, 15

Wt, Ht, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W, X; flipsign=false, order=0, fixposdef=false,
    skew=false, c=0.5, β=0, useconstraint=false, γ=0, usepdratio=false, truehess=true,
    itermax=30, showfig=true, plotssign=1,plotiters=(1,30),objtol=1e-4, showdbg=true)
Wtt, Htt, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(Wt, X; flipsign=false, order=0, fixposdef=false,
    skew=false, c=0.5, β=0, useconstraint=true, γ=0, usepdratio=false, truehess=true,
    itermax=1, showfig=true, plotssign=1,plotiters=(1,30),objtol=1e-4, showdbg=true)

p = size(Wt, 2)
A, b = POCA.buildproblem(Wt, Ht; truehess=true)
Aapprox, bapprox = POCA.buildproblem(Wt, Ht; truehess=false)
btr = zero(b)
for i = 1:p btr[i+(i-1)*p] = 1; end
Asolve = svd(A); x= Asolve \ b
dM = POCA.reshapex(x, (p, p), false)
if true
    xtr = Asolve \ btr; dMtr = POCA.reshapex(xtr, (p, p), false)
    λ = - tr(dM)/tr(dMtr)
    dM += λ*dMtr; x += λ*xtr
end
if b'*x > 0
    x *= -1
    dM *= -1
end

approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x
α = 0.125
fig, ax = plt.subplots(1,1, figsize=(4,4),gridspec_kw=Dict("width_ratios"=>ones(1)))
objold = scapair(Wt, Ht)
objnew = scapair(Wt*(I+α*dM), (I+α*dM)\Ht)
truefn(α, W, H) = scapair(I + α*dM, W, H)
approxfn(α, W, H) = approxobj(α*vec(dM), scapair(W,H), A, b)
plotfigures(ax,α,Wt,Ht,objold,objnew,truefn,approxfn)
plt.show(block=false)

scapair_vec(x, W, H, p) = (dM = POCA.reshapex(x, (p,p), false); scapair(W*(I+dM),(I+dM)\H))

b = ForwardDiff.gradient(x -> scapair_vec(x, Wt, Ht, p), zeros(p^2))./2
bc = Calculus.gradient(x -> scapair_vec(x, Wt, Ht, p), zeros(p^2))./2
bapp = ForwardDiff.gradient(x -> approxobj(x, pWH, A, b), zeros(p^2))./2
norm(b-bapp)

balpha = ForwardDiff.derivative(alpha -> scapair_vec(vec(alpha*dM), Wt, Ht, p), 0)./2
balphaapp = ForwardDiff.derivative(alpha -> approxobj(vec(alpha*dM), pWH, A, b), 0)./2

balpha = ForwardDiff.derivative(alpha -> truefn(alpha, Wt, Ht), 0)./2
balphaapp = ForwardDiff.derivative(alpha -> approxfn(alpha, Wt, Ht), 0)./2

balpha = Calculus.derivative(alpha -> truefn(alpha, Wt, Ht), 0)./2
balphaapp = Calculus.derivative(alpha -> approxfn(alpha, Wt, Ht), 0)./2


Identity = Matrix(1.0I,p,p)
balpha = ForwardDiff.derivative(alpha -> scapair_vec(vec(alpha*Identity), Wt, Ht, p), 0)./2
balphaapp = ForwardDiff.derivative(alpha -> approxobj(vec(alpha*Identity), pWH, A, b), 0)./2

#====== Test true hessian =====================================================#
R = Matrix(1.0I,ncells,ncells)
scheduler = Sequential{typeof(zero(eltype(U))*zero(eltype(R)))}(size(U, 2)) # good(full(50sec))
#scheduler = Priority{typeof(zero(eltype(U))*zero(eltype(R)))}() # bad
#scheduler = PriorityRandom{typeof(zero(eltype(U))*zero(eltype(R))),Float64}(size(U, 2)) # best(full,ubound())
M, sign_flip_idx, minL, minLss = mostly_positive_combination_plot(minimizeLij,Rθ,U; 
        scheduler=scheduler, tol=1e-4, maxiter=1000, show_debug=false, enplot=false, finetune=true) # minimizeLij_ubound

W0 = U*M
H0= W0\X
sca2(H0)

W2, H2, objvalsorder2, Ms, A, endval, iters, idx  = semiscasolve!0_2ndstep(W0, H0; order=0, fixposdef=false,
    β=0, allcompW = false, allcompH = true, useconstraint=true, γ=0, itermax=5,
    showfig=false, plotiters=(1,30), objtol=0.01, showdbg=false, c=1, truehess=false)

Wt, Ht, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W2, X; flipsign=false, order=0, fixposdef=false,
    skew=false, c=0.5, β=0, useconstraint=true, γ=0, usepdratio=false, truehess=true,
    itermax=1, showfig=true, plotssign=1,plotiters=(1,30),objtol=1e-4, showdbg=true)


A, b = As[1], bs[2]
btr = zero(b)
for i = 1:p btr[i+(i-1)*p] = 1; end
Asolve = svd(A); x= Asolve \ b
dM = POCA.reshapex(x, (p, p))

bs1 = bs[1]'*(As[1]\bs[1])
bs2 = bs[2]'*(As[2]\bs[2])
bs3 = bs[3]'*(As[3]\bs[3])

@show all(eigen(As[1]).values.>=0), all(eigen(As[1]).values.<=0)
@show all(eigen(As[2]).values.>=0), all(eigen(As[2]).values.<=0)
@show all(eigen(As[3]).values.>=0), all(eigen(As[3]).values.<=0)

Wt1, Ht1 = copy(W2), copy(H2)
Wt1, Ht1 = copy(Wt), copy(Ht)
scapair(Wt1,Ht1)
sca2(Wt1)
normalizeWH!(Wt1,Ht1) # penalty = 68.67353905890415, sca2(W) = 0.0008951706849394717
imshowW(Wt1,imgsz)
imsaveW("trueHessian_with_constraint.png",W1,imgsz)
PyPlot.savefig("trueHessian_with_constraint_linesearch.png")

#====== Test true hessian =====================================================#
include(joinpath(pkgdir(PhaseOptimizedComponentAnalysis),"test","obsoletePOCA.jl"))
makeimg(dM) = (dMimg = dM.+(-minimum(dM)); dMimg./maximum(dMimg))
function interpolate(dM,n)
    dMinter = zeros(size(dM).*(n+1))
    offset = CartesianIndex(1,1)
    for i in CartesianIndices(dM)
        ii = (i-offset)*n+i
        for j in CartesianIndices((n+1,n+1))
            dMinter[ii+j-offset] = dM[i]
        end
    end
    dMinter
end

# best
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_mstfalse_(16).jld") # 16, 19

# worst
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_mstfalse_(15).jld") # 9, 15

# Wt, Ht, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0_2ndstep(W, H; order=0, fixposdef=false,
#     allcompH=true, c=0.5, β=0, useconstraint=false, γ=0, itermax=2000,
#     showfig=true, plotiters=(1,30),objtol=1e-4, showdbg=true)

Wt, Ht, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W, X; flipsign=true, order=0, fixposdef=false,
    skew=false, c=0.5, β=0, useconstraint=true, γ=0, usepdratio=false, itermax=100,
    showfig=false, plotssign=1,plotiters=(1,30),objtol=1e-4, showdbg=true, truehess=true)

W1, H1 = copy(Wt), copy(Ht)
scapair(W1,H1)
sca2(W1)
normalizeWH!(W1,H1) # penalty = 68.67353905890415, sca2(W) = 0.0008951706849394717
imshowW(W1,imgsz)
imsaveW("trueHessian_with_constraint.png",W1,imgsz)
PyPlot.savefig("trueHessian_with_constraint_linesearch.png")

Wtwoc, Htwoc, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W, X; flipsign=false, order=0, fixposdef=false,
    skew=false, c=0.5, β=0, useconstraint=false, γ=0, usepdratio=false, itermax=3,
    showfig=false, plotssign=1,plotiters=(1,30),objtol=1e-4, showdbg=true)

iter = 3; interp_num = 5; p = size(W,2)

A, b = As[iter], bs[iter]
B = POCA.reshapex(b, (p, p))
Bimg = interpolate(makeimg(B),interp_num)
save("true_b$(iter)_without_constraint.png",Bimg)

btr = zero(b)
for i = 1:p btr[i+(i-1)*p] = 1; end
Asolve = svd(A); x= Asolve \ b
dM = POCA.reshapex(x, (p, p))
dMimg = interpolate(makeimg(dM),interp_num)
#ImageView.imshow(dMimg)
save("true_dM$(iter)_without_constraint.png",dMimg)

sleep(1)
xtr = Asolve \ btr; dMtr = POCA.reshapex(xtr, (p, p))
λ = - tr(dM)/tr(dMtr)
dM += λ*dMtr
x += λ*xtr
dMimg = interpolate(makeimg(dM),interp_num)
save("true_dM$(iter)_with_constraint.png",dMimg)


W1, H1 = copy(Wtwoc), copy(Htwoc)
scapair(W1,H1)
sca2(W1)
normalizeWH!(W1,H1) # penalty = 25451.492416409514, sca2(W) = 123.73117071016868
imshowW(W1,imgsz)
imsaveW("approxHessian_with_constraint.png",W1,imgsz)
PyPlot.savefig("approxHessian_with_constraint_linesearch.png")

#====== Test true hessian =====================================================#

for i in 2:10
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_nonorth_o0_PFfalse_skfalse_c0.5_b0false_uctrue_r0_updrfalse_mstfalse_($i).jld") # 16, 19

#    W[:,1] .*= -1
    Wtwoc, Htwoc, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(W, X; flipsign=false, order=0, fixposdef=false,
        skew=false, c=0.5, β=0, useconstraint=false, γ=0, usepdratio=false, itermax=3,
        showfig=false, plotssign=1,plotiters=(1,30),objtol=1e-2, showdbg=false)

        btr = zero(b)
        for i = 1:p btr[i+(i-1)*p] = 1; end
        Asolve = svd(A); x= Asolve \ b
        dM = POCA.reshapex(x, (p, p))
        
    bs1 = bs[1]'*(As[1]\bs[1])
    bs2 = bs[2]'*(As[2]\bs[2])
    bs3 = bs[3]'*(As[3]\bs[3])
    @show i, bs1, bs2, bs3
    @show all(eigen(As[1]).values.>=0), all(eigen(As[1]).values.<=0)
    @show all(eigen(As[2]).values.>=0), all(eigen(As[2]).values.<=0)
    @show all(eigen(As[3]).values.>=0), all(eigen(As[3]).values.<=0)
end

#====== Orthogonal POCA test ===================================================#
include(joinpath(pkgdir(PhaseOptimizedComponentAnalysis),"test","obsoletePOCA.jl"))

order = 0; fixposdef = false; c = 0.5; β = 0; useconstraint=true; γ=0; usepdratio=false;
                itermax=2000; objtol=1e-3; showfig=false; plotssign=1; plotiters=(1,30); multiplestep=true;
i=40 # 6, 13
paramstrsuffix = "$(order)_PF$(fixposdef)_c$(c)_b$(β)_uc$(useconstraint)_r$(γ)_updr$(usepdratio)_mst$(multiplestep)_($i)"
datafilename = "fc_nonorth_o$paramstrsuffix.jld"
X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell(datafilename)

# Full POCA with Sequential scheduler
scheduler = Sequential{typeof(zero(eltype(U))*zero(eltype(R)))}(size(U, 2))
M, sign_flip_idx, minL = mostly_positive_combination_plot(minimizeLij,Rθ,U; 
    scheduler=scheduler, tol=1e-4, maxiter=1e5, show_debug=false, enplot=true, plotsignidx=0, plotiters = collect(201:201))
imshowW(U*M,imgsz)

# GUB POCA with PriorityRandom scheduler
scheduler = PriorityRandom{typeof(zero(eltype(U))*zero(eltype(R))),Float64}(size(U, 2))
M2, sign_flip_idx, minL = mostly_positive_combination_plot(minimizeLij_ubound_plot,Rθ,U*M; 
    scheduler=scheduler, tol=1e-4, maxiter=1e5, show_debug=false, enplot=false, plotsignidx=0, plotiters = collect(1:200))
imshowW(U*M*M2,imgsz)

#========== Find only E(H) = 0 ======================================================#

gt_ncells, imgrs, gtW, gtH, gtWimgc, gtbg, found = POCA.findcase(order=1,fixposdef=true, β = 0.0, γ=0, numtrial=200)
X = imgrs

#rs = 0.4:0.01:0.6; numinneriters = 10
rs = 0.4:0.1:0.6; numinneriters = 2
iternumfirst = []
for r = rs
    @show r
    is = 0
    for i = 1:numinneriters
        print("$i ")
        X, gtW, gtH, W, H, iter, found = POCA.findcase2(r, 1; order=1,fixposdef=true, β = 0.0, γ=0, numtrial=20000)
        is += iter
    end
    println("")
    push!(iternumfirst,is/numinneriters)
end

plot(rs,iternumfirst)
xlabel("r")
ylabel("iteration number")
title("non-othogonality vs. fail first iteration #")

WHrps = 1:0.2:5; numinneriters = 50
iternumfirst = []
for whrp = WHrps
    @show whrp
    is = 0
    for i = 1:numinneriters
        print("$i ")
        X, gtW, gtH, W, H, iter, found = POCA.findcase2(0.45, 10^whrp, order=1,fixposdef=true, β = 0.0, γ=0, numtrial=20000)
        is += iter
    end
    println("")
    push!(iternumfirst,is/numinneriters)
end

plot(WHrps,iternumfirst)
xlabel("log10(w)")
ylabel("iteration number")
title("power difference vs. fail first iteration #")

iternumfirst = []; ratio = []; i = 0
for i in 1:200
    print("$i ")
    X, gtW, gtH, W, H, iter, found = POCA.findcase2(0.45, 1, 5, order=1,fixposdef=true, β = 0.0, γ=0, numtrial=20000)
    W0 = svd(X).U
    H0 = W0\X
    push!(ratio,sca2(W0)/sca2(H0))
    push!(iternumfirst,iter)
end

scatter(ratio,iternumfirst)
xlabel("sca2(W0)/sca2(H0)")
ylabel("iteration number")
title("penalty ratio vs. fail first iteration #")

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

β=0; γ=0; order=0; fixposdef=false; showfig=false; plotssign=-1
W, H, objvalsorder2, Ms, As, endval, iters, idx  = POCA.semiscasolve!0(U,X,c=1.0,itermax=2000,
    plotssign=plotssign,plotiters=(1,30),β=β, γ=γ, showfig=showfig, order=order, fixposdef=fixposdef)

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
scapair0(M, W, H) = POCA.sca2(W*M) * POCA.sca2(svd(M)\H)
penW=[sca2(W0*M2(b,c)) for b in rng, c in rng]
penH=[sca2(svd(M2(b,c))\H0) for b in rng, c in rng]
penpair=[scapair0(M2(b,c),W0,H0) for b in rng, c in rng]
# Dataset 0
# POCA.plotL(penW,penH,penpair,rng;vmaxL1=0.5,vminL1=0,vmaxL2=0.01,vminL2=0,vmaxL3=0.004,vminL3=0)
# POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng; vmaxL1=2,vminL1=-5, vmaxL2=-1, vminL2=-5,vmaxL3 = -1, vminL3=-5)
# Dataset 1
# POCA.plotL(penW,penH,penpair,rng,bcs;vmaxL1=sca2(W0),vminL1=0,vmaxL2=0.002,vminL2=0,vmaxL3=0.002,vminL3=0,showtrace=showtrace)
# POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng,bcs; vmaxL1=1,vminL1=-0.2,vmaxL2=-2.0,vminL2=-5.0,vmaxL3 = -2.0, vminL3=-5.0,showtrace=showtrace)
POCA.plotL(penW,penH,penpair,rng;vmaxL1=sca2(W0),vminL1=0,vmaxL2=0.002,vminL2=0,vmaxL3=0.002,vminL3=0,showtrace=showtrace)
POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng; vmaxL1=1,vminL1=-0.2,vmaxL2=-2.0,vminL2=-5.0,vmaxL3 = -2.0, vminL3=-5.0,showtrace=showtrace)
# Dataset 2
# POCA.plotL(penW,penH,penpair,rng,bcs;vmaxL1=sca2(W0),vminL1=0,vmaxL2=4,vminL2=0,vmaxL3=4,vminL3=0)
# POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng,bcs; vmaxL2=2, vminL2=-0.2,vmaxL3 = 2, vminL3=-0.2)
POCA.plotL(penW,penH,penpair,rng;vmaxL1=sca2(W0),vminL1=0,vmaxL2=4,vminL2=0,vmaxL3=4,vminL3=0)
POCA.plotL(log10.(penW),log10.(penH),log10.(penpair),rng; vmaxL2=2, vminL2=-0.2,vmaxL3 = 2, vminL3=-0.2)

function Independency(W,H)
    WTW = W'W; HHT = H*H'
    for i in 1:size(W,2)
        WTW[i,i] = 0
        HHT[i,i] = 0
    end
    norm(WTW)*norm(HHT)/(norm(W)*norm(H))^2
end

#======== difference penalty ================================================#
p = size(W,2)
dM = zeros(p,p)
# difference penalty
A, b = add_difference(W, H)
pendiff(x, W, H) = (dM = reshape(x, (p,p)); (POCA.sca1n(W*(I+dM))-POCA.sca1n((I-dM)*H))^2)
grad = ForwardDiff.gradient(x -> pendiff(x, W, H), vec(dM))
Hess = ForwardDiff.hessian(x -> pendiff(x, W, H), vec(dM))
norm(grad-2b) # 1.1433020935327523e-9
norm(Hess-2A) # 9.330778150324901e-10

A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H; order=1,γ=1.0)
scapair_vec(x, W, H) = (dM = reshape(x, (p,p)); scapair(W*(I+dM),(I-dM)*H,γ=1.0))
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), vec(dM))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), vec(dM))
norm(grad-2b) # 1.1132775960109558e-9
norm(Hess-2A) # 9.331355921485172e-10

#=== compare gradient and hessian of E(W) for true obj and approximation obj ===#
AW, bW = zeros(p^2, p^2), zeros(p^2)
PhaseOptimizedComponentAnalysis.add_direct!(AW, bW, W)
# gradient and hessian of I-dM approximation (gradaproxH, HessaproxH) 
approxWf(x) = (dM = reshape(x, (p,p)); PhaseOptimizedComponentAnalysis.sca2(W*(I+dM)))
gradaproxW = ForwardDiff.gradient(approxWf,zeros(p*p))
HessaproxW = ForwardDiff.hessian(approxWf,zeros(p*p))
# True gradient and hessian (gradW, HessW : ForwardDiff)
norm(gradaproxW-2bW) # 0
norm(HessaproxW-2AW) # 0
CgradaproxW = Calculus.gradient(approxWf, zeros(p*p))
CHessaproxW = Calculus.hessian(approxWf, zeros(p*p))
# True gradient and hessian (CgradW, CHessW : Calculus)
norm(gradaproxW-2bW) # 0
norm(CHessaproxW-2AW) # 0.003060868933564446

#=== compare gradient and hessian of E(H) for true obj and approximation obj by (I+ΔM)⁻¹ ≈ I-ΔM  ===#
AH, bH = zeros(p^2, p^2), zeros(p^2)
PhaseOptimizedComponentAnalysis.add_transpose!(AH, bH, H)
# gradient and hessian of I-dM approximation (gradaproxH, HessaproxH) 
approxHf(x) = (dM = reshape(x, (p,p)); PhaseOptimizedComponentAnalysis.sca2((I-dM)*H))
gradaproxH = ForwardDiff.gradient(approxHf,zeros(p*p))
HessaproxH = ForwardDiff.hessian(approxHf,zeros(p*p))
# True gradient and hessian (gradH, HessH : ForwardDiff)
norm(gradaproxH-2bH) # 0
norm(HessaproxH-2AH) # 0
CgradaproxH = Calculus.gradient(approxHf, zeros(p*p))
CHessaproxH = Calculus.hessian(approxHf, zeros(p*p))
# True gradient and hessian (CgradH, CHessH : Calculus)
norm(gradaproxH-2bH) # 7.280942764139423e-7
norm(CHessaproxH-2AH) # 36807.07815721351 (fail)

#====================================================================#

W, H, objval, alphas, dMs = semiscasolve!(U, X, itermax=1)

normalizeWH!(W,H)
imshowW(W,imgsz)
imsaveW("btls_a2_random_rho_c0.9.png",W,imgsz)

approxobj(x, pW, AW, bW, pH, AH, bH) = pW*pH + 2*(pH*bW + pW*bH)'*x + x'*(pH*AW + (bW.*bH' + bH.*bW') + pW*AH)*x
approxobj(x, pW, pH, A, b) = pW*pH + 2*b'*x + x'*A*x

W = copy(U)
H = W\X
#αs = -0.12:0.001:0.12
αs = -1.2:0.0001:2
iter = 0
Wi, Hi = W, H
for (alpha, dM) in zip(alphas,dMs)
    iter += 1
    M = I + alpha*dM
    if iter == 7
        A, b = PhaseOptimizedComponentAnalysis.buildproblem(Wi, Hi, order=2)
        pW = sca2(Wi)
        pH = sca2(Hi)
        objs = [scapair(I + α*dM, Wi, Hi) for α in αs]
        objsapprox = [approxobj(α*vec(dM), pW, pH, A, b) for α in αs]
        plot(αs, [objs  objsapprox])
        plot([alpha,alpha], [min(minimum(objs),minimum(objsapprox)),max(maximum(objs),maximum(objsapprox)),])
        minobj = min(minimum(objs),minimum(objsapprox))
        maxobj = max(maximum(objs),maximum(objsapprox))
        plot([0,0], [minobj, maxobj])
        axis([αs[1],αs[end],minobj,maxobj])
        @show iter, alpha
        legend(["true obj.", "quadratic approx. obj", "α"],fontsize = 12,loc=1)
    end
    Wi, Hi = Wi*M, M\Hi
end

time = @belapsed semiscasolve!(U, X)

function approxobjcoeff(W, H)
    m, p = size(W)
    bW, bH = zeros(p^2), zeros(p^2)
    AW, AH = zeros(p^2, p^2), zeros(p^2, p^2)
    PhaseOptimizedComponentAnalysis.add_direct!(AW, bW, W)
    PhaseOptimizedComponentAnalysis.add_transpose!(AH, bH, H)
    pW, pH = sca2(W), sca2(H)
    # The following come from the second derivative of a product
    return pW, AW, bW, pH, AH, bH
end


W = copy(U)
H = W\X
p = size(W, 2)
#αs = -0.12:0.001:0.12
αs = -1.2:0.001:2

# Compute the gradient and Hessian
A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H; order=2)
pW = sca2(W)
pH = sca2(H)
# Add a constraint forcing ΔM to have tr(ΔM) = 0, which corresponds to det(I + α*ΔM) ≈ 1
# I actually don't know if this is necessary
btr = zero(b); for i = 1:p btr[i+(i-1)*p] = 1; end   # from constraint on the trace
Asolve = svd(A)
x, xtr = Asolve \ b, Asolve \ btr
dM, dMtr = reshape(x, (p, p)), reshape(xtr, (p, p))
λ = - tr(dM)/tr(dMtr)
dM += λ*dMtr
x += λ*xtr
if b'*x > 0
    x = -x
    dM = -dM
end
α0 = 1; ρlo = 0.1; ρhi = 0.9; c = 0.9; itermax = 50
α = α0; iter = 0
fx = scapair(W, H); cbx = c*b'*x
while (scapair(I + α*dM, W, H) > fx + α*cbx) && (iter < itermax)
    iter += 1
    ρ = rand()*(ρhi-ρlo) + ρlo
    α = ρ*α
end
M = I + α*dM
# Line search
objs = [scapair(I + α*dM, W, H) for α in αs]
objs2 = [scapair(W*(I + α*dM), (I-α*dM+α^2*dM^2)*H) for α in αs]
#fzobjs = [freezescapair(I + α*dM, W, H) for α in αs]
#appinvfzobjs = [approxinvscapair(α*dM, W, H) for α in αs]

#pW, AW, bW, pH, AH, bH = approxobjcoeff(W, H)
#objsapprox = [approxobj(α*x, pW, AW, bW, pH, AH, bH) for α in αs]
objsapprox = [approxobj(α*x, pW, pH, A, b) for α in αs]
#plot(αs, [objs fzobjs appinvfzobjs objsapprox])
#axis([αs[1],αs[end],0,200000])
#legend(["true obj.", "freezed true obj.", "inverse approximated freezed true obj.", "approx. obj."],fontsize = 12,loc=1)

plot(αs, [objs objs2 objsapprox])
axis([αs[1],αs[end],0,200000])
legend(["true obj.", "approx. obj.", "quadratic approx. obj"],fontsize = 12,loc=1)

if false
    truef(alpha) = PhaseOptimizedComponentAnalysis.scapair(I+alpha*dM,W,H)
    trueg = x -> ForwardDiff.derivative(truef,x)

    approxobj(x, pW, pH, A, b) = pW*pH + 2*b'*x + x'*A*x
    approxf(alpha) = approxobj(alpha*x,pW,pH,A,b)
    approxg = x -> ForwardDiff.derivative(approxf,x)

    @test trueg(0) ≈ approxg(0)
end

α = argmin(objs)
M = I + α*dM
W = W*M; H = M\H

#========= Why do we need approximation : how fast and accurate =======================#
W = copy(U)
H = W\X

# ForwardDiff
@belapsed begin
    truef(dM) = PhaseOptimizedComponentAnalysis.scapair(I+dM,W,H)
    trueg = x -> ForwardDiff.gradient(truef,x)
    trueg(zeros(14,14))
end # 0.001579sec

# POCA
@belapsed  A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H) # 0.0015543sec

# Approximation with delta = eps(Float64)
@belapsed begin
    penalty_0 = PhaseOptimizedComponentAnalysis.scapair(W,H)
    grads = []
    for i = 1:p, j = 1:p
        delta = eps(Float64)
        M = Matrix(1.0I,p,p); M[i,j] += delta
        penalty_delta = PhaseOptimizedComponentAnalysis.scapair(M,W,H)
        push!(grads, (penalty_delta-penalty_0)/delta)
    end
end # 0.025164sec

# check accuracy of scapair
trueg(zeros(14,14))
reshape(2*b, (p, p))
ImageView.imshow(trueg(zeros(14,14))-reshape(2*b, (p, p)))

# check W
truepenWf(dM) = PhaseOptimizedComponentAnalysis.sca2(W*(I+dM))
truepenWg = x -> ForwardDiff.gradient(truepenWf,x)
truepenWg(zeros(14,14))

m, p = size(W)
bW, bH = zeros(p^2), zeros(p^2)
AW, AH = zeros(p^2, p^2), zeros(p^2, p^2)
PhaseOptimizedComponentAnalysis.add_direct!(AW, bW, W)
PhaseOptimizedComponentAnalysis.add_transpose!(AH, bH, H)
reshape(2*bW, (p, p))
ImageView.imshow(truepenWg(zeros(14,14))-reshape(2*bW, (p, p)))

# check H
truepenHf(dM) = PhaseOptimizedComponentAnalysis.sca2((I+dM)\H)
truepenHg = x -> ForwardDiff.gradient(truepenHf,x)
truepenHg(zeros(14,14))

reshape(2*bH, (p, p))
ImageView.imshow(truepenHg(zeros(14,14))-reshape(2*bH, (p, p)))


#======== Symmetric POCA Hessian check : H = (I-dM)*H ===============================#
p = size(W,2)
approxf(x) = (dM = reshape(x, (p,p)); PhaseOptimizedComponentAnalysis.scapair(W*(I+dM),(I-dM)*H))
gradapprox = ForwardDiff.gradient(approxf,zeros(p*p))
Hessapprox = ForwardDiff.hessian(approxf,zeros(p*p))

A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H; order=1)

norm(gradapprox-2b) # 0.0
norm(Hessapprox-2A) # 7.0682005816640276e-12

#======== Symmetric POCA Hessian check : H = M\H ===============================#
M = Matrix(1.0I,p,p)

# Symmetric penalty
A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H)
scapair_vec(x, W, H) = PhaseOptimizedComponentAnalysis.scapair(reshape(x, (p,p)), W, H)
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), vec(M))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), vec(M))
norm(grad-2b) # 9261.603848561132 (fail)
norm(Hess-2A) # 120505.99775855533 (fail)
Cgrad = Calculus.gradient(x -> scapair_vec(x, W, H), vec(M))
CHess = Calculus.hessian(x -> scapair_vec(x, W, H), vec(M))
norm(Cgrad-2b) # 2.9618746826037796e-5
norm(CHess-2A) # 278237.07671132515 (fail)

# Penalty W
AW, bW= zeros(p^2, p^2), zeros(p^2) 
PhaseOptimizedComponentAnalysis.add_direct!(AW, bW, W)
truepenWf(x) = PhaseOptimizedComponentAnalysis.sca2(W*reshape(x, (p,p)))
gradW = ForwardDiff.gradient(truepenWf,vec(M))
HessW = ForwardDiff.hessian(truepenWf,vec(M))
norm(gradW-2bW) # 0
norm(HessW-2AW) # 0
CgradW = Calculus.gradient(truepenWf, vec(M))
CHessW = Calculus.hessian(truepenWf, vec(M))
norm(CgradW-2bW) # 7.251469002715105e-9
norm(CHessW-2AW) # 0.00286375646620506

# Penalty H
AH, bH = zeros(p^2, p^2), zeros(p^2)
PhaseOptimizedComponentAnalysis.add_transpose!(AH, bH, H)
truepenHf(x) = PhaseOptimizedComponentAnalysis.sca2(reshape(x, (p,p))\H)
gradH = ForwardDiff.gradient(truepenHf,vec(M))
HessH = ForwardDiff.hessian(truepenHf,vec(M))
norm(gradH-2bH) # 1225.1872527762455 (fail)
norm(HessH-2AH) # 17425.921601688766 (fail)
CgradH = Calculus.gradient(truepenHf, vec(M))
CHessH = Calculus.hessian(truepenHf, vec(M))
norm(CgradH-2bH) # 1.3807838768279382e-6
norm(CHessH-2AH) # 36807.07764767368 (fail)

# Symmetric penalty : DoubleFloat64
M = Matrix{Double64}(1.0I,p,p)
W = Double64.(W)
H = Double64.(H)
A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H)
scapair_vec(x, W, H) = PhaseOptimizedComponentAnalysis.scapair(reshape(x, (p,p)), W, H)
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), vec(M))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), vec(M))
norm(grad-2b) # 9261.603848561115 (fail)
norm(Hess-2A) # 274633.5274819741 (fail)
Cgrad = Calculus.gradient(x -> scapair_vec(x, W, H), vec(M))
CHess = Calculus.hessian(x -> scapair_vec(x, W, H), vec(M))
norm(Cgrad-2b) # 1.222695900422151e-11 (much accurate)
norm(CHess-2A) # 0.7544802764520611 (much accurate)

#=== approxobj(x, pW, pH, A, b) = pW*pH + 2*b'*x + x'*A*x ===#
approxobj(x, pWH, A, b) = pWH + 2*b'*x + x'*A*x

A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H)
pW = PhaseOptimizedComponentAnalysis.sca2(W)
pH = PhaseOptimizedComponentAnalysis.sca2(H)

AH, bH = zeros(p^2, p^2), zeros(p^2)
PhaseOptimizedComponentAnalysis.add_transpose!(AH, bH, H)

approxHf(x) = approxobj(x, pH, AH, bH)
FgradapproxHf = ForwardDiff.gradient(approxHf, zeros(p*p))
FHessapproxHf = ForwardDiff.hessian(approxHf, zeros(p*p))
# True gradient and hessian (CgradH, CHessH : Calculus)
norm(gradH-FgradapproxHf) # 3.7735855246332927e-7
norm(HessH-FHessapproxHf) # 0.19414960594651876
norm(2AH-FHessapproxHf) # 0.0
CgradapproxHf = Calculus.gradient(approxHf, zeros(p*p))
CHessapproxHf = Calculus.hessian(approxHf, zeros(p*p))
# True gradient and hessian (CgradH, CHessH : Calculus)
norm(CgradH-CgradapproxHf) # 3.7735855246332927e-7
norm(CHessH-CHessapproxHf) # 0.19414960594651876 (much better)
norm(2AH-CHessapproxHf) # 0.21512752848552755

n = 400
AR = rand(n,n)
AR = AR+AR'
bR = rand(n)
pR = rand()
approxf(x) = approxobj(x, pR, AR, bR)
Fgradapproxf = ForwardDiff.gradient(approxf, zeros(n))
FHessapproxf = ForwardDiff.hessian(approxf, zeros(n))
norm(2bR-Fgradapproxf) # 0
norm(2AR-FHessapproxf) # 0
Cgradapproxf = Calculus.gradient(approxf, zeros(n))
CHessapproxf = Calculus.hessian(approxf, zeros(n))
norm(2bR-Cgradapproxf) # 7.603075860143318e-11
norm(2AR-CHessapproxf) # 0.0001238184986037651

#=== compare gradient and hessian of E(H) for true obj and approximation obj by (I+ΔM)⁻¹ ≈ I-ΔM  ===#
# gradient and hessian of I-dM approximation (gradaproxH, HessaproxH) 
approxHf(x) = (dM = reshape(x, (p,p)); PhaseOptimizedComponentAnalysis.sca2((I-dM)*H))
gradaproxH = ForwardDiff.gradient(approxHf,zeros(p*p))
HessaproxH = ForwardDiff.hessian(approxHf,zeros(p*p))
# True gradient and hessian (gradH, HessH : ForwardDiff)
norm(gradH-gradaproxH) # 1225.1872527762455 (fail)
norm(HessH-HessaproxH) # 17687.070167021488 (fail)
CgradaproxH = Calculus.gradient(approxHf, zeros(p*p))
CHessaproxH = Calculus.hessian(approxHf, zeros(p*p))
# True gradient and hessian (CgradH, CHessH : Calculus)
norm(CgradH-CgradaproxH) # 7.280942764139423e-7
norm(CHessH-CHessaproxH) # 36807.07815721351 (fail)

#=== compare gradient and hessian of E(H) for true obj and approximation obj by (I+ΔM)⁻¹ ≈ I-ΔM+(ΔM)² ===#

# true : M\H
# truepenHf(x) = PhaseOptimizedComponentAnalysis.sca2(reshape(x, (p,p))\H)
# gradH = ForwardDiff.gradient(truepenHf,vec(M))
# HessH = ForwardDiff.hessian(truepenHf,vec(M))
truepenHf(x) = PhaseOptimizedComponentAnalysis.sca2((I+reshape(x, (p,p)))\H)
gradH = ForwardDiff.gradient(truepenHf,zeros(p*p))
HessH = ForwardDiff.hessian(truepenHf,zeros(p*p))
# approximation : (I-dM+dM^2)*H
approxHf(x) = (dM = reshape(x, (p,p)); PhaseOptimizedComponentAnalysis.sca2((I-dM+dM^2)*H))
gradapproxH = ForwardDiff.gradient(approxHf,zeros(p*p))
HessapproxH = ForwardDiff.hessian(approxHf,zeros(p*p))
# approximation more exactly : (I-dM+dM^2-dM^3)*H (according to Matrix Cookbook eq(187))
approxHf3(x) = (dM = reshape(x, (p,p)); PhaseOptimizedComponentAnalysis.sca2((I-dM+dM^2-dM^3)*H))
gradapproxH3 = ForwardDiff.gradient(approxHf3,zeros(p*p))
HessapproxH3 = ForwardDiff.hessian(approxHf3,zeros(p*p))
# POCA
AH, bH = zeros(p^2, p^2), zeros(p^2)
PhaseOptimizedComponentAnalysis.add_transpose!(AH, bH, H;order=2)

# true(M\H) vs approximation (I-dM+dM^2)*H
norm(gradH-gradapproxH) # 1225.1872527762455 (fail)
norm(HessH-HessapproxH) # 36469.79004462122 (fail)
# true(M\H) vs approximation (I-dM+dM^2-dM^3)*H
norm(gradH-gradapproxH3) # 1225.1872527762455 (fail)
norm(HessH-HessapproxH3) # 36469.79004462116 (fail)
# approximation vs POCA
norm(gradapproxH-2bH) # 0.0
norm(HessapproxH-2AH) # 1.2531068071105002e-11

#============ p X 2 test : suface plot =============================#
scapair0(M, W, H) = POCA.sca2(W*M) * POCA.sca2(svd(M)\H)

ncells = 2
gtW = rand(300,ncells)
wscale = sqrt.(sum(gtW.^2; dims=1))
gtW ./= wscale
gtH = rand(ncells,1000)
X = gtW*gtH

s = 1
U = svd(X).U
W = copy(U[:,1:ncells])
W[:,1] .= s * W[:,1]
H = W\X
POCA.scapair(W, H)

n = 1
objvals = zeros(200*n+1,200*n+1)
for (ridx,a) = enumerate(-n:0.01:n), (cidx,b) = enumerate(-n:0.01:n)
    M = [1 a; b -1]
    objvals[ridx,cidx] = scapair0(M, W, H)
end

n = 1
objvals = zeros(200*n+1,200*n+1,200*n+1)
as = bs = cs = -n:0.01:n
for (ridx,a) = enumerate(as), (cidx,b) = enumerate(bs), (tidx,c) = enumerate(cs)
    M = [1 a; b c]
    objvals[ridx,cidx,tidx] = scapair0(M, W, H)
end

agrid = repeat(as',length(as),1)
bgrid = repeat(bs,1,length(bs))

cidx = 97
vmax = 0.01

objval = objvals[:,:,cidx]
objval[objval.>vmax] .= NaN
fig = figure("surfaceplot",figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection="3d")
ax.plot_surface(agrid, bgrid, objval, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("RdBu"), alpha=0.8, linewidth=0.25, norm = true, vmax = 10000)

tickvs = collect(LinRange(0,vmax,5))
ax.set_zticks(tickvs)
ax.set_zlabel("penalty")

ax.set_xticks(collect(LinRange(as[1],as[end],5)))
ax.set_xlabel("a")
ax.set_yticks(collect(LinRange(bs[1],bs[end],5)))
ax.set_ylabel("b")

ax.set_title("penalty (c=$(cs[cidx]))")

#======= For each order check the shape of approximation =====================================#

normalizeWH!(W,H)
imshowW(W,imgsz)
imsaveW("linesearch_order3PF.png",W,imgsz)

W, H, objvals = semiscasolve!0(U,X,c=1.0,itermax=14,plotssign=1,plotiters=(1,100), order=1, fixposdef=false);
Worg, Horg = copy(W), copy(H)
approxobj(x, pW, pH, A, b) = pW*pH + 2*b'*x + x'*A*x
function issymmetrictol(A; tol=1e-10)
    for i in 1:size(A,2)
        if norm(A[:,i]-A[i,:]) > tol
            return false
        end
    end
    return true
end

# Order 3, without PositiveFactorizations
Wnew,Hnew, α, dM2, dM, A, b = scastep0(W, H;order=3, fixposdef=false)
issymmetrictol(A) # true
minimum(eigen(A).values) # -1910
scapair(Wnew,Hnew) # 4224.678959513273
xaxismin, xaxismax = max(-0.2,-α*3), min(α*1.2,1.2)
alphastep = 10^floor(log10(α)-1)
αs = xaxismin:alphastep:xaxismax
objs = [scapair(I + α*dM, W, H) for α in αs]
objsapprox = [approxobj(α*vec(dM), sca2(W), sca2(H), A, b) for α in αs]
plot(αs,objs)
plot(αs,objsapprox)
yaxismin, yaxismax = min(minimum(objs),minimum(objsapprox)), max(maximum(objs),maximum(objsapprox))
plot([α,α], [yaxismin, yaxismax])
plot([0,0], [yaxismin, yaxismax])
axis([xaxismin, xaxismax, 4200, 4500])
legend(["true obj.", "quadratic approx. obj", "α"],fontsize = 12,loc=2)
savefig("order3woPF.png")

# Order 3, with PositiveFactorizations
Wnew,Hnew, α, dM2, dM, A, b = scastep0(W, H;order=3, fixposdef=true)
issymmetrictol(A) # true
minimum(eigen(A).values) # 5
scapair(Wnew,Hnew) # 712.9974973236899
xaxismin, xaxismax = max(-0.2,-α*3), min(α*1.2,1.2)
alphastep = 10^floor(log10(α)-1)
αs = xaxismin:alphastep:xaxismax
objs = [scapair(I + α*dM, W, H) for α in αs]
objsapprox = [approxobj(α*vec(dM), sca2(W), sca2(H), A, b) for α in αs]
plot(αs,objs)
plot(αs,objsapprox)
yaxismin, yaxismax = min(minimum(objs),minimum(objsapprox)), max(maximum(objs),maximum(objsapprox))
plot([α,α], [yaxismin, yaxismax])
plot([0,0], [yaxismin, yaxismax])
axis([xaxismin, xaxismax, yaxismin, yaxismax])
legend(["true obj.", "quadratic approx. obj", "α"],fontsize = 12,loc=2)
savefig("order3wPF.png")

# Order 2, without PositiveFactorizations
Wnew,Hnew, α, dM2, dM, A, b = scastep0(W, H;order=2, fixposdef=false)
issymmetrictol(A) # true
minimum(eigen(A).values) # -4143
scapair(Wnew,Hnew) # 2433.8900785236956
xaxismin, xaxismax = max(-0.2,-α*3), min(α*1.2,1.2)
alphastep = 10^floor(log10(α)-1)
αs = xaxismin:alphastep:xaxismax
objs = [scapair(I + α*dM, W, H) for α in αs]
objsapprox = [approxobj(α*vec(dM), sca2(W), sca2(H), A, b) for α in αs]
plot(αs,objs)
plot(αs,objsapprox)
yaxismin, yaxismax = min(minimum(objs),minimum(objsapprox)), max(maximum(objs),maximum(objsapprox))
plot([α,α], [yaxismin, yaxismax])
plot([0,0], [yaxismin, yaxismax])
axis([xaxismin, xaxismax, yaxismin, yaxismax])
legend(["true obj.", "quadratic approx. obj", "α"],fontsize = 12,loc=9)
savefig("order2woPF.png")

# Order 2, with PositiveFactorizations
Wnew,Hnew, α, dM2, dM, A, b = scastep0(W, H;order=2, fixposdef=true)
issymmetrictol(A) # true
minimum(eigen(A).values) # 21
scapair(Wnew,Hnew) # 3668.864093144477
xaxismin, xaxismax = max(-0.2,-α*3), min(α*1.2,1.2)
alphastep = 10^floor(log10(α)-1)
αs = xaxismin:alphastep:xaxismax
objs = [scapair(I + α*dM, W, H) for α in αs]
objsapprox = [approxobj(α*vec(dM), sca2(W), sca2(H), A, b) for α in αs]
plot(αs,objs)
plot(αs,objsapprox)
yaxismin, yaxismax = min(minimum(objs),minimum(objsapprox)), max(maximum(objs),maximum(objsapprox))
plot([α,α], [yaxismin, yaxismax])
plot([0,0], [yaxismin, yaxismax])
axis([xaxismin, xaxismax, yaxismin, yaxismax])
legend(["true obj.", "quadratic approx. obj", "α"],fontsize = 12,loc=2)
savefig("order2wPF.png")

# Order 1
Wnew,Hnew, α, dM2, dM, A, b = scastep0(W, H;order=1, fixposdef=false)
issymmetrictol(A) # true
minimum(eigen(A).values) # 21
scapair(Wnew,Hnew) # 2566.6636233851705
xaxismin, xaxismax = max(-0.2,-α*3), min(α*1.2,1.2)
alphastep = 10^floor(log10(α)-1)
αs = xaxismin:alphastep:xaxismax
objs = [scapair(I + α*dM, W, H) for α in αs]
objsapprox = [approxobj(α*vec(dM), sca2(W), sca2(H), A, b) for α in αs]
plot(αs,objs)
plot(αs,objsapprox)
yaxismin, yaxismax = min(minimum(objs),minimum(objsapprox)), max(maximum(objs),maximum(objsapprox))
plot([α,α], [yaxismin, yaxismax])
plot([0,0], [yaxismin, yaxismax])
axis([xaxismin, xaxismax, yaxismin, yaxismax])
legend(["true obj.", "quadratic approx. obj", "α"],fontsize = 12,loc=2)
savefig("order1.png")

#======== Orthognality vs Eigenvalue =============================#
rng = 0:0.01:0.25
r = []
for i in rng
    M = [1. 0.; 0. 1.]
    M[:,1] = POCA.Rθ(2,1,2,i*pi)*M[:,1]
    M[:,2] = POCA.Rθ(2,1,2,-i*pi)*M[:,2]
    evs = norm.(eigen(M).values)
    push!(r, evs[1]/evs[2])
end
plot(rng, r)

#======== Two step minimization ==================================#
order=0; fixposdef=false
c = 0
β=0;
useconstraint=true
γ=0; usepdratio=false
i=9; itermax = 20000
paramstrsuffix = "$(order)_PF$(fixposdef)_c$(c)_b$(β)_uc$(useconstraint)_r$(γ)_updr$(usepdratio)_($i)"
datafilename = "fc_nonorth_o$paramstrsuffix.jld"
#"fc_nonorth_o0_PFfalse_b0false_uctrue_r0_updrfalse_(113)"

fakecells_dic = load(datafilename)
gt_ncells = fakecells_dic["gt_ncells"]
imgrs = fakecells_dic["imgrs"]
gtW = fakecells_dic["gtW"]
gtH = fakecells_dic["gtH"]
gtWimgc = fakecells_dic["gtWimgc"]
gtbg = fakecells_dic["gtbg"]
imgsz = fakecells_dic["imgsz"]

ncells = 14
F = svd(imgrs)
U = F.U[:,1:ncells]
T = F.V[:,1:ncells] * Diagonal(F.S[1:ncells])

D = Matrix(1.0I,ncells,ncells)
R = Matrix(1.0I,ncells,ncells)
UD = U*D
S = UD*R
X = imgrs

# Dataset 1 : orthogonality = 0.002126245342057767,  gtM = [ 1.0  0.067171; 0.668001  -0.0448669],  [-1.0 -0.0665589; 0.660016  -0.0444727]
gtW = [0.962224   0.0242178; 0.0116875  0.0595547]
gtH = [0.0270566  0.0198188; 0.0155136  0.49868]

X = gtW*gtH
U = svd(X).U
W0 = copy(U)
H0 = W0\X

sca2(W0) # 0.69
sca2(H0) # 5.337820693497308e-5

showfig=false; plotssign=1; plotiters=(1,30); objtol=0; itermax=2000
W1, H1, objvalsorder2, Mss, As, endval, iters, idx  = POCA.semiscasolve!0(U, X; order=order, fixposdef=fixposdef,
        ρ=0.5, c=c, β=β, useconstraint=useconstraint, γ=0, usepdratio=usepdratio, itermax=itermax,
        showfig=showfig, plotssign=plotssign,plotiters=plotiters,objtol=objtol,showdbg=true)

sca2(W1) # 0.6032479290830385
sca2(H1) # 2.300612813069667e-20

showfig=true; plotiters=(1,30); objtol=0
W2, H2, objvalsorder2, Mss, A, endval, iters, idx  = POCA.semiscasolve!0_2ndstep(W1, H1; order=order, fixposdef=fixposdef,
       β=β, allcompW = false, allcompH = true, useconstraint=useconstraint, γ=0, itermax=itermax,
       showfig=showfig, plotiters=plotiters,objtol=objtol,showdbg=true)

sca2(W2) # 3.9301514774078074e-14
sca2(H2) # 1.0568088121551135e-6

scapair(W2, H2; allcompW = false, allcompH = false) # 4.1534187144290097e-20

# Dataset 2 : orthognality = 0.22341729948373742, gtM = [ -1.0  -1.51289;  -0.655804   0.442679]
gtW = [0.584652  0.0508351; 0.447497  0.969165]
gtH = [0.581042  0.317224;  0.299005  0.836679]

sca2(W1) # 0.43169739296106485
sca2(H1) # 2.578625797273283e-18

sca2(W2) # 2.2852555298732637e-13
sca2(H2) # 0.05359848679981811

# Dataset4 : order=1, fixposdef=false
gtW = [0.606349  0.430703; 0.251698  0.91011]
gtH = [0.657172  0.278718; 0.113736  0.968676]

X = gtW*gtH
U = svd(X).U
W0 = copy(U)
H0 = W0\X

sca2(W0) # 1.6524517255193167
sca2(H0) # 1.5143495313367143

W1, H1, objvalsorder2, Ms, As, endval, iters, idx  = POCA.semiscasolve!0(U, X; order=1, fixposdef=false,
        ρ=0.5, β=0, useconstraint=true, γ=0, usepdratio=false, itermax=200,
        showfig=false, plotssign=1, plotiters=(1,30), objtol=0, showdbg=true)

sca2(W1) # 0.0
sca2(H1) # 0.6107830217498699
scapair(W1,H1;allcompW=true) # 2.800148763143664

W2, H2, objvalsorder2, Ms, A, endval, iters, idx  = POCA.semiscasolve!0_2ndstep(W1, H1; order=1, fixposdef=false,
       β=0, allcompW = true, allcompH = false, useconstraint=true, γ=0, itermax=200,
       showfig=true, plotiters=(1,30), objtol=0, showdbg=true, c=1)

sca2(W2) # 3.9812514389633153
sca2(H2) # 0.0
scapair(W2,H2;allcompW=true) # 0.0

function plotsca2WH(U,X,Mss,idx)
    W = copy(U)
    H = W\X
    Ms = Mss[idx+2]
    penW = []; penH = []; penWH = []
    for M in Ms
        W = W*M
        H = M\H
        push!(penW, sca2(W))
        push!(penH, sca2(H))
        push!(penWH, scapair(W,H))
    end
    plot(1:length(Ms),[penW penH penWH])
    legend(["penalty of W", "penalty of H", "penalty of (W,H)"],fontsize = 12,loc=1)
end

plotsca2WH(U,X,Ms,idx)

function normalizecol!(M)
    for i in 1:size(M,2)
        M[:,i] ./= norm(M[:,i]) 
    end
end

indep(M) = abs(det(M))/reduce(*,norm.(eachcol(M)))
orthog(M) = (sz=size(M,2); Mc = copy(M); normalizecol!(Mc); 1-norm(Mc'Mc-I)/sqrt(sz*(sz-1)))

function plotsca2WH(U,X,Mss,idx)
    W = copy(U)
    H = W\X
    Ms = Mss[idx+2]
    penW = []; penH = []; penWH = []
    for M in Ms
        W = W*M
        H = M\H
        push!(penW, sca2(W))
        push!(penH, sca2(H))
        push!(penWH, scapair(W,H))
    end
    plot(1:length(Ms),[penW penH penWH])
    legend(["penalty of W", "penalty of H", "penalty of (W,H)"],fontsize = 12,loc=1)
end

Ms = Mss[idx]
ind = []; orth = []
for M in Ms
    push!(ind,indep(M))
    push!(orth,orthog(M))
end
plot(1:length(Ms),[ind orth])
legend(["O1", "O2"],fontsize = 12,loc=1)

#======== Two step minimization Hessian check ==================================#
W = copy(U)
H = W\X
p = size(W,2)
dM = zeros(p,p)

allcompW=false; allcompH=false
A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H; order=1, allcompW=allcompW, allcompH=allcompH)
scapair_vec(x, W, H) = (dM = reshape(x, (p,p)); POCA.scapair(W*(I+dM),(I-dM)*H; allcompW=allcompW, allcompH=allcompH))
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), vec(dM))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), vec(dM))
norm(grad-2b) # 0.0
norm(Hess-2A) # 5.485343005328847e-12

allcompW=true; allcompH=false
A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H; order=1, allcompW=allcompW, allcompH=allcompH)
scapair_vec(x, W, H) = (dM = reshape(x, (p,p)); POCA.scapair(W*(I+dM),(I-dM)*H; allcompW=allcompW, allcompH=allcompH))
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), vec(dM))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), vec(dM))
norm(grad-2b) # 1.668255510789884e-11
norm(Hess-2A) # 9.92726450574069e-11

allcompW=false; allcompH=true
A, b = PhaseOptimizedComponentAnalysis.buildproblem(W, H; order=1, allcompW=allcompW, allcompH=allcompH)
scapair_vec(x, W, H) = (dM = reshape(x, (p,p)); POCA.scapair(W*(I+dM),(I-dM)*H; allcompW=allcompW, allcompH=allcompH))
grad = ForwardDiff.gradient(x -> scapair_vec(x, W, H), vec(dM))
Hess = ForwardDiff.hessian(x -> scapair_vec(x, W, H), vec(dM))
norm(grad-2b) # 1.6192763136350506e-12
norm(Hess-2A) # 8.190899671151902e-12

#========== E(H) fixed POCA Penalty Image with minimum point of E(W,H) =========================#
include(joinpath(pkgdir(SymmetricComponentAnalysis),"test","oldsca.jl"))

Wgt = [0.6063  0.4307; 0.2517  0.9101]
Hgt = [0.6572  0.2787; 0.1137  0.9687]

X = Wgt*Hgt
U = svd(X).U
W = copy(U)
H = W\X
Mw =
W1, H1, objvalsorder2, Ms, As, endval, iters, idx  = SCA.semiscasolve!(U, X; order=1, fixposdef=false,
        ρ=0.5, β=0, useconstraint=true, γ=0, usepdratio=false, itermax=2000,
        showfig=false, plotssign=1, plotiters=(1,30), objtol=0, showdbg=true)
semiscasolve!(W, H, Mw, Mh; stparams, lsparams, cparams)
M = Ms[idx][end]
M ./= abs(M[1,1])

showtrace = false
rng = -3:0.01:3
M2(b,c) = [M[1,1] b; c M[2,2]]
scapair0(M, W, H) = sca2(W*M) * sca2(svd(M)\H)
penW=[sca2(W0*M2(b,c)) for b in rng, c in rng]
penH=[sca2(svd(M2(b,c))\H0) for b in rng, c in rng]
penpair=[scapair0(M2(b,c),W0,H0) for b in rng, c in rng]

plotL(penW,penH,penpair,rng;vmaxL1=sca2(W0),vminL1=0,vmaxL2=0.002,vminL2=0,vmaxL3=0.002,vminL3=0,showtrace=showtrace)
axs = plotL(log10.(penW),log10.(penH),log10.(penpair),rng; vmaxL1=1,vminL1=-0.2,vmaxL2=-2.0,vminL2=-5.0,vmaxL3 = -2.0, vminL3=-5.0,showtrace=showtrace,
        title1="log(E(W))", title2="log(E(H))", title3="log(E(W)E(H))")
ax = axs[5]
ax.plot(M[2,1],M[1,2],color="red", marker="x", linewidth = 3)

if sca2(W1) == 0
    normW2=[norm(W0*M2(b,c))^2 for b in rng, c in rng]
    penHfixW=[sca2(M2(b,c)\H0)*norm(W0*M2(b,c))^2 for b in rng, c in rng]

    axs = plotL(normW2,penH,penHfixW,rng;vmaxL1=sca2(W0),vminL1=0,vmaxL2=2,vminL2=0,vmaxL3=2,vminL3=0,showtrace=showtrace,
                    title1="||W||²", title2="E(H)", title3="E(H)||W||²")
    axslog = plotL(log10.(normW2),log10.(penH),log10.(penHfixW),rng; vmaxL1=1,vminL1=-0.2,vmaxL2=-2.0,vminL2=-5.0,vmaxL3 = -2.0, vminL3=-5.0,showtrace=showtrace,
                    title1="log(||W||²)", title2="log(E(H))", title3="log(E(H)||W||²)")

 #   ax = axs[5]
 #   ax.plot(M[2,1],M[1,2],color="red", marker="x", linewidth = 3)
    ax = axslog[5]
    ax.plot(M[2,1],M[1,2],color="red", marker="x", linewidth = 3)
else
    normH2=[norm(M2(b,c)\H0)^2 for b in rng, c in rng]]
    penWfixH=[sca2(W0*M2(b,c))*norm(M2(b,c)\H0)^2 for b in rng, c in rng]
 
    axs = plotL(penW,normH2,penWfixH,rng;vmaxL1=sca2(W0),vminL1=0,vmaxL2=2,vminL2=0,vmaxL3=2,vminL3=0,showtrace=showtrace)
    axslog = plotL(log10.(penW),log10.(normH2),log10.(penWfixH),rng; vmaxL1=1,vminL1=-0.2,vmaxL2=-2.0,vminL2=-5.0,vmaxL3 = -2.0, vminL3=-5.0,showtrace=showtrace)

    ax = axs[5]
    ax.plot(M[2,1],M[1,2],color="red", marker="x", linewidth = 3)
end

#============= iter vs norm(b) ======================================================#
include(joinpath(pkgdir(PhaseOptimizedComponentAnalysis),"test","obsoletePOCA.jl"))

W3, H3, objvalsorder2, Ms, As, bs, endval, iters, idx  = semiscasolve!0(U, X; order=0, fixposdef=false,
        ρ=0.5, β=0, useconstraint=true, γ=0, usepdratio=false, itermax=2000,
        showfig=false, plotssign=1, plotiters=(1,30), objtol=0, showdbg=true)

plot(bs)

function imshowWiter(U,X,Ms,iternum,imgsz)
    W = copy(U)
    H = W\X
    W = W*Ms[iternum]
    H = Ms[iternum]\H
    normalizeWH!(W,H)
    imshowW(W,imgsz)
    plot(H')
end

imshowWiter(U,X,Ms[idx],16,imgsz)



#========= Freezed POCA ================================================#
function freezesca2(signA, A)
    objval = zero(eltype(A))
    for (s,a) in zip(signA, A)
        objval += s < 0 ? a^2 : 0
    end
    return objval
end

freezescapair(M, W, H) = freezesca2(sign.(W),W*M) * freezesca2(sign.(H),M\H)
approxinvscapair(dM, W, H) = freezesca2(sign.(W),W*(I+dM)) * freezesca2(sign.(H),(I-dM)*H)


#============ Makie =========================================#
using GLMakie
xs = 0 : 0.1 : 10
ys = xs
fs(x, y) = sin(x) * sin(y)

surface(xs, ys, fs; axis=(type=Axis3,perspectiveness=true,viewmode=:fit))

#---- scene and camera ----#
# create scene
GLMakie.activate!()
scene = Scene(backgroundcolor=:gray)
subwindow = Scene(scene, px_area=Rect(100, 100, 200, 200), clear=true, backgroundcolor=:white)
scene
# 2D camera
cam2d!(subwindow)
meshscatter!(subwindow, rand(Point2f, 10), color=:gray)
center!(subwindow)
scene
# 3D camera
cam3d!(subwindow)
meshscatter!(subwindow, rand(Point3f, 10), color=:gray)
center!(subwindow)
scene
# pixel camera
campixel!(subwindow)
w, h = size(subwindow) # get the size of the scene in pixels
# this draws a line at the scene window boundary
image!(subwindow, [sin(i/w) + cos(j/h) for i in 1:w, j in 1:h])


using CairoMakie, ElectronDisplay # GLMakie doesn't need viewer package but CairoMakie does
# Makie doesn't support zoom in and out?

f = Figure()
ax = CairoMakie.Axis(f[1, 1],
    title = "A Makie Axis",
    xlabel = "The x label",
    ylabel = "The y label"
)

# lines
x = range(0, 10, length=100)
y = sin.(x)
lines(x, y)

figure, axis, lineplot = lines(x, y)
figure

lines(0..10, sin)
lines(0:1:10, cos)
lines([Point(0, 0), Point(5, 10), Point(10, 5)])

f, ax, l1 = lines(x, sin)
l2 = lines!(ax, x, cos) # or just lines!(x, cos)

# scatter
x = range(0, 10, length=100)
y = sin.(x)
scatter(x, y;
    figure = (; resolution = (400, 400)),
    axis = (; title = "Scatter plot", xlabel = "x label")
)

f, ax, sc1 = scatter(x, sin, color = :red, markersize = 5)
sc2 = scatter!(ax, x, cos, color = :blue, markersize = 10)

sc1.marker = :utriangle
sc1.markersize = 20

sc2.color = :transparent
sc2.markersize = 20
sc2.strokewidth = 1
sc2.strokecolor = :purple

scatter(x, sin,
    markersize = range(5, 15, length=100),
    color = range(0, 1, length=100),
    colormap = :thermal#,
#    colorrange = (0.33, 0.66)
)

colors = repeat([:crimson, :dodgerblue, :slateblue1, :sienna1, :orchid1], 20)
scatter(x, sin, color = colors, markersize = 20)

# legend
x = range(0, 10, length=100)

lines(x, sin, color = :red, label = "sin")
lines!(x, cos, color = :blue, label = "cos")
axislegend()
current_figure()

# subplot
x = LinRange(0, 10, 100)
y = sin.(x)

fig = Figure()
lines(fig[1, 1], x, y, color = :red)
lines(fig[1, 2], x, y, color = :blue)
lines(fig[2, 1:2], x, y, color = :green)
fig

fig = Figure()
ax1 = CairoMakie.Axis(fig[1, 1]) # Axis is conflict with the one in ImageAxes
ax2 = CairoMakie.Axis(fig[1, 2])
ax3 = CairoMakie.Axis(fig[2, 1:2])
fig
lines!(ax1, 0..10, sin)
lines!(ax2, 0..10, cos)
lines!(ax3, 0..10, sqrt)
fig

#============= plot functions =================#


function plot_surf(ds, ms, penL1, penL2; penL1Max = 4, penL2Max = 1)
    mgrid = repeat(ms',length(ds),1)
    dgrid = repeat(ds,1,length(ms))

    penL1[penL1 .> penL1Max] .= NaN
    penL2[penL2 .> penL2Max] .= NaN

    tickds = [-π/4, -π/8, 0, π/8, π/4]
    tickdlabels = ["-π/4", "-π/8", "0", "π/8", "π/4"]
    tickms = [-π, -π/2, 0, π/2, π]
    tickmlabels = ["-π", "-π/2", "0", "π/2", "π"]

    fig = figure("surfaceplot",figsize=(10,10))
    ax = fig.add_subplot(1,2,1,projection="3d")
    plot_surface(mgrid, dgrid, penL1, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("RdBu"), alpha=0.8, linewidth=0.25, norm = true, vmax = 4)
    ax.set_xlabel("m")
    ax.set_xticks(tickms)
    ax.set_yticks(tickds)
    ax.set_xticklabels(tickmlabels)
    ax.set_yticklabels(tickdlabels)
    ax.set_title("")

    subplot(122)
    ax = fig.add_subplot(1,2,2,projection="3d")
    plot_surface(mgrid, dgrid, penL2, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("RdBu"), alpha=0.8, linewidth=0.25, vmax = 1)
    ax.set_xlabel("m")
    ax.set_xticks(tickms)
    ax.set_yticks(tickds)
    ax.set_xticklabels(tickmlabels)
    ax.set_yticklabels(tickdlabels)
    ax.set_title("L2 objective value")
    tight_layout()
end


function plotL2D(penL1, penL2; vminL1=nothing, vmaxL1=nothing, vminL2=nothing, vmaxL2=nothing)
    tickds = [-π/4, -π/8, 0, π/8, π/4]
    tickdlabels = ["-π/4", "-π/8", "0", "π/8", "π/4"]
    tickms = [-π, -π/2, 0, π/2, π]
    tickmlabels = ["-π", "-π/2", "0", "π/2", "π"]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw=Dict("width_ratios"=>[1,0.3,1,0.1]))

    ax = axs[1]
    vmaxL1==nothing && (vmaxL1 = ceil((minimum(penL1[.!isnan.(penL1)])+1)*15)/10)
    vminL1==nothing && (vminL1 = floor(minimum(penL1[.!isnan.(penL1)])*10)/10)
    himg1 = ax.imshow(reverse(penL1; dims=1); extent = (-π,π,-π/4,π/4), vmin=vminL1, vmax=vmaxL1)
    ax.set_xlabel("\$\\theta_1\$")
    ax.set_xticks(tickms)
    ax.set_yticks(tickds)
    ax.set_xticklabels(tickmlabels)
    ax.set_yticklabels(tickdlabels)
    ax.set_title("L1 objective value")

    ax = axs[2]
    plt.colorbar(himg1, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)

    ax = axs[3]
    vmaxL2==nothing && (vmaxL2 = ceil((minimum(penL2[.!isnan.(penL2)])+1)*15)/10)
    vminL2==nothing && (vminL1 = floor(minimum(penL2[.!isnan.(penL2)])*10)/10)
    himg2 = ax.imshow(reverse(penL2; dims=1); extent = (-π,π,-π/4,π/4), vmin=vminL2, vmax=vmaxL2)
    ax.set_xlabel("\$\\theta_1\$")
    ax.set_xticks(tickms)
    ax.set_yticks(tickds)
    ax.set_xticklabels(tickmlabels)
    ax.set_yticklabels(tickdlabels)
    ax.set_title("L2 objective value")

    ax = axs[4]
    plt.colorbar(himg2, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)

    axs
end

#===== Pass arguments to this document ===========================#
# $ julia product10.jl arg1 arg2
# then ARGS will get the arguments as String array as ["arg1", "arg2"]
@show ARGS

#======= check if this is noVNC graphical platform ================#
is_ImageView_available = true
try
    Sys.islinux() && run(`ls /usr/bin/x11vnc`) # check if this is noVNC graphical platform
    using ImageView
    using Gtk.ShortNames
catch # not a graphical platform
    @warn("Not a RIS noVNC graphical platform")
    global is_ImageView_available = false
end

#=== variable in inner fn overwrite the outer variable with same name ====#
function outer_f()
    x = 1; y = 1
    inner_inline_f(a)=(x=a*[2,3]; z=3) # this overwrites the outer x when this is called
    function inner_f(a)
        # can access outer variables
        @show y # first call y = 1 but later call y = [4,6]
        y = a*[2,3] # this overwrites the outer y when this fn is called
        h = 3 # but this doesn't still live at the outer scope
    end
    @show x, y # shows (x, y) = (1, 1)
    inner_inline_f_x=inner_inline_f(2)
    inner_f_y=inner_f(2) # show y = 1
    @show x, y # shows (x, y) = ([4, 6], [4, 6])
    inner_f_y=inner_f(2) # shows y=[4, 6]
    @show x, y # shows (x, y) = ([4, 6], [4, 6])
    @show z # ERROR: UndefVarError: z not defined
    @show h # ERROR: UndefVarError: h not defined
end

x = 1; y = 1
inline_f(a)=(x=a*[2,3]) # this doesn't overwrite the global x when this is called
function f(a)
    y = a*[2,3] # this doesn't overwrite the global y when this fn is called
end
inline_f_x=inline_f(2)
f_y=f(2)
@show x, y # shows (x, y) = (1, 1)
