using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    #cd("/home/daewoo/work/julia/sca")
end

Pkg.activate(".")

using Images, Convex, SCS, LinearAlgebra
using FakeCells, AxisArrays, ImageCore, MappedArrays
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
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

# ARGS = ["-10","500","2","false",":column","[100]","[0.0]"]

SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
@show SNRs, maxiter
@show order, Wonly, sd_group
@show λs, βs

imgsize = (40,20); lengthT=1000; jitter=0

fxs = []
for SNR = SNRs
X, imgsz, ncells, fakecells_dic, _ = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                    fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

for λ = λs
λ1 = λ; λ2 = λ
for β = βs
@show SNR, λ, β
β1 = β; β2 = Wonly ? 0. : β
tol = -1#1e-20
astparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=:WkHk, order=order, hfirst=true, processorder=:none,
        poweradjust=:none, method=:cbyc_uc, rectify=:pinv, objective=:normal)
lsparams = LineSearchParams(method=:sca_full, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
        iterations_to_show=[499,500])
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
fprefix = "W2_SNR$(SNR)_reg$(stparams.reg)_L$(order)_lmw$(λ1)_lmh$(λ2)_bw$(β1)_bh$(β2)"

rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, poweradjust=:normalize, use_nndsvd=true)
Mw0, Mh0 = copy(Mw), copy(Mh);#W1,H1 = copy(W0), copy(H0); 
#imsaveW("W0_SNR$(SNR)_rt$(rt1).png",sortWHslices(W1,H1)[1],imgsz,borderwidth=1)

rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
W2,H2 = copy(W1), copy(H1)
normalizeWH!(W2,H2); iter = length(trs)
imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)

x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss)],fprefix*"_iter$(iter)_rt$(rt2)_log10plot.png"; title="convergence (SCA)", xlbl = "iteration",
        ylbl = "log10(penalty)", legendstrs = ["log10(x_abs)", "log10(f_x)", "log10(xw_abs)", "log10(xh_abs)"], legendloc=1, separate_win=false)
        #,axis=[480,1000,-0.32,-0.28])
ax = plotW(f_xs./f_xs[end], fprefix*"_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "penalty", separate_win=false,axis=[1,100,1,2])

# ax = plotW([x_abss f_xs],fprefix*"_iter$(iter)_rt$(rt2)_plot.png"; title="convergence (SCA)", xlbl = "iteration",
#         ylbl = "penalty", legendstrs = ["x_abs", "f_x"], legendloc=1, separate_win=false)
pens = [norm(I-Mw0*Mh0)^2]; Mwspars = [norm(W0*Mw0,1)]; Mhspars = [norm(Mh0*H0,1)]
for i in 1:length(trs)
    Mw = trs[i].Mw; Mh = trs[i].Mh
    push!(pens, norm(I-Mw*Mh)^2)
    push!(Mwspars, norm(W0*Mw,1))
    push!(Mhspars, norm(Mh*H0,1))
end
ax = plotW([log10.(pens) log10.(Mwspars) log10.(Mhspars)],fprefix*"_iter$(iter)_rt$(rt2)_log10plot2.png"; title="convergence (SCA)", xlbl = "iteration",
          ylbl = "log10.(penalty)", legendstrs = ["Invertibility penalty", "W sparsity", "H sparsity"], legendloc=4, separate_win=false)
save(fprefix*"_iter$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
push!(fxs,f_xs[end])
end
end
end

# CD (with constraint rectify=:truncate, without constraint rectify=:none )
tol = 1e-6
stparams = StepParams(β1=0.3, β2=0.0, order=1, hfirst=true, processorder=:none, poweradjust=:none,
            rectify=:none) 
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=2, maxiter=100, store_trace=true, show_trace=true)
rt1 = @elapsed W0, H0, Mw, Mh, Wcd, Hcd = initsemisca(X, ncells, balance=false, use_nndsvd=true)
normalizeWH!(Wcd,Hcd)
rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
normalizeWH!(Wcd,Hcd);
imsaveW("Wcd_nor_wo_nn_WL$(stparams.order)_$(SNR)dB_a$(stparams.β1)_rt1$(rt1)_rt2$(rt2).png",sortWHslices(Wcd,Hcd)[1],imgsz,borderwidth=1)

# Original CD from NMF.jl
# regularization = :components(H only), :transformation (W only), :both
# l₁W = αW*l₁ratio; l₂W = αW*(1-l₁ratio)
# l₁H = αH*l₁ratio,; l₂H = αH*(1-l₁ratio),
using NMF
rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
Wcd, Hcd = copy(Wcd0), copy(Hcd0)
α=0.1
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=100, α=α, l₁ratio=1.0, regularization=:transformation), X, Wcd, Hcd)
# normalizeWH!(Wcd,Hcd)
imsaveW("Wnmfcd_wo_nn_WL1_a$(α)_rt1$(rtcd1)_rt2$(rtcd2)_SNR$(SNR).png",Wcd,imgsz,borderwidth=1)
normalizeWH!(Wcd,Hcd)
imsaveW("Wnmfcd_wo_nn_nor_WL1_a$(α)_rt1$(rtcd1)_rt2$(rtcd2)_SNR$(SNR).png",Wcd,imgsz,borderwidth=1)
Wcd, Hcd = copy(Wcd0), copy(Hcd0)
α=0.5
rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=100, α=α, l₁ratio=0.0, regularization=:transformation), X, Wcd, Hcd)
# normalizeWH!(Wcd,Hcd)
imsaveW("Wcd_WL2_a$(α)_rt1$(rtcd1)_rt2$(rtcd2)_SNR$(SNR).png",Wcd,imgsz,borderwidth=1)
normalizeWH!(Wcd,Hcd)
imsaveW("Wcd_nor_WL2_a$(α)_rt1$(rtcd1)_rt2$(rtcd2)_SNR$(SNR).png",Wcd,imgsz,borderwidth=1)

#=
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(SNRs, log10.(fxs))
ax1.legend(["minimum penalty"], fontsize = 12,loc=1)
xlabel("SNR",fontsize = 12)
ylabel("log10(penalty)",fontsize = 12)
savefig("W2_regWkHk_L$(order)_iter$(maxiter)_fxs_log10plot.png")
save("W2_regWkHk_L$(order)_iter$(maxiter)_fxs.jld", "fxs", fxs)

dd = load("W2_SNR60_regWkHk_L2_lmw100_lmh100_bw0.1_bh0.1_iter1000_rt5.088476965.jld")
ax = plotW([log10.(dd["x_abss"]) log10.(dd["f_xs"])],"test.png"; title="convergence (SCA)", xlbl = "iteration",
        ylbl = "log10(penalty)", legendstrs = ["log10(x_abs)", "log10(f_x)"], legendloc=1, separate_win=false,
        axis=[1,1000,-6.15,-6.1])


SNR = 60
X, imgsz, ncells, fakecells_dic, _ = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
        fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true, use_nndsvd=true)

stparams = StepParams(β1=0.1, β2=0.1, λ1=100, λ2=100, reg=:WkHk, order=2, hfirst=true, processorder=:none,
        poweradjust=:balance, method=:cbyc_uc, rectify=:pinv, objective=:normal)
k = 5, p = size(Mw,1)
hessdk, graddk, E = SCA.hessgradpenMwk(Mw, Mh, k; option=stparams.option)
W0f = Matrix{eltype(Mw)}(1.0I,p,p); W0TW0 = W0'W0
hesssk, gradsk = SCA.Hessgradsparsity(W0TW0, W0f, Mw, k, stparams)
mwk = Mw[:,k]; mhk = Mh[k,:]
wk = W0*mwk; nflag_wk = wk.<0; wkn = nflag_wk.*wk
W0n = Diagonal(nflag_wk)*W0; Hessnnk= W0n'*W0n; gradnnk = W0'*wkn

f_α_mpen(x) = (E=I-Mw*Mh; norm(E-x*mhk')^2)
f_α_mpenfull(x) = (Mwx = copy(Mw); Mwx[:,k].+=x; norm(I-Mwx*Mh)^2)
f_α_nnpen(x) = (mwkx = mwk+x; Wk = W0*mwkx; sca2(Wk))
f_α_sppen(x) = (mwkx = mwk+x; Wk = W0*mwkx; order = stparams.order; norm(Wk,order)^order)

fdHessdk = Calculus.hessian(f_α_mpen, zeros(p))
fdgraddk = Calculus.gradient(f_α_mpen, zeros(p))
fdHessfdk = Calculus.hessian(f_α_mpenfull, zeros(p))
fdgradfdk = Calculus.gradient(f_α_mpenfull, zeros(p))
norm(fdHessdk-fdHessfdk)
norm(fdgraddk-fdgradfdk)
norm(fdHessdk-2*hessdk*Matrix{eltype(Mw)}(1.0I,p,p))
norm(fdgraddk-2*graddk)

fdHessnnk = ForwardDiff.hessian(f_α_nnpen, zeros(p))
fdgradnnk = ForwardDiff.gradient(f_α_nnpen, zeros(p))
norm(fdHessnnk-2*Hessnnk)
norm(fdgradnnk-2*gradnnk)

fdHesssk = ForwardDiff.hessian(f_α_sppen, zeros(p))
fdgradsk = ForwardDiff.gradient(f_α_sppen, zeros(p))
norm(fdHesssk-2*hesssk*Matrix{eltype(Mw)}(1.0I,p,p))
norm(fdgradsk-2*gradsk)

SCA.plotfig(f_α_mpen, f_α_nnpen, f_α_sppen, 0.1, "test.png"; αrng = -0.5:0.01:1.5, titlestr="", axis=[480,1000,-1,0])
=#

λ=0; β=1.0
λ1 = λ2 = λ
β1 = β; β2 = Wonly ? 0. : β
for initpwradj = [:normalize, :balance]
    for pwradj = [:none, :normalize, :balance]
    @show SNR, λ, β
    β1 = β; β2 = Wonly ? 0. : β
    tol = -1#1e-20
    stparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=:WkHk, order=order, hfirst=true, processorder=:none,
            poweradjust=pwradj, method=:cbyc_uc, rectify=:pinv, objective=:normal)
    lsparams = LineSearchParams(method=:sca_full, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
            iterations_to_show=[499,500])
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
    fprefix = "W2_SNR$(SNR)_L$(order)_lm$(λ)_b$(β1)_$(β2)_$(initpwradj)_$(pwradj)"
    
    rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
    Mw0, Mh0 = copy(Mw), copy(Mh);#W1,H1 = copy(W0), copy(H0); 
    #imsaveW("W0_SNR$(SNR)_rt$(rt1).png",sortWHslices(W1,H1)[1],imgsz,borderwidth=1)
    
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); iter = length(trs)
    imsaveW(fprefix*".png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
    
    x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
    ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss)],fprefix*"_log10plot.png"; title="convergence (SCA)", xlbl = "iteration",
            ylbl = "log10(penalty)", legendstrs = ["log10(x_abs)", "log10(f_x)", "log10(xw_abs)", "log10(xh_abs)"], legendloc=1, separate_win=false)
            #,axis=[480,1000,-0.32,-0.28])
    end
end

    