using Pkg
import Base:pathof

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

# ARGS =  ["[\"face\"]","5","1","true",":column","[0]","[5]",":none",":decimate"]
# ARGS =  ["[10]","50","1","true",":column","[0]","[0.01]",":balance2",":none"]
SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
pwradj = eval(Meta.parse(ARGS[8])); weighted = eval(Meta.parse(ARGS[9]))
initpwradj = :balance; ncells = 15; gtncells = 7

@show SNRs, ncells, maxiter, order, Wonly, sd_group, λs, βs, initpwradj, pwradj, weighted
flush(stdout)

SNR = SNRs[1]; λ=λ1=λ2=λs[1]; β1=βs[1]; β2= Wonly ? 0 : βs[1]

for SNR in SNRs
    if SNR == "face"
        filepath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
        nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
        X = zeros(nRow*nCol,nFace)
        for i in 1:nFace
            fname = "face"*@sprintf("%05d",i)*".pgm"
            img = load(joinpath(filepath,fname))
            X[:,i] = vec(img)
        end
        ncells0 = 49;  borderwidth=1
        fprefix0 = "Wuc_face_nc$(ncells)"
        rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
        # W1,H1 = copy(W0), copy(H0); normalizeWH!(W1,H1)
        # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
        # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W1,imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
        # normalizeWH!(Wp,Hp)
        # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wp,imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
    else
        lengthT=1000; jitter=0
        if gtncells == 2
            imgsz = (20,30); fname = "obj2_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
        else
            imgsz = (40,20); fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
        end
        X, imgsz, ncells0, fakecells_dic, img_nl, maxSNR_X = loadfakecell(fname, gt_ncells=gtncells, fovsz=imgsz, imgsz=imgsz,
                                                ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
        gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
        fprefix0 = "Wuc_$(SNR)dB_nc$(ncells)"
        rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
        # W0n, H0n = copy(W0), copy(H0); normalizeWH!(W0n,H0n)
        # Wpn, Hpn = copy(Wp), copy(Hp); normalizeWH!(Wpn,Hpn)
        # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W0n,imgsz, borderwidth=1)
        # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wpn,imgsz, borderwidth=1)
    end
    Mw0, Mh0 = copy(Mw), copy(Mh);
    for λ in λs
        λ1 = λ; λ2 = λ
        for β in βs
            @show SNR, λ, β
            β1 = β; β2 = Wonly ? 0. : β
            flush(stdout)
            paramstr="_$(sd_group)_$(weighted)_L$(order)_lm$(λ)_bw$(β1)_bh$(β2)"
            fprefix = fprefix0*"_Convex_$(initpwradj)_$(pwradj)"*paramstr
            sd_group ∉ [:column, :component, :pixel] && error("Unsupproted sd_group")
            Mw, Mh = copy(Mw0), copy(Mh0);
            rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,order;
                        poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, SNR=SNR, store_trace=true, weighted=weighted, decifactor=4)
            W2,H2 = copy(W0*Mw), copy(Mh*H0)
            normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
            if SNR == "face"
                clamp_level=0.5; W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
                signedcolors = (colorant"green1", colorant"white", colorant"magenta")
                imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W2, imgsz, gridcols=7, colors=signedcolors, borderval=W2_max, borderwidth=1)
            else
                imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            end
            ax = plotW([log10.(f_xs) log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_rt$(rt2)_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
                legendstrs = ["log(x_abs)", "log(f_x)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
            save(fprefix*"_iter$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "xw_abss", xw_abss, "xh_abss", xh_abss, "SNR", SNR,
                    "order", order, "β1", β1, "β2", β2, "rt2", rt2, "trs", trs)
        end
    end
end

# plt.show()
#=
dd = load("Wuc_60dB_Convex_normalize_none_column_L1_lm0_bw5_bh0.0_iter30_rt234.06637445.jld")
trs = dd["trs"]
rows = length(trs); cols = length(trs[1].peninvMw)
pimws=zeros(rows,cols); pimhs=zeros(rows,cols)
pnmws=zeros(rows,cols); pnmhs=zeros(rows,cols)
psmws=zeros(rows,cols); psmhs=zeros(rows,cols)
for k=1:15
    pimwks=[]; pimhks=[]
    pnmwks=[]; pnmhks=[]
    psmwks=[]; psmhks=[]
    for tr in trs
        push!(pimwks,tr.peninvMw[k]); push!(pimhks,tr.peninvMh[k])
        push!(pnmwks,tr.pennnMw[k]); push!(pnmhks,tr.pennnMh[k])
        push!(psmwks,tr.pensparMw[k]); push!(psmhks,tr.pensparMh[k])
    end
    pimws[:,k] = pimwks; pimhs[:,k] = pimhks
    pnmws[:,k] = pnmwks; pnmhs[:,k] = pnmhks
    psmws[:,k] = psmwks; psmhs[:,k] = psmhks
end
plotcols = 7
ax = plotW(pimws[:,1:plotcols], fprefix*"_imw_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "delta penalty",
    legendstrs = collect(1:plotcols), legendloc=1, separate_win=false)
ax = plotW(pimhs[:,1:plotcols], fprefix*"_imh_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "delta penalty",
    legendstrs = collect(1:plotcols), legendloc=1, separate_win=false)
ax = plotW(pnmws[:,1:plotcols], fprefix*"_nmw_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "delta penalty",
    legendstrs = collect(1:plotcols), legendloc=1, separate_win=false)
ax = plotW(pnmhs[:,1:plotcols], fprefix*"_nmh_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "delta penalty",
    legendstrs = collect(1:plotcols), legendloc=1, separate_win=false)
ax = plotW(psmws[:,1:plotcols], fprefix*"_smw_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "delta penalty",
    legendstrs = collect(1:plotcols), legendloc=1, separate_win=false)
ax = plotW(psmhs[:,1:plotcols], fprefix*"_smh_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "delta penalty",
    legendstrs = collect(1:plotcols), legendloc=1, separate_win=false)
=#

#=
imgsz = (20,30); lengthT = 1000; SNR=10; distance = 10; overlap_rate = 0.3
X, imgsz, ncells0, fakecells_dic, img_nl, maxSNR_X = loadfakecell("obj2_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld";
        gt_ncells=2, fovsz=imgsz, imgsz=imgsz, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
ncells = 5
fprefix0 = "Wuc_$(SNR)dB_nc$(ncells)"
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
W0n, H0n = copy(W0), copy(H0); normalizeWH!(W0n,H0n)
imsaveW(fprefix0*"_SVD_rt$(rt1).png",W0n,imgsz, borderwidth=1)
img = reshape(imgrs,imgsize...,lengthT)
save("img.gif", RGB{N0f8}.(clamp01nan!(img./maximum(img))))



order = 1; weighted = :none
verbose=false; show_figure=false; xmin=-10; xmax=10; fn=""; cols=1;
poweradjust = :balance2

Mw,Mh=copy(Mw0),copy(Mh0)
nMh = norm(Mh,2)^order; Hi = Mh*H0; nMhH0 = norm(Hi,order)^order

λ1=λ2=0; β1=5.0; β2=0
m, p = size(W0); n = size(H0,2)
normw2 = norm(W0)^2; normh2 = norm(H0)^2
normwp = norm(W0,order)^order; normhp = norm(H0,order)^order
λw = λ1/normw2; λh = λ2/normh2
βw = β1/normwp; βh = β2/normhp

k=1

p = size(W0,2)
fw=gh=Matrix(1.0I,p,p)
mwk = Mw[:,k]; mhk = Mh[k,:]'
(Eprev,Ek) = weighted == :none ? (I-Mw*Mh, Mw[:,k]*mhk) : (fw*(I-Mw*Mh)*gh, fw*Mw[:,k]*mhk*gh)
E = Eprev+Ek
(poweradjust==:balance2 && order==1 && βh!=0) && (sqMwk = sqrt(norm(Mw)^2-norm(mwk)^2))
# Convex : set variable
x = Variable(p)
set_value!(x, Mw[:,k])
# Convex : set problem
invertibility = weighted == :none ? sumsquares(E-x*mhk) : sumsquares(E-fw*x*mhk*gh)
if poweradjust == :balance2
    sparw = βw==0 ? 0 : order == 1 ? βw*norm(W0*x, 1)*nMh : βw*sumsquares(W0*x)*nMh
    sparh = βh==0 ? 0 : order == 1 ? βh*nMhH0*norm(vcat(x,sqMwk),2) : βh*nMhH0*norm(x,2)^2
    sparsity = sparw+sparh
else
    sparsity = order == 1 ? βw*norm(W0*x, 1) : βw*sumsquares(W0*x)
end
nnegativity = λ*sumsquares(max(0,-W0*x))
expr = invertibility + nnegativity + sparsity
problem = minimize(expr)
Evalpre = Convex.evaluate(expr)
totalpen, _ = SCA.penaltyMw(Mw,Mh,W0,H0,fw,gh,λ,βw,βh; order=order, poweradjust=poweradjust, weighted=weighted)
totalpen -= βw*(norm(W0*Mw,1)-norm(W0*mwk,1))*nMh
Convex.evaluate(invertibility)
Convex.evaluate(sparsity)
Convex.evaluate(nnegativity)
# println("expression curvature = ", vexity(expr))
# println("expression sign = ", sign(expr))

# Convex : solve
solve!(problem, ECOS.Optimizer; warmstart = false, silent_solver = true, verbose=verbose) 
# other solver options : SCS, ECOS, (GLPK : run error), (Gurobi, Mosek : precompile error)
# verbose=false (turn off warning)
# warmstart doesn't work for SCS.GeometricConicForm and ECOS

# Convex : check the result
# @show round.(Convex.evaluate(x), digits = 2)
Eval = problem.optval # round(problem.optval, digits = 10)
Evalfromxsol = Convex.evaluate(expr) # round(Convex.evaluate(expr), digits = 10)

Mwnew = copy(Mw); Mwnew[:,k]=x.value
Evalfromxsol2 = SCA.penaltyMw(Mwnew,Mh,W0,H0,fw,gh,λ,βw,βh; order=order, poweradjust=poweradjust, weighted=weighted)
Evalfromxsol2 -= βw*(norm(W0*Mw,1)-norm(W0*mwk,1))*nMh


x = Variable(15)
set_value!(x, rand(15))
expr = norm(vcat(x,sqrt(2)),2)
# expr = sumsquares(x)
problem = minimize(expr)
Evalpre = Convex.evaluate(expr)
println("expression curvature = ", vexity(expr))
println("expression sign = ", sign(expr))
solve!(problem, ECOS.Optimizer; warmstart = false, silent_solver = true, verbose=true) 
problem.optval
x.value
=#