using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    #cd("/home/daewoo/work/julia/sca")
end

Pkg.activate(".")

#using MultivariateStats # for ICA
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

# ARGS =  ["[10]","2","1","true",":column","[0]","[0.5]"]
# julia C:\Users\kdw76\WUSTL\Work\julia\sca\convexcdexpr.jl [10] 50 1 true :column [0] [0.01,0.1,1.0,10,100]
SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument

@show SNRs, maxiter
@show order, Wonly, sd_group
@show λs, βs
flush(stdout)
initpwradj = :balance; pwradj = :none; tol=-1
SNR = SNRs[1]; λ = λs[1]; β = βs[1]
for SNR in SNRs
    if false # 7cells
        gtncells = 7; imgsz = (40,20); ncls = 15
    else
        gtncells = 2; imgsz = (20,30); ncls = 6
    end
    lengthT=1000; jitter=0
    @show SNRs, imgsz
    fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
    X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell(fname, gt_ncells=gtncells, fovsz=imgsz,
            imgsz=imgsz, ncells=ncls, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
    rt1 = @elapsed W0, H0, Mw, Mh, Wcd, Hcd = initsemisca(X, ncells, initmethod=:nndsvd, poweradjust=initpwradj)
    Wcd0, Hcd0 = copy(Wcd), copy(Hcd);
    for λ in λs
        λ1 = λ; λ2 = λ
        for β in βs
            @show SNR, λ, β
            β1 = β; β2 = Wonly ? 0. : β
            flush(stdout)
            paramstr="_L$(order)_lm$(λ)_bw$(β1)_bh$(β2)"
            fprefix = "Wcd_$(SNR)dB_CVX_$(initpwradj)_$(pwradj)"*paramstr
            sd_group ∉ [:column, :ac, :pixel] && error("Unsupproted sd_group")
            Wcd, Hcd = copy(Wcd0), copy(Hcd0);
            rt2 = @elapsed Wcd, Hcd, f_xs, x_abss, xw_abss, xh_abss, iter = minWH!(X,Wcd,Hcd,λ1,λ2,β1,β2,maxiter,order;
                            poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, SNR=SNR)
            save(fprefix*"_iter$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
            W2,H2 = copy(Wcd), copy(Hcd)
            normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
            imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
                legendstrs = ["log(x_abs)", "log(f_x)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
        end
    end
end

# plt.show()
