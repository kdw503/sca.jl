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
include( ENV["JULIA_DEPOT_PATH"]*"/dev/SymmetricComponentAnalysis/test/testutils.jl")

plt.ioff()

SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
@show SNRs, maxiter
@show order, Wonly, sd_group
@show λs, βs
flush(stdout)
initpoweradjust=:normalize; poweradjust = :none; tol=-1
imgsize = (40,20); lengthT=1000; jitter=0
for SNR in SNRs
    X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                                fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
    rt1 = @elapsed W0, H0, Mw, Mh, Wcd, Hcd = initsemisca(X, ncells, poweradjust=initpoweradjust, use_nndsvd=true)
    Mw0, Mh0 = copy(Mw), copy(Mh)
    for λ in λs
        λ1 = λ; λ2 = λ
        for β in βs
            @show SNR, λ, β
            β1 = β; β2 = Wonly ? 0. : β
            paramstr="_L$(order)_lm$(λ)_bw$(β1)_bh$(β2)"
            # convex
            fprefix = "W2_$(SNR)dB_Convex_$(sd_group)"*paramstr
            sd_group ∉ [:column, :ac, :pixel] && error("Unsupproted sd_group")
            normalizeWH!(W0,H0); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
            imsaveW("W0_$(SNR)dB.png",sortWHslices(W0,H0)[1],imgsz,borderwidth=1)
            Mw0, Mh0 = copy(Mw), copy(Mh);
            rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter = minMwMh!(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,order;
                            poweradjust=poweradjust, fprefix=fprefix, sd_group=sd_group, SNR=SNR)
            save(fprefix*"_iter$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
            W2,H2 = copy(W0*Mw), copy(Mh*H0)
            normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
            imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
                legendstrs = ["log(x_abs)", "log(f_x)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
            # ssca
            reg = :WkHk; method=:cbyc_uc
            stparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=reg, order=order, hfirst=true, processorder=:none,
                    poweradjust=poweradjust, method=method, rectify=:pinv, objective=:normal)
            lsparams = LineSearchParams(method=:sca_full, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
                    iterations_to_show=[499,500])
            cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                    x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
            fprefix = "W2_$(SNR)dB_$(method)"*paramstr
            Mw, Mh = copy(Mw0), copy(Mh0)
            rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
            W2,H2 = copy(W1), copy(H1)
            normalizeWH!(W2,H2); iter = length(trs)
            imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            
            x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
            ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss)],fprefix*"_iter$(iter)_rt$(rt2)_log10plot.png"; title="convergence (SCA)", xlbl = "iteration",
                    ylbl = "log10(penalty)", legendstrs = ["log10(x_abs)", "log10(f_x)", "log10(xw_abs)", "log10(xh_abs)"], legendloc=1, separate_win=false)
                    #,axis=[480,1000,-0.32,-0.28])
            # hals
            stparams = StepParams(β1=β1, β2=β2, order=order, hfirst=true, processorder=:none, poweradjust=poweradjust,
                        rectify=:none) 
            cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                        x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
            normalizeWH!(Wcd,Hcd)
            rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
            normalizeWH!(Wcd,Hcd); iter = length(trs)
            imsaveW("Wcd_no_nn_L$(order)_$(SNR)dB_a$(β1)_rt1$(rt1)_rt2$(rt2).png",sortWHslices(Wcd,Hcd)[1],imgsz,borderwidth=1)
        end
    end
end

# plt.show()
