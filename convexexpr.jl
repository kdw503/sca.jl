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

ARGS =  ["[\"face\"]","50","1","true",":column","[0]","[5]"]

SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
@show SNRs, maxiter
@show order, Wonly, sd_group
@show λs, βs
flush(stdout)
initpwradj = :normalize; pwradj = :none
for SNR in SNRs
    if SNR == "face"
        path = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
        nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
        X = zeros(nRow*nCol,nFace)
        for i in 1:nFace
            fname = "face"*@sprintf("%05d",i)*".pgm"
            img = load(joinpath(path,fname))
            X[:,i] = vec(img)
        end
        ncells = 49;  borderwidth=1
        fprefix = "Wuc_face"

    else
        imgsize = (40,20); lengthT=1000; jitter=0
        X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                                    fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
        gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
        fprefix = "Wuc_$(SNR)dB"
    end
    rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
    Mw0, Mh0 = copy(Mw), copy(Mh);
    for λ in λs
        λ1 = λ; λ2 = λ
        for β in βs
            @show SNR, λ, β
            β1 = β; β2 = Wonly ? 0. : β
            flush(stdout)
            paramstr="_$(sd_group)_L$(order)_lm$(λ)_bw$(β1)_bh$(β2)"
            fprefix = fprefix*"_Convex_$(initpwradj)_$(pwradj)"*paramstr
            sd_group ∉ [:column, :ac, :pixel] && error("Unsupproted sd_group")
            Mw, Mh = copy(Mw0), copy(Mh0);
            rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter = minMwMh!(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,order;
                            poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group,SNR=SNR)
            save(fprefix*"_iter$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
            W2,H2 = copy(W0*Mw), copy(Mh*H0)
            normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
            if SNR == "face"
                clamp_level=0.5; W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
                colors = (colorant"white", colorant"white", colorant"black")
                imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W2, imgsz, gridcols=7, colors=colors, borderval=W2_max, borderwidth=1)
                imshowW(W2_clamped, imgsz, gridcols=7, colors=colors, borderval=W2_max, borderwidth=1)
            else
                imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            end
            ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
                legendstrs = ["log(x_abs)", "log(f_x)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
        end
    end
end

# plt.show()
