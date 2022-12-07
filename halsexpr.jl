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
using FakeCells, AxisArrays, ImageCore, MappedArrays, NMF
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

# ARGS =  ["[:face]","30","1","true","[0.1]"]
SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
αs = eval(Meta.parse(ARGS[5])) # be careful spaces in the argument
@show SNRs, maxiter
@show order, Wonly, αs
flush(stdout)
gtncells = 7; tol=-1 # until maxiter
SNR = SNRs[1]; α = αs[1]
for SNR in SNRs
    if SNR == :face
        filepath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
        nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
        X = zeros(nRow*nCol,nFace)
        for i in 1:nFace
            fname = "face"*@sprintf("%05d",i)*".pgm"
            img = load(joinpath(filepath,fname))
            X[:,i] = vec(img)
        end
        ncells = 49;  borderwidth=1
        fprefix0 = "Hals_face"
        rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
        # W1,H1 = copy(W0), copy(H0); normalizeWH!(W1,H1)
        # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
        # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W1,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
        # normalizeWH!(Wp,Hp)
        # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wp,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
    else
        lengthT=1000; jitter=0
        if gtncells == 2
            imgsz = (20,30); fname = "obj2_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
        else
            imgsz = (40,20); fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
        end
        X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell(fname, gt_ncells=gtncells, fovsz=imgsize, imgsz=imgsz,
                                                ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
        gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
        fprefix0 = "Whals_$(SNR)dB_nc$(ncells)"
        rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
    end
    Wcd0, Hcd0 = copy(Wcd), copy(Hcd);
    for α in αs
        @show SNR, α
        flush(stdout)
        paramstr="_a$(α)"
        fprefix = fprefix0*paramstr
        Wcd, Hcd = copy(Wcd0), copy(Hcd0);
        rt2 = @elapsed results = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
        normalizeWH!(Wcd,Hcd)
        iter = results.niters
        if SNR == :face
            clamp_level=0.5; Wcd_max = maximum(abs,Wcd)*clamp_level; Wcd_clamped = clamp.(Wcd,0.,Wcd_max)
            signedcolors = (colorant"green1", colorant"white", colorant"magenta")
            imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", Wcd, imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
        else
            mssdcd, mlcd, ssds = matchednssd(gtW,Wcd)
            mssdHcd = ssdH(mlcd,gtH,Hcd')
            imsaveW(fprefix*"_iter$(iter)_rt2$(rt2).png",sortWHslices(Wcd,Hcd)[1],imgsz,borderwidth=1)
        end
        # ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss)],fprefix*"_iter$(iter)_rt$(rt2)_log10plot.png"; title="convergence (SCA)", xlbl = "iteration",
        #         ylbl = "log10(penalty)", legendstrs = ["log10(x_abs)", "log10(f_x)", "log10(xw_abs)", "log10(xh_abs)"], legendloc=1, separate_win=false)
    end
end

# plt.show()
