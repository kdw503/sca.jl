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

#ARGS =  ["[\"face\"]","1","1","true",":column","[100]","[0]"]

SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
@show SNRs, maxiter
@show order, Wonly, sd_group
@show λs, βs
flush(stdout)
initpwradj = :balance; pwradj = :none; tol=-1
imgsize = (40,20); lengthT=1000; jitter=0
SNR = SNRs[1]; λ = λs[1]; λ1 = λ; λ2 = λ; β = βs[1]; β1 = β; β2 = Wonly ? 0. : β
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
        ncells = 49;  borderwidth=1
        fprefix0 = "Wuc_face"
        rt1 = @elapsed W0, H0, Mw, Mh, Wcd, Hcd = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
        # W1,H1 = copy(W0), copy(H0); normalizeWH!(W1,H1)
        # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
        # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W1,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
        # normalizeWH!(Wp,Hp)
        # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wp,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
    else
        imgsize = (40,20); lengthT=1000; jitter=0
        X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                    fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
        gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
        fprefix0 = "Wuc_$(SNR)dB"
        rt1 = @elapsed W0, H0, Mw, Mh, Wcd, Hcd = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
    end
    Mw0, Mh0 = copy(Mw), copy(Mh);
    for λ in λs
        λ1 = λ; λ2 = λ
        for β in βs
            @show SNR, λ, β
            β1 = β; β2 = Wonly ? 0. : β
            flush(stdout)
            paramstr="_L$(order)_lm$(λ)_aw$(β1)_ah$(β2)"
            fprefix =fprefix0*"_$(initpwradj)_$(pwradj)"*paramstr
            reg = :WkHk; method=:cbyc_uc
            stparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=reg, order=order, hfirst=true, processorder=:none,
                    poweradjust=pwradj, method=method, rectify=:pinv, objective=:normal)
            lsparams = LineSearchParams(method=:sca_full, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
                    iterations_to_show=[499,500])
            cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                    x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
            fprefix = "Wuc_$(SNR)dB_$(initpwradj)_$(pwradj)"*paramstr
            Mw, Mh = copy(Mw0), copy(Mh0)
            rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
            W2,H2 = copy(W1), copy(H1)
            normalizeWH!(W2,H2); iter = length(trs)
            if SNR == "face"
                clamp_level=0.5; Wcd_max = maximum(abs,Wcd)*clamp_level; Wcd_clamped = clamp.(Wcd,0.,Wcd_max)
                signedcolors = (colorant"green1", colorant"white", colorant"magenta")
                imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W2, imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
            else
                imsaveW(fprefix*"_iter$(iter)_rt2$(rt2).png",sortWHslices(Wcd,Hcd)[1],imgsz,borderwidth=1)
            end
            x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
            ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss)],fprefix*"_iter$(iter)_rt$(rt2)_log10plot.png"; title="convergence (SCA)", xlbl = "iteration",
                    ylbl = "log10(penalty)", legendstrs = ["log10(x_abs)", "log10(f_x)", "log10(xw_abs)", "log10(xh_abs)"], legendloc=1, separate_win=false)
                    #,axis=[480,1000,-0.32,-0.28])
        end
    end
end

# plt.show()
