using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")

include(joinpath(workpath,"setup_light.jl"))

num_experiments = 10
subdir = "compnmf"
subworkpath = joinpath(workpath,"paper",subdir)
methods = "compnmf"
SNR = -10
factor = 1
ncells = 15
compnmf_maxiter = 1000

dataset = :fakecells; inhibitindices=0; bias=0.1
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; maskth=0.25; makepositive = true; tol=-1
imgsz0 = (40,20)
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

for iter in 1:num_experiments
    @show iter
    # generate data
    X, imsz, lhT, ncs, gtncells, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
            inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

    (m,n,p) = (size(X)...,ncells)
    gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
    X = LCSVD.noisefilter(filter,X)

    if subtract_bg
        rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
        NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
        LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
        close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
        bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
    end
    m, n = size(X)

    prefix = "compnmf"; maxiter = compnmf_maxiter
    for initmethod in [:nndsvd, :rand]
        dd = Dict()
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        if initmethod == :nndsvd
            rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar)
        else
            Wcn0, Hcn0 = rand(m,ncells), rand(ncells,n)
            rt1 = 0
        end
        Wcn, Hcn = copy(Wcn0), copy(Hcn0);
        result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn;
                            gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
        Wcn, Hcn = copy(Wcn0), copy(Hcn0);
        rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
        rt1 += rst0.inittime # add calculation time for compression matrices L and R
        rt2 -= rst0.inittime
        avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wcn, Hcn; clamp=false)
        LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
        fprex = "$(prefix)$(SNR)db$(factor)f$(ncells)s$(initmethod)"
        fname = joinpath(subworkpath,prefix,"$(fprex)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
        #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
        TestData.imsave_data_gt(dataset,fname*"_gt", Wcn,Hcn,gtW,gtH,imgsz,100; saveH=false, verbose=false)

        rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
        dd["niters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
        dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
        if true#iter == num_experiments
            metadata = Dict()
            metadata["maxiter"] = maxiter
        end
        save(joinpath(subworkpath,prefix,"$(fprex)LXYR_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end

for i in 1:5
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=1000, tol=tol, verbose=false), X, Wcn, Hcn;
                        gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
    LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    makepositive && TestData.flip2makepos!(Wcn,Hcn)
    Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
    #TestData.imshowW(Wcn,imgsz)
    TestData.imsaveW(joinpath(subworkpath,"compnmf_rand_UV$(i).png"),Wno,imgsz)
    Wcn = result.L*result.X_tilde; Hcn = result.Y_tilde*result.R
    LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    makepositive && TestData.flip2makepos!(Wcn,Hcn)
    Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
    #TestData.imshowW(Wcn,imgsz)
    TestData.imsaveW(joinpath(subworkpath,"compnmf_rand_LXYR$(i).png"),Wno,imgsz)

    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    X_tilde, Y_tilde = CompNMF.compressive_nmf(result.A_tilde, result.L, result.R, Wcn, Hcn, ncells; max_iter=1000, ls=0)
    Wcn = result.L*X_tilde; Hcn = Y_tilde*result.R
    LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    makepositive && TestData.flip2makepos!(Wcn,Hcn)
    Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
    #TestData.imshowW(Wcn,imgsz)
    TestData.imsaveW(joinpath(subworkpath,"cnmf_rand$(i).png"),Wno,imgsz)
end

for i in 1:5
    @show i
    rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);

    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=1000, tol=tol, verbose=false), X, Wcn, Hcn;
                        gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
    LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    makepositive && TestData.flip2makepos!(Wcn,Hcn)
    Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
    #TestData.imshowW(Wcn,imgsz)
    TestData.imsaveW(joinpath(subworkpath,"compnmf_nndsvd_UV$(i).png"),Wno,imgsz)
    Wcn = result.L*result.X_tilde; Hcn = result.Y_tilde*result.R
    LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    makepositive && TestData.flip2makepos!(Wcn,Hcn)
    Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
    #TestData.imshowW(Wcn,imgsz)
    TestData.imsaveW(joinpath(subworkpath,"compnmf_nndsvd_LXYR$(i).png"),Wno,imgsz)

    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    L, R, X_tilde0, Y_tilde0, A_tilde = CompNMF.compmat(X, Wcn, Hcn; w=4)
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    X_tilde, Y_tilde = CompNMF.compressive_nmf(result.A_tilde, result.L, result.R, Wcn, Hcn, ncells; max_iter=1000, ls=0)
    Wcn = result.L*X_tilde; Hcn = Y_tilde*result.R
    LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
    makepositive && TestData.flip2makepos!(Wcn,Hcn)
    Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
    #TestData.imshowW(Wcn,imgsz)
    TestData.imsaveW(joinpath(subworkpath,"cnmf_nndsvd$(i).png"),Wno,imgsz)
end


Wcn0, Hcn0 = rand(m,ncells), rand(ncells,n)

Wcn, Hcn = copy(Wcn0), copy(Hcn0);
result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=1000, tol=tol, verbose=false), X, Wcn, Hcn;
                    gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
# L, R, X_tilde, Y_tilde, A_tilde = CompNMF.compmat(X, Wcn, Hcn)
# X_tilde0, Y_tilde0 = CompNMF.compressive_nmf(A_tilde, L, R, Wcn, Hcn, ncells; max_iter=1000, ls=0)
Wcn = result.L*result.X_tilde; Hcn = result.Y_tilde*result.R
LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
makepositive && TestData.flip2makepos!(Wcn,Hcn)
Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
#TestData.imshowW(Wcn,imgsz)
TestData.imsaveW(joinpath(subworkpath,"compnmf_rand_LXYR$(i).png"),Wno,imgsz)

Wcn, Hcn = copy(Wcn0), copy(Hcn0);
X_tilde, Y_tilde = CompNMF.compressive_nmf(result.A_tilde, result.L, result.R, Wcn, Hcn, ncells; max_iter=1000, ls=0)
Wcn = result.L*X_tilde; Hcn = Y_tilde*result.R
LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
makepositive && TestData.flip2makepos!(Wcn,Hcn)
Wno, Hno, mssd, mssdH = TestData.match_order_to_gt(Wcn,Hcn,gtW,gtH)
#TestData.imshowW(Wcn,imgsz)
TestData.imsaveW(joinpath(subworkpath,"cnmf_rand$(i).png"),Wno,imgsz)

@show norm(result.X_tilde-X_tilde)


Wcn0, Hcn0 = rand(m,ncells), rand(ncells,n)

Wcn, Hcn = copy(Wcn0), copy(Hcn0);
L, R, X_tilde, Y_tilde, A_tilde = CompNMF.compmat(X, Wcn, Hcn)
L0, R0, A_tilde0 = copy(L), copy(R), copy(A_tilde)
X_tilde0, Y_tilde0 = CompNMF.compressive_nmf(A_tilde, L, R, Wcn, Hcn, ncells; max_iter=1000, ls=0)
@show norm(L0-L), norm(R0-R), norm(A_tilde0-A_tilde)

Wcn, Hcn = copy(Wcn0), copy(Hcn0);
X_tilde, Y_tilde = CompNMF.compressive_nmf(A_tilde0, L0, R0, Wcn, Hcn, ncells; max_iter=1000, ls=0)

@show norm(X_tilde0-X_tilde)

# PyCall
using PyCall
np = pyimport("numpy")
Xpy = np.asarray(X)
np.save(joinpath(subworkpath,"X-10dB_factor1.npy"),Xpy)
tmp = np.load(joinpath(subworkpath,"X-10dB_factor1.npy"))


# now python

# activate environment at the power shell prompt
Scripts\activate.bat
# To deactivate a virtual environment, type:
# deactivate

# launch python
py

#
import time
import numpy as np
from numpy import random

x = random.randint(100, size=(3, 5))

from sklearn.utils.extmath import randomized_svd, squared_norm, randomized_range_finder

X = np.load("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca\\paper\\compnmf\\X-10dB_factor1.npy")
nr = 15
r_ov = 10

rt0 = time.time()
L = randomized_range_finder(X, size = nr + r_ov, n_iter = 3)
R = randomized_range_finder(X.T, size = nr + r_ov, n_iter = 3)
rt = time.time()-rt0

np.savez("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca\\paper\\compnmf\\LR-10dB_factor1.npz",L,R)

rt = 0
for i in range(1,50):
#    print(i)
    rt0 = time.time()
    L = randomized_range_finder(X, size = nr + r_ov, n_iter = 3)
    R = randomized_range_finder(X.T, size = nr + r_ov, n_iter = 3)
    rt += time.time()-rt0

rt/50


# julia again
using NPZ
LR = npzread(joinpath(subworkpath,"LR-10dB_factor1.npz"))
print(LR.files)
L = LR["arr_0"]
R = LR["arr_1"]'

# lcsvd uing initialization of scikit-learn implementation
Wcn0, Hcn0 = rand(m,ncells), rand(ncells,n)
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=1000, tol=tol, verbose=true), X, Wcn, Hcn;
   L=L, R=R, gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
rt2 = @elapsed CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=1000, tol=tol, verbose=false), X, Wcn, Hcn;
   L=L, R=R, gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
lt = length(result.avgfits)
trng_sk = range(0,rt2,length=lt)
avgfit_sk = copy(result.avgfits)

# lcsvd uing initialization of our implementation
Wcn0, Hcn0 = rand(m,ncells), rand(ncells,n)
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=1000, tol=tol, verbose=true), X, Wcn, Hcn;
   gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
rt2 = @elapsed CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=1000, tol=tol, verbose=false), X, Wcn, Hcn;
   gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
lt = length(result.avgfits)
trng = range(0,rt2,length=lt)
avgfit = copy(result.avgfits)


include(joinpath(workpath,"setup_plot.jl"))

fig = Figure(resolution=(400,300))
ax = AMakie.Axis(fig[1, 1], limits = ((0,1.0), nothing), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
ln1 = lines!(ax, trng_sk, avgfit_sk, color=mtdcolors[2], label="scikit-learn")
ln2 = lines!(ax, trng, avgfit, color=mtdcolors[3], label="our implem.")
axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db$(ncells)s_compare_Q.png"),fig,px_per_unit=2)
