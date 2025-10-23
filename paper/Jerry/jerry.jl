using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","jerry")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))

using ImagineFormat
imgwrp = load("C:\\Users\\kdw76\\WUSTL\\Work\\Data\\JerryOCPI\\DongHoon1L05_Apertures_warp.imagine")
imgorg = load("C:\\Users\\kdw76\\WUSTL\\Work\\Data\\JerryOCPI\\DongHoon1.imagine")
imshow(imgwrp)

# z = 3, t=30:end, x = 300:end, y = 200:end
imgwrpcrop = imgwrp[200:end-2,300:end-2,3,[collect(1:24)...,collect(30:end)...]]
imshow(imgwrpcrop)
imgwrpcropmean = dropdims(mean(imgwrpcrop,dims=3),dims=3)

y,x,t = size(imgwrpcrop); imgsz = (y,x)

X = Array(reshape(imgwrpcrop,x*y,t))
rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
NMF.solve!(NMF.CoordinateDescent{eltype(W)}(maxiter=60, α=0), X, W, H)
LCSVD.normalizeW!(W,H)
wmin = minimum(W); Wimg = W.-wmin; wmax = maximum(Wimg); Wimg = Wimg ./ wmax
bgimg = reshape(Wimg, imgsz)
save(joinpath(subworkpath,"bgW.png"),bgimg)
plotH_data(joinpath(subworkpath,"bgH"),H)
bgmeanH = W*fill(mean(H),1,t)
bgH = W*H
#        bg = W*H
Xsbgmh = X .- bgmeanH
Xsbgh = X .- bgH
save(joinpath(subworkpath,"X.jld2"),"Xsbgmh",Xsbgmh,"Xsbgh",Xsbgh,"Xwbg",X)
bgmtd = :nosbg
X = bgmtd == :sbgmh ? load(joinpath(subworkpath,"X.jld2"),"Xsbgmh") :
    bgmtd == :sbgh ? load(joinpath(subworkpath,"X.jld2"),"Xsbgh") :
    bgmtd == :nosbg ? load(joinpath(subworkpath,"X.jld2"),"Xwbg") : error("Unknown bgmtd $(bgmtd)")

noc = 30 # true=27
nac = 0

# https://docs.dandiarchive.org/example-notebooks/tutorials/cosyne_2023/advanced_asset_search/
# Need to find wide field single photon imaging data

# PCB
prefix = "PCB"
initmethod = :sbc
rt0 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=:isvd, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
rt11 = 0
if initmethod == :sbc
    try
        rt11 = @elapsed M0 = sbc(U)
    catch e
        save(joinpath(subworkpath,"sbc_error$(iter).jld2"),"U",W0)
        error("SBC failed with error: $(e)")
    end
    rt12 = @elapsed N0 = M0\D
    rt13 = @elapsed LCSVD.balanceWH!(M0, N0)
    rt1 = rt0+rt11+rt12+rt12
    W1 = U*M0; H1 = N0*H0
    LCSVD.normalizeW!(W1,H1);
    fv = LCSVD.fitd(X,W1*H1)
    fprex = "$(prefix)_$(bgmtd)_$(initmethod)"
    LCSVD.flip2makepos!(W1,H1)
    fname = joinpath(subworkpath,"$(fprex)_f$(fv)_rt$(rt1)") 
    imsave_data(:ocpi,fname,W1,H1,imgsz,100; gridcols=6, borderwidth=4, saveH=false, verbose=false)
else
    rt1 = rt0
end

β = 0; α = 0.005
β1 = β2 = β; α1 = α2 = α
β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
r = 0.3; useprecond = false
usecalparams = false
tol = 1e-5; inner_tol = 1e-6; maxiter = Int(ceil(log(eps(eltype(X)))/log(r))); inner_maxiter = 1000
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    # α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    r=r, useprecond=useprecond, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter,
    store_trace = false, store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol,
    f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

W1, H1 = rst0.W, rst0.Ht'
LCSVD.normalizeW!(W1,H1);
fv = LCSVD.fitd(X,W1*H1)
fprex = "$(prefix)_$(bgmtd)_$(initmethod)"
LCSVD.flip2makepos!(W1,H1)
fname = usecalparams ? joinpath(subworkpath,"$(fprex)_av[1]$(α1vec[1])_av[2]$(α1vec[2])_bv[1]$(β1vec[1])_bv[2]$(β1vec[2])_f$(fv)_it$(rst0.niters)_rt$(rt2)") :
                       joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_f$(fv)_it$(rst0.niters)_rt$(rt2)_nocal") 
imsave_data(:ocpi,fname,W1,H1,imgsz,100; gridcols=6, borderwidth=4, saveH=false, verbose=false)

# HALS
prefix = "HALS"
rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, noc, variant=:ar)
maxiter = 100; α=0
W1, H1 = copy(Wcd0), copy(Hcd0)
rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{eltype(W1)}(maxiter=maxiter, α=α, l₁ratio=1,
                tol=tol, verbose=false), X, W1, H1)
LCSVD.normalizeW!(W1,H1)
fv = LCSVD.fitd(X,W1*H1)
fprex = "$(prefix)_$(bgmtd)"
fname = joinpath(subworkpath,"$(fprex)_a$(α)_f$(fv)_it$(rst0.niters)_rt$(rt2)")
imsave_data(:ocpi,fname,W1,H1,imgsz,100; gridcols=6, borderwidth=4, saveH=false)
