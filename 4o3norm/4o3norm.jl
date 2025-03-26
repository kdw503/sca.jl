
using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"4o3norm")

include(joinpath(workpath,"setup_light.jl"))
#include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using ForwardDiff

#=========== with FakeCells ==============#
dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.1
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))
bias = 0.1

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncs, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
(m,n,p) = (size(X)...,ncs)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

subtract_bg = false
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    bg = W*fill(mean(H),1,n)
    X .-= bg
end

# PCB
prefix = "pcb"
noc = ncs; nac = 0
(tailstr,initmethod,α,β) = ("_sp_nn",:isvd,0.005,5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')

β1 = β2= β; α1 = α2 = α
β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
r=0.3; useprecond=false; uselv=false; tol=1e-6; optim_method = :lbfgs4o3norm
maxiter = optim_method == :lbfgs4o3norm ? 1 : Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    r=r, useprecond=useprecond, usedenoiseW0H0=false, optim_method = optim_method,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = false,
    store_inner_trace = false, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rst1 = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed LCSVD.solve!(alg, X, U, V, D, M1, N1t);
alg.α1=α1; alg.α2=α2; alg.β1=0.; alg.β2=0.; alg.useprecond=true
M1, N1t = copy(M0), copy(N0t)
rt3 = @elapsed LCSVD.solve!(alg, X, U, V, D, M1, N1t);

using BenchmarkTools
M1, N1t = copy(M0), copy(N0t); @btime LCSVD.solve!(alg, X, U, V, D, $M1, $N1t);

W1, H1 = rst1.W, rst1.Ht'
LCSVD.normalizeW!(W1,H1);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
# dataset != :fakecells && (fv = LCSVD.fitd(X,W1*H1))
nodr = LCSVD.matchedorder(ml,noc); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
LCSVD.flip2makepos!(Wlc1,Hlc1)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)_$(initmethod)_$(optim_method)"
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_a$(α)_b$(β)"
fname = joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)

#========= Gradient Test =================#
m = 10; noc = 4; nac = 2; nc = noc+nac
U = rand(m,nc)
M = rand(nc,noc)
fx(M) = norm(U*M,4/3)
fdG = ForwardDiff.gradient(fx,M)
normp(x,l,p) = sum(abs.(x).^l)^p
W = U*M
# G = normp(W,4/3,-1/4)*U'*(abs.(W).^(1/3)) # sum((U*M).^(4/3))^(-1/4)*U'*(U*M).^(1/3)
G = normp(W,4/3,-1/4)*U'*(abs.(W).^(1/3).*sign.(W)) # sum((U*M).^(4/3))^(-1/4)*U'*(U*M).^(1/3)
norm(fdG - G)

m = 10; noc = 4; nac = 2; nc = noc+nac
U = rand(m,nc)
M = rand(nc,noc)
fx(M) = norm(U*M,4/3)^(4/3)
fdG = ForwardDiff.gradient(fx,M)
W = U*M
G = 4/3*U'*abs.(W).^(1/3) # 4/3**U'*(U*M).^(1/3)
norm(fdG - G)

#========= Gradient runtime Test =================#
noc = ncs; nac = 0
(tailstr,initmethod,α,β) = ("_sp_nn",:isvd,0.005,5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')

m,nc = size(U); n,nc = size(V); noc = size(M0,2); lt = noc*nc
αw = rand(); αh = rand(); βw = rand(); βh = rand(); σ0 = rand(); r = rand()
alg = LCSVD.LinearCombSVD(Float64,α1=αw, α2=αh, β1=βw, β2=βh, σ0=σ0, r=r, useprecond=false,
                            uselv=false, imgsz=imgsz)
M = copy(M0); Nt = copy(N0t)
state = LCSVD.prepare_state(U, V, M, Nt, alg)
updater = LCSVD.LinearCombSVDUpd{Float64}(D, state, noc, alg)

using BenchmarkTools
WTW0 = similar(state.WTW0); HH0T = similar(state.HH0T)
# relaxed L1
@btime begin
    nM, nNt, aw1n, ah1n, aw3n, ah3n, regW1, regH1, regW3, regH3 =
        LCSVD.cal_params(U, V, state.W, state.Ht, M, Nt, updater.σw2, updater.σh2, updater, alg) # 45.910 μs (3 allocations: 128 bytes)
@btime LCSVD.gradRelxEs1!(state.WTW0,view(state.TmpW,1:m,:),U,view(state.W,1:m,:),updater.σw2,uselv=false) # 39.679 μs (11 allocations: 400 bytes)
@btime LCSVD.sprod!(WTW0,state.WTW0,aw1n) # 62.310 ns (0 allocations: 0 bytes)
@btime state.grad[1:lt] .+= vec(WTW0) # 457.974 ns (6 allocations: 2.17 KiB)

@btime LCSVD.gradRelxEs1!(state.HH0T,view(state.TmpHt,1:n,:),V,view(state.Ht,1:n,:),updater.σh2;uselv=false) # 49.261 μs (11 allocations: 400 bytes)
@btime LCSVD.sprod!(HH0T,state.HH0T,ah1n) # 63.377 ns (0 allocations: 0 bytes)
@btime state.grad[lt+1:end] .+= vec(HH0T) # 437.031 ns (6 allocations: 2.17 KiB)
end # 139.437 μs (39 allocations: 5.28 KiB)


function cal_params(W0::AbstractMatrix{T}, H0t, W, Ht, M, Nt, σw2, σh2, updater, alg) where T
    m, nc = size(W0); n, nc = size(H0t)
    aw1, ah1, aw3, ah3 = updater.αw, updater.αh, updater.βw, updater.βh
    nM = (updater.αh != 0.) || (updater.βh != 0.) ? norm(M) : zero(T)
    nNt = (updater.αw != 0.) || (updater.βw != 0.) ? norm(Nt) : zero(T)
    (aw1n, ah1n) = (updater.αw*nNt,updater.αh*nM)
    (aw3n, ah3n) = (updater.βw*nNt^2,updater.βh*nM^2)
    regW1 = updater.αw != 0 ? relaxedL1(view(W,1:m,:), σw2; uselv=alg.uselv) : zero(T)
    regH1 = updater.αh != 0 ? relaxedL1(view(Ht,1:n,:), σh2; uselv=alg.uselv) : zero(T)
    regW3 = updater.βw != 0 ? sca2(view(W,1:m,:); uselv=alg.uselv) : zero(T)
    regH3 = updater.βh != 0 ? sca2(view(Ht,1:n,:); uselv=alg.uselv) : zero(T)
    nM, nNt, aw1n, ah1n, aw3n, ah3n, regW1, regH1, regW3, regH3
end

@btime LCSVD.relaxedL1(view(state.W,1:m,:), updater.σw2; uselv=false) # 19.000 μs (7 allocations: 224 bytes)

# L4o3
@btime begin
    nM, nNt, aw1n, ah1n, aw3n, ah3n, regW1, regH1, regW3, regH3, smW4o3, smHt4o3 =
        LCSVD.cal_params_4o3norm(U, V, state.W, state.Ht, M, Nt, updater, alg) # 559.937 μs (5 allocations: 211.14 KiB)
@btime LCSVD.gradEs4o3!(state.WTW0,view(state.TmpW,1:m,:),U,view(state.W,1:m,:);uselv=false) # 232.192 μs (10 allocations: 384 bytes)
@btime LCSVD.sprod!(WTW0,state.WTW0,aw1n*smW4o3^(-1/4)) # 127.101 ns (2 allocations: 32 bytes)
@btime state.grad[1:lt] .+= vec(WTW0) # 440.371 ns (6 allocations: 2.17 KiB)

@btime LCSVD.gradEs4o3!(state.HH0T,view(state.TmpHt,1:n,:),V,view(state.Ht,1:n,:);uselv=false) # 289.501 μs (10 allocations: 384 bytes)
@btime LCSVD.sprod!(HH0T,state.HH0T,ah1n*smHt4o3^(-1/4)) # 131.340 ns (2 allocations: 32 bytes)
@btime state.grad[lt+1:end] .+= vec(HH0T) # 451.735 ns (6 allocations: 2.17 KiB)
end # 1.108 ms (45 allocations: 216.36 KiB)

function cal_params_4o3norm(W0::AbstractMatrix{T}, H0t, W, Ht, M, Nt, updater, alg) where T
    m, nc = size(W0); n, nc = size(H0t)
    aw1, ah1, aw3, ah3 = updater.αw, updater.αh, updater.βw, updater.βh
    nM = (updater.αh != 0.) || (updater.βh != 0.) ? norm(M) : zero(T)
    nNt = (updater.αw != 0.) || (updater.βw != 0.) ? norm(Nt) : zero(T)
    (aw1n, ah1n) = (updater.αw*nNt,updater.αh*nM)
    (aw3n, ah3n) = (updater.βw*nNt^2,updater.βh*nM^2)
    smW4o3 = sum(abs.(W).^(4/3)); smHt4o3 = sum(abs.(Ht).^(4/3)); 
    regW1 = updater.αw != 0 ? smW4o3^(3/4) : zero(T)
    regH1 = updater.αh != 0 ? smHt4o3^(3/4) : zero(T)
    regW3 = updater.βw != 0 ? sca2(W; uselv=alg.uselv) : zero(T)
    regH3 = updater.βh != 0 ? sca2(Ht; uselv=alg.uselv) : zero(T)
    nM, nNt, aw1n, ah1n, aw3n, ah3n, regW1, regH1, regW3, regH3, smW4o3, smHt4o3
end

@btime smW4o3 = sum(abs.(state.W).^(4/3)) # 247.139 μs (5 allocations: 93.86 KiB)
@btime smW4o3 = sum(abs.(state.W).^(2)) # 8.562 μs (12 allocations: 94.06 KiB)

@btime 2.5^2 # 1.542 ns (0 allocations: 0 bytes) (integer exponent)
@btime 2.5^(4/3) # 1.499 ns (0 allocations: 0 bytes) (fractional exponent)
@btime 4/3 # 1.496 ns (0 allocations: 0 bytes)
a = 2.5
@btime a^2 # 24.891 ns (1 allocation: 16 bytes)
@btime a^(4/3) # 82.122 ns (1 allocation: 16 bytes)
@btime 4/3 # 1.496 ns (0 allocations: 0 bytes)
v = rand(100)
@btime v.^2 # 1.921 μs (8 allocations: 2.20 KiB)
@btime v.^(4/3) # 5.315 μs (2 allocations: 2.02 KiB)
@btime abs.(v).^(4/3) # 5.386 μs (3 allocations: 2.03 KiB)
v = rand(800*15)
@btime v.^2 # 7.230 μs (9 allocations: 94.02 KiB)
@btime v.^(4/3) # 254.006 μs (3 allocations: 93.83 KiB)
@btime abs.(v).^(4/3) # 255.122 μs (4 allocations: 93.84 KiB)
v = rand(100000)
@btime v.^2 # 80.882 μs (9 allocations: 781.52 KiB)
@btime v.^(4/3) # 1.959 ms (3 allocations: 781.33 KiB)
@btime abs.(v).^(4/3) # 255.122 μs (4 allocations: 93.84 KiB)
@btime cbrt.(v) # 734.982 μs (3 allocations: 781.31 KiB)
@btime v.^(1/3) # 1.957 ms (3 allocations: 781.33 KiB)
@btime cbrt.(v.^4) # 1.620 ms (10 allocations: 781.55 KiB)
@btime v.^(1//3) # 1.958 ms (3 allocations: 781.33 KiB)
@btime v.^(4//3) # 1.957 ms (3 allocations: 781.33 KiB)

@btime abs.(state.W) # 6.325 μs (3 allocations: 93.81 KiB)
@btime smW4o3 = sum((state.W.^(2)).^(2/3)) # 247.139 μs (5 allocations: 93.86 KiB)
@btime smW4o3 = sum(abs.(state.W).^(1.3333333333333333)) # 247.501 μs (5 allocations: 93.86 KiB)
@btime smW4o3 = norm(state.W,4/3)^(3/4) # 239.027 μs (2 allocations: 32 bytes)
@btime smW4o3^(3/4) # 42.184 ns (1 allocation: 16 bytes)

function norm4o3_test!(x)
    sum = 0.
    @inbounds @simd for i in eachindex(x)
        sum += abs(x[i])^(4/3)
    end
    sum
end
@btime smW4o3 = norm4o3_test!(state.W) # 216.476 μs (1 allocation: 16 bytes)

#============ Plot ==============================#
xrng = -2:0.001:2
y1s = norm.(xrng,1)
y4o3s = norm.(xrng,4//3).^(4//3)
y2s = norm.(xrng,2).^2

fig = Figure()
ax = AMakie.Axis(fig[1, 1], xlabel = "x", ylabel = "∥x∥ₗᵖ", title = "",
        xminorgridvisible=true,xminorticks = IntervalsBetween(10),xminorticksvisible=true)

lns = Dict()
ln = lines!(ax, xrng, y1s, color=mtdcolors[2], label="l=1")
lns["l=1"] = ln
ln = lines!(ax, xrng, y4o3s, color=mtdcolors[3], label="l=4/3")
lns["l=4/3"] = ln
ln = lines!(ax, xrng, y2s, color=mtdcolors[5], label="l=2")
lns["l=2"] = ln

axislegend(ax, position = :ct) # halign = :left, valign = :top
save(joinpath(".","4o3norm","4o3norm2.png"),fig,px_per_unit=2)

