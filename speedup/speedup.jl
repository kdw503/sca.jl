
using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"speedup")

include(joinpath(workpath,"setup_light.jl"))
#include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using BenchmarkTools

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
r=0.3; useprecond=false; uselv=false; tol=1e-6
maxiter = Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    r=r, useprecond=useprecond, usedenoiseW0H0=false,
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
# rt2 = @elapsed LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW = gtW, gtH = gtH);
# @profview LCSVD.solve!(alg, X, U, V, D, M1, N1t);
# M1, N1t = copy(M0), copy(N0t); @btime LCSVD.solve!(alg, X, U, V, D, $M1, $N1t);
# before : 35.218 ms (11620 allocations: 87.70 MiB)
# change W0TW to WTW0 : 33.343 ms (10694 allocations: 78.98 MiB)
# using view : 22.194 ms (8228 allocations: 3.82 MiB)
# Reduce memory : 21.788 ms (8225 allocations: 3.81 MiB)
# after order=2 for sca2 : 16.357 ms (8221 allocations: 3.81 MiB)
# after reimplement sca2 : 15.802 ms (8316 allocations: 3.83 MiB)
# after remove duplicated calculations : 15.754 ms (8312 allocations: 3.68 MiB)
# after type inference fix and x0 = vcat(.,.) : 15.471 ms (8138 allocations: 3.63 MiB), 19.165 ms (8887 allocations: 6.36 MiB on RIS)

# RIS
# before : 84.885 ms (26194 allocations: 165.73 MiB)
# after remove duplicated calculations : 53.557 ms (28634 allocations: 25.85 MiB)
# after type inference fix and x0 = vcat(.,.) : 49.897 ms (26279 allocations: 24.92 MiB)
# precond (sp_nn)
# before : 109.364 ms (29594 allocations: 195.16 MiB)
# after remove duplicated calculations : 61.398 ms (28842 allocations: 20.43 MiB)
# after type inference fix and x0 = vcat(.,.) : 47.481 ms (20713 allocations: 14.71 MiB)
# precond(sp)
# before : 85.872 ms (42152 allocations: 198.67 MiB)
# after remove duplicated calculations : 54.275 ms (33078 allocations: 33.97 MiB)
# after type inference fix and x0 = vcat(.,.) : 52.369 ms (31405 allocations: 34.81 MiB)


W1, H1 = rst1.W, rst1.Ht'
LCSVD.normalizeW!(W1,H1);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
dataset == :fakecells && (fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false))
dataset != :fakecells && (fv = LCSVD.fitd(X,W1*H1))
nodr = LCSVD.matchedorder(ml,noc); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
LCSVD.flip2makepos!(Wlc1,Hlc1)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)_$(initmethod)"
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_a$(α)_b$(β)"
fname = joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)

#========= View Test =================#
A0 = zeros(4,4)
B0 = zeros(20000,4)
C0 = rand(10000,4)
D0 = rand(20000,4)
function mem_test!(A,B,C,D)
    for i in eachindex(D)
        B[i] = D[i]^2
    end
    LCSVD.mmul!(A,C',B)
end
m = 10000
@btime mem_test!(A0,B0[1:m,:],C0,D0[1:m,:]) # 78.300 μs (12 allocations: 625.34 KiB)
@btime mem_test!(A0,view(B0,1:m,:),C0,view(D0,1:m,:)) # mem_test!(A0,view(B0,1:m,:),C0,view(D0,1:m,:))


#========= gradEsM2 =================#
# function gradEsM2(M::AbstractMatrix{T}; power=1) where T
#     nc, noc = size(M); lt = nc*noc
#     if power == 1
#         nM = norm(M); x = vec(M)
#         x/nM
#     else
#         2*vec(M)
#     end
# end
# function gradEsM2!(gm::AbstractVector{T}, M::AbstractMatrix{T}; power=1) where T
#     nc, noc = size(M); lt = nc*noc
#     x = vec(M)
#     if power == 1
#         LCSVD.sdiv!(gm,x,norm(M))
#     else
#         LCSVD.sprod!(gm,x,2)
#     end
# end
# M = rand(100,100)
# gm = rand(10000) 
# gradEsM2(M, power=1)
# gradEsM2!(gm, M, power=1)
# @btime gradEsM2(M, power=1); # 10.100 μs (4 allocations: 78.25 KiB)
# @btime gradEsM2!(gm, M, power=1); # 6.420 μs (2 allocations: 80 bytes)
# @btime gradEsM2(M, power=2); # 7.833 μs (4 allocations: 78.25 KiB)
# @btime gradEsM2!(gm, M, power=2); # 1.280 μs (2 allocations: 80 bytes)

#========= relaxedL1 =================#
function relaxedL1_nlv_old(x, σ2)
    result = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        result += sqrt(x[i]^2 + σ2)
    end
    return result
end
function relaxedL1_nlv(x, σ2)
    result = zero(eltype(x))
    @inbounds @simd for a in x
        result += sqrt(a^2 + σ2)
    end
    return result
end

A = rand(800,1000) .- 0.5
σ2 = 0.1

relaxedL1_nlv_old(A,σ2)
relaxedL1_nlv(A,σ2)
@btime relaxedL1_nlv_old(A,σ2) #1.024 ms (1 allocation: 16 bytes)
@btime relaxedL1_nlv(A,σ2) # 1.024 ms (1 allocation: 16 bytes)

#========= sca2 =================#
function sca2_no_order(x; allcomp=false)
    allcomp && return sum(abs2.(x))
    objval = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        objval += abs(min(0, x[i]))^2
    end
    return objval
end

function sca2_order(x, order=2; allcomp=false)
    allcomp && return sum(abs2.(x))
    objval = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        objval += abs(min(0, x[i]))^order
    end
    return objval
end

function sca2_no_allcomp(x)
    objval = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        objval += abs(min(0, x[i]))^2
    end
    return objval
end

function sca2_new(x)
    objval = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        objval += x[i] > 0 ? 0 : x[i]^2
    end
    return objval
end

function sca2_new2(x)
    objval = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        x[i] > 0 ? nothing : objval += x[i]^2
    end
    return objval
end

function sca2_new3(x)
    objval = zero(eltype(x))
    @inbounds @simd for a in x
        objval += a > 0 ? 0 : a^2
    end
    return objval
end

A = rand(800,1000) .- 0.5

@btime sca2_order($A) # 4.747 ms (0 allocations: 0 bytes)
@btime sca2_no_order($A) # 115.800 μs (0 allocations: 0 bytes)
@btime sca2_no_allcomp($A) # 115.500 μs (0 allocations: 0 bytes)
@btime sca2_new($A) # 75.000 μs (0 allocations: 0 bytes) ----> best
@btime sca2_new2($A) # 384.000 μs (0 allocations: 0 bytes)
@btime sca2_new3($A) # 77.000 μs (0 allocations: 0 bytes)

#========= Compare before and after =================#
subworkpath = joinpath(workpath,"speedup")
subworkpath_b4 = joinpath(workpath,"paper","ncells")
subworkpath_after = joinpath(workpath,"paper","ncells_fast")

z = 0.5
ddpcbb4=load(joinpath(subworkpath_b4,"pcb","pcb0db1f10s_runtime_vs_avgfits.jld2"))
pcbb4rng = ddpcbb4["rng"]
pcbb4_sp_nn_means = ddpcbb4["stat_sp_nn"][1]
pcbb4_sp_nn_stds = ddpcbb4["stat_sp_nn"][2]
pcbb4_sp_nn_upper = pcbb4_sp_nn_means+z*pcbb4_sp_nn_stds
pcbb4_sp_nn_lower = pcbb4_sp_nn_means-z*pcbb4_sp_nn_stds

ddpcbaft=load(joinpath(subworkpath_after,"pcb","pcb0db1f10s_runtime_vs_avgfits.jld2"))
pcbaftrng = ddpcbaft["rng"]
pcbaft_sp_nn_means = ddpcbaft["stat_sp_nn"][1]
pcbaft_sp_nn_stds = ddpcbaft["stat_sp_nn"][2]
pcbaft_sp_nn_upper = pcbaft_sp_nn_means+z*pcbaft_sp_nn_stds
pcbaft_sp_nn_lower = pcbaft_sp_nn_means-z*pcbaft_sp_nn_stds

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon()
# compare PCB with other methods
fig = Figure(size=(400,300))
ax = AMakie.Axis(fig[1, 1], limits = ((0,0.2), (0.5,1.0)), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
ln = lines!(ax, pcbb4rng[plotrng], pcbb4_sp_nn_means[plotrng], color=mtdcolors[2], label="before", linestyle=:dash)
bnd = band!(ax, pcbb4rng[plotrng], pcbb4_sp_nn_lower[plotrng], pcbb4_sp_nn_lower[plotrng], color=mtdcoloras[2])
ln = lines!(ax, pcbaftrng[plotrng], pcbaft_sp_nn_means[plotrng], color=mtdcolors[4], label="after", linestyle=nothing)
bnd = band!(ax, pcbaftrng[plotrng], pcbaft_sp_nn_lower[plotrng], pcbaft_sp_nn_lower[plotrng], color=mtdcoloras[4])
axislegend(ax, labelsize=10, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits0db1f10s_speedup.png"),fig,px_per_unit=2)

