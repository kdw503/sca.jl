using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"StochasticLM")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using StochasticLM
using LinearAlgebra

#========= Natural dataset (sparse coding) ==========#
include(joinpath(subworkpath,"pcb_slm.jl"))

dataset = :natural
imgsz = (12,12); lengthT = 100000; noc = ncs = 72; nac = 0
patch_size = imgsz[1]

# dd = load(joinpath(subworkpath, "X_whitened_Hspar","natural_SC_l3.0_iter50.jld2"))
# sD = dd["D"]; αs = dd["αs"]; X_whitened = dd["X_whitened"]

prefix = "pcb"
dataset = :natural; p = 72; nac = 0; k = p+nac; imgsz=(12,12); lengthT = 100000; (m,n) = (*(imgsz...), lengthT)

dd = load(joinpath(subworkpath,"allinit.jld2"))
X_whitened = dd["X_whitened"][1]
U, Vt, D = dd["SVD"]; V = Vt'
(m,n,p) = (size(X_whitened)...,ncs); imgsz = (12,12)
gtW, gtH = (Matrix{eltype(X_whitened)}(undef,0,0),Matrix{eltype(X_whitened)}(undef,0,0))
optim_method = :stochasticLM
β = 0

for initmethod in [:isvd,:sbc,:BPDN]
    @show initmethod
    Winit, Hinit, M0, N0t, _ = dd[String(initmethod)]
    for αpow in [-4,-1]
        @show αpow
        maxiter = 1000
        tol=1e-6; inner_tol = 1e-6; inner_maxiter = 100
        M0, N0 = copy(M0), copy(N0t')
θ0 = vcat(vec(M0), vec(N0))

bd = 2.0 # batchsize divide rate
βw=βh=β; αw=0; αh=10.0^(αpow); σw = σh = σ = 1e-16
W0 = U*M0; H0 = N0*Vt
rtαh = sqrt(αh); Gh = dg(H0, σh)*rtαh

myminibatcher = makescminibather(k,p,n)
rjop = SLMCache(makescop(M0, N0, Vt, Gh); Hd = ones(Float64,2k*p), minibatcher=myminibatcher) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
nmf_rjop! = make_sc_fcache_op!(M0, N0, Vt, D, Gh, rtαh, σh)

rt2 = @elapsed θopt, objval = stochasticlm(nmf_rjop!, θ0, k^2+Int((n*p)÷bd), rjop; itermax=maxiter, solver_kwargs=(; itmax=inner_maxiter), verbose=false) # length(rjop.r) -> full batch
@show objval
M = reshape(θopt[1:k*p], k, p)
N = reshape(θopt[k*p+1:end], p, k)
sum(abs2, M0 * N0 - D) / 2
sum(abs2, M * N - D) / 2
W1 = U*M; H1 = N*Vt

        L1h = norm(H1,1)
        fv = LCSVD.fitd(X_whitened,W1*H1)
        LCSVD.normalizeWH!(W1,H1); norm1nH = norm(H1,1)
        fprex = "$(prefix)_$(dataset)_$(initmethod)_$(optim_method)"
        #fprex = "$(prefix)_BPDN"
        regstr = "_aw$(αw)_ah$(αh)_b$(β)"
        fname = joinpath(subworkpath,"$(fprex)$(regstr)_objval$(objval)_f$(fv)_rt$(rt2)")
        imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
        imsave_data(dataset,fname*"_c",W1,H1,imgsz,200; saveH=true, signedcolors=TestData.g1wm())
        # Xest = W1*H1; mse = norm(X_whitened[:,1:72]-Xest[:,1:72])^2/length(X_whitened[:,1:72])
        # imsave_data(dataset,joinpath(subworkpath,"$(fprex)$(regstr)_mse$(mse)_Xest1to72.png"),Xest[:,1:72],Xest[1:72,:],imgsz,lengthT; saveH=false)
    end
end



#========= fakecells dataset ==========#
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

noc = ncs; nac = 0; nc = noc+nac; initmethod = :isvd
rt1 = @elapsed U, Vt, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(Vt'); N0t = copy(N0')
W0 = copy(U); H0 = copy(Vt)

#=========================== PCB ========================================#
# using random weights on matching D? We could write a custom minibatcher to randomize the weights on each iteration.
include(joinpath(subworkpath,"pcb_slm.jl"))

# prepare data
k = p = 15
U, s, V = tsvd(X, k); Vt = V'; D = Diagonal(s); Dsq = Diagonal(sqrt.(s))
W0, H0 = NMF.nndsvd(X, p); x = vec(X)
M0 = W0\U; N0 = Vt/H0

# PCB with StochasticLM
θ0 = vcat(vec(M0), vec(N0))
Mcache, Ncache = copy(M0), copy(N0)

α = 0.01; β = 0.001
for α in [#=0, 5e-7,=# 1e-2, 1e-6, 0.1#=, 0.5=#] , β in [#=0, 5e-5,=# 1e-3, 0.05, 1e-4, 5e-4 #=, 1.0=#]
#for α in [0, 5e-7, 1e-6, 1e-2, 0.1, 0.5] , β in [0, 5e-5, 1e-4, 5e-4, 1e-3, 0.05, 1.0]
    try
        #α*=100; β*=100
    @show α, β
bd = 2.0 # batchsize divide rate
βw=βh=β; αw=αh=α; σw = σh = σ = 1e-16
W0 = U*M0; H0 = N0*Vt
rtβw = sqrt(βw); rtαw = sqrt(αw); Sw = (W0.<0)*rtβw; Gw = dg(W0, σw)*rtαw
rtβh = sqrt(βh); rtαh = sqrt(αh); Sh = (H0.<0)*rtβh; Gh = dg(H0, σh)*rtαh

myminibatcher = makeminibather(k,p,m,n)
rjop = SLMCache(makeop(Mcache, Ncache, U, Vt, Sw, Sh, Gw, Gh); Hd = ones(Float64,2k*p), minibatcher=myminibatcher) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
nmf_rjop! = make_fcache_op!(Mcache, Ncache, U, Vt, D, Sw, Sh, Gw, Gh, rtβw, rtβh, rtαw, rtαh, σw, σh)
nbatch = sum(calculate_whbatchsize(Int((2(m+n)*p)÷bd), m/(m+n), p).*p)*2 + k^2

rt = @elapsed θopt, objval = stochasticlm(nmf_rjop!, θ0, nbatch, rjop; itermax=1000, solver_kwargs=(; itmax=100), verbose=true) # length(rjop.r) -> full batch
@show objval
M = reshape(θopt[1:k*p], k, p)
N = reshape(θopt[k*p+1:end], p, k)
sum(abs2, M0 * N0 - D) / 2
sum(abs2, M * N - D) / 2
W = U*M; H = N*Vt
sum(abs2, W0*H0 - X) / 2
sum(abs2, W * H - X) / 2

LCSVD.normalizeWH!(W,H)
fprex = "PCB_slm_batchdivide$(bd)_ysnormal0.1weight_sigma$(σ)_a$(α)_b$(β)_f$(objval)_rt$(rt)"
#fprex = "$(prefix)_BPDN"
fname = joinpath(subworkpath,"$(fprex)")
imsave_data(dataset,fname,W,H,imgsz,lengthT; saveH=false)
    catch e
        @warn e
        continue
    end
end

# minibatcher test
βw=βh=0.5; αw=αh=0.05; σw = σh = σ = 1e-16
W0 = U*M0; H0 = N0*Vt
rtβw = sqrt(βw); rtαw = sqrt(αw); Sw = (W0.<0)*rtβw; Gw = dg(W0, σw)*rtαw
rtβh = sqrt(βh); rtαh = sqrt(αh); Sh = (H0.<0)*rtβh; Gh = dg(H0, σh)*rtαh
myminibatcher = makeminibather(k,p,m,n)
rjop = SLMCache(makeop(Mcache, Ncache, U, Vt, Sw, Sh, Gw, Gh); Hd = ones(Float64,2k*p))
mb = myminibatcher(rjop, k^2+4p) # choose only one column and row for each nonnegativity and sparsity
@test mbnw = mb[k^2+1:k^2+p]; mbnh = mb[k^2+p+1:k^2+2p]; mbsw = mb[k^2+2p+1:k^2+3p]; mbsh = mb[k^2+3p+1:k^2+4p]
@test mbnw[2:end] == mbnw[1:end-1] .+ m; mbnh[2:end] == mbnh[1:end-1] .+ 1
@test mbsw[2:end] == mbsw[1:end-1] .+ m; mbsh[2:end] == mbsh[1:end-1] .+ 1
@test mbsw == mbnw .+ (m+n)*p; mbsh == mbnh .+ (m+n)*p

# Jacobian Check: <Jv, w> == <v, Jᵀ w>
k, p = size(M0); m, n = size(X)
J = makeop(M0,N0,U,Vt,Sw,Sh,Gw,Gh)
f(x) = (M = reshape(x[1:k*p], k, p); N = reshape(x[k*p+1:end], p, k);
        Ys = M*N-D; W = U*M; H = N*Vt;
        Sw = (W.<0)*rtβw; Sh = (H.<0)*rtβh; Ynw = Sw.*W; Ynh = Sh.*H;
        Ysw = g(W, σw)*rtαw; Ysh = g(H, σh)*rtαh;
        [vec(Ys); vec(Ynw); vec(Ynh); vec(Ysw); vec(Ysh)])
x0 = [vec(M0);vec(N0)]
Jfd = ForwardDiff.jacobian(f,x0)
Jfdv = Jfd*x0
Jv = J*x0
norm(Jfdv-Jv) # 3.5743888218162846e-16

# Transpose Consistency Check: <Jv, w> == <v, Jᵀ w>
v = randn(2k*p)
w = randn(k^2+2*(m+n)*p)

lhs = dot(J*v, w)
rhs = dot(v, J' * w)
abs(lhs-rhs) # 7.105427357601002e-15
issymmetric(J) # false


#======== Sparse Coding with scale invariant term =========#
# minibatcher test
αh=0.05; σw = σh = σ = 1e-16
H0 = N0*Vt
rtαh = sqrt(αh); Ph = g(H0, σh)*rtαh; Gh = dg(H0, σh)*rtαh
nM = Ref(norm(M0))

# Jacobian Check: <Jv, w> == <v, Jᵀ w>
k, p = size(M0); m, n = size(X)

J = makesc2op(M0,N0,Vt,Gh,Ph,nM)
f(x) = (M = reshape(x[1:k*p], k, p); N = reshape(x[k*p+1:end], p, k);
        Ys = M*N-D; H = N*Vt; Ysh = g(H, σh)*rtαh*sqrt(norm(M));
        [vec(Ys); vec(Ysh)])
x0 = [vec(M0);vec(N0)]
Jfd = ForwardDiff.jacobian(f,x0)
Jfdv = Jfd*x0
Jv = J*x0
norm(Jfdv-Jv) # 1.1964907807068579e-14

# Transpose Consistency Check: <Jv, w> == <v, Jᵀ w>
v = randn(2k*p)
w = randn(k^2+n*p)

lhs = dot(J*v, w)
rhs = dot(v, J' * w)
abs(lhs-rhs) # 8.526512829121202e-14
issymmetric(J) # false




rjop = SLMCache(makeop(Mcache, Ncache, U, Vt, Sw, Sh); Hd = Vector{Float64}(undef, 2k*p), minibatcher=myminibatcher!) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
nmf_rjop! = make_fcache_op!(Mcache, Ncache, U, Vt, D, Sw, Sh, rtβw, rtβh)
θ = [vec(Mcache); vec(Ncache)]

# stochasticlm(...)
f! = nmf_rjop!                               # f!(cache, θ, idx) -> objval; must also accept `nothing` as cache
θ0 = θ                               # initial guess
batchsize = length(rjop.r)                   # number of samples per minibatch
cache = rjop                        # holds residual, Jacobian, and other useful precomputated values
(lb, ub) = (nothing, nothing)    # lower and upper bounds on θ
solver! = minres_solver!         # solver!(d, cache, λ, Hd)
solver_kwargs = (; itmax=20)
λmin=sqrt(eps(float(eltype(θ0))))  # minimum regularization coefficient (multiplies the diagonal of the Hessian)
λ0=max(λmin, sqrt(λmin))         # initial regularization coefficient
η0=:auto                         # initial barrier coefficient for bound constraints
ηmin=eps(float(eltype(θ0)))      # minimum barrier coefficient
Hdminratio=0                     # minimum diagonal value of the Hessian
γ=4                              # regularization coefficient multiplier/divider for failure/success
p0=0.0001                        # minimum fraction of expected improvement needed for step-acceptance
p1=0.25                          # if fractional expected improvement <p1, increase λ
p2=0.75                          # if fractional expected improvement >p2, decrease λ
itermax=1000
convergence=ConvergenceParams()
fvalpairs=nothing
iswarn=true
verbose=true

SLM = StochasticLM

   Base.@constprop :none
    SLM.checkp(p0, p1, p2)
    bd = (lb === nothing && ub === nothing) ? nothing : BoundsData(lb, ub)
    SLM.validate_start(θ0, bd)
    batchsize <= length(SLM.residual(cache)) || throw(ArgumentError("batchsize ($batchsize) must be less than or equal to the number of residuals ($(length(SLM.residual(cache))))"))
    T = eltype(cache.r)
    θ = copyto!(similar(θ0, T), θ0)
    d = similar(θ)
    θtmp = similar(θ)
    Hd = similar(θ)   # hessian diagonal
    sd = SLM.SolverData{T}(batchsize, θ)

 # Select the initial minibatch
    idx = SLM.minibatcher(cache, batchsize)
    # Initialize the cache
    objval = SLM.ObjBarVals(SLM.checked_update!(f!, cache, θ, idx))   # cache now includes all objective terms, but not any elements from the constraints
    SLM.hessprep!(Hd, cache, idx; Hdminratio)       # prepare the Hessian diagonal (the Levenberg-Marquardt damping); should occur before incorporating the barrier
    if bd !== nothing
        if η0 === :auto
            η0 = SLM.average_complementarity(SLM.total_neggradient(cache, idx), θ0, bd)
        end
        verbose && @info("Initial barrier coefficient: η0 = $η0")
        # Incorporate the barrier contribution into the objective value and set/update the appropriate elements in the cache
        objval = SLM.ObjBarVals(objval, SLM.barrier!(SLM.invalidate!(bd), θ, η0))
    else
        η0 = 0
    end
    verbose && @info("Initial objective value: $objval")
    if fvalpairs === nothing
        fvalpairs = Tuple{typeof(objval), typeof(objval)}[]
    else
        empty!(fvalpairs)
    end
    iter, λ, η = 0, λ0, η0
    nreset = 100
    ηnew = typemax(η)
       npairs = length(fvalpairs)    # keep track of when we started this block of η
            rtol = SLM.default_rtol(fvalpairs, nreset, npairs)
            iter += 1
            # ret = SLM.step_slm!(f!, θ, (bd, η), sd, cache, λ, Hd, idx, objval, rtol; solver!, solver_kwargs, λmin, γ, p0, p1, p2, iswarn, d, θtmp)
                d=similar(θ)
                θtmp=similar(θ)
                predcache=Tuple{eltype(objval), eltype(objval)}[]
                forceupdate=false
                empty!(predcache)

    λmax = 1/eps(T)
    canshrink = true
    λref = Ref(λ)
    sd, J, A, M = SLM.build!(sd, Val(SLM.solvermode(solver!)), cache, bd, λref, Hd, idx)
    if forceupdate   # for testing purposes
        sd, A, M = SLM.update_λ!(sd, A, M, cache, bd, λref, Hd)
    end
        if λref[] != λ
            λref[] = λ
            sd, A, M = SLM.update_λ!(sd, A, M, cache, bd, λref, Hd)
        end
        d0, A0, M0 = deepcopy(d), deepcopy(A), deepcopy(M)
        sd0 = deepcopy(sd)
        cache0, Hd0 = deepcopy(cache), copy(Hd)
        solver!(d0, sd0, A0, M0, rtol; solver_kwargs...)
        SLM.scalemin!(d0, sd0, J, cache0, bd, λref, Hd0)
        α = SLM.trim_valid!(SLM.result(d), θ, bd)


#=========================== PCB (using soft thresholding) ========================================#
# using random weights on matching D? We could write a custom minibatcher to randomize the weights on each iteration.
# vec(AXB)=(Bᵀ⊗A)vec(X) # ⊗ : Kronecker product (A⊗B = [a11*B a12*B; a21*B a22*B]) = kron(A,B)
# Y(W,H) = WH; dY=dWH+WdH
# Y(X)=A∘(BXC)∈Rᵐˣⁿ; dY=A∘(BdXC) -> A.*(B*dX*C) # ∘ : Hadamard product
# vec(dY) = diag(vec(A))(C'⊗B)vec(dX)
using SparseArrays, ForwardDiff, LinearOperators, Test, Random

function makeop(M::AbstractMatrix{T}, N::AbstractMatrix{T}) where T
    m′, k′ = size(M)
    k″, n′ = size(N)
    @assert k′ == k″
    return LinearOperator{T}(m′ * n′, (m′ + n′) * k′, false, false,
            function(res, v, α, β)    # Jacobian-vector product
                dM = reshape(v[1:m′*k′], m′, k′)
                dN = reshape(v[m′*k′+1:end], k′, n′)
                resmtrx = reshape(res, m′, n′)
                #resmtrx = reshape(view(res,m′*n′), m′, n′)
                mul!(resmtrx, dM, N, α, β)
                mul!(resmtrx, M, dN, α, true)
                return res
            end,
            function(res, v, α, β)  # Jacobian-transpose-vector product
                dD = reshape(v, m′, n′)
                dM = reshape(@view(res[1:m′*k′]), m′, k′)
                dN = reshape(@view(res[m′*k′+1:end]), k′, n′)
                mul!(dM, dD, N', α, β)
                mul!(dN, M', dD, α, β)
                return res
            end,
            nothing)
end
function make_fcache_op!(M::AbstractMatrix{T}, N::AbstractMatrix{T}, D::AbstractMatrix{T}) where T
    x = vec(D)
    Dpred = M * N
    Mscratch, Nscratch, Dpredscratch = copy(M), copy(N), copy(Dpred)
    return function(rj, θ, idx)
        if rj === nothing
            # Don't modify M, N, Dpred
            copyto!(Mscratch, view(θ, 1:m*k))
            copyto!(Nscratch, @view(θ[m*k+1:end]))
            mul!(Dpredscratch, Mscratch, Nscratch)
            return sum((D[i] - Dpredscratch[i])^2 for i in idx) / 2
        end
        copyto!(M, view(θ, 1:m*k))
        copyto!(N, @view(θ[m*k+1:end]))
        mul!(Dpred, M, N)
        @inbounds r = x[idx] - vec(Dpred)[idx]
        rj.r[idx] = r
        # Cache Hd
        for j = 1:k
            s = sum(abs2, @view(N[j, :]))
            for i = 1:m
                rj.Hd[i + (j - 1) * m] = s
            end
        end
        offset = m * k
        for i = 1:k
            s = sum(abs2, @view(M[:, i]))
            for j = 1:n
                rj.Hd[offset + i + (j - 1) * k] = s
            end
        end
        return dot(r, r) / 2
    end
end


function myminibatcher!(cache, _)
    rand!(cache.Gdiag)
    copyto!(cache.Ddiag, cache.Gdiag)   # must be matched to Gdiag
    return Base.OneTo(length(cache.Gdiag))
end

k = 15
U, s, V = tsvd(X, k); Vt = V'; D = Diagonal(s); Dsq = Diagonal(sqrt.(s))
W0, H0 = NMF.nndsvd(X, k); x = vec(X)
M0 = W0\U; N0 = Vt/H0
θ0 = vcat(vec(M0), vec(N0))
Mcache, Ncache = copy(M0), copy(N0)
m, k = size(M0); n = size(N0,2)
rjop = SLMCache(makeop(Mcache, Ncache); Hd = Vector{Float64}(undef, (m + n) * k), minibatcher=myminibatcher!) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
nmf_rjop! = make_fcache_op!(Mcache, Ncache, D)
θ = copy(θ0)
local objval
for i = 1:20
    θopt, objval = stochasticlm(nmf_rjop!, θ, length(D), rjop; itermax=100, solver_kwargs=(; itmax=20), verbose=false)
    M = reshape(view(θopt,1:m*k), m, k)
    N = reshape(@view(θopt[m*k+1:end]), k, n)
    W = U*M; H = N*Vt
    LCSVD.flip2makepos!(W,H)
    W[W.<0] .= 0.; H[H.<0] .= 0 # soft thresholding
    M .= U\W; N .= H/Vt
    θ .= θopt
    @show i, objval, sum(abs2, M * N - D) / 2
end
@test objval < 1e-8 # 358.5590176266584(250 iteration),
M = reshape(θ[1:m*k], m, k)
N = reshape(θ[m*k+1:end], k, n)
sum(abs2, M0 * N0 - D) / 2 # 33879.19093610554
@test sum(abs2, M * N - D) / 2 < 1e-8 # 177.584815557117
W = U*M; H = N*Vt
sum(abs2, W0*H0 - X) / 2 # 15145.3731323332
sum(abs2, W * H - X) / 2 # 1.7270287550765262e9

LCSVD.normalizeWH!(W,H)
fprex = "PCB_slm_$(dataset)"
#fprex = "$(prefix)_BPDN"
fname = joinpath(subworkpath,"$(fprex)")
imsave_data(dataset,fname,W,H,imgsz,lengthT; saveH=false)

#=================== using LinearOperator ===================================#
using SparseArrays, ForwardDiff, LinearOperators, Test, Random

k = size(W0, 2)

function nmf_rj!(rj, θ, idx)
    W = reshape(θ[1:m*k], m, k)
    H = reshape(θ[m*k+1:end], k, n)
    Xpred = W * H
    r = x[idx] - vec(Xpred)[idx]
    if rj !== nothing
        # Poison the cache to ensure that only the entries in `idx` are used (testing only!)
        fill!(rj.r, NaN)
        issparse(rj.J) && fill!(rj.J.nzval, NaN)
        rj.r[idx] .= r
        for l in idx
            j0, i0 = divrem(l - 1, m)
            for kk in 1:k
                rj.J[l, i0 + 1 + (kk - 1) * m] = H[kk, j0 + 1]
                rj.J[l, m * k + j0 * k + kk] = W[i0 + 1, kk]
            end
        end
    end
    return dot(r, r) / 2
end
# Set up the sparsity pattern for the Jacobian
JI, JJ, JV = Int[], Int[], Float64[]
for l in 1:(m * n)
    j0, i0 = divrem(l - 1, m)
    for kk in 1:k
        push!(JI, l); push!(JJ, i0 + 1 + (kk - 1) * m); push!(JV, 0.0)
        push!(JI, l); push!(JJ, m * k + j0 * k + kk); push!(JV, 0.0)
    end
end
J = sparse(JI, JJ, JV, m * n, (m + n) * k) # S[JI[k], JJ[k]] = JV[k]
rj = SLMCache(J)

function makeopWH(W::AbstractMatrix{T}, H::AbstractMatrix{T}) where T
    m′, k′ = size(W)
    k″, n′ = size(H)
    @assert k′ == k″
    return LinearOperator{T}(m′ * n′, (m′ + n′) * k′, false, false,
            function(res, v, α, β)    # Jacobian-vector product
                dW = reshape(v[1:m′*k′], m′, k′)
                dH = reshape(v[m′*k′+1:end], k′, n′)
                resmtrx = reshape(res, m′, n′)
                mul!(resmtrx, dW, H, α, β)
                mul!(resmtrx, W, dH, α, true)
                return res
            end,
            function(res, v, α, β)  # Jacobian-transpose-vector product
                dX = reshape(v, m′, n′)
                dW = reshape(@view(res[1:m′*k′]), m′, k′)
                dH = reshape(@view(res[m′*k′+1:end]), k′, n′)
                mul!(dW, dX, H', α, β)
                mul!(dH, W', dX, α, β)
                return res
            end,
            nothing)
end
function make_fcache_op!(W::AbstractMatrix{T}, H::AbstractMatrix{T}) where T
    Xpred = W * H
    Wscratch, Hscratch, Xpredscratch = copy(W), copy(H), copy(Xpred)
    return function(rj, θ, idx)
        if rj === nothing
            # Don't modify W, H, Xpred
            copyto!(Wscratch, view(θ, 1:m*k))
            copyto!(Hscratch, @view(θ[m*k+1:end]))
            mul!(Xpredscratch, Wscratch, Hscratch)
            return sum((x[i] - Xpredscratch[i])^2 for i in idx) / 2
        end
        copyto!(W, view(θ, 1:m*k))
        copyto!(H, @view(θ[m*k+1:end]))
        mul!(Xpred, W, H)
        @inbounds r = x[idx] - vec(Xpred)[idx]
        rj.r[idx] = r
        # Cache Hd
        for j = 1:k
            s = sum(abs2, @view(H[j, :]))
            for i = 1:m
                rj.Hd[i + (j - 1) * m] = s
            end
        end
        offset = m * k
        for i = 1:k
            s = sum(abs2, @view(W[:, i]))
            for j = 1:n
                rj.Hd[offset + i + (j - 1) * k] = s
            end
        end
        return dot(r, r) / 2
    end
end
# Create the common backing store and share it between the operator and residual function
W0, H0 = NMF.nndsvd(X, k); x = vec(X)
θ0 = vcat(vec(W0), vec(H0))
Wcache, Hcache = copy(W0), copy(H0)
m, k = size(W0); n = size(H0,2)
rjop = SLMCache(makeopWH(Wcache, Hcache); Hd = Vector{Float64}(undef, (m + n) * k)) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
nmf_rjop! = make_fcache_op!(Wcache, Hcache)
# Check that the operator- and matrix-Jacobian give the same results
rp = randperm(m*n)
nmf_rjop!(StochasticLM.invalidate!(rjop), θ0 .+ 0.001, rp)
nmf_rj!(StochasticLM.invalidate!(rj), θ0 .+ 0.001, rp)
@test rj.r ≈ rjop.r
@test StochasticLM.neggradient(rjop, rp) ≈ StochasticLM.neggradient(rj, rp)
Hd = zeros((m + n) * k)
StochasticLM.hessdiag!(Hd, rj, rp)
@test Hd ≈ StochasticLM.hessdiag!(similar(Hd), rjop, rp)
bd = StochasticLM.BoundsData(zeros(length(θ0)), fill(Inf, length(θ0)))
λ, η = 1/5, 1e-3
StochasticLM.barrier!(StochasticLM.invalidate!(bd), θ0 .+ 0.001, η)
Am, _, bm = StochasticLM.build_system(true, rj, bd, λ, Hd, rp)
Aop, _, bop = StochasticLM.build_system(false, rjop, bd, λ, Hd, rp)
v = randn(size(Am, 1))
y1 = Am * v
y2 = Aop * v
@test y1 ≈ y2
@test bm ≈ bop

# Test that we can optimize using the operator framework
W0, H0 = NMF.nndsvd(X, k); x = vec(X)
θ0 = vcat(vec(W0), vec(H0))
Wcache, Hcache = copy(W0), copy(H0)
m, k = size(W0); n = size(H0,2)
rjop = SLMCache(makeop(Wcache, Hcache); Hd = Vector{Float64}(undef, (m + n) * k)) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
nmf_rjop! = make_fcache_op!(Wcache, Hcache)
θopt, objval = stochasticlm(nmf_rjop!, θ0 .+ 0.001, length(W0) + length(H0), rjop, (zeros(length(θ0)), fill(Inf, length(θ0))); ηmin=1e-12, itermax=10)
@test objval < 1e-8 # 358.5590176266584(250 iteration),
W = reshape(θopt[1:m*k], m, k)
H = reshape(θopt[m*k+1:end], k, n)
sum(abs2, W0 * H0 - X) / 2 # 43064.868023189076
@test sum(abs2, W * H - X) / 2 < 1e-8 # 42488.81795944464

LCSVD.normalizeWH!(W,H)
fprex = "slm_$(dataset)"
#fprex = "$(prefix)_BPDN"
fname = joinpath(subworkpath,"$(fprex)")
imsave_data(dataset,fname,W,H,imgsz,lengthT; saveH=false)

θopt, objval = stochasticlm(nmf_rjop!, θ0 .+ 0.001, length(W0) + length(H0), rjop, (zeros(length(θ0)), fill(Inf, length(θ0))); ηmin=1e-12, itermax=50,verbose=true)
θopt, objval = stochasticlm(nmf_rjop!, θ0 .+ 0.001, fill(length(W0) + length(H0), 100), rjop, (zeros(length(θ0)), fill(Inf, length(θ0))); ηmin=1e-7, verbose=true)
θopt, objval = stochasticlm(nmf_rjop!, θ0 .+ 0.001, length(W0) + length(H0), rjop, (zeros(length(θ0)), fill(Inf, length(θ0))); itermax=1000, solver_kwargs=(; itmax=20), ηmin=1e-7, verbose=true)
W = reshape(θopt[1:m*k], m, k)
H = reshape(θopt[m*k+1:end], k, n)
sum(abs2, W * H - X) / 2 # 9742.417501566157
LCSVD.normalizeWH!(W,H)
fprex = "slm_1000_$(dataset)"
#fprex = "$(prefix)_BPDN"
fname = joinpath(subworkpath,"$(fprex)")
imsave_data(dataset,fname,W,H,imgsz,lengthT; saveH=false)

# compare with HALS final objective value
W, H = NMF.nndsvd(X, k); x = vec(X)
result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=50, α=0., l₁ratio=1, tol=1e-6, verbose=false), X, W, H)
sum(abs2, W * H - X) / 2 # 9697.595281487327
fprex = "hals_$(dataset)"
#fprex = "$(prefix)_BPDN"
fname = joinpath(subworkpath,"$(fprex)")
imsave_data(dataset,fname,W,H,imgsz,lengthT; saveH=false)

#================ readme example ========================================#
a, τ = 1.2, 12.0               # model parameters (ground truth)
t = 0:50                       # independent parameters `ξ[i]` that affect each predicted value
y = a .* exp.(-t ./ τ)         # dependent values (we try to predict these)

# Initialize the cache for intermediate values. `RJCache` is the default type, and it stores
# the residuals and Jacobian.
J = zeros(length(y), 2)  # allocate space to store the Jacobian
rj = SLMCache(J)

# Create the function `fbatch!` that we'll use in optimization.
# `fbatch!` should take three arguments, `fbatch!(rj, θ, idx)`, where
#   - `θ` is the current value of the parameter vector (here `[a, τ]`)
#   - `idx` lists the indices of observations included in the minibatch (i.e., at times `t[idx]`)
# `fbatch!` should always return the value of the objective function evaluated with parameters `θ`
# on minibatch `idx`.
# For the first argument, you must support two syntaxes:
#    fbatch!(nothing, θ, idx)        # just compute the objective value on the minibatch (no caching performed)
#    fbatch!(rj, θ, ix)              # populate the cache `rj` for later derivative computation
#
# The first syntax will be called when new settings of `θ` are being tested for whether
# they improve the objective value; the second will be called when picking a descent direction
# for the next step.
#
# For performance reasons, enclose any global variables (here, `t` and `y`) by passing them as
# arguments to a function that creates `fbatch!`.
# Here we write out the Jacobian by hand; below, we'll see how you can use Automatic Differentiation for these computations.
function create_fbatch!(t, y)          # pass in the evaluation times and the values we are trying to match (`y[i] ≈ f(t[i]; θ...)`)
    return function(rj, θ, idx)        # this is `fbatch!`
        tidx, yidx = t[idx], y[idx]    # we only need to compute residual and Jacobian for the ones in `idx`
        a′, τ′ = θ                     # unpack the parameter vector
        expvals = exp.(-tidx ./ τ′)
        ypred = a′ * expvals
        r = yidx - ypred               # compute the residual at times `tidx`
        if rj !== nothing              # Important! The `rj` input might be `nothing` (if there is no need to update the cache)
        @show rj.r[0 .+ idx]
            rj.r[idx] .= r             # store the residuals we computed in the appropriate slots
            rj.J[idx,1] .= expvals     # likewise for the two columns of the Jacobian
            rj.J[idx,2] .= tidx ./ τ′^2 .* ypred
        end
        return dot(r, r) / 2           # return the objective value
    end
end

fbatch! = create_fbatch!(t, y)    # fbatch!(rj, θ, idx)

# Here we use a minibatch size equal to the whole dataset, which makes the algorithm non-stochastic.
# (This is just to show that we can extract the correct values of `a` and `τ`.)
batchsize = length(t)
θ0 = [0.9, 10.0]    # not the true minimum
θ, objval = stochasticlm(fbatch!, θ0, batchsize, rj)

#============ StochasticLM ===============#
prefix = "slm"
# Initialize the cache for intermediate values. `RJCache` is the default type, and it stores
# the residuals and Jacobian.
Jw = zeros(nc^2+m*noc, nc*noc)  # allocate space to store the Jacobian for W components
rjw = SLMCache(Jw)
Jh = zeros(nc^2+n*noc, nc*noc)  # allocate space to store the Jacobian for W components
rjh = SLMCache(Jh)

using SparseArrays

function create_f!(U,N,D,β)
    return function(rj, x, idx)        # this is `f!`
        m = size(U,1)
        l = m*noc
        idx[idx.>l] .-= nc^2
        M = reshape(x,nc,noc)
        Rs = M*N-D; rs = vec(Rs)
        bdU = blockdiag(ntuple(_ -> sparse(U), noc)...)
        w = vec(U*M); rn = β*min.(w,0); rnidx=rn[idx]; sgn = -sign.(rn)
        dsgn = spdiagm(0 => sgn)
        r = vcat(rs,rnidx)
        if rj !== nothing              # Important! The `rj` input might be `nothing` (if the cache
            #rj.r .= 0
            rj.r[1:nc^2] .= rs             #  does not yet need updating)
            rj.r[nc^2 .+ idx] .= rnidx # bdU[idx,:]*x
            #rj.J .= 0.
            for i=1:nc
                rj.J[i:nc:nc^2,i:noc:end] .= N'
            end
            rj.J[nc^2 .+ idx, :] .= β*dsgn[idx,idx]*bdU[idx,:]
        end
        return dot(r, r) / 2
    end
end

β = 5.0; bsr = 0.1
batchsize_w = Int(floor(m*noc*bsr)); batchsize_h = Int(floor(n*noc*bsr))
# x0 = vec(M0);
# f! = create_f!(U,N0,D,β)    # f!(rj, x, idx)
# x, objval = stochasticlm(f!, x0, batchsize_w, rjw) # test
maxiter = 50
Jw = zeros(nc^2+m*noc, nc*noc)  # allocate space to store the Jacobian for W components
rjw = SLMCache(Jw)
Jh = zeros(nc^2+n*noc, nc*noc)  # allocate space to store the Jacobian for W components
rjh = SLMCache(Jh)
for β = [10.0, 100.0, 50.0, 500.0, 5000.0]
    @show β
    i = 0
    M = copy(M0); N = copy(N0)
    rt1 = @elapsed while (i<maxiter)
        i += 1
        f! = create_f!(U,N,D,β)
        xw = vec(M)
        xw, objvalw = stochasticlm(f!, xw, batchsize_w, rjw)
        M .= reshape(xw,nc,noc)
        f! = create_f!(V,M',D',β)
        xht = vec(N')
        xht, objvalh = stochasticlm(f!, xht, batchsize_h, rjh)
        N .= reshape(xht,nc,noc)'
        @show i, objvalw, objvalh
    end
    W1, H1 = U*M, N*Vt
    L1h = norm(H1,1)
    fv = LCSVD.fitd(X,W1*H1)
    LCSVD.normalizeWH!(W1,H1)
    fprex = "$(prefix)_$(initmethod)"
    #fprex = "$(prefix)_BPDN"
    regstr = "_b$(β)_bsr$(bsr)"
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_it$(maxiter)_rt$(rt1)")
    imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
end

#=====================================#
using ForwardDiff, Test

# Symmetric penalty
f(x) = (M = reshape(x,nc,noc); Rs = M*N0-D; vec(Rs))
x0 = vec(M0)
fdJ = ForwardDiff.jacobian(f, x0)

J = zeros(nc^2, nc*noc)
J .= 0.
for i=1:nc
    J[i:nc:nc^2,i:noc:end] .= N0'
end

@test fdJ == J

# Sparsity W penalty
idx = 1:10; β = 5.0
fdJ = zeros(m*noc, nc*noc)
f(x) = (M = reshape(x,nc,noc); w = vec(U*M); rn = β*min.(w,0); rn[idx])
x0 = vec(M0)
fwdJ = ForwardDiff.jacobian(f, x0)
fdJ[idx,:] .= fwdJ

J = zeros(m*noc, nc*noc)
J .= 0.
bdU = blockdiag(ntuple(_ -> sparse(U), noc)...)
w = vec(U*M0); rn = min.(w,0); βsgn = -β*sign.(rn)
dβsgn = spdiagm(0 => βsgn)
J[idx, :] .= dβsgn[idx,idx]*bdU[idx,:]

@test fdJ[idx,:] == J[idx,:]

# Symmetric penalty and Sparsity W penalty
f(x) = (M = reshape(x,nc,noc); Rs = M*N0-D; rs = vec(Rs);
        w = vec(U*M); rn = β*min.(w,0); rnidx = rn[idx];
        rns = similar(rn); fill!(rns,0); rns[idx] .= rnidx;
        vcat(rs,rns))
fdJ = ForwardDiff.jacobian(f, x0)

f! = create_f!(U,N,D,β)
f!(rjw, x0, idx)
@test fdJ[1:nc^2,:] == rjw.J[1:nc^2,:]
@test fdJ[nc^2 .+ idx,:] == rjw.J[nc^2 .+ idx,:]
fx = f(x0)
@test dot(fx,fx)/2 == f!(rjw, x0, idx)
@test f(x0)[1:nc^2] == rjw.r[1:nc^2]
@test f(x0)[nc^2 .+ idx] == rjw.r[nc^2 .+ idx]

# Symmetric penalty and Sparsity H penalty
f(x) = (Nt = reshape(x,nc,noc); Rs = Nt*M0'-D'; rs = vec(Rs);
        h = vec(V*Nt); rn = β*min.(h,0); rnidx = rn[idx];
        rns = similar(rn); fill!(rns,0); rns[idx] .= rnidx;
        vcat(rs,rns))
x0 = vec(N0')
fdJ = ForwardDiff.jacobian(f, x0)

f! = create_f!(V,M0',D',5.0)
f!(rjh, x0, idx)
@test fdJ[1:nc^2,:] == rjh.J[1:nc^2,:]
@test fdJ[nc^2 .+ idx,:] == rjh.J[nc^2 .+ idx,:]

fx = f(x0)
@test isapprox(dot(fx,fx)/2, f!(rjh, x0, idx); atol = 1e-10)
@test f(x0)[1:nc^2] == rjh.r[1:nc^2]
@test f(x0)[nc^2 .+ idx] == rjh.r[nc^2 .+ idx]

#===================== source code =========================#

function stochasticlm(
        f!,                               # f!(cache, θ, idx) -> objval; must also accept `nothing` as cache
        θ0,                               # initial guess
        nbatchs::AbstractVector{Int},     # number of samples per minibatch
        cache::AbstractSLMCache{T},       # holds residual, Jacobian, and other useful precomputated values
        (lb, ub) = (nothing, nothing);    # lower and upper bounds on θ
        solver! = minres_solver!,         # solver!(d, cache, λ, Hd)
        solver_kwargs = (;),
        λmin=sqrt(eps(float(eltype(θ0)))),  # minimum regularization coefficient (multiplies the diagonal of the Hessian)
        λ0=max(λmin, sqrt(λmin)),         # initial regularization coefficient
        η0=:auto,                         # initial barrier coefficient for bound constraints
        ηmin=eps(float(eltype(θ0))),      # minimum barrier coefficient
        Hdminratio=0,                     # minimum diagonal value of the Hessian
        γ=4,                              # regularization coefficient multiplier/divider for failure/success
        p0=0.0001,                        # minimum fraction of expected improvement needed for step-acceptance
        p1=0.25,                          # if fractional expected improvement <p1, increase λ
        p2=0.75,                          # if fractional expected improvement >p2, decrease λ
        iswarn::Bool=false,
        verbose::Bool=false,
    ) where T
    Base.@constprop :none
    checkp(p0, p1, p2)
    bd = (lb === nothing && ub === nothing) ? nothing : BoundsData(lb, ub)
    validate_start(θ0, bd)
    mxbatch = maximum(nbatchs)
    mxbatch <= length(residual(cache)) || throw(ArgumentError("all batchsizes (maximum $mxbatch) must be less than or equal to the number of residuals ($(length(residual(cache))))"))

    θ = copyto!(similar(θ0, T), θ0)
    d = similar(θ)
    θtmp = similar(θ)
    Hd = similar(θ)   # hessian diagonal
    sd = SolverData{T}(mxbatch, θ)

    λ, fvalpairs = λ0, nothing
    η = η0
    ηdec = isa(η, Real) ? (η/ηmin)^(1/length(nbatchs)) : nothing
    rtol = sqrt(eps(T))
    nreset = min(100, length(nbatchs) ÷ 2 + 1)
    local objval
    for (iter, batchsize) in enumerate(nbatchs)
        idx = minibatcher(cache, batchsize)
        if batchsize != length(sd.dw)
            resize_r!(sd, batchsize)
        end
        objval = ObjBarVals(checked_update!(f!, cache, θ, idx))
        hessprep!(Hd, cache, idx; Hdminratio)
        if bd !== nothing
            if η === :auto
                η = average_complementarity(total_neggradient(cache, idx), θ0, bd)
                verbose && @info("Initial barrier coefficient: η = $η")
                ηdec = (η/ηmin)^(1/length(nbatchs))
            end
            objval = ObjBarVals(objval, barrier!(invalidate!(bd), θ, η))
        else
            η = 0
        end
        verbose && @info("Iter $iter (batchsize=$batchsize): objective value = $objval, λ = $λ, η = $η")
        ret = step_slm!(f!, θ, (bd, η), sd, cache, λ, Hd, idx, objval, rtol; solver!, solver_kwargs, λmin, γ, p0, p1, p2, iswarn, d, θtmp)
        ret === nothing && break
        objvalnew, λ = ret
        if fvalpairs === nothing
            fvalpairs = Tuple{typeof(objval), typeof(objval)}[]
        end
        push!(fvalpairs, (objval, objvalnew))
        rtol = default_rtol(fvalpairs, nreset)
        if !iszero(η)
            η /= ηdec
            η = max(η, ηmin)
        end
    end
    return θ, objval.objective, map(vp -> (T(vp[1]), T(vp[2])), fvalpairs)
end

function step_slm!(
        f!, θ, (bd, η)::Tuple{Union{Nothing, BoundsData}, Real}, sd, cache::AbstractSLMCache{T}, λ, Hd, idx, objval, rtol; solver!, solver_kwargs, λmin, γ, p0, p1, p2, iswarn, d=similar(θ), θtmp=similar(θ), predcache=Tuple{eltype(objval), eltype(objval)}[], forceupdate::Bool=false,
    ) where T
    empty!(predcache)
    λmax = 1/eps(T)
    canshrink = true
    λref = Ref(λ)
    sd, J, A, M = build!(sd, Val(solvermode(solver!)), cache, bd, λref, Hd, idx)
    if forceupdate   # for testing purposes
        sd, A, M = update_λ!(sd, A, M, cache, bd, λref, Hd)
    end
    while true
        if λref[] != λ
            λref[] = λ
            sd, A, M = update_λ!(sd, A, M, cache, bd, λref, Hd)
        end
        solver!(d, sd, A, M, rtol; solver_kwargs...)
        scalemin!(d, sd, J, cache, bd, λref, Hd)
        α = trim_valid!(result(d), θ, bd)
        if iszero(result(d)) || !isvalidstep(result(d), θ, cache)
            λ = γ * λ
            canshrink = false
            λ > λmax && return nothing
            continue
        end
        # @show λ dot(result(d), neggradient(cache, idx)) / (norm(result(d) .* sqrt.(Hd)) * norm(neggradient(cache, idx) ./ sqrt.(Hd)))
        # @show λ norm(result(d) .* sqrt.(Hd)) / norm(θ .* sqrt.(Hd))
        θtmp .= θ .+ result(d)
        objvalnew = ObjBarVals{T}(checked_update!(f!, nothing, θtmp, idx), barrier_objective(θtmp, (bd, η)))
        objpred = ObjBarVals{T}(predict(result(d), objval.objective, cache, idx), predict_barrier(result(d), objval.barrier, bd))
        @assert objpred <= objval + (1 + length(θ)) * eps(objval) "Value: $objval, Prediction: $objpred, New value: $objvalnew, λ: $λ, α: $α\nd = $(sprint(show, result(d); context=(:compact=>true, :limit=>true)))"
        # Trust-region criterion
        Δobj = objval - objvalnew
        Δpred = objval - objpred
        iseps(Δobj, objval, length(θ)) && iseps(Δpred, objpred, length(θ)) && return nothing
        ρ = Δobj / Δpred
        push!(predcache, (Δobj.objective, Δpred.objective))
        # @show ρ Δobj Δpred
        if ρ < p1
            # insufficient improvement, suggesting the Hessian underestimated the local curvature. Use more damping.
            λ = γ * λ
            canshrink = false
            if λ > λmax
                if iswarn
                    # check for user error (e.g., defining the residual as `predicted - observed` instead of `observed - predicted`)
                    nrev = 0
                    for (dobj, dpred) in reverse(predcache)
                        if -1.1 * dpred <= dobj <= -0.9 * dpred
                            nrev += 1
                            if nrev >= 3
                                @warn("The actual change is approximately equal to `-(predicted change)`, possibly suggesting a sign discrepancy between the residual and the Jacobian.")
                                break
                            end
                        else
                            nrev = 0   # reset counter if not sequential
                        end
                    end
                end
                return nothing
            end
        elseif ρ > p2 && canshrink && λ > λmin
            # good improvement, enough to suggest we don't need this much damping
            λ = max(λmin, λ / γ)
        end
        if ρ > p0
            # accept the step
            copyto!(θ, θtmp)
            objval = objvalnew
            break
        end
    end
    return objval, λ
end

#===================================================#
# g'(X)
function gprime!(gp, X, σ)
    @. gp = 0.5 * X * (X^2 + σ^2)^(-3/4)
    return gp
end

# J(M)[H]
function jac_mul!(Y, H, U, M, σ)
    X  = U * M          # m×p
    UH = U * H          # m×p
    gp = similar(X)
    gprime!(gp, X, σ)
    @. Y = gp * UH      # y .= gp .* UH
    return Y
end

# J(M)'[Y]
function jac_t_mul!(Z, Y, U, M, σ)
    X  = U * M
    gp = similar(X)
    gprime!(gp, X, σ)
    tmp = gp .* Y
    Z .= U' * tmp
    return Z
end

k, p = size(M0); m, n = size(X)
J = makeop(Mcache,Ncache,U,Vt,Sw,Sh)

# Jacobian Check: <Jv, w> == <v, Jᵀ w>
k, p = size(M0); m, n = size(X)
#J = makeop(Mcache,Ncache,U,Vt,Sw,Sh,Gw,Gh)
f(x) = (M = reshape(x[1:k*p], k, p); N = reshape(x[k*p+1:end], p, k);
        W = U*M; H = N*Vt;
        Ysw = g(W, σw)*rtαw; Ysh = g(H, σh)*rtαh;
        [vec(Ysw); vec(Ysh)])
x0 = [vec(M0);zeros(k*p)]
Jfd = ForwardDiff.jacobian(f,x0)
Jfdv = Jfd*x0; Jfdv = reshape(Jfdv[1:m*p],m,p)
Jv = jac_mul!(zeros(m,p), M0, U, M0, σw)
norm(Jfdv-Jv) # 3.5743888218162846e-16
