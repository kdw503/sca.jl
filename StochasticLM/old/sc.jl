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

# ARGS = [":isvd","0.0005","0","1","2000", "10"]
initmethod = eval(Meta.parse(ARGS[1])) # :randcolX
αh_user = eval(Meta.parse(ARGS[2])) # 5e-4
sd = eval(Meta.parse(ARGS[3])) # 1
batch_rate = eval(Meta.parse(ARGS[4])) # 0.5
maxiter = eval(Meta.parse(ARGS[5])) # 2000
inner_maxiter = eval(Meta.parse(ARGS[6])) # 100
@show initmethod, αh_user, sd, batch_rate, maxiter, inner_maxiter
flush(stdout) 

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
optim_method = :stochasticLM2
β = 0

#======== L1 ==========#
# initmethod = :randcolX
Winit, Hinit, Minit, Ninitt, _ = dd[String(initmethod)]
fvinit = LCSVD.fitd(X_whitened,Winit*Hinit)

M, N = copy(Minit), copy(Ninitt')
θ0 = vcat(vec(copy(M)), vec(copy(N)))
W = U*M; H = N*Vt

βw=βh=β; αw=0
σh = 1e-16
M .= copy(Minit)
N .= copy(Ninitt')
W0 = U*M; H0 = N*Vt # norm(H0,1) = 15953.009989164451
batch_size = Int(round(100000*batch_rate))
αh = αh_user*norm(D)^2/norm(H0[:,1:batch_size],1)/norm(M) # norm(H0,1), norm(H0[:,1:50000],1), norm(H0[:,1:30000],1)
θ0 = vcat(vec(copy(M)), vec(copy(N)))
W = U*M; H = N*Vt
rtαh = sqrt(αh); Ph = g(H, σh)*rtαh; Gh = dg(H, σh)*rtαh
nM = Ref(norm(M))
myminibatcher = makescminibather(k,p,n; sd=sd)
rjop = SLMCache(makesc2op(M, N, Vt, Gh, Ph, nM); Hd = ones(Float64,2k*p), minibatcher=myminibatcher) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
nmf_rjop! = make_sc2_fcache_op!(M, N, Vt, D, Gh, Ph, nM, rtαh, σh)
nbatch = calculate_sc_hbatchsize(Int((n*p)*batch_rate),p)*p + k^2
rt2 = @elapsed θopt, objval = stochasticlm(nmf_rjop!, θ0, nbatch, rjop; itermax=maxiter,
                                        solver_kwargs=(; itmax=inner_maxiter),
                                        convergence=ConvergenceParams(; dprime=1e-100),
                                        verbose=true) # length(rjop.r) -> full batch
M .= reshape(θopt[1:k*p], k, p)
N .= reshape(θopt[k*p+1:end], p, k)

W = U*M; H = N*Vt
normWHmX = norm(W*H-X_whitened)
L1h = norm(H,1)
LCSVD.normalizeWH!(W,H)
norm1nH = norm(H,1) 
@show normWHmX, norm1nH; flush(stdout) 
fprex = "$(prefix)_$(dataset)_$(initmethod)_$(optim_method)_sd$(sd)_br$(batch_rate)_ii$(inner_maxiter)_mi$(maxiter)"
#fprex = "$(prefix)_BPDN"
regstr = "_aw$(αw)_ah$(αh_user)_nWHX$(normWHmX)_nH1$(norm1nH)"
fv = LCSVD.fitd(X_whitened,W*H)
fname = joinpath(subworkpath,"$(fprex)$(regstr)_objval$(objval)_f$(fv)_rt$(rt2)")
imsave_data(dataset,fname,W,H,imgsz,lengthT; saveH=false)
