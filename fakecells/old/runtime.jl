using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath, "fakecells")

include(joinpath(workpath,"setup_light.jl"))
#include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

# ARGS = [":lbfgs_admm", ":isvd", "0.005","0", "100"]
optim_method = eval(Meta.parse(ARGS[1])) # :lbfgs_admm
initmethod = eval(Meta.parse(ARGS[2])) # :isvd
α = eval(Meta.parse(ARGS[3])) # 0.005
β = eval(Meta.parse(ARGS[4])) # 5.0
maxiter = eval(Meta.parse(ARGS[5])) # 500
@show optim_method, α, β, maxiter
flush(stdout) 

dataset = :fakecells; inhibitindices=0; bias=0.1; SNR=10.0; factor=1; noc=15; nac=0; nc=noc+nac
subtract_bg=false; maskth=0.25; makepositive = true; tol=-1
imgsz0 = (40,20)
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

X, imsz, lhT, ncs, gtnoc, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,noc)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

#============== PCB ==============================#
prefix = "pcb"

rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
for α in [0.0005,0.0010,0.0015,0.0020,0.0025,0.0030]
    @show α
# ur = 1e-3; nr = 1e-2 # :sgd_injectnoise
β1 = β2 = β; α1 = α2 = α
β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
#r = eps(Float64)^(1/(2*maxiter))
σ0 = 1; r = 0.3; useprecond = false
# maxiter = 500 # 500 #Int(ceil(log(eps(eltype(X_whitened)))/log(r))) #lcsvd_maxiter # 
# inner_maxiter = 10 #Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) #
inner_maxiter = 50; inner_tol = 1e-6

alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0,
    #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    r=r, useprecond=useprecond, usedenoiseUVt=false, optim_method=optim_method,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = false, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0) #, ur=ur, nr=nr); # ur=0.00001, nr=0.001
# M1, N1t = copy(M0), copy(N0t)
# rst1 = LCSVD.solve!(alg, X_whitened, U, V, D, M1, N1t; gtW=gtW, gtH=gtH, use_σ2_cal_pen=false);
alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed rst2 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

W1, H1 = rst2.W, rst2.Ht'
# L1h = norm(H1,1)
LCSVD.normalizeW!(W1,H1);
fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
nodr = LCSVD.matchedorder(ml,noc)
W1, H1 = W1[:,nodr], H1[nodr,:]
makepositive && LCSVD.flip2makepos!(W1,H1)

# Einit = rst1.traces[1].f_x; Eend = rst1.traces[end].f_x
# Esym = rst1.traces[end].sympen
# Esh = rst1.traces[end].sparseH
# norm1nH = norm(H1,1)
fprex = "$(prefix)_$(dataset)_$(initmethod)_$(optim_method)_W1H1"
#fprex = "$(prefix)_BPDN"
regstr = "_s0$(σ0)_r$(r)_a$(α)_b$(β)_intol$(inner_tol)_initer$(inner_maxiter)"
# fname = joinpath(subworkpath,"$(fprex)$(regstr)_Einit$(Einit)_Eend$(Eend)_Esy$(Esym)_Esh$(Esh)_nH$(norm1nH)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
fname = joinpath(subworkpath,"$(fprex)$(regstr)_f$(fv)_it$(rst2.niters)_rt$(rt2)")
imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
# niters = map(t->t.niters,rst1.traces); f_x = map(t->t.f_x,rst1.traces)
# save(joinpath(subworkpath,"$(fprex)$(regstr)_mit$(alg.maxiter).jld2"),"niters",niters,"f_x",f_x)
end
