using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","rnaseq")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"utils.jl"))

noc = eval(Meta.parse(ARGS[1]))
α1 = eval(Meta.parse(ARGS[2]));
α2 = eval(Meta.parse(ARGS[3]));
β1 = eval(Meta.parse(ARGS[4]));
β2 = eval(Meta.parse(ARGS[5]));
lcsvd_maxiter = eval(Meta.parse(ARGS[6]));

@show noc, α1, α2, β1, β2, lcsvd_maxiter

dataset = :rnaseq
#X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
using FileIO, IncrementalSVD
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)s
file_name = feature_name*"-raw"
Xraw = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X
X = sqrt.(Xraw)

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)

initisvd(X,noc) = ((U,s)=isvd(X,noc); H = U'*X ; (U, H, copy(U), copy(H))) # H isn't normalized one

mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter; tol=-1 

fname = joinpath(subworkpath,"$(file_name)_isvd_init$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    W0, H0, M0, N0, Wp, Hp, D, rt1 = dd["W0"], dd["H0"], dd["M0"], dd["N0"], dd["Wp"], dd["Hp"], dd["D"], dd["rt1"]
    # noc = 500, norm(X-W0*M0*N0*H0) = 15375.242f0
else
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, noc; initmethod=:custom, initfn=initisvd)
    save(fname, "W0",W0,"H0",H0,"Wp",Wp,"Hp",Hp,"M0",M0,"N0",N0,"D",D,"rt1",rt1)
end

σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false,
    denoisefilter=:avg, uselv=false, maxiter = maxiter, store_trace = false,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
M, N = copy(M0), copy(N0)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
W, H = rst0.W, rst0.H
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X,W*H)
fname = joinpath(subworkpath,"$(file_name)_lcsvd_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(rst0.niters).jld2")
save(fname, "W",W,"H",H,"M",M,"N",N,"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)
