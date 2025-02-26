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
#include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

feature_name = eval(Meta.parse(ARGS[1]))
ncells = eval(Meta.parse(ARGS[2]))
noc = eval(Meta.parse(ARGS[3]))
nac = eval(Meta.parse(ARGS[4]))
α1 = eval(Meta.parse(ARGS[5]));
α2 = eval(Meta.parse(ARGS[6]));
β1 = eval(Meta.parse(ARGS[7]));
β2 = eval(Meta.parse(ARGS[8]));
pcb_maxiter = eval(Meta.parse(ARGS[9]));

@show feature_name, ncells, noc, nac, α1, α2, β1, β2, pcb_maxiter

memsize = Int(Sys.total_memory())/1e9
@show memsize; flush(stdout)

# load data
using NRRD
feature_group=first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-raw"
Xraw = load(joinpath(fgpath,file_name*".nhdr")).data
m,nraw = size(Xraw)
ncells = ncells == 0 ? nraw : (file_name*="_n$(ncells)"; ncells); @show ncells; flush(stdout)
ncells > nraw && error("ncells must be smaller than $(nraw)")
X = view(Xraw,:,1:ncells); n=ncells

# LCSVD
method = "pcb"
@show method; flush(stdout)

initmethod = :random; initmtdstr="init$(initmethod)"
nc = noc+nac
W0=rand(Float32,size(X,1),nc)
H0=rand(Float32,nc,size(X,2))
M0=rand(Float32,nc,noc)
N0=rand(Float32,noc,nc)
D=rand(Float32,nc,nc)
rt1=0.

@show "solve", Int(Sys.total_memory()-Sys.free_memory())/1e9
useprecond=false; uselv=false; tol=0#3e-5
σ0=std(W0*M0); r=0.3; maxiter = pcb_maxiter == 0 ? Int(ceil(log(eps(eltype(X)))/log(r))) : pcb_maxiter
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond,
    denoisefilter=:avg, uselv=false, maxiter = maxiter, store_trace = false,
    store_inner_trace = false, show_trace = true, allow_f_increases = true,
    f_abstol=0, f_reltol=0, f_inctol=1e2, x_abstol=0, x_reltol=tol, successive_f_converge=0)
M, N = copy(M0), copy(N0)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
@show "2", Int(Sys.total_memory()-Sys.free_memory())/1e9
iter = rst0.niters
fname = joinpath(subworkpath,"$(file_name)_$(initmtdstr)_noc$(noc)_nac$(nac)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_tol$(tol)_it$(iter).jld2")
save(fname, "W",rst0.W,"H",rst0.H,"M",M,"N",N,"rt1",rt1,"rt2",rt2,"iter",iter,"fitval",0,"rst",rst0)
