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
compnmf_maxiter = eval(Meta.parse(ARGS[2]));

@show noc, compnmf_maxiter

dataset = :rnaseq
#X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
using FileIO, IncrementalSVD
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)
file_name = feature_name*"-raw"
Xraw = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X
X = sqrt.(Xraw)

# COMPNMF
prefix = "compnmf"
@show prefix; flush(stdout)
fname = joinpath(subworkpath,"$(file_name)_nndrsvd_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    Wcn0, Hcn0, rt1 = dd["Whals0"], dd["Hhals0"], dd["rt1"]
else
    rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, noc, variant=:ar);
    save(fname, "Whals0",Wcn0,"Hhals0",Hcn0,"rt1",rt1)
end
W, H = copy(Wcn0), copy(Hcn0);
fname = joinpath(subworkpath,"$(file_name)_compmat_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    L, R, rt2 = dd["L"], dd["R"], dd["rt2"]
else
    rt2 = @elapsed L, R, X_tilde, Y_tilde, A_tilde = compmat(X, W, H, w=2)
    save(fname, "L",L,"R",R,"rt2",rt2)
end
method = :COMPNMF; maxiter = compnmf_maxiter
(tailstr,initmethod) = ("_nn",:lowrank_nndsvd)
dd = Dict(); tol=-1
rt3 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{eltype(X)}(maxiter=maxiter, tol=tol, verbose=false), X, W, H, L=L, R=R)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X,W*H)
fname = joinpath(subworkpath,"$(file_name)_compnmf_noc$(noc).jld2")
save(fname, "W",W,"H",H,"rt1",rt1,"rt2",rt2,"rt3",rt3,"iter",maxiter,"fitval",fitval)
