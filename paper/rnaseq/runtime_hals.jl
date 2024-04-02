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
αhals = eval(Meta.parse(ARGS[2]));
hals_maxiter = eval(Meta.parse(ARGS[3]));

@show noc, αhals, hals_maxiter

dataset = :rnaseq
#X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
using FileIO, IncrementalSVD
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)
file_name = feature_name*"-raw"
Xraw = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X
X = sqrt.(Xraw)

# HALS
prefix="hals"; @show prefix
fname = joinpath(subworkpath,"$(file_name)_nndrsvd_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    Whals0, Hhals0, rt1 = dd["Whals0"], dd["Hhals0"], dd["rt1"]
else
    rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar);
    save(fname, "Whals0",Whals0,"Hhals0",Hhals0,"rt1",rt1)
end
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1
W, H = copy(Whals0), copy(Hhals0);
rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, W, H)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X,W*H)
fname = joinpath(subworkpath,"$(file_name)_hals_noc$(noc)_a$(αhals)_iter$(maxiter).jld2")
save(fname, "W",W,"H",H,"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)
