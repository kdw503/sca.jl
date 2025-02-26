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

feature_name = eval(Meta.parse(ARGS[1]))
ncells = eval(Meta.parse(ARGS[2]))
noc = eval(Meta.parse(ARGS[3]))
αhals = eval(Meta.parse(ARGS[4]));
hals_maxiter = eval(Meta.parse(ARGS[5]));

@show feature_name, ncells, noc, αhals, hals_maxiter

memsize = Int(Sys.total_memory())/1e9
@show memsize

# load data
using NRRD
feature_group=first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-raw"
Xraw = load(joinpath(fgpath,file_name*".nhdr")).data
m,nraw = size(Xraw)
ncells = ncells == 0 ? nraw : (file_name*="_n$(ncells)"; ncells); @show ncells; flush(stdout)
ncells < nraw && error("ncells must be greater than $(nraw)")
X = view(Xraw,:,1:ncells); n=ncells

# HALS
method="hals"
@show method; flush(stdout)
# Initialization
initstr = "initrand"
m, n = size(X) 
W, H = rand(Float32,m,noc), rand(Float32,noc,n); # random init
# solve
@show "solve"; flush(stdout)
tol = 0; maxiter = hals_maxiter
rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, W, H)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
LCSVD.normalizeW!(W,H)
# fitval = LCSVD.fitd(X,W*H)
fitval = 0
# @show "s3"
fname = joinpath(subworkpath,"$(file_name)_$(initstr)_$(method)_noc$(noc)_a$(αhals)_tol$(tol)_mit$(maxiter)_nrrd.jld2")
save(fname, "W",W,"H",H,"rt2",rt2,"iter",rst0.niters,"fitval",fitval, "rst", rst0, "memsize", memsize)
# WMB-10Xv3 (non-interactive): XHt(1093sec), WX'(4028sec), total()
# W, H, rt1, rt2, iter, fitval = dd["W"], dd["H"], dd["rt1"], dd["rt2"], dd["iter"], dd["fitval"]
