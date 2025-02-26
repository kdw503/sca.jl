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

@show feature_name, ncells, noc, αhals, hals_maxiter; flush(stdout)

memsize = Int(Sys.total_memory())/1e9
@show memsize; flush(stdout)

# sparse array
@show Int(Sys.free_memory())/1e9 #
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)s
file_name = feature_name*"-raw"
Xraw = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X' # 1.4G
@show Int(Sys.free_memory())/1e9 #
m, n = size(Xraw)
Int(Sys.free_memory())/1e9 #
W = rand(Float32,n,noc)
XW = Matrix{Float32}(undef,m,noc)
@show Int(Sys.free_memory())/1e9 # 
@show @elapsed mul!(XW, Xraw, W) # 
@show Int(Sys.free_memory())/1e9 # 
@show @elapsed mul!(XW, Xraw, W) # 
@show Int(Sys.free_memory())/1e9 # 

Xraw=0
GC.gc()

# NRRD
# load data
using NRRD
@show Int(Sys.free_memory())/1e9 # 
feature_group=first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-raw"
Xraw = load(joinpath(fgpath,file_name*".nhdr")).data
@show Int(Sys.free_memory())/1e9 #
m,nraw = size(Xraw)
ncells = ncells == 0 ? nraw : (file_name*="_n$(ncells)"; ncells); @show ncells; flush(stdout)
ncells > nraw && error("ncells must be smaller than $(nraw)")
X = view(Xraw,:,1:ncells); n=ncells

# calculate X*H'
W = rand(Float32,n,noc)
XW = Matrix{Float32}(undef,m,noc)
@show Int(Sys.free_memory())/1e9 #
@show "Calculating $(file_name) ($m,$n) XHt..."; flush(stdout)
# rt1 = @elapsed sum(X)
# @show rt1; flush(stdout)
# rt2 = @elapsed sum(X)
# @show rt2; flush(stdout)
# save(joinpath(subworkpath,"$(file_name)_interactive_sum(X)_$(memsize)GB_$(rt1)_$(rt2).jld2"), "rt1", rt1, "rt2", rt2)

@show @elapsed mul!(XW, X, W)
@show Int(Sys.free_memory())/1e9 #
# @show rt1; flush(stdout)
# save(joinpath(subworkpath,"$(file_name)_interactive_XHt_$(memsize)GB_runtime.jld2"), "rt1", rt1)
@show @elapsed mul!(XW, X, W)
@show Int(Sys.free_memory())/1e9 #
# @show rt2; flush(stdout)
# save(joinpath(subworkpath,"$(file_name)_interactive_XHt_$(memsize)GB_runtime.jld2"), "rt1", rt1, "rt2", rt2)
