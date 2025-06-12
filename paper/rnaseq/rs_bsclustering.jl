using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
allenbrainversion = "20241130"
subworkpath = joinpath(workpath,"paper","rnaseq",allenbrainversion)

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"utils.jl"))
include(joinpath(workpath,"clustering.jl"))

# Bootstrap clustering
# Base.ARGS = ["\"WMB-10Xv2-HY\"", "\"log2\"", ":pcb_sp", "0.0001", "100"]
feature_name = eval(Meta.parse(ARGS[1]))
pp = eval(Meta.parse(ARGS[2]))
mf_method = eval(Meta.parse(ARGS[3]))
pvalue = eval(Meta.parse(ARGS[4]))
nresample = eval(Meta.parse(ARGS[5]))
@show mf_method; flush(stdout)

#=================== Bootstrap Clustering =================#
feature_name_pp = feature_name*"-"*pp
if mf_method == :pcb_sp
    fname = joinpath(subworkpath,"$(feature_name_pp)_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw0.0_bh0.0_tol1.0e-6_it13.jld2")
    dd = load(fname)
    Gene = dd["W"]; Cell = dd["H"]
elseif mf_method == :pcb_sp_nn
    fname = joinpath(subworkpath,"$(feature_name_pp)_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it11.jld2")
    dd = load(fname)
    Gene = dd["W"]; Cell = dd["H"]
elseif mf_method == :svd
    fname = joinpath(subworkpath,"$(feature_name_pp)_initisvd_noc500_X.jld2")
    dd = load(fname)
    Gene = dd["U"]'; Cell = (dd["V"]*dd["D"])'
elseif mf_method == :hals
    fname = joinpath(subworkpath,"$(feature_name_pp)_hals_noc500_a0.1_iter100.jld2")
    dd = load(fname)
    Gene = dd["W"]; Cell = dd["H"]
else
    error("Unknown mf_method: $mf_method")
end
rtclust = @elapsed clustn = cluster_resample(Cell, nresample, pvalue)
save(joinpath(subworkpath,"$(feature_name_pp)_$(mf_method)_p$(pvalue)_n$(nresample)_bsclass.jld2"), "clustn",clust,"rtclust",rtclust)

