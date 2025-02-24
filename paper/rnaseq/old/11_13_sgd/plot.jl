using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

# sizestep, pcb_maxiter, pcb_inner_maxiter, tol, per_component = 10, 1, 1, 1e-6, false
sizestep = eval(Meta.parse(ARGS[1]))
pcb_maxiter = eval(Meta.parse(ARGS[2]))
pcb_inner_maxiter = eval(Meta.parse(ARGS[3]))
tol = eval(Meta.parse(ARGS[4]))
per_component = eval(Meta.parse(ARGS[5]))
memsize = Int(Sys.total_memory())/1e9
per_com_str = per_component ? "cw" : ""
subworkpath = joinpath(workpath,"paper","rnaseq","old","11_13_sgd","$(sizestep)to1")

# sgd
jldfname = "paper/rnaseq/old/11_13_sgd/WMB-10Xv2-HY-rawt_ss10_pcb_sgd_noc500_a0.005_b5.0_r0.3_tol1.0e-6_it14.jld2"
# slbfgs
jldfname = "paper/rnaseq/old/11_13_sgd/WMB-10Xv2-HY-rawt_ss10_pcb_slbfgs_noc500_aw0.0001_bh1.0_r0.3_tol1.0e-6_it14.jld2"
# lbfgs
jldfname = "paper/rnaseq/old/11_13_sgd/WMB-10Xv2-HY-rawt_ss10_pcb_lbfgs_noc500_a0.005_b5.0_r0.6_tol1.0e-6_it11.jld2"

fname = joinpath(workpath,jldfname)
dd = load(fname)
