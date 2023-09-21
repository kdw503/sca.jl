using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
include(joinpath(workpath,"setup_plot.jl"))
subworkpath = joinpath(workpath,"paper","ncells")

z = 0.5
ncls = [15,30,60]
for ncells = ncls
plottime = Inf
for (mtdstr, submtdstrs) in [("sca",["_sp", "_sp_nn"]),("admm",["_nn"]),("hals",["_nn", "_sp_nn"])]
    ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,\"$(mtdstr)$(ncells)_runtime_vs_avgfits.jld2\"))"))
    rng = eval(Meta.parse("$(ddsym)[\"rng\"]"))
    eval(Meta.parse("$(mtdstr)rng=$(ddsym)[\"rng\"]"))
    plottime = plottime > rng[end] ? rng[end] : plottime
    for submtdstr in submtdstrs
        frpx = "$(mtdstr)$(submtdstr)"; dickeystr = "stat$(submtdstr)"
        # @eval ($(Symbol("$(frpx)_means")) = ($(ddsym)["stat$(submtdstr)"][1]))
        # @eval ($(Symbol("$(frpx)_stds")) = ($(ddsym)["stat$(submtdstr)"][2]))
        eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat$(submtdstr)\"][1]"))
        eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat$(submtdstr)\"][2]"))
        @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + z*$(Symbol("$(frpx)_stds"))))
        @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - z*$(Symbol("$(frpx)_stds"))))
    end
end

# ddhals = load("hals_runtime_vs_avgfits.jld2"); rng = ddhals["rng"]
# hals_nn_means = ddhals["stat_nn"][1]; hals_nn_stds = ddhals["stat_nn"][2]
# hals_sp_nn_means = ddhals["stat_sp_nn"][1]; hals_sp_nn_stds = ddhals["stat_sp_nn"][2]
# hals_nn_upper = hals_nn_means + z*hals_nn_stds; hals_nn_lower = hals_nn_means - z*hals_nn_stds
# hals_sp_nn_upper = hals_sp_nn_means + z*hals_sp_nn_stds; hals_sp_nn_lower = hals_sp_nn_means - z*hals_sp_nn_stds

# ddsca = load("sca_runtime_vs_avgfits.jld2"); rng = ddsca["rng"]
# sca_nn_means = ddsca["stat_nn"][1]; sca_nn_stds = ddsca["stat_nn"][2]
# sca_sp_means = ddsca["stat_sp"][1]; sca_sp_stds = ddsca["stat_sp"][2]
# sca_sp_nn_means = ddsca["stat_sp_nn"][1]; sca_sp_nn_stds = ddsca["stat_sp_nn"][2]
# sca_nn_upper = sca_nn_means + z*sca_nn_stds; sca_nn_lower = sca_nn_means - z*sca_nn_stds
# sca_sp_upper = sca_sp_means + z*sca_sp_stds; sca_sp_lower = sca_sp_means - z*sca_sp_stds
# sca_sp_nn_upper = sca_sp_nn_means + z*sca_sp_nn_stds; sca_sp_nn_lower = sca_sp_nn_means - z*sca_sp_nn_stds

# ddadmm = load("admm_runtime_vs_avgfits.jld2"); rng = ddadmm["rng"]
# admm_nn_means = ddadmm["stat_nn"][1]; admm_nn_stds = ddadmm["stat_nn"][2]
# admm_sp_means = ddadmm["stat_sp"][1]; admm_sp_stds = ddadmm["stat_sp"][2]
# admm_sp_nn_means = ddadmm["stat_sp_nn"][1]; admm_sp_nn_stds = ddadmm["stat_sp_nn"][2]
# admm_nn_upper = admm_nn_means + z*admm_nn_stds; admm_nn_lower = admm_nn_means - z*admm_nn_stds
# admm_sp_upper = admm_sp_means + z*admm_sp_stds; admm_sp_lower = admm_sp_means - z*admm_sp_stds
# admm_sp_nn_upper = admm_sp_nn_means + z*admm_sp_nn_stds; admm_sp_nn_lower = admm_sp_nn_means - z*admm_sp_nn_stds

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)

plotlength = min(length(rng),50); plotrng = Colon()
fig = Figure(resolution=(400,300))
ax = AMakie.Axis(fig[1, 1], limits = ((0,min(0.3,plottime)), nothing), xlabel = "time(sec)", ylabel = "average fit", title = "Average Fit Value vs. Running Time")

lns = Dict(); bnds=Dict()
# for (i,(frpx, lbl)) in enumerate([("sca_sp","SMF (α=100,β=0)"),("hals_nn","HALS (α=0)"),("hals_sp_nn","HALS (α=0.1)"),
#                                  ("admm_nn","Comp. NMF (α=0)"),("admm_sp","Comp. SMF (α=10)"),
#                                  ("sca_nn","SMF (α=0,β=1000)"),("sca_sp_nn","SMF (α=100,β=1000)"),("admm_sp_nn","Comp. NMF (α=10)"),])
for (i,(mtdstr, submtdstr, lbl, clridx)) in enumerate([("sca","_sp","SMF (α=100,β=0)",2),("sca","_sp_nn","SMF (α=100,β=1000)",5),
                                                       ("admm","_nn","Compressed NMF",3),("hals","_nn","HALS",4)])#,("hals","_sp_nn","HALS",6)])
    # eval(print("$(frpx)_means"))
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(ncells).png"),fig,px_per_unit=2)

end # for ncells
