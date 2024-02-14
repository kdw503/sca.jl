using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","ncells")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
using Interpolations

SNR = 0; ncellss = [10,20,30,50]; num_experiments = 50; factor=1

for (prefix, initmethods, tailstrs) in [("lcsvd", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("lcsvd_LPF", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("lcsvd_precon", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("lcsvd_precon_LPF", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("hals",["nndsvd","nndsvd"],["_nn","_sp_nn"]),("compnmf",["lowrank_nndsvd"],["_nn"])]
    for ncells = ncellss
        rt2_min = Inf
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            fprex="$(prefix)$(SNR)db$(factor)f$(ncells)s$(initmethod)"
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; @show rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
            end
        end
        rt2_min = floor(rt2_min, digits=4)
        rng = range(0,stop=rt2_min,length=100)
        
        stat_nn=[]; stat_sp=[]; stat_sp_nn=[]
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            afs=[]
            fprex="$(prefix)$(SNR)db$(factor)f$(ncells)s$(initmethod)"
            for iter in 1:num_experiments
                @show tailstr, iter
                dd = load(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"))
                rt2s = dd["data"]["rt2s"]; avgfits = dd["data"]["avgfits"]
                lr = length(rt2s); la = length(avgfits)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; avgfits=avgfits[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
                push!(afs,itp(rng))
            end
            avgfits = hcat(afs...)
            means = dropdims(mean(avgfits,dims=2),dims=2)
            stds = dropdims(std(avgfits,dims=2),dims=2)
            tailstr == "_nn" && (push!(stat_nn,means); push!(stat_nn,stds))
            tailstr == "_sp" && (push!(stat_sp,means); push!(stat_sp,stds))
            tailstr == "_sp_nn" && (push!(stat_sp_nn,means); push!(stat_sp_nn,stds))
        end
        fprex="$(prefix)$(SNR)db$(factor)f$(ncells)s"
        save(joinpath(subworkpath,prefix,"$(fprex)_runtime_vs_avgfits.jld2"),"rng",rng, "stat_nn", stat_nn,
                        "stat_sp", stat_sp, "stat_sp_nn", stat_sp_nn)
    end
end

tmppath = ""
z = 0.5
for (idx,ncells) = enumerate(ncellss)
plottime = Inf
for (mtdstr, submtdstrs) in [("lcsvd",["_sp", "_sp_nn", "_nn"]),
                            ("lcsvd_LPF",["_sp", "_sp_nn", "_nn"]),
                            ("lcsvd_precon",["_sp", "_sp_nn", "_nn"]),
                            ("lcsvd_precon_LPF",["_sp", "_sp_nn", "_nn"]),
                            ("compnmf",["_nn"]),("hals",["_nn", "_sp_nn"])]
    @show mtdstr
    fprex="$(mtdstr)$(SNR)db$(factor)f$(ncells)s"
    ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(mtdstr)\",\"$(mtdstr)$(SNR)db$(factor)f$(ncells)s_runtime_vs_avgfits.jld2\"))"))
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

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon()

# compare LCSVD with different options (_sp)
fig = Figure(resolution=(400,300))
maxplottimes = [0.40,0.40,0.40]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), (0.6,1.0)), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("lcsvd","_sp","LCSVD",2,nothing),
                                                                ("lcsvd_LPF","_sp","LCSVD LPF",6,nothing),
                                                                ("lcsvd_precon","_sp","LCSVD predond",4,nothing),
                                                                ("lcsvd_precon_LPF","_sp","precond LPF",5,nothing)]) # all
    # eval(print("$(frpx)_means"))
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end
idx == 3 && axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits_sp_$(SNR)db$(factor)f$(ncells)s_all.png"),fig,px_per_unit=2)

# compare LCSVD with different options (_nn)
fig = Figure(resolution=(400,300))
maxplottimes = [0.40,0.40,0.40]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), (0.6,1.0)), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("lcsvd","_nn","LCSVD",2,nothing),
                                                                ("lcsvd_LPF","_nn","LCSVD LPF",6,nothing),
                                                                ("lcsvd_precon","_nn","LCSVD predond",4,nothing),
                                                                ("lcsvd_precon_LPF","_nn","precond LPF",5,nothing)]) # all
    # eval(print("$(frpx)_means"))
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

idx == 3 && axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits_nn_$(SNR)db$(factor)f$(ncells)s_all.png"),fig,px_per_unit=2)

# compare LCSVD with different options (_sp_nn)
fig = Figure(resolution=(400,300))
maxplottimes = [0.40,0.40,0.40]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), (0.6,1.0)), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("lcsvd","_sp_nn","LCSVD",2,nothing),
                                                                ("lcsvd_LPF","_sp_nn","LCSVD LPF",6,nothing),
                                                                ("lcsvd_precon","_sp_nn","LCSVD predond",4,nothing),
                                                                ("lcsvd_precon_LPF","_sp_nn","precond LPF",5,nothing)]) # all
    # eval(print("$(frpx)_means"))
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

idx == 3 && axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits_sp_nn_$(SNR)db$(factor)f$(ncells)s_all.png"),fig,px_per_unit=2)

# compare LCSVD with other methods
fig = Figure(resolution=(400,300))
maxplottimes = [0.40,0.40,0.40]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), (0.6,1.0)), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
# for (i,(frpx, lbl)) in enumerate([("sca_sp","SMF (α=100,β=0)"),("hals_nn","HALS (α=0)"),("hals_sp_nn","HALS (α=0.1)"),
#                                  ("admm_nn","Comp. NMF (α=0)"),("admm_sp","Comp. SMF (α=10)"),
#                                  ("sca_nn","SMF (α=0,β=1000)"),("sca_sp_nn","SMF (α=100,β=1000)"),("admm_sp_nn","Comp. NMF (α=10)"),])
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("lcsvd","_sp","SMF (α=0.005,β=0)",2,nothing),
                                                                  ("lcsvd","_nn","SMF (α=0,β=5.0)",6,:dashdot),
                                                                  ("lcsvd","_sp_nn","SMF (α=0.005,β=5.0)",4,:dash),
                                                                  ("compnmf","_nn","Compressed NMF",5,nothing),
                                                                  ("hals","_nn","HALS (α=0)",3,nothing),
                                                                  ("hals","_sp_nn","HALS (α=0.1)",7,:dash)]) # all
# for (i,(mtdstr, submtdstr, lbl, clridx)) in enumerate([("lcsvd","_sp","SMF (α=0.005,β=0)",2),("lcsvd","_sp_nn","SMF (α=0.005,β=5.0)",4),
#                                                        ("compnmf","_nn","Compressed NMF",5), ("hals","_sp_nn","HALS (α=0.1)",7)]) # selective
    # eval(print("$(frpx)_means"))
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

idx == 3 && axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db$(factor)f$(ncells)s_all.png"),fig,px_per_unit=2)

end # for ncellss



ncellss = [10]; num_experiments = 50; SNR=0
for (prefix, tailstrs) in [("lcsvd", ["_sp"])]
    for ncells = ncellss
        fprex="$(prefix)isvd$(SNR)db$(ncells)s"
        rt2_min = Inf
        for tailstr in tailstrs
            for iter in 1:num_experiments
                dd = load(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"data")
                rt2s = dd["rt2s"]; @show rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
            end
        end
        rt2_min = floor(rt2_min, digits=4)
        rng = range(0,stop=rt2_min,length=100)
        
        stat_nn=[]; stat_sp=[]; stat_sp_nn=[]
        for tailstr in tailstrs
            afs=[]
            for iter in 1:num_experiments
                @show tailstr, iter
                dd = load(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"))
                rt2s = dd["data"]["rt2s"]; avgfits = dd["data"]["avgfits"]
                lr = length(rt2s); la = length(avgfits)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; avgfits=avgfits[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
                push!(afs,itp(rng))
            end
            avgfits = hcat(afs...)
            means = dropdims(mean(avgfits,dims=2),dims=2)
            stds = dropdims(std(avgfits,dims=2),dims=2)
            tailstr == "_nn" && (push!(stat_nn,means); push!(stat_nn,stds))
            tailstr == "_sp" && (push!(stat_sp,means); push!(stat_sp,stds))
            tailstr == "_sp_nn" && (push!(stat_sp_nn,means); push!(stat_sp_nn,stds))
        end
        save(joinpath(subworkpath,prefix,"$(fprex)_runtime_vs_avgfits.jld2"),"rng",rng,"stat_sp", stat_sp)
    end
end


z = 0.5
ncellss = [10]; SNR=0
for (idx,ncells) = enumerate(ncellss)
plottime = Inf
for (mtdstr, submtdstrs) in [("lcsvd",["_sp"]),("lcsvdisvd",["_sp"])]
    @show mtdstr
    ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,\"$(mtdstr)\",\"$(mtdstr)$(SNR)db$(ncells)s_runtime_vs_avgfits.jld2\"))"))
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

plotrng = Colon()
fig = Figure(resolution=(400,300))
maxplottimes = [0.09]
ax = AMakie.Axis(fig[1, 1], limits = ((0,min(maxplottimes[idx],plottime)), nothing), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
# for (i,(frpx, lbl)) in enumerate([("sca_sp","SMF (α=100,β=0)"),("hals_nn","HALS (α=0)"),("hals_sp_nn","HALS (α=0.1)"),
#                                  ("admm_nn","Comp. NMF (α=0)"),("admm_sp","Comp. SMF (α=10)"),
#                                  ("sca_nn","SMF (α=0,β=1000)"),("sca_sp_nn","SMF (α=100,β=1000)"),("admm_sp_nn","Comp. NMF (α=10)"),])
for (i,(mtdstr, submtdstr, lbl, clridx)) in enumerate([("lcsvd","_sp","SMF NNDSVD (α=0.005,β=0)",2),("lcsvdisvd","_sp","SMF ISVD (α=0.005,β=0)",7)]) # all
# for (i,(mtdstr, submtdstr, lbl, clridx)) in enumerate([("lcsvd","_sp","SMF (α=0.005,β=0)",2),("lcsvd","_sp_nn","SMF (α=0.005,β=5.0)",4),
#                                                        ("compnmf","_nn","Compressed NMF",5), ("hals","_sp_nn","HALS (α=0.1)",7)]) # selective
    # eval(print("$(frpx)_means"))
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db$(ncells)s_compare_initmethod.png"),fig,px_per_unit=2)

end # for factor


# check the each experiment
for i in 1:num_experiments
    dd1 = load(joinpath(subworkpath,"lcsvd","lcsvd-10db15sisvd_sp_results$(i).jld2"))
    @show dd1["data"]["avgfits"][end]
end
dd2 = load(joinpath(subworkpath,"lcsvd","lcsvd-10db15s_runtime_vs_avgfits.jld2"))