using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","linesearch")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
using Interpolations

SNRs = [-10, 0, 10]; num_experiments = 30; ncells=15; factor=1

for (prefix, initmethods, tailstrs) in [("backtracking", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("morethuente", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("hagerzhang", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("none", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"])]
    for SNR = SNRs
        rt2_min = Inf
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            fprex="$(prefix)$(SNR)db$(factor)f$(ncells)s$(initmethod)"
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; # @show rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
            end
        end
        rt2_min = floor(rt2_min, digits=4)
        rng = range(0,stop=rt2_min,length=100)
        # @show prefix, tailstrs, SNR, rt2_min

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
z = 0.5; ylimits=(0.5,1.0)
for (idx,SNR) = enumerate(SNRs)
plottime = Inf
for (mtdstr, submtdstrs) in [("backtracking",["_sp", "_sp_nn", "_nn"]),
                            ("morethuente",["_sp", "_sp_nn", "_nn"]),
                            ("hagerzhang",["_sp", "_sp_nn", "_nn"]),
                            ("none",["_sp", "_sp_nn", "_nn"])]
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
position = :rb #SNR < 0 ? :lt : :rb
# compare LCSVD with other methods
fig = Figure(resolution=(400,300))
maxplottimes = [0.2,0.2,0.2]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), ylimits), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("backtracking","_sp","BackTracking",2,nothing),
                                                                  ("morethuente","_sp","MoreThuente",3,:dashdot),
                                                                  ("hagerzhang","_sp","HagerZhang",5,:dashdot)])
                                                                  # ("none","_sp","No LineSearch",6,:dash)])
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

#axislegend(ax, labelsize=10, position = position) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db$(factor)f$(ncells)s_sp_$(maxplottimes[idx]).png"),fig,px_per_unit=2)

fig = Figure(resolution=(400,300))
maxplottimes = [0.2,0.2,0.2]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), ylimits), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("backtracking","_nn","BackTracking",2,nothing),
                                                                  ("morethuente","_nn","MoreThuente",3,:dashdot),
                                                                  ("hagerzhang","_nn","HagerZhang",5,:dashdot)])
                                                                  # ("none","_nn","No LineSearch",6,:dash)])
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

#axislegend(ax, labelsize=10, position = position) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db$(factor)f$(ncells)s_nn_$(maxplottimes[idx]).png"),fig,px_per_unit=2)

fig = Figure(resolution=(400,300))
maxplottimes = [0.2,0.2,0.2]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), ylimits), xlabel = "time(sec)", ylabel = "average fit")#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("backtracking","_sp_nn","BackTracking",2,nothing),
                                                                  ("morethuente","_sp_nn","MoreThuente",3,:dashdot),
                                                                  ("hagerzhang","_sp_nn","HagerZhang",5,:dashdot)])
                                                                  # ("none","_sp_nn","No LineSearch",6,:dash)])
    frpx = "$(mtdstr)$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, labelsize=10, position = position) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db$(factor)f$(ncells)s_sp_nn_$(maxplottimes[idx]).png"),fig,px_per_unit=2)

end # for SNR
