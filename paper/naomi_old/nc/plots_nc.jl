using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","naomi/nc")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
using Interpolations

pavg = 5.0; num_experiments = 50; ncellss=[10,20,40,50,500]; factor=1
itp_time_resol = 1000
for (method, initmethods, tailstrs) in [("pcb_tsvd", ["tsvd","tsvd"], ["_sp","_sp_nn"]),
                # ("pcb_LPF", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                # ("pcb_precon", ["isvd","isvd"], ["_sp","_sp_nn"]),
                # ("pcb_precon_LPF", ["isvd","isvd", "nndsvd"], ["_sp","_sp_nn", "_nn"]),
                ("hals",["nndsvd","nndsvd"],["_nn","_sp_nn"]),("compnmf",["lowrank_nndsvd"],["_nn"])]
    for ncells in ncellss
        rt2_min = Inf
        fprex = "$(method)$(pavg)mW$(factor)f$(ncells)s"
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,method,"$(fprex)$(initmethod)$(tailstr)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]
                rt2_min = min(rt2_min,rt2s[end])
            end
        end
        rt2_min = floor(rt2_min, digits=4); @show ncells, method, rt2_min
        rng = range(0,stop=rt2_min,length=itp_time_resol)
        
        stat_nn=[]; stat_sp=[]; stat_sp_nn=[]
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            afs=[]
            for iter in 1:num_experiments
                # @show tailstr, iter
                dd = load(joinpath(subworkpath,method,"$(fprex)$(initmethod)$(tailstr)_results$(iter).jld2"))
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
        save(joinpath(subworkpath,method,"$(fprex)_runtime_vs_avgfits.jld2"),"rng",rng, "stat_nn", stat_nn,
                        "stat_sp", stat_sp, "stat_sp_nn", stat_sp_nn)
    end
end

tmppath = ""
z = 0.5; ylimits=(0.48,1.01); resol=(800,600); fntsize1 = 30; fntsize2 = 30
for (idx, ncells) = enumerate(ncellss)
plottime = Inf
for (method, submethods) in [("pcb_tsvd",["_sp", "_sp_nn"]),
                            #("pcb_LPF",["_sp", "_sp_nn", "_nn"]),
                            #("pcb_precon",["_sp", "_sp_nn"]),
                            #("pcb_precon_LPF",["_sp", "_sp_nn", "_nn"]),
                            ("compnmf",["_nn"]),("hals",["_nn", "_sp_nn"])]
    @show method
    fprex="$(method)$(pavg)mW$(factor)f$(ncells)s"
    ddstr = "dd$(method)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(method)_runtime_vs_avgfits.jld2"))) # this doens't work 'method' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(method)\",\"$(method)$(pavg)mW$(factor)f$(ncells)s_runtime_vs_avgfits.jld2\"))"))
    rng = eval(Meta.parse("$(ddsym)[\"rng\"]"))
    eval(Meta.parse("$(method)rng=$(ddsym)[\"rng\"]"))
    plottime = plottime > rng[end] ? rng[end] : plottime
    for submethod in submethods
        frpx = "$(method)$(submethod)"; dickeystr = "stat$(submethod)"
        # @eval ($(Symbol("$(frpx)_means")) = ($(ddsym)["stat$(submethod)"][1]))
        # @eval ($(Symbol("$(frpx)_stds")) = ($(ddsym)["stat$(submethod)"][2]))
        eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat$(submethod)\"][1]"))
        eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat$(submethod)\"][2]"))
        @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + z*$(Symbol("$(frpx)_stds"))))
        @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - z*$(Symbol("$(frpx)_stds"))))
    end
end

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon()

# compare PCB with other methods
fig = Figure(size=resol)
maxplottimes = [0.08, 0.2, 0.3, 0.4, 50.0]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), ylimits),
                xlabel = "time(sec)", ylabel = "weighted average fit", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2)#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
# for (i,(frpx, lbl)) in enumerate([("sca_sp","PCB (α=100,β=0)"),("hals_nn","HALS (α=0)"),("hals_sp_nn","HALS (α=0.1)"),
#                                  ("admm_nn","Comp. NMF (α=0)"),("admm_sp","Comp. PCB (α=10)"),
#                                  ("sca_nn","PCB (α=0,β=1000)"),("sca_sp_nn","PCB (α=100,β=1000)"),("admm_sp_nn","Comp. NMF (α=10)"),])
for (i,(method, submethod, lbl, clridx, linestyle)) in enumerate([("pcb_tsvd","_sp","PCB (α=0.005,β=0)",2,nothing),
                                                                  ("pcb_tsvd","_sp_nn","PCB (α=0.005,β=5.0)",4,:dash),
#                                                                  ("pcb_LPF","_sp_nn","PCB LPF (α=0.005,β=5.0)",6,:dashdot),
                                                                  ("compnmf","_nn","Compressed NMF",5,nothing),
                                                                  ("hals","_nn","HALS (α=0)",3,nothing),
                                                                  ("hals","_sp_nn","HALS (α=0.1)",7,:dash)]) # all
# for (i,(method, submethod, lbl, clridx)) in enumerate([("pcb","_sp","PCB (α=0.005,β=0)",2),("pcb","_sp_nn","PCB (α=0.005,β=5.0)",4),
#                                                        ("compnmf","_nn","Compressed NMF",5), ("hals","_sp_nn","HALS (α=0.1)",7)]) # selective
    # eval(print("$(frpx)_means"))
    frpx = "$(method)$(submethod)"
    ln = lines!(ax, eval(Symbol("$(method)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(method)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, labelsize=fntsize1, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(pavg)mW$(factor)f$(ncells)s_all.png"),fig,px_per_unit=2)

end # for pavgs
