using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","audio")

z = 0.5

#================================= plot fixed alpha ===================#
for mtdstr in ["sca","admm","hals"]
    ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld"))) # this doens't work 'mtdstr' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,\"$(mtdstr)_cbcl_runtime_vs_fits.jld\"))"))
    @eval (rng=($(ddsym)["rng"]))
    submtdstrs = mtdstr == "sca" ? ["_sp"] : ["_sp_nn"]
    for submtdstr in submtdstrs
        frpx = "$(mtdstr)$(submtdstr)"; dickeystr = "stat$(submtdstr)"
        # @eval ($(Symbol("$(frpx)_means")) = ($(ddsym)["stat$(submtdstr)"][1]))
        # @eval ($(Symbol("$(frpx)_stds")) = ($(ddsym)["stat$(submtdstr)"][2]))
        eval(Meta.parse("$(frpx)1_means=$(ddstr)[\"stat$(submtdstr)1\"][1]"))
        eval(Meta.parse("$(frpx)1_stds=$(ddstr)[\"stat$(submtdstr)1\"][2]"))
        @eval ($(Symbol("$(frpx)1_upper")) = ($(Symbol("$(frpx)1_means")) + z*$(Symbol("$(frpx)1_stds"))))
        @eval ($(Symbol("$(frpx)1_lower")) = ($(Symbol("$(frpx)1_means")) - z*$(Symbol("$(frpx)1_stds"))))
        eval(Meta.parse("$(frpx)2_means=$(ddstr)[\"stat$(submtdstr)2\"][1]"))
        eval(Meta.parse("$(frpx)2_stds=$(ddstr)[\"stat$(submtdstr)2\"][2]"))
        @eval ($(Symbol("$(frpx)2_upper")) = ($(Symbol("$(frpx)2_means")) + z*$(Symbol("$(frpx)2_stds"))))
        @eval ($(Symbol("$(frpx)2_lower")) = ($(Symbol("$(frpx)2_means")) - z*$(Symbol("$(frpx)2_stds"))))
    end
end

# ddhals = load("hals_runtime_vs_avgfits.jld"); rng = ddhals["rng"]
# hals_nn_means = ddhals["stat_nn"][1]; hals_nn_stds = ddhals["stat_nn"][2]
# hals_sp_nn_means = ddhals["stat_sp_nn"][1]; hals_sp_nn_stds = ddhals["stat_sp_nn"][2]
# hals_nn_upper = hals_nn_means + z*hals_nn_stds; hals_nn_lower = hals_nn_means - z*hals_nn_stds
# hals_sp_nn_upper = hals_sp_nn_means + z*hals_sp_nn_stds; hals_sp_nn_lower = hals_sp_nn_means - z*hals_sp_nn_stds

# ddsca = load("sca_runtime_vs_avgfits.jld"); rng = ddsca["rng"]
# sca_nn_means = ddsca["stat_nn"][1]; sca_nn_stds = ddsca["stat_nn"][2]
# sca_sp_means = ddsca["stat_sp"][1]; sca_sp_stds = ddsca["stat_sp"][2]
# sca_sp_nn_means = ddsca["stat_sp_nn"][1]; sca_sp_nn_stds = ddsca["stat_sp_nn"][2]
# sca_nn_upper = sca_nn_means + z*sca_nn_stds; sca_nn_lower = sca_nn_means - z*sca_nn_stds
# sca_sp_upper = sca_sp_means + z*sca_sp_stds; sca_sp_lower = sca_sp_means - z*sca_sp_stds
# sca_sp_nn_upper = sca_sp_nn_means + z*sca_sp_nn_stds; sca_sp_nn_lower = sca_sp_nn_means - z*sca_sp_nn_stds

# ddadmm = load("admm_runtime_vs_avgfits.jld"); rng = ddadmm["rng"]
# admm_nn_means = ddadmm["stat_nn"][1]; admm_nn_stds = ddadmm["stat_nn"][2]
# admm_sp_means = ddadmm["stat_sp"][1]; admm_sp_stds = ddadmm["stat_sp"][2]
# admm_sp_nn_means = ddadmm["stat_sp_nn"][1]; admm_sp_nn_stds = ddadmm["stat_sp_nn"][2]
# admm_nn_upper = admm_nn_means + z*admm_nn_stds; admm_nn_lower = admm_nn_means - z*admm_nn_stds
# admm_sp_upper = admm_sp_means + z*admm_sp_stds; admm_sp_lower = admm_sp_means - z*admm_sp_stds
# admm_sp_nn_upper = admm_sp_nn_means + z*admm_sp_nn_stds; admm_sp_nn_lower = admm_sp_nn_means - z*admm_sp_nn_stds

mtdcolors = [RGB{N0f8}(0.00,0.00,0.00),RGB{N0f8}(0.00,0.45,0.70),RGB{N0f8}(0.90,0.62,0.00),
             RGB{N0f8}(0.00,0.62,0.45),RGB{N0f8}(0.80,0.47,0.65),RGB{N0f8}(0.34,0.71,0.91),
             RGB{N0f8}(0.84,0.37,0.00),RGB{N0f8}(0.94,0.89,0.26)]

alpha = 0.2; mtdcoloras = convert.(RGBA,mtdcolors,alpha)

fig = Figure()
ax1 = AMakie.Axis(fig[1, 1], xlabel = "time(sec)", ylabel = "fit", title = "Average Fit Value vs. Running Time")
ax2 = AMakie.Axis(fig[1, 1], yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
hidespines!(ax2)
hidexdecorations!(ax2)

lns = Dict(); bnds=Dict()
for (i,(frpx, lbl)) in enumerate([("sca_sp","SMF (α=100)"),("admm_sp_nn","Comp. NMF (α=10)"),("hals_sp_nn","HALS (α=0.1)")])
    # eval(print("$(frpx)_means"))
    ln1 = lines!(ax1, rng, eval(Symbol("$(frpx)1_means")), color=mtdcolors[i+1], label=lbl)
    bnd1 = band!(ax1, rng, eval(Symbol("$(frpx)1_lower")), eval(Symbol("$(frpx)1_upper")), color=mtdcoloras[i+1])
    ln2 = lines!(ax2, rng, eval(Symbol("$(frpx)2_means")), color=mtdcolors[i+1], label=lbl, linestyle = :dash, linewidth = 2)
    bnd2 = band!(ax2, rng, eval(Symbol("$(frpx)2_lower")), eval(Symbol("$(frpx)2_upper")), color=mtdcoloras[i+1])
    lns["$(frpx)_line1"] = ln1; bnds["$(frpx)_band1"] = bnd1;
    lns["$(frpx)_line2"] = ln1; bnds["$(frpx)_band2"] = bnd1;
end

axislegend(ax1, position = :rt) # halign = :left, valign = :top
axislegend(ax2, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"cbclface_fits.png"),fig,px_per_unit=2)


# # hals nn
# lhnn=lines!(ax, rng, hals_nn_means, color=cls[1], label="HALS (α=0)")
# bhnn=band!(ax, rng, hals_nn_lower, hals_nn_upper, color=RGBA(0,0,1,0.2))
# # delete!(ax,lhnn); delete!(ax,bhnn) # or delete!.(ax,[lhnn,bhnn])
# # hals sp nn
# lines!(ax, rng, hals_sp_nn_means, color=:green, label="HALS (α=0.1)")
# band!(ax, rng, hals_sp_nn_lower, hals_sp_nn_upper, color=RGBA(0,1,0,0.2))

# # sca nn
# lines!(ax, rng, sca_nn_means, color=:red, label="SMF (α=0,β=1000)")
# band!(ax, rng, sca_nn_lower, sca_nn_upper, color=RGBA(1,0,0,0.2))
# # sca sp
# lines!(ax, rng, sca_sp_means, color=:magenta, label="SMF (α=100,β=0)")
# band!(ax, rng, sca_sp_lower, sca_sp_upper, color=RGBA(1,0,1,0.2))
# # sca sp nn
# lines!(ax, rng, sca_sp_nn_means, color=:magenta, label="SMF (α=100,β=1000)")
# band!(ax, rng, sca_sp_nn_lower, sca_sp_nn_upper, color=RGBA(1,0,1,0.2))

# # admm nn
# color = (0.5,0.5,0)
# lines!(ax, rng, admm_nn_means, color=RGBA(color...,1), label="Comp. NMF (α=0)")
# band!(ax, rng, admm_nn_lower, admm_nn_upper, color=RGBA(color...,0.2))
# # admm sp nn
# color = (1,0.5,0.5)
# lines!(ax, rng, admm_sp_nn_means, color=RGBA(color...,1), label="Comp. NMF (α=10)")
# band!(ax, rng, admm_sp_nn_lower, admm_sp_nn_upper, color=RGBA(color...,0.2))
# # admm sp
# color = (0.7,0.1,0.5)
# lines!(ax, rng, admm_sp_means, color=RGBA(color...,1), label="Comp. SMF (α=10)")
# band!(ax, rng, admm_sp_lower, admm_sp_upper, color=RGBA(color...,0.2))



#================================= plot ranged alpha ===================#
for mtdstr in ["sca","admm","hals"]
    ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld"))) # this doens't work 'mtdstr' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,\"$(mtdstr)_cbcl_alpha_runtime_vs_fits.jld\"))"))
    @eval (rng=($(ddsym)["rng"]))
    submtdstrs = mtdstr == "sca" ? ["_sp"] : ["_sp_nn"]
    for submtdstr in submtdstrs
        frpx = "$(mtdstr)$(submtdstr)"; dickeystr = "stat$(submtdstr)"
        # @eval ($(Symbol("$(frpx)_means")) = ($(ddsym)["stat$(submtdstr)"][1]))
        # @eval ($(Symbol("$(frpx)_stds")) = ($(ddsym)["stat$(submtdstr)"][2]))
        eval(Meta.parse("$(frpx)1_means=$(ddstr)[\"stat$(submtdstr)1\"][1]"))
        eval(Meta.parse("$(frpx)1_stds=$(ddstr)[\"stat$(submtdstr)1\"][2]"))
        @eval ($(Symbol("$(frpx)1_upper")) = ($(Symbol("$(frpx)1_means")) + z*$(Symbol("$(frpx)1_stds"))))
        @eval ($(Symbol("$(frpx)1_lower")) = ($(Symbol("$(frpx)1_means")) - z*$(Symbol("$(frpx)1_stds"))))
        eval(Meta.parse("$(frpx)2_means=$(ddstr)[\"stat$(submtdstr)2\"][1]"))
        eval(Meta.parse("$(frpx)2_stds=$(ddstr)[\"stat$(submtdstr)2\"][2]"))
        @eval ($(Symbol("$(frpx)2_upper")) = ($(Symbol("$(frpx)2_means")) + z*$(Symbol("$(frpx)2_stds"))))
        @eval ($(Symbol("$(frpx)2_lower")) = ($(Symbol("$(frpx)2_means")) - z*$(Symbol("$(frpx)2_stds"))))
    end
end

# mtdcolors = [RGB{N0f8}(0.00,0.00,0.00),RGB{N0f8}(0.00,0.45,0.70),RGB{N0f8}(0.90,0.62,0.00),
#              RGB{N0f8}(0.00,0.62,0.45),RGB{N0f8}(0.80,0.47,0.65),RGB{N0f8}(0.34,0.71,0.91),
#              RGB{N0f8}(0.84,0.37,0.00),RGB{N0f8}(0.94,0.89,0.26)]
mtdcolors = [RGB{N0f8}(0.00,0.00,0.00),RGB{N0f8}(0.00,0.45,0.70),RGB{N0f8}(0.90,0.62,0.00),
             RGB{N0f8}(0.00,0.62,0.45),RGB{N0f8}(0.30,0.35,0.60),RGB{N0f8}(0.80,0.52,0.30),
             RGB{N0f8}(0.30,0.52,0.35),RGB{N0f8}(0.94,0.89,0.26)]

alpha = 0.2; mtdcoloras = convert.(RGBA,mtdcolors,alpha)

fig = Figure(resolution = (800,400))
ax11_1 = AMakie.Axis(fig[1, 1], xlabel = "time(sec)", ylabel = "fit", title = "Fit and Sparsity Values vs. Running Time")
ax11_2 = AMakie.Axis(fig[1, 1], yaxisposition = :right, ylabel = "sparsity of W" #= yticklabelcolor = :red =# )
hidespines!(ax11_2)
hidexdecorations!(ax11_2)

ln1s = OrderedDict(); ln2s = OrderedDict(); bnd1s=OrderedDict(); bnd2s=OrderedDict(); lbls=OrderedDict{String,String}()
for (i,(frpx, lbl)) in enumerate([("sca_sp","SMF (α∈[1,1000])"),("admm_sp_nn","Comp. NMF (α∈[1,100])"),("hals_sp_nn","HALS (α∈[0,5])")])
    # eval(print("$(frpx)_means"))
    ln1 = lines!(ax11_1, rng, eval(Symbol("$(frpx)1_means")), color=mtdcolors[i+1], label=lbl)
    bnd1 = band!(ax11_1, rng, eval(Symbol("$(frpx)1_lower")), eval(Symbol("$(frpx)1_upper")), color=mtdcoloras[i+1])
    ln2 = lines!(ax11_2, rng, eval(Symbol("$(frpx)2_means")), color=mtdcolors[i+4], label=lbl, linestyle = :dash, linewidth = 2)
    bnd2 = band!(ax11_2, rng, eval(Symbol("$(frpx)2_lower")), eval(Symbol("$(frpx)2_upper")), color=mtdcoloras[i+4])
    ln1s["$(frpx)_line1"] = ln1; bnd1s["$(frpx)_band1"] = bnd1;
    ln2s["$(frpx)_line2"] = ln2; bnd2s["$(frpx)_band2"] = bnd2;
    lbls["$(frpx)_label"] = lbl
end

fig[1,2] = Legend(fig,[collect(values(ln1s)),collect(values(ln2s))],[collect(values(lbls)),collect(values(lbls))],["Fit", "Sparsity of W"])
# axislegend(ax1, position = :rt) # halign = :left, valign = :top
# axislegend(ax2, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"cbclface_alpha_fits.png"),fig,px_per_unit=2)







#================================= admm initialization test ===================#
ddadmmw1 = load(joinpath(subworkpath,"admm_cbcl_alpha_w1_runtime_vs_fits.jld")); rng = ddadmmw1["rng"]
admmw1_sp_nn1_means = ddadmmw1["stat_sp_nn1"][1]; admmw1_sp_nn1_stds = ddadmmw1["stat_sp_nn1"][2]
admmw1_sp_nn1_upper = admmw1_sp_nn1_means + z*admmw1_sp_nn1_stds; admmw1_sp_nn1_lower = admmw1_sp_nn1_means - z*admmw1_sp_nn1_stds
admmw1_sp_nn2_means = ddadmmw1["stat_sp_nn2"][1]; admmw1_sp_nn2_stds = ddadmmw1["stat_sp_nn2"][2]
admmw1_sp_nn2_upper = admmw1_sp_nn2_means + z*admmw1_sp_nn2_stds; admmw1_sp_nn2_lower = admmw1_sp_nn2_means - z*admmw1_sp_nn2_stds

ddadmmw8 = load(joinpath(subworkpath,"admm_cbcl_alpha_w8_runtime_vs_fits.jld")); rng = ddadmmw8["rng"]
admmw8_sp_nn1_means = ddadmmw8["stat_sp_nn1"][1]; admmw8_sp_nn1_stds = ddadmmw8["stat_sp_nn1"][2]
admmw8_sp_nn1_upper = admmw8_sp_nn1_means + z*admmw8_sp_nn1_stds; admmw8_sp_nn1_lower = admmw8_sp_nn1_means - z*admmw8_sp_nn1_stds
admmw8_sp_nn2_means = ddadmmw8["stat_sp_nn2"][1]; admmw8_sp_nn2_stds = ddadmmw8["stat_sp_nn2"][2]
admmw8_sp_nn2_upper = admmw8_sp_nn2_means + z*admmw8_sp_nn2_stds; admmw8_sp_nn2_lower = admmw8_sp_nn2_means - z*admmw8_sp_nn2_stds

fig = Figure(resolution = (800,400))
ax11_1 = AMakie.Axis(fig[1, 1], xlabel = "time(sec)", ylabel = "fit", title = "Fit and Sparsity Values vs. Running Time")
ax11_2 = AMakie.Axis(fig[1, 1], yaxisposition = :right, ylabel = "Sparsity of W" #= yticklabelcolor = :red =# )
hidespines!(ax11_2)
hidexdecorations!(ax11_2)

ln1s = OrderedDict(); ln2s = OrderedDict(); bnd1s=OrderedDict(); bnd2s=OrderedDict(); lbls=OrderedDict{String,String}()
for (i,(frpx, lbl)) in enumerate([("admmw1_sp_nn","Comp. NMF (w=1,α∈[1,100])"),("admm_sp_nn","Comp. NMF (w=4,α∈[1,100])"),("admmw8_sp_nn","Comp. NMF (w=8,α∈[1,100])")])
    # eval(print("$(frpx)_means"))
    ln1 = lines!(ax11_1, rng, eval(Symbol("$(frpx)1_means")), color=mtdcolors[i+1], label=lbl)
    bnd1 = band!(ax11_1, rng, eval(Symbol("$(frpx)1_lower")), eval(Symbol("$(frpx)1_upper")), color=mtdcoloras[i+1])
    ln2 = lines!(ax11_2, rng, eval(Symbol("$(frpx)2_means")), color=mtdcolors[i+4], label=lbl, linestyle = :dash, linewidth = 2)
    bnd2 = band!(ax11_2, rng, eval(Symbol("$(frpx)2_lower")), eval(Symbol("$(frpx)2_upper")), color=mtdcoloras[i+4])
    ln1s["$(frpx)_line1"] = ln1; bnd1s["$(frpx)_band1"] = bnd1;
    ln2s["$(frpx)_line2"] = ln2; bnd2s["$(frpx)_band2"] = bnd2;
    lbls["$(frpx)_label"] = lbl
end

fig[1,2] = Legend(fig,[collect(values(ln1s)),collect(values(ln2s))],[collect(values(lbls)),collect(values(lbls))],["Fit", "Sparsity of W"])
save(joinpath(subworkpath,"cbclface_admm_w_fits.png"),fig,px_per_unit=2)


# # admm sp nn
# color = (1,0.5,0.5)
# lines!(ax, rng, admm_sp_nn_means, color=RGBA(color...,1), label="Comp. NMF (α=10)")
# band!(ax, rng, admm_sp_nn_lower, admm_sp_nn_upper, color=RGBA(color...,0.2))
# # admm sp


