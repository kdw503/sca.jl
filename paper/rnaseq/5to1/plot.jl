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

# sizestep, pcb_maxiter, pcb_inner_maxiter, tol, per_component = 5, 1, 1, 1e-6, false
sizestep = eval(Meta.parse(ARGS[1]))
pcb_maxiter = eval(Meta.parse(ARGS[2]))
pcb_inner_maxiter = eval(Meta.parse(ARGS[3]))
tol = eval(Meta.parse(ARGS[4]))
per_component = eval(Meta.parse(ARGS[5]))
memsize = Int(Sys.total_memory())/1e9
per_com_str = per_component ? "cw" : ""
subworkpath = joinpath(workpath,"paper","rnaseq","$(sizestep)to1")

#colormap
# Makie.available_gradients()
# Plasma, Inferno, Magma, Cividis, Jet, grays, heat, :Spectral
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# #======== Load Data ===========#
# dataset = :rnaseq
# #X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
# feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)
# file_name = feature_name*"-raw"
# adata = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad"))
# Xraw = adata.X # cell_label(adata.obs_names), gene_identifier(adata.var_names)

noc = 500; nac = 0; nc = noc+nac
prefix = "pcb"
r = 0.3
inner_tol = tol; inner_maxiter = pcb_inner_maxiter
optim_method = :lbfgs; smaxiter = 500
maxiter = pcb_maxiter == 0 ? Int(ceil(log(eps(eltype(X)))/log(r))) : pcb_maxiter
jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb_noc$(noc)_a$(α)_b$(β)_it$(rst0.niters)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")

jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb_noc$(noc)_βrng"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "βrng", βrng,"fitvals",fitvals,"sws",sws ,"per_component", per_component, "optim_method", optim_method, "α", α, "r",r,"tol",tol)

#======== both sparsity and nonnegativity ============#
# αrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_αrng.jld2")
dd = load(fname)
αrng = dd["αrng"]; fitvals = dd["fitvals"]; nsws = dd["nsws"]; β = dd["β"]; noc=500

labels = ["fitval","norm(sw,1)/noc"]
fig = Figure(resolution = (500,300))
ax1 = AMakie.Axis(fig[1, 1], limits=(nothing, (0.924,0.927) ), xlabel = "α", ylabel = "fit", title = "Fit Value vs. α")
ax2 = AMakie.Axis(fig[1, 1], limits=(nothing, (21,24) ), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, αrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, αrng, nsws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_αrng_noc$(noc)_b$(β).png"),fig)

# βrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_βrng.jld2")
dd = load(fname)
βrng = dd["βrng"]; fitvals = dd["fitvals"]; nsws = dd["nsws"]; α = dd["α"]; noc=500

labels = ["fitval","norm(W,1)/noc"]
fig = Figure(resolution = (500,300))
fitymin = floor(minimum(fitvals)*1000)/1000
fitymax = ceil(maximum(fitvals)*1000)/1000
nswymin = floor(minimum(nsws))
nswymax = ceil(maximum(nsws))
βmin = minimum(βrng)
βmax = maximum(βrng)
βmax = 50
ax1 = AMakie.Axis(fig[1, 1], limits=((βmin,βmax), (fitymin,fitymax)), xlabel = "noc", ylabel = "fit", title = "Fit Value vs. noc")
ax2 = AMakie.Axis(fig[1, 1], limits=((βmin,βmax), (nswymin,nswymax)), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, βrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, βrng, nsws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_βrng_noc$(noc)_a$(α).png"),fig)

# nocrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_nocrng700_nsw.jld2")
dd = load(fname)
nocrng = dd["nocrng"]; fitvals = dd["fitvals"]; nsws = dd["nsws"]; β = dd["β"]; α = dd["α"]

# fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_noc300_a0.005_b3.0_it10.jld2")
# save(fname, "nocrng", nocrng,"fitvals",fitvals,"sws",sws ,"per_component", per_component, "optim_method", optim_method, "α", α, "β", β, "r",r,"tol",tol)

nocrngfnames = [(300,"WMB-10Xv2-HY-rawt_ss5_pcb_noc300_a0.005_b3.0_it10.jld2"),
                (400,"WMB-10Xv2-HY-rawt_ss5_pcb_noc400_a0.005_b3.0_it9.jld2"),
                (500,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_a0.005_b3.0_it9.jld2"),
                (600,"WMB-10Xv2-HY-rawt_ss5_pcb_noc600_a0.005_b3.0_it8.jld2"),
                (700,"WMB-10Xv2-HY-rawt_ss5_pcb_noc700_a0.005_b3.0_it7.jld2")]
nocrng = Int[]; fitvals = Float64[]; nsws = Float64[]
for (noc,jldfname) in nocrngfnames
    fname = joinpath(subworkpath,jldfname)
    dd = load(fname)
    push!(nocrng,noc)
    push!(fitvals,dd["fitval"])
    W = dd["W"]; H = dd["H"]
    d=LCSVD.normalizeWH!(W,H)
    nsw = norm(W,1)/noc
    push!(nsws,nsw)
end
β = 3.0; α = 0.005

labels = ["fitval","norm(sw,1)/noc"]
fig = Figure(resolution = (500,300))
ax1 = AMakie.Axis(fig[1, 1], limits=((300,700), (0.91,0.95)), xlabel = "noc", ylabel = "fit", title = "Fit Value vs. noc")
ax2 = AMakie.Axis(fig[1, 1], limits=((300,700), (19.,24.)), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, nocrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, nocrng, nsws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_nocrng_a$(α)_b$(β)_nsw_700.png"),fig)

optim_method = :lbfgs; per_component = false; r = 3.0; tol = 1e-6
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_nocrng700_nsw.jld2")
save(fname, "nocrng", nocrng,"fitvals",fitvals,"nsws",nsws ,"per_component", per_component, "optim_method", optim_method, "α", α, "β", β, "r",r,"tol",tol)


# nocrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_nocrng1000.jld2")
dd = load(fname)
nocrng = dd["nocrng"]; fitvals = dd["fitvals"]; sws = dd["sws"]; β = dd["β"]; α = dd["α"]

nocrngfnames = [(300,"WMB-10Xv2-HY-rawt_ss5_pcb_noc300_a0.005_b3.0_it10.jld2"),
                (400,"WMB-10Xv2-HY-rawt_ss5_pcb_noc400_a0.005_b3.0_it9.jld2"),
                (500,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_a0.005_b3.0_it9.jld2"),
                (600,"WMB-10Xv2-HY-rawt_ss5_pcb_noc600_a0.005_b3.0_it8.jld2"),
                (700,"WMB-10Xv2-HY-rawt_ss5_pcb_noc700_a0.005_b3.0_it7.jld2")#=,
                (800,"WMB-10Xv2-HY-rawt_ss5_pcb_noc800_a0.005_b3.0_it11.jld2"),
                (900,"WMB-10Xv2-HY-rawt_ss5_pcb_noc900_a0.005_b3.0_it10.jld2"),
                (1000,"WMB-10Xv2-HY-rawt_ss5_pcb_noc1000_a0.005_b3.0_it11.jld2")=#]
nocrng = Int[]; fitvals = Float64[]; nsws = Float64[]
for (noc,jldfname) in nocrngfnames
    fname = joinpath(subworkpath,jldfname)
    dd = load(fname)
    push!(fitvals,dd["fitval"])
    W = dd["W"]; H = dd["H"]
    d=LCSVD.normalizeWH!(W,H)
    nsw = norm(W,1)/noc
    push!(nsws,nsw)
    push!(nocrng,noc)
end
β = 3.0; α = 0.005; r=0.3; tol = 1e-6

labels = ["fitval","norm(W,1)/noc"]
fig = Figure(resolution = (500,300))
fitymin = floor(minimum(fitvals)*1000)/1000
fitymax = ceil(maximum(fitvals)*1000)/1000
nswymin = floor(minimum(nsws))
nswymax = ceil(maximum(nsws))
nocmin = minimum(nocrng)
nocmax = maximum(nocrng)
ax1 = AMakie.Axis(fig[1, 1], limits=((nocmin,nocmax), (fitymin,fitymax)), xlabel = "noc", ylabel = "fit", title = "Fit Value vs. noc")
ax2 = AMakie.Axis(fig[1, 1], limits=((nocmin,nocmax), (nswymin,nswymax)), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, nocrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, nocrng, nsws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_nocrng_a$(α)_b$(β)_nsw_700.png"),fig)

fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_nocrng700_nsw.jld2")
save(fname, "nocrng", nocrng,"fitvals",fitvals,"sws",sws ,"per_component", false, "optim_method", :lbfgs, "α", α, "β", β, "r",r,"tol",tol)

# nacrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss5_pcb_nacrng.jld2")
dd = load(fname)
nacrng = dd["nacrng"]; fitvals = dd["fitvals"]; nsws = dd["nsws"]; β = dd["β"]; α = dd["α"]; noc=500

nacrngfnames = [(300,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac300_a0.005_b3.0_it6.jld2"),
                (400,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac400_a0.005_b3.0_it5.jld2"),
                (500,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac500_a0.005_b3.0_it8.jld2")#=,
                (600,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac600_a0.005_b3.0_it7.jld2"),
                (700,"./nac_old/WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac700_a0.005_b3.0_it11.jld2"),
                (800,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac800_a0.005_b3.0_it9.jld2"),
                (900,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac900_a0.005_b3.0_it8.jld2"),
                (1000,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac1000_a0.005_b3.0_it11.jld2")=#
                ]
nacrng = Float64[]; fitvals = Float64[]; nsws = Float64[]
for (nac,jldfname) in nacrngfnames
#    @show nac
    fname = joinpath(subworkpath,jldfname)
    dd = load(fname)
    push!(fitvals,dd["fitval"])
    push!(nsws,dd["nsw"])
    push!(nacrng,nac)
end
noc = 500; α = 0.005; β = 3.0

labels = ["fitval","norm(W,1)/noc"]
fig = Figure(resolution = (500,300))
fitymin = floor(minimum(fitvals)*1000)/1000
fitymax = ceil(maximum(fitvals)*1000)/1000
nswymin = floor(minimum(nsws))
nswymax = ceil(maximum(nsws))
nacmin = minimum(nacrng)
nacmax = maximum(nacrng)
ax1 = AMakie.Axis(fig[1, 1], limits=((nacmin,nacmax), (fitymin,fitymax)), xlabel = "nac", ylabel = "fit", title = "Fit Value vs. nac")
ax2 = AMakie.Axis(fig[1, 1], limits=((nacmin,nacmax), (nswymin,nswymax)), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, nacrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, nacrng, nsws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_pcb_nacrng_noc$(noc)_a$(α)_b$(β).png"),fig)


# noc vs nac
nocrngfnames = [(300,"WMB-10Xv2-HY-rawt_ss5_pcb_noc300_a0.005_b3.0_it10.jld2"),
                (400,"WMB-10Xv2-HY-rawt_ss5_pcb_noc400_a0.005_b3.0_it9.jld2"),
                (500,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_a0.005_b3.0_it9.jld2"),
                (600,"WMB-10Xv2-HY-rawt_ss5_pcb_noc600_a0.005_b3.0_it8.jld2"),
                (700,"WMB-10Xv2-HY-rawt_ss5_pcb_noc700_a0.005_b3.0_it7.jld2")#=,
                (800,"WMB-10Xv2-HY-rawt_ss5_pcb_noc800_a0.005_b3.0_it11.jld2"),
                (900,"WMB-10Xv2-HY-rawt_ss5_pcb_noc900_a0.005_b3.0_it10.jld2"),
                (1000,"WMB-10Xv2-HY-rawt_ss5_pcb_noc1000_a0.005_b3.0_it11.jld2")=#]
noctms = Float64[]; nocfitvals = Float64[]; nocnsws = Float64[]
for (noc,jldfname) in nocrngfnames
    fname = joinpath(subworkpath,jldfname)
    dd = load(fname)
    push!(nocfitvals,dd["fitval"])
    W = dd["W"]; H = dd["H"]
    tm = dd["rt1"]+dd["rt2"]
    push!(noctms,tm)
    d=LCSVD.normalizeWH!(W,H)
    nsw = norm(W,1)/noc
    push!(nocnsws,nsw)
end

nacrngfnames = [(300,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac300_a0.005_b3.0_it6.jld2"),
                (400,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac400_a0.005_b3.0_it5.jld2"),
                (500,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac500_a0.005_b3.0_it8.jld2")#=,
                (600,"WMB-10Xv2-HY-rawt_ss5_pcb_noc500_nac600_a0.005_b3.0_it7.jld2"),
                (700,"./nac_old/WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac700_a0.005_b3.0_it11.jld2"),
                (800,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac800_a0.005_b3.0_it9.jld2"),
                (900,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac900_a0.005_b3.0_it8.jld2"),
                (1000,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_nac1000_a0.005_b3.0_it11.jld2")=#
                ]
nactms = Float64[]; nacfitvals = Float64[]; nacnsws = Float64[]; noc = 500
for (nac,jldfname) in nacrngfnames
#    @show nac
    fname = joinpath(subworkpath,jldfname)
    dd = load(fname)
    push!(nacfitvals,dd["fitval"])
    push!(nacnsws,dd["nsw"])
    tm1 = dd["rt1"]
    tm2 = dd["rt2"]
    @show tm1, tm2
    tm = tm1+tm2
    push!(nactms,tm)
end
α=0.005; β=3.0

labels = ["noc fitval","noc norm(W,1)/noc", "nac fitval","nac norm(W,1)/noc"]
fig = Figure(resolution = (550,300))
fitymin = floor(min(minimum(nocfitvals),minimum(nacfitvals))*1000)/1000
#fitymin = 0.995
fitymax = ceil(max(maximum(nocfitvals),maximum(nacfitvals))*1000)/1000
nswymin = floor(min(minimum(nocnsws),minimum(nacnsws)))
nswymax = ceil(max(maximum(nocnsws),maximum(nacnsws)))
tmin = floor(min(minimum(noctms),minimum(nactms)))
tmax = ceil(max(maximum(noctms),maximum(nactms)))
xtickstep = Int(floor((tmax-tmin)/3))
ax1 = AMakie.Axis(fig[1, 1], limits=((tmin,tmax), (fitymin,fitymax) ), xticks = tmin:xtickstep:tmax,
                xtickformat = values -> ["$(round(value/3600,digits=2))" for value in values], 
                xlabel = "time(hr)", ylabel = "fit", title = "Fit Value vs. time")
ax2 = AMakie.Axis(fig[1, 1], limits=((tmin,tmax), (nswymin,nswymax) ), xticks = tmin:xtickstep:tmax,
                xtickformat = values -> ["$(round(value/3600,digits=2))" for value in values],
                yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, noctms, nocfitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, noctms, nocnsws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
ln3 = lines!(ax1, nactms, nacfitvals, color=mtdcolors[2], label=labels[3])
ln4 = lines!(ax2, nactms, nacnsws, color=mtdcolors[2], label=labels[4], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2,ln3,ln4],labels)
save(joinpath(subworkpath,"fit_and_sparsity_time_nocrng_nacrng_a$(α)_b$(β).png"),fig)


halsnocrngfnames = [(300,"WMB-10Xv2-HY-rawt_ss5_hals_noc300_a0.1_mit500.jld2"),
                    (400,"WMB-10Xv2-HY-rawt_ss5_hals_noc400_a0.1_mit500.jld2"),
                    (500,"WMB-10Xv2-HY-rawt_ss5_hals_noc500_a0.1_mit500.jld2")#=,
                    (600,"WMB-10Xv2-HY-rawt_ss5_hals_noc600_a0.1_mit500.jld2"),
                    (700,"WMB-10Xv2-HY-rawt_ss5_hals_noc700_a0.1_mit500.jld2"),
                    (800,"WMB-10Xv2-HY-rawt_ss5_hals_noc800_a0.1_mit500.jld2"),
                    (900,"WMB-10Xv2-HY-rawt_ss5_hals_noc900_a0.1_mit500.jld2"),
                    (1000,"WMB-10Xv2-HY-rawt_ss5_hals_noc1000_a0.1_mit500.jld2")=#]

halsnoctms = Float64[]; halsnocfitvals = Float64[]; halsnocnsws = Float64[]; noc=500
for (noc,jldfname) in halsnocrngfnames
    fname = joinpath(subworkpath,jldfname)
    dd = load(fname)
    push!(halsnocfitvals,dd["fitval"])
    W = dd["W"]; H = dd["H"]
    tm = dd["rt1"]+dd["rt2"]
    push!(halsnoctms,tm)
    d=LCSVD.normalizeWH!(W,H)
    nsw = norm(W,1)/noc
    push!(halsnocnsws,nsw)
end

labels = ["PCB noc fitval","PCB noc norm(W,1)/noc", "PCB nac fitval","PCB nac norm(W,1)/noc", "HALS noc fitval","HALS noc norm(W,1)/noc"]
fig = Figure(resolution = (600,300))
fitymin = floor(min(minimum(nocfitvals),minimum(nacfitvals),minimum(halsnocfitvals))*1000)/1000
#fitymin = 0.995
fitymax = ceil(max(maximum(nocfitvals),maximum(nacfitvals),maximum(halsnocfitvals))*1000)/1000
nswymin = floor(min(minimum(nocnsws),minimum(nacnsws),minimum(halsnocnsws)))
nswymax = ceil(max(maximum(nocnsws),maximum(nacnsws),maximum(halsnocnsws)))
tmin = floor(min(minimum(noctms),minimum(nactms),minimum(halsnoctms)))
tmax = ceil(max(maximum(noctms),maximum(nactms),maximum(halsnoctms)))
xtickstep = Int(floor((tmax-tmin)/3))
ax1 = AMakie.Axis(fig[1, 1], limits=((tmin,tmax), (fitymin,fitymax) ), xticks = tmin:xtickstep:tmax,
                xtickformat = values -> ["$(round(value/3600,digits=2))" for value in values], 
                xlabel = "time(hr)", ylabel = "fit", title = "Fit Value vs. time")
ax2 = AMakie.Axis(fig[1, 1], limits=((tmin,tmax), (nswymin,nswymax) ), xticks = tmin:xtickstep:tmax,
                xtickformat = values -> ["$(round(value/3600,digits=2))" for value in values],
                yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, noctms, nocfitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, noctms, nocnsws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
ln3 = lines!(ax1, nactms, nacfitvals, color=mtdcolors[2], label=labels[3])
ln4 = lines!(ax2, nactms, nacnsws, color=mtdcolors[2], label=labels[4], linestyle = :dash, linewidth = 2)
ln5 = lines!(ax1, halsnoctms, halsnocfitvals, color=mtdcolors[3], label=labels[5])
ln6 = lines!(ax2, halsnoctms, halsnocnsws, color=mtdcolors[3], label=labels[6], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2,ln3,ln4,ln5,ln6],labels)
save(joinpath(subworkpath,"fit_and_sparsity_time_pcb_vs_hals_nocrng_nacrng_a$(α)_b$(β).png"),fig)




#======== Gene sparsity and cell nonnegativity ============#
# αrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_αwrng.jld2")
dd = load(fname)
αrng = dd["αrng"]; fitvals = dd["fitvals"]; sws = dd["sws"]; β = dd["β"]; noc=500

labels = ["fitval","norm(sw,1)"]
fig = Figure(resolution = (500,300))
ax1 = AMakie.Axis(fig[1, 1], limits=(nothing, (0.995,1.0) ), xlabel = "αw", ylabel = "fit", title = "Fit Value vs. αw")
ax2 = AMakie.Axis(fig[1, 1], limits=(nothing, (4000,9000) ), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
labels = ["fit","norm(W,1)"] # foreach(i->labels[i] *= " (inhibited)" ,inhibitindices)
ln1 = lines!(ax1, αrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, αrng, sws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_αwrng_noc$(noc)_bw$(β).png"),fig)

# βrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_βhrng.jld2")
dd = load(fname)
βrng = dd["βrng"]; fitvals = dd["fitvals"]; sws = dd["sws"]; α = dd["α"]; noc=500

labels = ["fitval","norm(sw,1)"]
fig = Figure(resolution = (500,300))
ax1 = AMakie.Axis(fig[1, 1], limits=((0,50), (0.9,1.0)), xlabel = "β", ylabel = "fit", title = "Fit Value vs. β")
ax2 = AMakie.Axis(fig[1, 1], limits=((0,50), (2000.,7000) ), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
labels = ["fit","norm(W,1)"] # foreach(i->labels[i] *= " (inhibited)" ,inhibitindices)
ln1 = lines!(ax1, βrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, βrng, sws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_βrng_noc$(noc)_a$(α).png"),fig)

βrng = [0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0, 30.0, 50.0, 70.0, 90.0, 100.0, 200, 400, 600, 800, 1000, 2000, 4000, 5000, 6000, 8000, 10000]
fitvals = [0.995571, 0.995187, 0.994457, 0.992279, 0.990214, 0.98844, 0.986793, 0.98599, 0.975468, 0.969159, 0.964912, 0.962009, 0.960679, 0.952928, 0.946409, 0.943178, 0.941188, 0.939647, 0.935867, 0.932633, 0.931874, 0.931356, 0.930333, 0.929777]
sws = [2242.18, 2379.49, 2376.04, 2359.18, 2291.9, 2322.68, 2313.51, 2298.65, 2461.84, 2455.77, 2536.32, 2635.93, 2683.95, 3011.59, 3512.46, 4031.69, 4580.25, 4498.81, 4498.22, 6022.21, 6509.41, 5997.07, 5710.26, 5403.12]
save(fname, "βrng", βrng,"fitvals",fitvals,"sws",sws ,"per_component", per_component, "optim_method", optim_method, "α", α, "r",r,"tol",tol)

# nocrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss20_pcb_nocrng1000.jld2")
dd = load(fname)
nocrng = dd["nocrng"]; fitvals = dd["fitvals"]; sws = dd["sws"]; β = dd["β"]; α = dd["α"]

labels = ["fitval","norm(sw,1)"]
fig = Figure(resolution = (500,300))
ax1 = AMakie.Axis(fig[1, 1], limits=((300,1200), (0.95,1.05)), xlabel = "noc", ylabel = "fit", title = "Fit Value vs. noc")
ax2 = AMakie.Axis(fig[1, 1], limits=((300,1200), (1000.,3000)), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
labels = ["fit","norm(W,1)"] # foreach(i->labels[i] *= " (inhibited)" ,inhibitindices)
ln1 = lines!(ax1, nocrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, nocrng, sws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_nocrng_noc$(noc)_a$(α)_b$(β)_1200.png"),fig)

nocrng = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
fitvals = [0.966555, 0.982728, 0.992279, 0.996846, 0.998702, 0.999454, 0.999759, 0.999864, 0.999939, 0.999970]
sws = [2347.05, 2453.88, 2359.18, 2145.99, 2162.67, 1786.27, 1630.25, 1465.82, 1342.80, 1264.64]

# fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss20_pcb_noc1200_a0.005_b3.0_it11.jld2")
# dda = load(fname)
# fitval = dda["fitval"]; W = dda["W"]; H = dda["H"]
# d=LCSVD.normalizeWH!(W,H)
# sw = norm(W,1)

fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss20_pcb_nocrng1200.jld2")
save(fname, "nocrng", nocrng,"fitvals",fitvals,"sws",sws ,"per_component", per_component, "optim_method", optim_method, "α", α, "β", β, "r",r,"tol",tol)

# "WMB-10Xv2-HY-rawt_ss20_pcb_noc500_a0.005_b3.0_it10.jld2"
# "WMB-10Xv2-HY-rawt_ss20_pcb_noc500_a0.005_b3.0_it11.jld2"

nocrngfnames = [(300,"WMB-10Xv2-HY-rawt_ss20_pcb_noc300_a0.005_b3.0_it7.jld2"),
                (400,"WMB-10Xv2-HY-rawt_ss20_pcb_noc400_a0.005_b3.0_it7.jld2"),
                (500,"WMB-10Xv2-HY-rawt_ss20_pcb_noc500_a0.005_b3.0_it7.jld2"),
                (600,"WMB-10Xv2-HY-rawt_ss20_pcb_noc600_a0.005_b3.0_it6.jld2"),
                (700,"WMB-10Xv2-HY-rawt_ss20_pcb_noc700_a0.005_b3.0_it13.jld2"),
                (800,"WMB-10Xv2-HY-rawt_ss20_pcb_noc800_a0.005_b3.0_it11.jld2"),
                (900,"WMB-10Xv2-HY-rawt_ss20_pcb_noc900_a0.005_b3.0_it10.jld2"),
                (1000,"WMB-10Xv2-HY-rawt_ss20_pcb_noc1000_a0.005_b3.0_it11.jld2"),
                (1100,"WMB-10Xv2-HY-rawt_ss20_pcb_noc1100_a0.005_b3.0_it11.jld2"),
                (1200,"WMB-10Xv2-HY-rawt_ss20_pcb_noc1200_a0.005_b3.0_it11.jld2")]
nocrng = Int[]; fitvals = Float64[]; sws = Float64[]
for (noc,jldfname) in nocrngfnames
    fname = joinpath(subworkpath,jldfname)
    dd = load(fname)
    push!(nocrng,noc)
    push!(fitvals,dd["fitval"])
    W = dd["W"]; H = dd["H"]
    d=LCSVD.normalizeWH!(W,H)
    sw = norm(W,1)/noc
    push!(sws,sw)
end
β = 3.0; α = 0.005

labels = ["fitval","norm(sw,1)"]
fig = Figure(resolution = (500,300))
ax1 = AMakie.Axis(fig[1, 1], limits=((300,1200), (0.95,1.05)), xlabel = "noc", ylabel = "fit", title = "Fit Value vs. noc")
ax2 = AMakie.Axis(fig[1, 1], limits=((300,1200), (1.,8)), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
labels = ["fit","norm(W,1)/noc"] # foreach(i->labels[i] *= " (inhibited)" ,inhibitindices)
ln1 = lines!(ax1, nocrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, nocrng, sws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_nocrng_a$(α)_b$(β)_nsw_1200.png"),fig)

fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss20_pcb_nocrng1200_nsw.jld2")
save(fname, "nocrng", nocrng,"fitvals",fitvals,"sws",sws ,"per_component", per_component, "optim_method", optim_method, "α", α, "β", β, "r",r,"tol",tol)


# nacrng
fname = joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss20_pcb_nocrng1000.jld2")
dd = load(fname)
nocrng = dd["nocrng"]; fitvals = dd["fitvals"]; sws = dd["sws"]; β = dd["β"]; α = dd["α"]

labels = ["fitval","norm(sw,1)"]
fig = Figure(resolution = (500,300))
ax1 = AMakie.Axis(fig[1, 1], limits=((300,1200), (0.95,1.05)), xlabel = "noc", ylabel = "fit", title = "Fit Value vs. noc")
ax2 = AMakie.Axis(fig[1, 1], limits=((300,1200), (1000.,3000)), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
labels = ["fit","norm(W,1)"] # foreach(i->labels[i] *= " (inhibited)" ,inhibitindices)
ln1 = lines!(ax1, nocrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, nocrng, sws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_nocrng_noc$(noc)_a$(α)_b$(β)_1200.png"),fig)

# noc vs nac


