using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","sparse_coding")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

# ARGS = [":lbfgs_admm","0.0005","400", "100", "1e-7"]
optim_method = eval(Meta.parse(ARGS[1])) # :lbfgs_admm
αh = eval(Meta.parse(ARGS[2])) # 5e-4
maxiter = eval(Meta.parse(ARGS[3])) # 500
inner_maxiter = eval(Meta.parse(ARGS[4])) # 10
inner_tol = eval(Meta.parse(ARGS[5])) # 1e-7
@show αh, maxiter, inner_maxiter, inner_tol
flush(stdout) 

#========= sparse coding with natural images =============#
dataset = :natural
imgsz = (12,12); lengthT = 100000; noc = ncs = 72; nac = 0
dataset = :natural; noc = 72; nac = 0; (m,n) = (*(imgsz...), lengthT)

dd = load(joinpath(subworkpath,"allinit.jld2"))
X_whitened = dd["X_whitened"][1]
U, Vt, D = dd["SVD"]; V = Vt'
gtW, gtH = (Matrix{eltype(X_whitened)}(undef,0,0),Matrix{eltype(X_whitened)}(undef,0,0))

#============== PCB ==============================#
prefix = "pcb"

(ur, nr) = (0, 0)
# αh = 5e-4; inner_maxiter = 100; inner_tol = 1e-7
for initmethod in [:randcolX, :isvd, :randH]
    Winit, Hinit, M0, N0t, _ = dd[String(initmethod)]
    #for αh in [0.0000001, 0.0000005]
    σ0=1; r=0.5; useprecond=false; uselv=false

    @show initmethod
    # ur = 1e-3; nr = 1e-2 # :sgd_injectnoise
    β1 = β2 = β = 0; α1 = 0; α2 = αh
    β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
    α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
    #r = eps(Float64)^(1/(2*maxiter))
    # maxiter = 500 # 500 #Int(ceil(log(eps(eltype(X_whitened)))/log(r))) #lcsvd_maxiter # 
    tol=0

    # inner_maxiter = 10 #Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) #
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0,
        #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
        r=r, useprecond=useprecond, usedenoiseUVt=false, optim_method=optim_method,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
        store_inner_trace = false, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0) #, ur=ur, nr=nr); # ur=0.00001, nr=0.001
    # M1, N1t = copy(M0), copy(N0t)
    # rst1 = LCSVD.solve!(alg, X_whitened, U, V, D, M1, N1t; gtW=gtW, gtH=gtH, use_σ2_cal_pen=false);
    # alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
    M1, N1t = copy(M0), copy(N0t)
    rt2 = @elapsed rst2 = LCSVD.solve!(alg, X_whitened, U, V, D, M1, N1t);

    W1, H1 = rst2.W, rst2.Ht'
    # L1h = norm(H1,1)
    fv = LCSVD.fitd(X_whitened,W1*H1)
    # Einit = rst1.traces[1].f_x; Eend = rst1.traces[end].f_x
    # Esym = rst1.traces[end].sympen
    # Esh = rst1.traces[end].sparseH
    LCSVD.normalizeWH!(W1,H1); norm1nH = norm(H1,1)
    fprex = "$(prefix)_$(initmethod)_$(optim_method)_intol$(inner_tol)_initer$(inner_maxiter)"
    #fprex = "$(prefix)_BPDN"
    regstr = "_s0$(σ0)_r$(r)_ah$(α2)"
    # fname = joinpath(subworkpath,"$(fprex)$(regstr)_Einit$(Einit)_Eend$(Eend)_Esy$(Esym)_Esh$(Esh)_nH$(norm1nH)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_f$(fv)_it$(rst2.niters)_rt$(rt2)")
    imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
    niters = map(t->t.niters,rst2.traces); f_x = map(t->t.f_x,rst2.traces)
    save(joinpath(subworkpath,"$(fprex)$(regstr)_mit$(alg.maxiter)_iit$(alg.inner_maxiter)_itol$(alg.inner_tol).jld2"),"niters",niters,"f_x",f_x)
end

#=
fname = joinpath(subworkpath,"$(pr[efix)_$(initmethod)_$(optim_method)_inner_maxiter_compare.jld2")
save(fname, "niters10_7",niters10_7,"f_x10_7",f_x10_7,
            "niters100_7",niters100_7,"f_x100_7",f_x100_7,
            "niters100_10",niters100_10,"f_x100_10",f_x100_10)

resol=(800,600); fntsize1 = 30; fntsize2 = 30

fig = Figure(size=resol)
ax = AMakie.Axis(fig[1, 1], limits = ((0,100), (48000,50000)),
                xlabel = "iteration", ylabel = "penalty", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2, yscale = log10)#, title = "Average Fit Value vs. Running Time")

lines!(ax, f_x10_7[2:end], color=mtdcolors[2], label="i_maxiter=10, i_tol=1e-7", linestyle=nothing)
lines!(ax, f_x100_7[2:end], color=mtdcolors[4], label="i_maxiter=100, i_tol=1e-7", linestyle=nothing)
lines!(ax, f_x100_10[2:end], color=mtdcolors[5], label="i_maxiter=100, i_tol=1e-10", linestyle=nothing)

axislegend(ax, labelsize=fntsize1, position = :cb)
save(joinpath(subworkpath,"$(prefix)_$(initmethod)_$(optim_method)_inner_maxiter_compare_penalty.png"),fig, px_per_unit=2)


fig = Figure(size=resol)
ax = AMakie.Axis(fig[1, 1], limits = ((0,100), (0,150)),
                xlabel = "iteration", ylabel = "number of inner iterations", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2)#, title = "Average Fit Value vs. Running Time")


lines!(ax, niters10_7[2:end], color=mtdcolors[2], label="i_maxiter=10, i_tol=1e-7", linestyle=nothing)
lines!(ax, niters100_7[2:end], color=mtdcolors[4], label="i_maxiter=100, i_tol=1e-7", linestyle=nothing)
lines!(ax, niters100_10[2:end], color=mtdcolors[5], label="i_maxiter=100, i_tol=1e-10", linestyle=nothing)

axislegend(ax, labelsize=fntsize1, position = :rt)
save(joinpath(subworkpath,"$(prefix)_$(initmethod)_$(optim_method)_inner_maxiter_compare_niters.png"),fig, px_per_unit=2)


resol=(800,600); fntsize1 = 30; fntsize2 = 30
fig1 = Figure(size=resol)
ax1 = AMakie.Axis(fig1[1, 1], limits = ((0,100), (0,150)),
                xlabel = "iteration", ylabel = "number of inner iterations", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2)#, title = "Average Fit Value vs. Running Time")
fig2 = Figure(size=resol)
ax2 = AMakie.Axis(fig2[1, 1], limits = ((0,100), (9.9e6,1.0e7)),
                xlabel = "iteration", ylabel = "penalty/α", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2, yscale=log10)#, title = "Average Fit Value vs. Running Time")

initmethod = :randcolX
colors = [1,2,3,4,5,6,7,8]
for (i,α2) in enumerate([0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005])
    fprex = "$(prefix)_$(initmethod)_$(optim_method)_intol$(inner_tol)_initer$(inner_maxiter)"
    #fprex = "$(prefix)_BPDN"
    regstr = "_s0$(σ0)_r$(r)_ah$(α2)"
    ddd = load(joinpath(subworkpath,"$(fprex)$(regstr)_it$(alg.maxiter).jld2"))
    niters = ddd["niters"]; f_x = ddd["f_x"]
    lines!(ax1, niters[2:end], color=mtdcolors[colors[i]], label="αh$(α2)", linestyle=nothing)
    lines!(ax2, f_x[2:end]./α2, color=mtdcolors[colors[i]], label="αh$(α2)", linestyle=nothing)
end

axislegend(ax1, labelsize=fntsize1, position = :rt)
axislegend(ax2, labelsize=fntsize1, position = :rt)
save(joinpath(subworkpath,"$(prefix)_$(initmethod)_α_vs_niters2.png"),fig1, px_per_unit=2)
save(joinpath(subworkpath,"$(prefix)_$(initmethod)_α_vs_f_x2.png"),fig2, px_per_unit=2)

for (i,α2) in enumerate([0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005])
    fprex = "$(prefix)_$(initmethod)_$(optim_method)_intol$(inner_tol)_initer$(inner_maxiter)"
    #fprex = "$(prefix)_BPDN"
    regstr = "_s0$(σ0)_r$(r)_ah$(α2)"
    ddd = load(joinpath(subworkpath,"$(fprex)$(regstr)_it$(alg.maxiter).jld2"))
    niters = ddd["niters"]; f_x = ddd["f_x"]
    @show minimum(f_x[2:end])/α2, maximum(f_x[2:end])/α2
end


resol=(800,600); fntsize1 = 30; fntsize2 = 30
fig1 = Figure(size=resol)
ax1 = AMakie.Axis(fig1[1, 1], limits = ((0,100), (0,150)),
                xlabel = "iteration", ylabel = "number of inner iterations", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2)#, title = "Average Fit Value vs. Running Time")
fig2 = Figure(size=resol)
ax2 = AMakie.Axis(fig2[1, 1], limits = ((0,100), (6.9e6,8.3e9)),
                xlabel = "iteration", ylabel = "penalty/α", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2, yscale=log10)#, title = "Average Fit Value vs. Running Time")

colors = [1,2,3,4,5,6,7,8]; inner_tol = 1e-10; inner_maxiter = 100; α2 = 0.0005
for (i,initmethod) in enumerate([:randcolX, :BPDN, :isvd, :randH])
    fprex = "$(prefix)_$(initmethod)_$(optim_method)_intol$(inner_tol)_initer$(inner_maxiter)"
    #fprex = "$(prefix)_BPDN"
    regstr = "_s0$(σ0)_r$(r)_ah$(α2)"
    ddd = load(joinpath(subworkpath,"$(fprex)$(regstr)_it$(alg.maxiter).jld2"))
    niters = ddd["niters"]; f_x = ddd["f_x"]
    lines!(ax1, niters[2:end], color=mtdcolors[colors[i]], label="initmethod = $(initmethod)", linestyle=nothing)
    lines!(ax2, f_x[2:end]./α2, color=mtdcolors[colors[i]], label="initmethod = $(initmethod)", linestyle=nothing)
end

axislegend(ax1, labelsize=fntsize1, position = :rt)
axislegend(ax2, labelsize=fntsize1, position = :rt)
save(joinpath(subworkpath,"$(prefix)_imaxiter100_initmethod_vs_niters2.png"),fig1, px_per_unit=2)
save(joinpath(subworkpath,"$(prefix)_imaxiter100_initmethod_vs_f_x2.png"),fig2, px_per_unit=2)

for (i,initmethod) in enumerate([:randcolX, :BPDN, :isvd, :randH])
    fprex = "$(prefix)_$(initmethod)_$(optim_method)_intol$(inner_tol)_initer$(inner_maxiter)"
    #fprex = "$(prefix)_BPDN"
    regstr = "_s0$(σ0)_r$(r)_ah$(α2)"
    ddd = load(joinpath(subworkpath,"$(fprex)$(regstr)_it$(alg.maxiter).jld2"))
    niters = ddd["niters"]; f_x = ddd["f_x"]
    @show minimum(f_x[2:end])/α2, maximum(f_x[2:end])/α2
end

# alpha vs. penalty and niters for different αh values for lbfgs_admm
resol=(800,600); fntsize1 = 30; fntsize2 = 30
for inner_tol in [1e-6, 1e-7]
    @show inner_tol
for inner_maxiter in [100, 1000]
    @show inner_maxiter
fig1 = Figure(size=resol)
ax1 = AMakie.Axis(fig1[1, 1], limits = ((0,1000), (0,150)),
                xlabel = "iteration", ylabel = "number of inner iterations", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2)#, title = "Average Fit Value vs. Running Time")
fig2 = Figure(size=resol)
ax2 = AMakie.Axis(fig2[1, 1], limits = ((0,1000), (8.0e6,5.3e7)),
                xlabel = "iteration", ylabel = "penalty/α", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2, yscale=log10)#, title = "Average Fit Value vs. Running Time")

colors = [1,2,3,4,5,6,7,8]; maxiter = 1000; optim_method = :lbfgs_admm
initmethod = :randcolX; σ0=1; r=0.5
for (i,α2) in enumerate([0.0005, 0.005, 0.05, 0.5])
    fprex = "$(prefix)_$(initmethod)_$(optim_method)_intol$(inner_tol)_initer$(inner_maxiter)"
    regstr = "_s0$(σ0)_r$(r)_ah$(α2)"
    ddd = load(joinpath(subworkpath,"$(fprex)$(regstr)_mit$(maxiter)_iit$(inner_maxiter)_itol$(inner_tol).jld2"))
    niters = ddd["niters"]; f_x = ddd["f_x"]
    lines!(ax1, niters[2:end], color=mtdcolors[colors[i]], label="α = $(α2)", linestyle=nothing)
    lines!(ax2, f_x[2:end]./α2, color=mtdcolors[colors[i]], label="α = $(α2)", linestyle=nothing)
    @show minimum(f_x[2:end])/α2, maximum(f_x[2:end])/α2
end

axislegend(ax1, labelsize=fntsize1, position = :rt)
axislegend(ax2, labelsize=fntsize1, position = :rt)
save(joinpath(subworkpath,"alpha_vs_niters_itol$(inner_tol)_iiter$(inner_maxiter).png"),fig1, px_per_unit=2)
save(joinpath(subworkpath,"alpha_vs_f_x_itol$(inner_tol)_iiter$(inner_maxiter).png"),fig2, px_per_unit=2)
end
end

for (i,α2) in enumerate([0.0005, 0.005, 0.05, 0.5])
    fprex = "$(prefix)_$(initmethod)_$(optim_method)_intol$(inner_tol)_initer$(inner_maxiter)"
    regstr = "_s0$(σ0)_r$(r)_ah$(α2)"
    ddd = load(joinpath(subworkpath,"$(fprex)$(regstr)_mit$(maxiter)_iit$(inner_maxiter)_itol$(inner_tol).jld2"))
    niters = ddd["niters"]; f_x = ddd["f_x"]
    @show minimum(f_x[2:end])/α2, maximum(f_x[2:end])/α2
end
=#