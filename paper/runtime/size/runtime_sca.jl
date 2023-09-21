using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","size")

include(joinpath(workpath,"setup_light.jl"))

dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 50
    sca_maxiter = 400; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 200
else
    num_experiments = 2
    sca_maxiter = 2; sca_inner_maxiter = 2; sca_ls_maxiter = 2
    admm_maxiter = 2; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 2
end

for factor = [1,2,4,8]
    fovsz = (40,20)
    imgsz = (factor*fovsz[1],fovsz[2]); lengthT = factor*100
for iter in 1:num_experiments
    @show fovsz, iter; flush(stdout)
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; imgsz=imgsz, fovsz=fovsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# SCA
@show "SCA"
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; α = 100; β = 1000
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
try
for (tailstr,initmethod,α,β) in [("_sp",:nndsvd,100.,0.), ("_sp_nn",:nndsvd,100.,1000.)] #("_nn",:nndsvd,0.,1000.), 
    @show tailstr; flush(stdout)
    dd = Dict()
    rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    σ0=s*std(W0) #=10*std(W0)=#
    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
        useprecond=useprecond, uselv=uselv)
    lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    Mw, Mh = copy(Mw0), copy(Mh0);
    rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                    penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                     penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    f_xs = getdata(trs,:f_x); niters = getdata(trs,:niter); totalniters = sum(niters)
    avgfitss = getdata(trs,:avgfits); fxss = getdata(trs,:fxs)
    avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
    for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
        isempty(afs) && continue
        append!(avgfits,afs); append!(inner_fxs,fxs)
        if iter == 1
            rt2i = 0.
        else
            rt2i = collect(range(start=laps[iter-1],stop=laps[iter],length=length(afs)+1))[2:end].-laps[1]
        end
        append!(rt2s,rt2i)
    end
    @show length(rt2s); flush(stdout)

    dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
        metadata["alpha"] = α; metadata["beta"] = β; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
    end
    save(joinpath(subworkpath,"sca","sca$(factor)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
end
catch e
    save(joinpath(subworkpath,"sca$(factor)_error_$(iter).jld2"),datadic)
    @warn e
    iter -= 1
end
end # for iter
end # for factor


using Interpolations

for factor = [1,2,4,8]
num_expriments=50
rt2_min = Inf
for tailstr in ["_sp","_sp_nn"]
    for iter in 1:num_expriments
        dd = load(joinpath(subworkpath,"sca","sca$(tailstr)_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]; @show rt2s[end]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_min = floor(rt2_min, digits=2)
rng = range(0,stop=rt2_min,length=100)

stat_nn=[]; stat_sp=[]; stat_sp_nn=[]
for tailstr in ["_sp","_sp_nn"]
    afs=[]
    for iter in 1:num_expriments
        @show tailstr, iter
        dd = load(joinpath(subworkpath,"sca","sca$(tailstr)_results$(iter).jld2"))
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
save(joinpath(subworkpath,"sca$(factor)_runtime_vs_avgfits.jld2"),"rng",rng, "stat_nn", stat_nn, "stat_sp", stat_sp, "stat_sp_nn", stat_sp_nn)
end

#==== New Initialization (Sparse W) ===========================#
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

initpwradj=:wh_normalize
penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; useRelaxedL1=true; s=10*0.3^0; r=(0.3)^1
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; uselv=false
maxiter = 50; inner_maxiter = 50; ls_maxiter = 100; poweradjust = :none

α = 100; α1=α2=α
initmethods = (:sparseW, :isvd, :nndsvd)
usepreconds = (false, )
submtdstrs = ("_sp_nn", ) #"_sp_nn"
for initmethod in initmethods
    for useprecond in usepreconds
        precondstr = useprecond ? "_pcond" : ""
        for submtdstr = submtdstrs
            ptpx = "$(initmethod)$(precondstr)$(submtdstr)"
            @show ptpx
            β = submtdstr == "_sp" ? 0 : 1000
            β1=β2=β
            @eval ($(Symbol("avgfits_$(ptpx)")) =  Float64[])
            @eval ($(Symbol("inner_fxs_$(ptpx)")) =  Float64[])
            @eval ($(Symbol("rt2s_$(ptpx)")) =  Float64[])

rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#

stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
f_xs = getdata(trs,:f_x); niters = getdata(trs,:niter); totalniters = sum(niters)
avgfitss = getdata(trs,:avgfits); fxss = getdata(trs,:fxs)

for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
    isempty(afs) && continue
    @eval (append!($(Symbol("avgfits_$(ptpx)")),$afs)) # eval(Meta.parse("append!(avgfits_$(ptpx),$(afs))"))
    @eval (append!($(Symbol("inner_fxs_$(ptpx)")),$fxs))
    if iter == 1
        rt2i = 0.
    else
        rt2i = collect(range(start=laps[iter-1],stop=laps[iter],length=length(afs)+1))[2:end].-laps[1]
    end
    @eval (append!($(Symbol("rt2s_$(ptpx)")),$rt2i))
end
            # @show "avgfits_$(ptpx)", eval(Symbol("avgfits_$(ptpx)"))
            # @show "inner_fxs_$(ptpx)", eval(Symbol("inner_fxs_$(ptpx)"))
            # @show "rt2s_$(ptpx)", eval(Symbol("rt2s_$(ptpx)"))
        end
    end
end

include(joinpath(workpath,"setup_plot.jl"))
alpha = 0.2; cls = distinguishable_colors(20); clbs = convert.(RGBA,cls,alpha)

plotrng = :
fig = Figure(resolution=(600,400))
ax = GLMakie.Axis(fig[1, 1], limits = ((0,0.3), nothing), xlabel = "time(sec)", ylabel = "average fit", title = "Average Fit Value vs. Running Time")

lns = Dict(); clridx = 0
for initmethod in initmethods
    for useprecond in usepreconds
        precondstr = useprecond ? "_pcond" : ""
        for submtdstr = submtdstrs
            clridx += 1
            ptpx = "$(initmethod)$(precondstr)$(submtdstr)"
            ln = lines!(ax, eval(Symbol("rt2s_$(ptpx)"))[plotrng], eval(Symbol("avgfits_$(ptpx)"))[plotrng], color=cls[clridx+2], label=ptpx)
        end
    end
end
axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"init_compare2.png"),fig,px_per_unit=2)

#===============================#
initpwradj=:wh_normalize
penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; useRelaxedL1=true; s=10*0.3^0; r=(0.3)^1
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; uselv=false
maxiter = 50; inner_maxiter = 50; ls_maxiter = 100; poweradjust = :none

α = 100; α1=α2=α 
for initmethod in (:isvd, :nndsvd)
    for useprecond in (false, true)
        precondstr = useprecond ? "_pcond" : ""
        for submtdstr = ("_sp", "_sp_nn")
            ptpx = "$(initmethod)$(precondstr)$(submtdstr)"
            @show ptpx
            β = submtdstr == "_sp" ? 0 : 1000
            β1=β2=β
            @eval ($(Symbol("avgfits_$(ptpx)")) =  Float64[])
            @eval ($(Symbol("inner_fxs_$(ptpx)")) =  Float64[])
            @eval ($(Symbol("rt2s_$(ptpx)")) =  Float64[])

rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#

stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
f_xs = getdata(trs,:f_x); niters = getdata(trs,:niter); totalniters = sum(niters)
avgfitss = getdata(trs,:avgfits); fxss = getdata(trs,:fxs)

for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
    isempty(afs) && continue
    @eval (append!($(Symbol("avgfits_$(ptpx)")),$afs)) # eval(Meta.parse("append!(avgfits_$(ptpx),$(afs))"))
    @eval (append!($(Symbol("inner_fxs_$(ptpx)")),$fxs))
    if iter == 1
        rt2i = 0.
    else
        rt2i = collect(range(start=laps[iter-1],stop=laps[iter],length=length(afs)+1))[2:end].-laps[1]
    end
    @eval (append!($(Symbol("rt2s_$(ptpx)")),$rt2i))
end
            # @show "avgfits_$(ptpx)", eval(Symbol("avgfits_$(ptpx)"))
            # @show "inner_fxs_$(ptpx)", eval(Symbol("inner_fxs_$(ptpx)"))
            # @show "rt2s_$(ptpx)", eval(Symbol("rt2s_$(ptpx)"))
        end
    end
end

include(joinpath(workpath,"setup_plot.jl"))
alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)

plotrng = :
fig = Figure(resolution=(600,400))
ax = GLMakie.Axis(fig[1, 1], limits = ((0,0.3), nothing), xlabel = "time(sec)", ylabel = "average fit", title = "Average Fit Value vs. Running Time")

lns = Dict(); clridx = 0
for initmethod in (:isvd, :nndsvd)
    for useprecond in (false, true)
        precondstr = useprecond ? "_pcond" : ""
        for submtdstr = ("_sp", "_sp_nn")
            clridx += 1
            ptpx = "$(initmethod)$(precondstr)$(submtdstr)"
            ln = lines!(ax, eval(Symbol("rt2s_$(ptpx)"))[plotrng], eval(Symbol("avgfits_$(ptpx)"))[plotrng], color=mtdcolors[clridx], label=ptpx)
        end
    end
end
axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"precond_compare.png"),fig,px_per_unit=2)


#===============================#
initpwradj=:wh_normalize
penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; useRelaxedL1=true; s=10*0.3^0; r=(0.3)^1
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; uselv=false
maxiter = 50; inner_maxiter = 50; ls_maxiter = 100; poweradjust = :none

α = 100; α1=α2=α
initmethod = :nndsvd
for (useprecond, precondtype, submtdstr) in [(false, :nothing, "_sp"), (true, :invert,"_sp"), (true, :sparse,"_sp"),
                        (false, :nothing, "_sp_nn"), (true, :invert,"_sp_nn"), (true, :sparse,"_sp_nn"), (true, :full, "_sp_nn")]
    precondstr = useprecond ? "_pcond" : ""
    precondtypestr = useprecond ? "_"*String(precondtype) : ""
    ptpx = "$(precondstr)$(precondtypestr)$(submtdstr)"
    @show ptpx
    β = submtdstr == "_sp" ? 0 : 1000
    β1=β2=β
    @eval ($(Symbol("avgfits_$(ptpx)")) =  Float64[])
    @eval ($(Symbol("inner_fxs_$(ptpx)")) =  Float64[])
    @eval ($(Symbol("rt2s_$(ptpx)")) =  Float64[])

rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
σ0=s*std(W0) #=10*std(W0)=#

stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    useprecond=useprecond, precondtype=precondtype, uselv=uselv)
lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0);
cparams.store_trace = false; cparams.store_inner_trace = false;
cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
f_xs = getdata(trs,:f_x); niters = getdata(trs,:niter); totalniters = sum(niters)
avgfitss = getdata(trs,:avgfits); fxss = getdata(trs,:fxs)

for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
    isempty(afs) && continue
    @eval (append!($(Symbol("avgfits_$(ptpx)")),$afs)) # eval(Meta.parse("append!(avgfits_$(ptpx),$(afs))"))
    @eval (append!($(Symbol("inner_fxs_$(ptpx)")),$fxs))
    if iter == 1
        rt2i = 0.
    else
        rt2i = collect(range(start=laps[iter-1],stop=laps[iter],length=length(afs)+1))[2:end].-laps[1]
    end
    @eval (append!($(Symbol("rt2s_$(ptpx)")),$rt2i))
end

end

include(joinpath(workpath,"setup_plot.jl"))
alpha = 0.2; cls = distinguishable_colors(12); clbs = convert.(RGBA,cls,alpha)

plotrng = :
fig = Figure(resolution=(600,400))
ax = GLMakie.Axis(fig[1, 1], limits = ((0,0.3), nothing), xlabel = "time(sec)", ylabel = "average fit", title = "Average Fit Value vs. Running Time")

lns = Dict(); clridx = 0
for (useprecond, precondtype, submtdstr) in [(false, :nothing, "_sp"), (true, :invert,"_sp"), (true, :sparse,"_sp"),
        (false, :nothing, "_sp_nn"), (true, :invert,"_sp_nn"), (true, :sparse,"_sp_nn"), (true, :full, "_sp_nn")]
    precondstr = useprecond ? "_pcond" : ""
    precondtypestr = useprecond ? "_"*String(precondtype) : ""
    ptpx = "$(precondstr)$(precondtypestr)$(submtdstr)"
    clridx += 1
    ln = lines!(ax, eval(Symbol("rt2s_$(ptpx)"))[plotrng], eval(Symbol("avgfits_$(ptpx)"))[plotrng], color=cls[clridx], label=ptpx)
end
axislegend(ax, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"precond_compare2.png"),fig,px_per_unit=2)
