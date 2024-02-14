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
subworkpath = joinpath(workpath,"SBC")

dataset = :fakecells; SNR = 0
imgsz = (40,20); lengthT = 1000; prefix = "lcsvd"; num_experiments=10000
iters=[1,4,11,14,18,24,28,32,38,39,40,46,47,50]; rt1s = 0; nsuccess = 0

for iter in 1:num_experiments
    @show iter
    X, imsz, lhT, ncells, gtncells, datadic = load_data(:fakecells; sigma=5.0, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=0.1, useCalciumT=true,
            inhibitindices=0, issave=false, isload=false, gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false);

    (m,n,p) = (size(X)...,ncells)
    gtW, gtH = (datadic["gtW"], datadic["gtH"])

    useprecond = true; usedenoiseW0H0 = false; uselv = false
    s = 10; r=0.3; maxiter = 100
    for initmethod in [:sbc]#,:isvd,:nndsvd]
        (tailstr,α,β) = ("_sp",0.005,0.)
        fprex = "$(prefix)$(SNR)db$(initmethod)"

        dd = Dict()
        α1=α2=α; β1=β2=β
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        rt0 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=:isvd, svdmethod=:isvd)
        rt11 = 0
        if initmethod == :sbc
            try
                rt11 = @elapsed M0 = sbc(W0)
            catch e
                @warn e
                save(joinpath(subworkpath,"sbc_error$(iter).jld2"),"U",W0)
                continue
            end
            rt12 = @elapsed N0 = M0\D
            rt13 = @elapsed LCSVD.balanceWH!(M0, N0)
            rt1 = rt11+rt12+rt12
            rt1s += rt1; nsuccess += 1
        else
            rt1 = rt0
        end
#=        Wlc0 = W0*M0; Hlc0 = N0*H0
        LCSVD.normalizeW!(Wlc0,Hlc0)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
        fname = joinpath(subworkpath,"$(fprex)_rt$(rt1)")
        TestData.imsave_data_gt(dataset,fname*"_gt", Wlc0,Hlc0,gtW,gtH,imgsz,100; saveH=false, verbose=false)

        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=s*std(W0), r=r, useprecond=useprecond, usedenoiseW0H0=usedenoiseW0H0,
            denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, store_trace = true,
            store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
        M, N = copy(M0), copy(N0)
        rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
        alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        M, N = copy(M0), copy(N0)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
        Wlc, Hlc = rst0.W, rst0.H
        avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wlc, Hlc; clamp=false)
        LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
        # precondstr = useprecond ? "_precond" : ""
        # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
        fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
        #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
        TestData.imsave_data_gt(dataset,fname*"_gt", Wlc,Hlc,gtW,gtH,imgsz,100; saveH=false, verbose=false)

        f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niter); totalniters = sum(niters)
        avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
        avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
        for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
            isempty(afs) && continue
            append!(avgfits,afs); append!(inner_fxs,fxs)
            if iter == 1
                rt2i = 0.
            else
                rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
            end
            append!(rt2s,rt2i)
        end
        dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
        dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
        if true#iter == num_experiments
            metadata = Dict()
            metadata["sigma0"] = σ0; metadata["r"] = r; metadata["initmethod"] = initmethod
            metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
            metadata["usedenoiseW0H0"] = usedenoiseW0H0; metadata["denoisefilter"] = alg.denoisefilter; 
            metadata["alpha"] = α; metadata["beta"] = β
        end
        save(joinpath(subworkpath,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
=#    end
end
rt1avg = rt1s/nsuccess

include(joinpath(workpath,"setup_plot.jl"))
using Interpolations

SNRs = [0]; num_experiments = 50; ncells=15; factor=1
itp_time_resol=1000

for (prefix, initmethods, tailstrs) in [("lcsvd", ["isvd","sbc", "nndsvd"], ["_sp","_sp", "_sp"])]
    for SNR = SNRs
        rt2_min = Inf
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            fprex="$(prefix)$(SNR)db$(initmethod)"
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,"$(fprex)$(tailstr)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; @show rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
            end
        end
        rt2_min = floor(rt2_min, digits=4)
        rng = range(0,stop=rt2_min,length=itp_time_resol)
        
        stat_af_sbc=[]; stat_af_isvd=[]; stat_af_nndsvd=[]
        stat_fx_sbc=[]; stat_fx_isvd=[]; stat_fx_nndsvd=[]
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            afs=[]; fxs=[]
            fprex="$(prefix)$(SNR)db$(initmethod)"
            for iter in 1:num_experiments
                @show tailstr, iter
                dd = load(joinpath(subworkpath,"$(fprex)$(tailstr)_results$(iter).jld2"))
                rt2s = dd["data"]["rt2s"]
                # avgfits
                avgfits = dd["data"]["avgfits"]
                lr = length(rt2s); la = length(avgfits)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; avgfits=avgfits[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
                push!(afs,itp(rng))
                # fxs
                inner_fxs = dd["data"]["inner_fxs"]
                lr = length(rt2s); la = length(inner_fxs)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; inner_fxs=inner_fxs[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, inner_fxs, Gridded(Linear()))
                push!(fxs,itp(rng))
            end
            # avgfits
            avgfits = hcat(afs...)
            means = dropdims(mean(avgfits,dims=2),dims=2)
            stds = dropdims(std(avgfits,dims=2),dims=2)
            initmethod == "sbc" && (push!(stat_af_sbc,means); push!(stat_af_sbc,stds))
            initmethod == "isvd" && (push!(stat_af_isvd,means); push!(stat_af_isvd,stds))
            initmethod == "nndsvd" && (push!(stat_af_nndsvd,means); push!(stat_af_nndsvd,stds))
            # fxs
            fxs = hcat(fxs...)
            means = dropdims(mean(fxs,dims=2),dims=2)
            stds = dropdims(std(fxs,dims=2),dims=2)
            initmethod == "sbc" && (push!(stat_fx_sbc,means); push!(stat_fx_sbc,stds))
            initmethod == "isvd" && (push!(stat_fx_isvd,means); push!(stat_fx_isvd,stds))
            initmethod == "nndsvd" && (push!(stat_fx_nndsvd,means); push!(stat_fx_nndsvd,stds))
        end
        fprex="$(prefix)$(SNR)db"
        save(joinpath(subworkpath,"$(fprex)_runtime_vs_avgfits.jld2"),"rng",rng,
            "stat_af_sbc", stat_af_sbc, "stat_af_isvd", stat_af_isvd, "stat_af_nndsvd", stat_af_nndsvd,
            "stat_fx_sbc", stat_fx_sbc, "stat_fx_isvd", stat_fx_isvd, "stat_fx_nndsvd", stat_fx_nndsvd)
    end
end

tmppath = ""
z = 0.5; ylimits=(0.6,1.1)
for (idx,SNR) = enumerate(SNRs)
plottime = Inf
for (mtdstr, submtdstrs) in [("lcsvd",["_sbc", "_isvd", "_nndsvd"])]
    @show mtdstr
    fprex="$(mtdstr)$(SNR)db"
    ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(mtdstr)$(SNR)db_runtime_vs_avgfits.jld2\"))"))
    rng = eval(Meta.parse("$(ddsym)[\"rng\"]"))
    eval(Meta.parse("$(mtdstr)rng=$(ddsym)[\"rng\"]"))
    plottime = plottime > rng[end] ? rng[end] : plottime
    for submtdstr in submtdstrs
        # avgfits
        frpx = "$(mtdstr)_af$(submtdstr)"
        eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat_af$(submtdstr)\"][1]"))
        eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat_af$(submtdstr)\"][2]"))
        @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + z*$(Symbol("$(frpx)_stds"))))
        @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - z*$(Symbol("$(frpx)_stds"))))
        # fxs
        frpx = "$(mtdstr)_fx$(submtdstr)"
        eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat_fx$(submtdstr)\"][1]"))
        eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat_fx$(submtdstr)\"][2]"))
        @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + z*$(Symbol("$(frpx)_stds"))))
        @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - z*$(Symbol("$(frpx)_stds"))))
    end
end

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon()

# avgfits
fig = Figure(resolution=(600,450))
maxplottimes = [0.2,0.2,0.2]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), ylimits),
                xlabel = "time(sec)", ylabel = "average fit", xlabelsize=20, ylabelsize=20,
                xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("lcsvd","_sbc","SBC",2,nothing),
                                                                  ("lcsvd","_isvd","ISVD",3,nothing),
                                                                  ("lcsvd","_nndsvd","NNDSVD",5,nothing)]) # all
    frpx = "$(mtdstr)_af$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, labelsize=20, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db_all.png"),fig,px_per_unit=2)

# fxs
fig = Figure(resolution=(600,450))
maxplottimes = [0.2,0.2,0.2]
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), nothing),
                xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("lcsvd","_sbc","SBC",2,nothing),
                                                                  ("lcsvd","_isvd","ISVD",3,nothing),
                                                                  ("lcsvd","_nndsvd","NNDSVD",5,nothing)]) # all
    frpx = "$(mtdstr)_fx$(submtdstr)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, labelsize=20, position = :rt) # halign = :left, valign = :top
save(joinpath(subworkpath,"penalty$(SNR)db_all.png"),fig,px_per_unit=2)

end # for SNR





iters=[79]; rt1s = 0; nsuccess = 0
for iter in iters
    @show iter
    U = load(joinpath(subworkpath,"sbc_error$(iter).jld2"),"U")
    try
    rt11 = @elapsed M0 = sbc(U)
    catch e
        @warn e
    end
end
