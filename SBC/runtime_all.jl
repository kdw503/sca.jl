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
subworkpath = joinpath(workpath,"SBC/naomi/gbg_no_subbg_hungaf")
weighted=false; delta_f = false; subtract_bg = false; bias=0.1;

dataset = :naomi; SNR = 0; pavg = 5.0
(imgsz0, lengthT0, hplot_space) = dataset == :naomi ?     ((25,25), 5000, -10000) :
                                  dataset == :fakecells ? ((40,20), 1000, -5) :
                                                          ((40,20), 1000, -5)
issaveimg = true; figsize=(900,600)
inhibitindices=0; factor = 1; orthogonal=false
lpfilter = dataset ∈ [:neurofinder] ? :meanT : :none; filterstr = "_$(lpfilter)"
noisestr = dataset == :fakecells ? "$(SNR)dB" : dataset == :naomi ? "$(pavg)mW" : ""
avg_rad=7.0; pavg=5.0; psf_NA=0.3

maskth=0.25; makepositive = true; tol=-1
sqfactor = Int(floor(sqrt(factor)))
vres0 = 1.0; vres = sqfactor*vres0 # v resolution for naomi dataset
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*lengthT0; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

prefix = "pcb"; num_experiments=10
rt1s = 0; nsuccess = 0
inner_tol = 1e-7; inner_maxiter0 = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100))
r=0.3; tol=0

savedata = true; maxiter0 = 100 # isvd(2), sbc(10), varimax(13)
for α in [0.0005]
    @show α
(tailstr,β) = ("_sp",0.)
for iter in 1:num_experiments
    @show iter
    seed = iter
    X, imsz, lhT, noc, gtnoc, datadic = load_data(dataset; seed=seed, sigma=5.0, imgsz=imgsz,
            lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, orthogonal=orthogonal, inhibitindices=0,
            avg_rad=avg_rad, pavg=pavg, psf_NA=psf_NA,
            issave=true, isload=true, gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false);
    (m,n,p) = (size(X)...,noc)
    nac =0; nc = noc + nac
    gtW, gtH = (datadic["gtW"], datadic["gtH"])

    if dataset in [:naomi]
        if delta_f
            powers, mod_vals = datadic["powers"], datadic["mod_vals"]
            gtH_nobase = (gtH./powers.-mod_vals).*powers
        else
            gtH_nobase = gtH
        end
        dt = datadic["dt"]; inh_idx = datadic["inh_idx"]
    else
        gtH_nobase = gtH
    end

    if subtract_bg
        rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
        NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
        LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
    #    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
        bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
    end

    useprecond = true; usedenoiseUVt = false; uselv = false
    for initmethod in [:sbc,:isvd,:varimax]#,:sbc_p2,:nndsvd
        maxiter = savedata ? maxiter0 :
                  initmethod == :sbc ? maxiter0 :
                  initmethod == :isvd ? maxiter0 :
                  initmethod == :varimax ? 1 : maxiter0
        inner_maxiter = savedata ? inner_maxiter0 :
                  initmethod == :sbc ? inner_maxiter0 :
                  initmethod == :isvd ? inner_maxiter0 :
                  initmethod == :varimax ? inner_maxiter0 : inner_maxiter0
        fprex = dataset==:fakecells ? "$(prefix)$(SNR)db$(seed)_$(initmethod)" : "$(prefix)_$(dataset)$(seed)_$(initmethod)"

        dd = Dict()
        α1=α2=α; β1=β2=β
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
        LCSVD.normalizeW!(Wp,Hp);
        dataset in [:fakecells, :naomi] && begin
            if (dataset == :naomi) && delta_f
                Hp_nobase = subtract_baseline(Hp; q=0.01)
            else
                Hp_nobase = Hp
            end
            avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH_nobase, Wp, Hp_nobase; weighted=weighted, clamp=false)
            nodr = matchedorder(ml,noc)
            Wp1, Hp1 = Wp[:,nodr], Hp[nodr,:]
        end
        fname = joinpath(subworkpath,"$(fprex)_af$(avgfit)_rt$(rt1)")
        imsave_data(dataset,fname,Wp1,Hp1,imgsz,100; saveH=false, verbose=false)

        V = copy(H0'); N0t = copy(N0')
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseUVt=usedenoiseUVt, maskW=maskW, maskH = maskH,
            denoisefilter=:avg, uselv=uselv, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
            store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
if savedata
        M1, N1t = copy(M0), copy(N0t)
        rst = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH, weighted=weighted, delta_f=delta_f);
end
        alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

        @show tailstr, rt2
        W1, H1 = rst0.W, rst0.Ht'
        LCSVD.normalizeW!(W1,H1);
        dataset in [:fakecells, :naomi] && begin
            if (dataset == :naomi) && delta_f
                H1_nobase = subtract_baseline(H1; q=0.01)
            else
                H1_nobase = H1
            end
            avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH_nobase, W1, H1_nobase; weighted=weighted, clamp=false)
            nodr = matchedorder(ml,noc)
            W1, H1 = W1[:,nodr], H1[nodr,:]
        end
        fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_it$(rst0.niters)_iit$(inner_maxiter)_f$(rst0.objvalue)_af$(avgfit)_rt$(rt2)")
        imsave_data(dataset,fname,W1,H1,imgsz,100; saveH=false, verbose=false)
if savedata
        f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niters); totalniters = sum(niters)
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
            metadata["seed"] = seed; metadata["r"] = r; metadata["initmethod"] = initmethod
            metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
            metadata["usedenoiseUVt"] = usedenoiseUVt; metadata["denoisefilter"] = alg.denoisefilter
            metadata["alpha"] = α; metadata["beta"] = β
        end
        save(joinpath(subworkpath,"$(fprex)$(tailstr)_a$(α)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end
    end # for initmethod
end # for iter
end # for α

include(joinpath(workpath,"setup_plot.jl"))
using Interpolations

SNRs = [0]; noc=15; factor=1
itp_time_resol=1000
for α in [0.0005]
for (prefix, initmethods, tailstrs) in [("pcb", ["isvd","sbc","varimax" #=, "sbc_p2", "nndsvd"=#], ["_sp","_sp","_sp"#=, "_sp", "_sp"=#])]
    for SNR = SNRs
        rt2_min = Inf
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            for iter in 1:num_experiments
                seed = iter
                fprex = dataset==:fakecells ? "$(prefix)$(SNR)db$(seed)_$(initmethod)" : "$(prefix)_$(dataset)$(seed)_$(initmethod)"
                fn = joinpath(subworkpath,"$(fprex)$(tailstr)_a$(α)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; @show initmethod, rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
            end
        end
        rt2_min = floor(rt2_min, digits=4)
        rng = range(0,stop=rt2_min,length=itp_time_resol)
        
        stat_af_sbc=[]; stat_af_sbc_p2=[]; stat_af_isvd=[]; stat_af_nndsvd=[]; stat_af_varimax=[]
        stat_fx_sbc=[]; stat_fx_sbc_p2=[]; stat_fx_isvd=[]; stat_fx_nndsvd=[]; stat_fx_varimax=[]
        rt1_sbc=[];     rt1_sbc_p2=[];     rt1_isvd=[];     rt1_nndsvd=[];     rt1_varimax=[]
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            afs=[]; fxs=[]
            rt1sum = 0.
            for iter in 1:num_experiments
                @show tailstr, iter
                seed = iter
                fprex = dataset==:fakecells ? "$(prefix)$(SNR)db$(seed)_$(initmethod)" : "$(prefix)_$(dataset)$(seed)_$(initmethod)"
                dd = load(joinpath(subworkpath,"$(fprex)$(tailstr)_a$(α)_results$(iter).jld2"))
                rt1sum += dd["data"]["rt1"]
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
            rt1 = rt1sum/num_experiments
             # avgfits
            avgfits = hcat(afs...)
            means = dropdims(mean(avgfits,dims=2),dims=2)
            stds = dropdims(std(avgfits,dims=2),dims=2)
            initmethod == "sbc" && (push!(stat_af_sbc,means); push!(stat_af_sbc,stds); rt1_sbc = rt1)
            initmethod == "sbc_p2" && (push!(stat_af_sbc_p2,means); push!(stat_af_sbc_p2,stds); rt1_sbc_p2 = rt1)
            initmethod == "isvd" && (push!(stat_af_isvd,means); push!(stat_af_isvd,stds); rt1_isvd = rt1)
            initmethod == "nndsvd" && (push!(stat_af_nndsvd,means); push!(stat_af_nndsvd,stds); rt1_nndsvd = rt1)
            initmethod == "varimax" && (push!(stat_af_varimax,means); push!(stat_af_varimax,stds); rt1_varimax = rt1)
            # fxs
            fxs = hcat(fxs...)
            means = dropdims(mean(fxs,dims=2),dims=2)
            stds = dropdims(std(fxs,dims=2),dims=2)
            initmethod == "sbc" && (push!(stat_fx_sbc,means); push!(stat_fx_sbc,stds))
            initmethod == "sbc_p2" && (push!(stat_fx_sbc_p2,means); push!(stat_fx_sbc_p2,stds))
            initmethod == "isvd" && (push!(stat_fx_isvd,means); push!(stat_fx_isvd,stds))
            initmethod == "nndsvd" && (push!(stat_fx_nndsvd,means); push!(stat_fx_nndsvd,stds))
            initmethod == "varimax" && (push!(stat_fx_varimax,means); push!(stat_fx_varimax,stds))
        end
        fprex = dataset==:fakecells ? "$(prefix)$(SNR)db" : "$(prefix)_$(dataset)"
        save(joinpath(subworkpath,"$(fprex)_a$(α)_runtime_vs_avgfits.jld2"), "rng", rng,
            "rt1_sbc", rt1_sbc, "rt1_sbc_p2", rt1_sbc_p2, "rt1_isvd", rt1_isvd, "rt1_nndsvd", rt1_nndsvd, "rt1_varimax", rt1_varimax, 
            "stat_af_sbc", stat_af_sbc, "stat_fx_sbc", stat_fx_sbc, "stat_af_sbc_p2", stat_af_sbc_p2, "stat_fx_sbc_p2", stat_fx_sbc_p2,
            "stat_af_isvd", stat_af_isvd, "stat_fx_isvd", stat_fx_isvd, "stat_af_nndsvd", stat_af_nndsvd, "stat_fx_nndsvd", stat_fx_nndsvd,
            "stat_af_varimax", stat_af_varimax, "stat_fx_varimax", stat_fx_varimax)
    end
end
end

tmppath = ""
z = 0.5; ylimits= (0.4,0.7) # naomi (0.6,1.1)
for (α, maxplottime) in [(0.0005,1.4)]#, (0.05,0.28)] # , (0.5,0.03), (5.0,0.2), (50.0,0.2)
for (idx,SNR) = enumerate(SNRs)
    plottime = Inf
    for (mtdstr, submtdstrs) in [("$(prefix)",["_sbc", "_isvd", "_varimax"#=, "_sbc_p2", "_nndsvd"=#])]
        @show mtdstr
        fprex = dataset==:fakecells ? "$(mtdstr)$(SNR)db" : "$(mtdstr)_$(dataset)"
        ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
    #    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
        eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(fprex)_a$(α)_runtime_vs_avgfits.jld2\"))"))
        rng = eval(Meta.parse("$(ddsym)[\"rng\"]"))
        eval(Meta.parse("$(mtdstr)rng=$(ddsym)[\"rng\"]"))
        plottime = plottime > rng[end] ? rng[end] : plottime
        for submtdstr in submtdstrs
            eval(Meta.parse("rt1$(submtdstr)=$(ddsym)[\"rt1$(submtdstr)\"]"))
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
    fprex = dataset==:fakecells ? "$(SNR)db" : "$(dataset)"

    # avgfits
    fig = Figure(resolution=(600,450))
    wstring = weighted ? "weighted" : ""
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), ylimits),
                    xlabel = "time(sec)", ylabel = "$(wstring) average fit", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb","_sbc","SBC",2,nothing),
                                                                    # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                    ("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb","_varimax","VariMax",4,nothing),]) # all
        frpx = "$(mtdstr)_af$(submtdstr)"; lbl *= " ($(round(eval(Symbol("rt1$(submtdstr)")),digits=3))sec)"
        ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
        bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
        lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
    end

    axislegend(ax, labelsize=20, position = :rb) # halign = :left, valign = :top
    save(joinpath(subworkpath,"$(wstring)avgfits_$(fprex)_a$(α)_numexpr$(num_experiments).png"),fig,px_per_unit=2)

    # fxs
    fig = Figure(resolution=(600,450))
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), nothing),
                    xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb","_sbc","SBC",2,nothing),
                                                                    # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                    ("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb","_varimax","VariMax",4,nothing),]) # all
        frpx = "$(mtdstr)_fx$(submtdstr)"; lbl *= " ($(round(eval(Symbol("rt1$(submtdstr)")),digits=3))sec)"
        ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
        bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
        lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
    end

    axislegend(ax, labelsize=20, position = :rt) # halign = :left, valign = :top
    save(joinpath(subworkpath,"penalty_$(fprex)db_a$(α)_numexpr$(num_experiments).png"),fig,px_per_unit=2)

end # for SNR
end # for α




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

function sbc(U::AbstractMatrix; kwargs...)
    r = size(U, 2); noc = 500
    M = Matrix{eltype(U)}(undef, r, noc)
    m2 = zeros(r)
    q = zeros(r)
    for j in 1:noc
        @show j
        fill!(q, 0)
        q[j] = 1
        # Perform a Gramm-Schmidt orthogonalization of q against the columns of M[:,1:j-1]
        for k in 1:j-1
            q .-= (M[j,k] / m2[k]) .* M[:,k]
        end
        M[:,j], _ = sbc(U, q; kwargs...)
        m2[j] = sum(abs2, @view(M[:,j]))
    end
    return M
end

α = 0.0005; seed = iter = 1; tailstr="_sp"

z = 0.5; ylimits=(0.6,1.1)
(α, maxplottime) = (0.0005,0.32)
alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon()

# avgfits
fig = Figure(resolution=(600,450))
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), ylimits),
                xlabel = "time(sec)", ylabel = "average fit", xlabelsize=20, ylabelsize=20,
                xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb","_sbc","SBC",2,nothing),
                                                                # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                ("pcb","_isvd","ISVD",3,nothing),
                                                                ("pcb","_varimax","VariMax",4,nothing),]) # all
    frpx = "$(mtdstr)_af$(submtdstr)"; lbl *= " ($(round(eval(Symbol("rt1$(submtdstr)")),digits=3))sec)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

for initmethod in [:sbc,:isvd,:varimax]
    fprex = "$(prefix)$(SNR)db$(seed)_$(initmethod)"
    dd = load(joinpath(subworkpath,"$(fprex)$(tailstr)_a$(α)_results$(iter).jld2"))
end
axislegend(ax, labelsize=20, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db_a$(α)_numexpr$(num_experiments).png"),fig,px_per_unit=2)

# fxs
fig = Figure(resolution=(600,450))
ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), nothing),
                xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
lns = Dict(); bnds=Dict()
for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb","_sbc","SBC",2,nothing),
                                                                # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                ("pcb","_isvd","ISVD",3,nothing),
                                                                ("pcb","_varimax","VariMax",4,nothing),]) # all
    frpx = "$(mtdstr)_fx$(submtdstr)"; lbl *= " ($(round(eval(Symbol("rt1$(submtdstr)")),digits=3))sec)"
    ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
end

axislegend(ax, labelsize=20, position = :rt) # halign = :left, valign = :top
save(joinpath(subworkpath,"penalty$(SNR)db_a$(α)_numexpr$(num_experiments).png"),fig,px_per_unit=2)
