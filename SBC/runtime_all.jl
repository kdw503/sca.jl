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
imgsz = (40,20); lengthT = 1000; prefix = "pcb"; num_experiments=30
rt1s = 0; nsuccess = 0
maskth=0.25; maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
inner_tol = 1e-7; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100))
    r=0.3; maxiter = 100

for iter in 1:num_experiments
    @show iter
    X, imsz, lhT, noc, gtnoc, datadic = load_data(:fakecells; sigma=5.0, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=0.1, useCalciumT=true,
            inhibitindices=0, issave=false, isload=false, gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false);
    (m,n,p) = (size(X)...,noc)
    nac =0; nc = noc + nac
    gtW, gtH = (datadic["gtW"], datadic["gtH"])

    useprecond = true; usedenoiseUVt = false; uselv = false
    for initmethod in [:sbc,:sbc_p2,:isvd,:nndsvd]
        (tailstr,α,β) = ("_sp",0.005,0.)
        fprex = "$(prefix)$(SNR)db_$(initmethod)"

        dd = Dict()
        α1=α2=α; β1=β2=β
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:tsvd)
        LCSVD.normalizeW!(Wp,Hp);
        dataset == :fakecells && begin
            avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wp, Hp; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            Wp1, Hp1 = Wp[:,nodr], Hp[nodr,:]
        end
        fname = joinpath(subworkpath,"$(fprex)_af$(avgfit)_rt$(rt1)")
        #imsave_data(dataset,fname,Wp1,Hp1,imgsz,100; saveH=false, verbose=false)

        V = copy(H0'); N0t = copy(N0')
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseUVt=usedenoiseUVt, maskW=maskW, maskH = maskH,
            denoisefilter=:avg, uselv=uselv, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
            store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
        M1, N1t = copy(M0), copy(N0t)
        rst = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
        alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

        @show tailstr, rt2
        W1, H1 = rst0.W, rst0.Ht'
        LCSVD.normalizeW!(W1,H1);
        dataset == :fakecells && begin
            avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            W1, H1 = W1[:,nodr], H1[nodr,:]
        end
        fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
        #imsave_data(dataset,fname,W1,H1,imgsz,100; saveH=false, verbose=false)

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
            metadata["r"] = r; metadata["initmethod"] = initmethod
            metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
            metadata["usedenoiseUVt"] = usedenoiseUVt; metadata["denoisefilter"] = alg.denoisefilter
            metadata["alpha"] = α; metadata["beta"] = β
        end
        save(joinpath(subworkpath,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end
rt1avg = rt1s/nsuccess

include(joinpath(workpath,"setup_plot.jl"))
using Interpolations

SNRs = [0]; noc=15; factor=1
itp_time_resol=1000

for (prefix, initmethods, tailstrs) in [("pcb", ["isvd","sbc", "sbc_p2", "nndsvd"], ["_sp","_sp", "_sp", "_sp"])]
    for SNR = SNRs
        rt2_min = Inf
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            fprex="$(prefix)$(SNR)db_$(initmethod)"
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,"$(fprex)$(tailstr)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; @show initmethod, rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
            end
        end
        rt2_min = floor(rt2_min, digits=4)
        rng = range(0,stop=rt2_min,length=itp_time_resol)
        
        stat_af_sbc=[]; stat_af_sbc_p2=[]; stat_af_isvd=[]; stat_af_nndsvd=[]
        stat_fx_sbc=[]; stat_fx_sbc_p2=[]; stat_fx_isvd=[]; stat_fx_nndsvd=[]
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            afs=[]; fxs=[]
            fprex="$(prefix)$(SNR)db_$(initmethod)"
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
            initmethod == "sbc_p2" && (push!(stat_af_sbc_p2,means); push!(stat_af_sbc_p2,stds))
            initmethod == "isvd" && (push!(stat_af_isvd,means); push!(stat_af_isvd,stds))
            initmethod == "nndsvd" && (push!(stat_af_nndsvd,means); push!(stat_af_nndsvd,stds))
            # fxs
            fxs = hcat(fxs...)
            means = dropdims(mean(fxs,dims=2),dims=2)
            stds = dropdims(std(fxs,dims=2),dims=2)
            initmethod == "sbc" && (push!(stat_fx_sbc,means); push!(stat_fx_sbc,stds))
            initmethod == "sbc_p2" && (push!(stat_fx_sbc_p2,means); push!(stat_fx_sbc_p2,stds))
            initmethod == "isvd" && (push!(stat_fx_isvd,means); push!(stat_fx_isvd,stds))
            initmethod == "nndsvd" && (push!(stat_fx_nndsvd,means); push!(stat_fx_nndsvd,stds))
        end
        fprex="$(prefix)$(SNR)db"
        save(joinpath(subworkpath,"$(fprex)_runtime_vs_avgfits.jld2"),"rng",rng,
            "stat_af_sbc", stat_af_sbc, "stat_af_sbc_p2", stat_af_sbc_p2, "stat_af_isvd", stat_af_isvd, "stat_af_nndsvd", stat_af_nndsvd,
            "stat_fx_sbc", stat_fx_sbc, "stat_fx_sbc_p2", stat_fx_sbc_p2, "stat_fx_isvd", stat_fx_isvd, "stat_fx_nndsvd", stat_fx_nndsvd)
    end
end

tmppath = ""
z = 0.5; ylimits=(0.6,1.1); maxplottime = 0.2
for (idx,SNR) = enumerate(SNRs)
    plottime = Inf
    for (mtdstr, submtdstrs) in [("$(prefix)",["_sbc", "_sbc_p2", "_isvd", "_nndsvd"])]
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
    maxplottimes = [maxplottime,0.1,0.1]
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), ylimits),
                    xlabel = "time(sec)", ylabel = "average fit", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb","_sbc","SBC",2,nothing),
                                                                    ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    ("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb","_nndsvd","NNDSVD",5,nothing)]) # all
        frpx = "$(mtdstr)_af$(submtdstr)"
        ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
        bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
        lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
    end

    axislegend(ax, labelsize=20, position = :rb) # halign = :left, valign = :top
    save(joinpath(subworkpath,"avgfits$(SNR)db_all.png"),fig,px_per_unit=2)

    # fxs
    fig = Figure(resolution=(600,450))
    maxplottimes = [maxplottime,0.1,0.1]
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottimes[idx]#=min(maxplottimes[idx],plottime)=#), nothing),
                    xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb","_sbc","SBC",2,nothing),
                                                                    ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    ("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb","_nndsvd","NNDSVD",5,nothing)]) # all
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
