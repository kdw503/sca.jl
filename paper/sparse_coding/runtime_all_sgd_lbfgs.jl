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
subworkpath = joinpath(workpath,"paper","sparse_coding")

# in julia REPL> ARGS = ["\"tsvd_test\"","[\"pcb_tsvd\"]", "1", "2","0","1","15","150","120","0.1","800"]
# ARGS = ["0.3", "1e-3","1e-2","1e-6","100","100"]
r = eval(Meta.parse(ARGS[1]))
ur = eval(Meta.parse(ARGS[2]))
nr = eval(Meta.parse(ARGS[3]))
inner_tol = eval(Meta.parse(ARGS[4]))
smaxiter = eval(Meta.parse(ARGS[5]))
maxiter = eval(Meta.parse(ARGS[6]))
inner_maxiter = 1000

subdirname = "sa10_store_r$(r)_ur$(ur)_nr$(nr)_itol$(inner_tol)_iit$(inner_maxiter)_sit$(smaxiter)_it$(maxiter)"
@show subdirname
dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.1
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))
bias = 0.1

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncs, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
(m,n,p) = (size(X)...,ncs)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

subtract_bg = false
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    bg = W*fill(mean(H),1,n)
    X .-= bg
end

prefix = "pcb"
noc = ncs; nac = 0

num_experiments=30
rt1s = 0; nsuccess = 0
maskth=0.25; maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
tol=-1; maxiter = maxiter # Int(ceil(log(eps(eltype(X)))/log(r)))
αrng = [0.005]

# for (inner_maxiter, r) in [(10,0.3),(10,0.5),(10,0.99),
#                            (100,0.3),(100,0.5),(100,0.99),
#                            (1000,0.3),(1000,0.5),(1000,0.99), ]
# for maxiter in [1,2,3,5,10]
for α in αrng
    @show α; flush(stdout)
    (tailstr,β) = ("_sp",0.)
    α1=α2=α; β1=β2=0
for iter in 1:num_experiments
    @show iter; flush(stdout)
    useprecond = false; usedenoiseUVt = false; uselv = false
    for optim_method in [:lbfgs, :sgd_injectnoise]
        @show optim_method
        initmethod = :isvd
        fprex = "$(prefix)_sa1_$(dataset)_$(initmethod)"
        rt1 = @elapsed U, Vt, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
        V = copy(Vt'); N0t = copy(N0')

        dd = Dict()
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseUVt=usedenoiseUVt, maskW=maskW, maskH = maskH, optim_method = optim_method,
            denoisefilter=:avg, uselv=uselv, imgsz=imgsz, maxiter = maxiter, smaxiter = smaxiter, inner_maxiter=inner_maxiter,
            store_trace = true, store_inner_trace = true, show_trace = true, store_sparsity_nneg = true, allow_f_increases = true,
            f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0,
            ur=ur, nr=nr);
        M1, N1t = copy(M0), copy(N0t)
        rst = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
        alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        store_sparsity_nneg = false
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

        W1, H1 = rst0.W, rst0.Ht'
        LCSVD.normalizeW!(W1,H1)
        fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
        nodr = LCSVD.matchedorder(ml,noc)
        W1, H1 = W1[:,nodr], H1[nodr,:]
        fname = joinpath(subworkpath,subdirname,"$(fprex)_$(alg.optim_method)_a$(α)_b$(β)_af$(fv)_it$(rst0.niters)_rt$(rt2)")
        imsave_data(dataset,fname,W1,H1,imgsz,100; saveH=false, verbose=false)

        f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niters); totalniters = sum(niters)
        avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
        symss = LCSVD.getdata(rst.traces,:invs); sparWss = LCSVD.getdata(rst.traces,:sparseWs); sparHss = LCSVD.getdata(rst.traces,:sparseHs)
        avgfits = Float64[]; inner_fxs = Float64[]; sympens = Float64[]; sparWpens = Float64[]; sparHpens = Float64[]; rt2s = Float64[]
        for (iter,(afs,fxs,syms,sparWs,sparHs)) in enumerate(zip(avgfitss, fxss, symss, sparWss, sparHss))
            isempty(afs) && continue
            append!(avgfits,afs); append!(inner_fxs,fxs); append!(sympens,syms); append!(sparWpens,sparWs); append!(sparHpens,sparHs)
            if iter == 1
                rt2i = 0.
            else
                rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
            end
            append!(rt2s,rt2i)
        end
        dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = 0; dd["rt2s"] = rt2s
        dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
        dd["sympens"] = sympens; dd["sparWpens"] = sparWpens; dd["sparHpens"] = sparHpens
        dd["laps"] = rst0.laps[2:end]-rst0.laps[1:end-1]
        if true#iter == num_experiments
            metadata = Dict()
            metadata["r"] = r; metadata["ur"] = ur; metadata["nr"] = nr; metadata["initmethod"] = initmethod
            metadata["inner_tol"] = inner_tol; metadata["inner_maxiter"] = inner_maxiter; metadata["smaxiter"] = smaxiter
            metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond; metadata["usedenoiseUVt"] = usedenoiseUVt
            metadata["denoisefilter"] = alg.denoisefilter; metadata["alpha"] = α; metadata["beta"] = β
        end
        save(joinpath(subworkpath,subdirname,"data","$(fprex)$(tailstr)_$(alg.optim_method)_a$(α)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end # for iter
rt1avg = rt1s/nsuccess
end # for α 
#=
include(joinpath(workpath,"setup_plot.jl"))
using Interpolations

subdirnames = ["innertol1.0e-6_inneriter10_iter50_r0.3",
               "innertol1.0e-6_inneriter10_iter100_r0.5",
               "innertol1.0e-6_inneriter10_iter4000_r0.99",
               "innertol1.0e-6_inneriter100_iter50_r0.3",
               "innertol1.0e-6_inneriter100_iter100_r0.5",
               "innertol1.0e-6_inneriter100_iter4000_r0.99",
               "innertol1.0e-6_inneriter1000_iter50_r0.3",
               "innertol1.0e-6_inneriter1000_iter100_r0.5",
               "innertol1.0e-6_inneriter1000_iter4000_r0.99"
              ]
for subdirname in subdirnames
    @show subdirname
itp_time_resol=10000 # if the time resolution is too sparse, increase this.
for α in αrng
    for (prefix, initmethods, tailstrs) in [("pcb", ["DICT","isvd","sbc","BPDN"], ["_sp","_sp","_sp","_sp"])]
        rt2_min = Inf; rt2sdict = []; rt2sisvd = []; rt2ssbc = []
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            fprex = "$(prefix)_$(dataset)_$(initmethod)"
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,subdirname,"data","$(fprex)$(tailstr)_a$(α)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; #@show initmethod, rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
                initmethod == "DICT" ? push!(rt2sdict,rt2s[end]) :
                              "isvd" ? push!(rt2sisvd,rt2s[end]) :
                                       push!(rt2ssbc,rt2s[end])
            end
        end
        rt2sdict_mean = isempty(rt2sdict) ? NaN : mean(rt2sdict)
        rt2sisvd_mean = isempty(rt2sisvd) ? NaN : mean(rt2sisvd)
        rt2ssbc_mean  = isempty(rt2ssbc) ?  NaN : mean(rt2ssbc)
        @show α, rt2sdict_mean, rt2sisvd_mean, rt2ssbc_mean
        rt2_min = floor(rt2_min, digits=4)
        rng = range(0,stop=rt2_min,length=itp_time_resol)
        
        stat_af_dict=[]; stat_af_sbc=[]; stat_af_sbc_p2=[]; stat_af_isvd=[]; stat_af_nndsvd=[]
        stat_fx_dict=[]; stat_fx_sbc=[]; stat_fx_sbc_p2=[]; stat_fx_isvd=[]; stat_fx_nndsvd=[]
        for (initmethod,tailstr) in zip(initmethods,tailstrs)
            afs=[]; fxs=[]
            fprex = "$(prefix)_$(dataset)_$(initmethod)"
            for iter in 1:num_experiments
                @show tailstr, iter
                dd = load(joinpath(subworkpath,subdirname,"data","$(fprex)$(tailstr)_a$(α)_results$(iter).jld2"))
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
            initmethod == "DICT" && (push!(stat_af_dict,means); push!(stat_af_dict,stds))
            initmethod == "sbc" && (push!(stat_af_sbc,means); push!(stat_af_sbc,stds))
            initmethod == "sbc_p2" && (push!(stat_af_sbc_p2,means); push!(stat_af_sbc_p2,stds))
            initmethod == "isvd" && (push!(stat_af_isvd,means); push!(stat_af_isvd,stds))
            initmethod == "nndsvd" && (push!(stat_af_nndsvd,means); push!(stat_af_nndsvd,stds))
            # fxs
            fxs = hcat(fxs...)
            means = dropdims(mean(fxs,dims=2),dims=2)
            stds = dropdims(std(fxs,dims=2),dims=2)
            initmethod == "DICT" && (push!(stat_fx_dict,means); push!(stat_fx_dict,stds))
            initmethod == "sbc" && (push!(stat_fx_sbc,means); push!(stat_fx_sbc,stds))
            initmethod == "sbc_p2" && (push!(stat_fx_sbc_p2,means); push!(stat_fx_sbc_p2,stds))
            initmethod == "isvd" && (push!(stat_fx_isvd,means); push!(stat_fx_isvd,stds))
            initmethod == "nndsvd" && (push!(stat_fx_nndsvd,means); push!(stat_fx_nndsvd,stds))
        end
        fprex="$(prefix)"
        save(joinpath(subworkpath,subdirname,"$(fprex)_a$(α)_runtime_vs_avgfits.jld2"),"rng",rng,
            "stat_af_dict", stat_af_dict, "stat_af_sbc", stat_af_sbc, "stat_af_sbc_p2", stat_af_sbc_p2,
            "stat_af_isvd", stat_af_isvd, "stat_af_nndsvd", stat_af_nndsvd, "stat_fx_dict", stat_fx_dict,
            "stat_fx_sbc", stat_fx_sbc, "stat_fx_sbc_p2", stat_fx_sbc_p2, "stat_fx_isvd", stat_fx_isvd,
            "stat_fx_nndsvd", stat_fx_nndsvd)
    end # for (prefix, initmethods, tailstrs)
end # for α
end # for subdirname

tmppath = "delme"
for subdirname in subdirnames
    @show subdirname
z = 0.5;
for (α, maxplottime, ybtm) in [(0.0001,10.0,0.0)]
    ylimits=(ybtm,1.1)
    plottime = Inf
    for (mtdstr, submtdstrs) in [("$(prefix)",["_dict"])]
        @show mtdstr
        ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
    #    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
        # dirname = eval(Meta.parse("joinpath(subworkpath,\"$(subdirname)\")"))
        # @show dirname
        eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(subdirname)\",\"$(mtdstr)_a$(α)_runtime_vs_avgfits.jld2\"))"))
        rng = eval(Meta.parse("$(ddsym)[\"rng\"]"))
        eval(Meta.parse("$(mtdstr)rng=$(ddsym)[\"rng\"]"))
        plottime = plottime > rng[end] ? rng[end] : plottime
        for submtdstr in submtdstrs
            # avgfits
            frpx = "$(mtdstr)_af$(submtdstr)"
            eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat_af$(submtdstr)\"][1]"))
            eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat_af$(submtdstr)\"][2]"))
            @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + $(z)*$(Symbol("$(frpx)_stds"))))
            @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - $(z)*$(Symbol("$(frpx)_stds"))))
            # fxs
            frpx = "$(mtdstr)_fx$(submtdstr)"
            eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat_fx$(submtdstr)\"][1]"))
            eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat_fx$(submtdstr)\"][2]"))
            @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + $(z)*$(Symbol("$(frpx)_stds"))))
            @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - $(z)*$(Symbol("$(frpx)_stds"))))
        end
    end

    alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
    plotrng = Colon()

    # avgfits
    fig = Figure(size=(600,450))
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), ylimits),
                    xlabel = "time(sec)", ylabel = "fit", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([#("pcb","_sbc","SBC",2,nothing),
                                                                    # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                    #("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb","_dict","DICT",2,nothing)]) # all
        frpx = "$(mtdstr)_af$(submtdstr)"
        ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
        bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
        lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
    end

    #axislegend(ax, labelsize=20, position = :rb) # halign = :left, valign = :top
    save(joinpath(subworkpath,tmppath,subdirname,"fits_a$(α)_all.png"),fig,px_per_unit=2)

    # fxs
    fig = Figure(size=(600,450))
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), nothing),
                    yscale=log10,xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([#("pcb","_sbc","SBC",2,nothing),
                                                                    # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                    #("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb","_dict","DICT",2,nothing)]) # all
        frpx = "$(mtdstr)_fx$(submtdstr)"
        ln = lines!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
        bnd = band!(ax, eval(Symbol("$(mtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
        lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
    end

    #axislegend(ax, labelsize=20, position = :rt) # halign = :left, valign = :top
    save(joinpath(subworkpath,tmppath,subdirname,"penalty_a$(α)_all.png"),fig,px_per_unit=2)

end # for α

end 
=#