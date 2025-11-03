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

inner_tol = eval(Meta.parse(ARGS[1]))
inner_maxiter = eval(Meta.parse(ARGS[2]))
maxiter = eval(Meta.parse(ARGS[3]))
r = eval(Meta.parse(ARGS[4]))
@show  inner_tol, inner_maxiter, maxiter, r; flush(stdout)

function init_dictionary(X, K)
    D = X[:, rand(1:end, K)]
    D ./= sqrt.(sum(D .^ 2, dims=1))  # Normalize columns
    return D
end

dataset = :natural
ddinit = load(joinpath(subworkpath,"allinit.jld2"))
X_whitened = ddinit["X_whitened"][1]
U, Vt, D = ddinit["SVD"]; V = Vt'; ncs = size(U,2)
(m,n,p) = (size(X_whitened)...,ncs)
gtW, gtH = (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
imgsz=(12,12); lengthT=size(Vt,2)

prefix = "pcb"
noc = ncs; nac = 0

num_experiments=1
rt1s = 0; nsuccess = 0
maskth=0.25; maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
tol=-1; maxiter = maxiter # Int(ceil(log(eps(eltype(X)))/log(r)))
αrng = [1e-4]

for (inner_maxiter, r) in [(10,0.3),(10,0.5),(10,0.99),
                           (100,0.3),(100,0.5),(100,0.99),
                           (1000,0.3),(1000,0.5),(1000,0.99), ]
for maxiter in [20,30,50,100]
subdirname = "innertol$(inner_tol)_inneriter$(inner_maxiter)_iter$(maxiter)_r$(r)"
@show subdirname
for α in αrng
    @show α; flush(stdout)
    (tailstr,β) = ("_sp",0.)
    α1=0; α2=α; β1=β2=0
for iter in 1:num_experiments
    @show iter; flush(stdout)
    useprecond = false; usedenoiseUVt = false; uselv = false
    for initmethod in [:DICT]#,:sbc_p2,:nndsvd
        @show initmethod
        fprex = "$(prefix)_$(dataset)_$(initmethod)"
        if initmethod == :DICT
            Winit = init_dictionary(X_whitened, noc)
            M0 = U'Winit; N0t = rand(noc,noc)
            Hinit = N0t'*Vt
        else
            Winit, Hinit, M0, N0t, _ = ddinit[String(initmethod)]
            LCSVD.normalizeW!(Wp,Hp);
        end
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseUVt=usedenoiseUVt, maskW=maskW, maskH = maskH,
            denoisefilter=:avg, uselv=uselv, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
            store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
        alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X_whitened, U, V, D, M1, N1t);

        W1, H1 = rst0.W, rst0.Ht'
        LCSVD.normalizeW!(W1,H1);
        fit = LCSVD.fitd(X_whitened,W1*H1); @show fit; flush(stdout)
        fname = joinpath(subworkpath,"figures",subdirname,"$(fprex)_a$(α)_b$(β)_f$(fit)_it$(rst0.niters)_rt$(rt2)")
        imsave_data(dataset,fname,W1,H1,imgsz,100; saveH=false, verbose=false)
    end
end # for iter
rt1avg = rt1s/nsuccess
end # for α 
end
end

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
    for (prefix, initmethods, tailstrs) in [("pcb", ["DICT"], ["_sp"])]
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
