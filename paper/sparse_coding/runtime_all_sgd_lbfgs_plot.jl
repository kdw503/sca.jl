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

for (inner_maxiter, r, maxiter) in [(10,0.3,100),(10,0.5,500),(10,0.99,4000),
                           (100,0.3,100),(100,0.5,500),(100,0.99,4000),
                           (1000,0.3,100),(1000,0.5,500),(1000,0.99,4000), ]
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
subdirnames = ["innertol1.0e-7_inneriter10_iter50_r0.3",
               "innertol1.0e-7_inneriter10_iter100_r0.5",
               #"innertol1.0e-7_inneriter10_iter4000_r0.99",
               "innertol1.0e-7_inneriter100_iter50_r0.3",
               "innertol1.0e-7_inneriter100_iter100_r0.5",
               "innertol1.0e-7_inneriter100_iter500_r0.5",
               #"innertol1.0e-7_inneriter100_iter4000_r0.99",
               #"innertol1.0e-7_inneriter1000_iter50_r0.3",
               #"innertol1.0e-7_inneriter1000_iter100_r0.5",
               "innertol1.0e-7_inneriter1000_iter4000_r0.99"
              ]
subdirnames = ["r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50",
               "r0.3_ur0.003_nr0.001_itol1.0e-6_iit1000_sit200_it50",
               "sa10_r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50",
               "sa10_r0.3_ur0.003_nr0.001_itol1.0e-6_iit1000_sit200_it50"
              ]

subdirnames = ["sa10_store_r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50"
              ]
dataset = :fakecells; num_experiments=30
for subdirname in subdirnames
    @show subdirname
itp_time_resol=10000 # if the time resolution is too sparse, increase this.
for α in [0.005]
    for (prefix, optim_methods) in [("pcb_sa1", ["lbfgs", "sgd_injectnoise"])]
        initmethod = "isvd"; tailstr = "sp"
        rt2slbfgs = []; rt2ssgdin = []; rngdic = Dict()
        for optim_method in optim_methods
            rt2_min = Inf; 
            fprex = "$(prefix)_$(dataset)_$(initmethod)_$(tailstr)_$(optim_method)"
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,subdirname,"data","$(fprex)_a$(α)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; #@show initmethod, rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
                optim_method == "sgd_injectnoise" ? push!(rt2ssgdin,rt2s[end]) : push!(rt2slbfgs,rt2s[end])
            end
            rt2_min = floor(rt2_min, digits=4)
            rngdic[optim_method] = range(0,stop=rt2_min,length=itp_time_resol)
        end
        rt2slbfgs_mean = isempty(rt2slbfgs) ? NaN : mean(rt2slbfgs)
        rt2ssgdin_mean = isempty(rt2ssgdin) ? NaN : mean(rt2ssgdin)
        @show α, rt2slbfgs_mean, rt2ssgdin_mean
        
        stat_af_lbfgs=[]; stat_af_sgdin=[]
        stat_fx_lbfgs=[]; stat_fx_sgdin=[]
        for optim_method in optim_methods
            afs=[]; fxs=[]
            fprex = "$(prefix)_$(dataset)_$(initmethod)_$(tailstr)_$(optim_method)"
            for iter in 1:num_experiments
                @show optim_method, iter
                dd = load(joinpath(subworkpath,subdirname,"data","$(fprex)_a$(α)_results$(iter).jld2"))
                rt2s = dd["data"]["rt2s"]
                # avgfits
                avgfits = dd["data"]["avgfits"]
                lr = length(rt2s); la = length(avgfits)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; avgfits=avgfits[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
                push!(afs,itp(rngdic[optim_method]))
                # fxs
                inner_fxs = dd["data"]["inner_fxs"]
                lr = length(rt2s); la = length(inner_fxs)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; inner_fxs=inner_fxs[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, inner_fxs, Gridded(Linear()))
                push!(fxs,itp(rngdic[optim_method]))
            end
            # avgfits
            avgfits = hcat(afs...)
            means = dropdims(mean(avgfits,dims=2),dims=2)
            stds = dropdims(std(avgfits,dims=2),dims=2)
            optim_method == "lbfgs" && (push!(stat_af_lbfgs,means); push!(stat_af_lbfgs,stds))
            optim_method == "sgd_injectnoise" && (push!(stat_af_sgdin,means); push!(stat_af_sgdin,stds))
            # fxs
            fxs = hcat(fxs...)
            means = dropdims(mean(fxs,dims=2),dims=2)
            stds = dropdims(std(fxs,dims=2),dims=2)
            optim_method == "lbfgs" && (push!(stat_fx_lbfgs,means); push!(stat_fx_lbfgs,stds))
            optim_method == "sgd_injectnoise" && (push!(stat_fx_sgdin,means); push!(stat_fx_sgdin,stds))
        end
        fprex="$(prefix)"
        save(joinpath(subworkpath,subdirname,"$(fprex)_a$(α)_runtime_vs_avgfits.jld2"),
            "rng_lbfgs",rngdic["lbfgs"], "rng_sgdin", rngdic["sgd_injectnoise"],
            "stat_af_lbfgs", stat_af_lbfgs, "stat_af_sgdin", stat_af_sgdin,
            "stat_fx_lbfgs", stat_fx_lbfgs, "stat_fx_sgdin", stat_fx_sgdin)
    end # for (prefix, initmethods, tailstrs)
end # for α
end # for subdirname

tmppath = ""
for subdirname in subdirnames
    @show subdirname
z = 0.5;
for (α, maxplottime, ybtm) in [(0.005,3.0,0.0)]
    ylimits=(ybtm,1.1)
    for (mtdstr, submtdstrs) in [("pcb_sa1",["lbfgs","sgdin"])]
        @show mtdstr
        ddstr = "dd$(mtdstr)"; ddsym = Symbol(ddstr)
    #    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
        # dirname = eval(Meta.parse("joinpath(subworkpath,\"$(subdirname)\")"))
        # @show dirname
        eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(subdirname)\",\"$(mtdstr)_a$(α)_runtime_vs_avgfits.jld2\"))"))
        for submtdstr in submtdstrs
            eval(Meta.parse("$(submtdstr)rng=$(ddsym)[\"rng_$(submtdstr)\"]"))
            # avgfits
            frpx = "$(mtdstr)_af_$(submtdstr)"
            eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat_af_$(submtdstr)\"][1]"))
            eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat_af_$(submtdstr)\"][2]"))
            @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + $(z)*$(Symbol("$(frpx)_stds"))))
            @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - $(z)*$(Symbol("$(frpx)_stds"))))
            # fxs
            frpx = "$(mtdstr)_fx_$(submtdstr)"
            eval(Meta.parse("$(frpx)_means=$(ddstr)[\"stat_fx_$(submtdstr)\"][1]"))
            eval(Meta.parse("$(frpx)_stds=$(ddstr)[\"stat_fx_$(submtdstr)\"][2]"))
            @eval ($(Symbol("$(frpx)_upper")) = ($(Symbol("$(frpx)_means")) + $(z)*$(Symbol("$(frpx)_stds"))))
            @eval ($(Symbol("$(frpx)_lower")) = ($(Symbol("$(frpx)_means")) - $(z)*$(Symbol("$(frpx)_stds"))))
        end
    end

    alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
    plotrng = Colon()

    # avgfits
    fig = Figure(size=(600,450))
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), ylimits),
                    xlabel = "time(sec)", ylabel = "average fit", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb_sa1","lbfgs","LBFGS",2,nothing),
                                                                    # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                    #("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb_sa1","sgdin","SGD",3,nothing)]) # all
        frpx = "$(mtdstr)_af_$(submtdstr)"
        ln = lines!(ax, eval(Symbol("$(submtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
        bnd = band!(ax, eval(Symbol("$(submtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
        lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
    end

    axislegend(ax, labelsize=20, position = :rb) # halign = :left, valign = :top
    save(joinpath(subworkpath,tmppath,subdirname,"fits_a$(α)_all.png"),fig,px_per_unit=2)

    # fxs
    fig = Figure(size=(600,450))
    ax = AMakie.Axis(fig[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), nothing),
                    yscale=log10,xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    lns = Dict(); bnds=Dict()
    for (i,(mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate([("pcb_sa1","lbfgs","LBFGS",2,nothing),
                                                                    # ("pcb","_sbc_p2","SBC P2",6,nothing),
                                                                    # ("pcb","_nndsvd","NNDSVD",5,nothing),
                                                                    #("pcb","_isvd","ISVD",3,nothing),
                                                                    ("pcb_sa1","sgdin","SGD",3,nothing)]) # all
        frpx = "$(mtdstr)_fx_$(submtdstr)"
        ln = lines!(ax, eval(Symbol("$(submtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
        bnd = band!(ax, eval(Symbol("$(submtdstr)rng"))[plotrng], eval(Symbol("$(frpx)_lower"))[plotrng], eval(Symbol("$(frpx)_upper"))[plotrng], color=mtdcoloras[clridx])
        lns["$(frpx)_line"] = ln; bnds["$(frpx)_band"] = bnd;
    end

    axislegend(ax, labelsize=20, position = :rb) # halign = :left, valign = :top
    save(joinpath(subworkpath,tmppath,subdirname,"penalty_a$(α)_all.png"),fig,px_per_unit=2)

end # for α

end 

#=========== Compare all penalties ====================#

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon(); z = 0.5
(α, maxplottime, ybtm) = (0.005,0.5,0.6); ylimits=(ybtm,1.1)
# avgfits
fig1 = Figure(size=(600,450))
ax1 = AMakie.Axis(fig1[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), ylimits),
                xlabel = "time(sec)", ylabel = "average fit", xlabelsize=20, ylabelsize=20,
                xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
fig2 = Figure(size=(600,450))
ax2 = AMakie.Axis(fig2[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), nothing),
                yscale=log10,xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
lns1 = Dict(); bnds1=Dict(); lns2 = Dict(); bnds2=Dict()
for (i,(ddstr, subdirname, mtdstr, submtdstr, lbl, clridx, linestyle)) in enumerate(
        [("ddlbfgs","r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50","pcb_sa1","lbfgs","LBFGS",1,nothing),
        ("ddsgdin1","r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50","pcb_sa1","sgdin","SGD nr=0.01",2,nothing),
        ("ddsgdin2","r0.3_ur0.003_nr0.001_itol1.0e-6_iit1000_sit200_it50","pcb_sa1","sgdin","SGD nr=0.001",6,nothing),
        ("ddsgdinsa1","sa10_r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50","pcb_sa1","sgdin","SGD nr=0.01(SA)",5,nothing),
        ("ddsgdinsa2","sa10_r0.3_ur0.003_nr0.001_itol1.0e-6_iit1000_sit200_it50","pcb_sa1","sgdin","SGD nr=0.001(SA)",3,nothing)]) # all
    ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
    # dirname = eval(Meta.parse("joinpath(subworkpath,\"$(subdirname)\")"))
    # @show dirname
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(subdirname)\",\"$(mtdstr)_a$(α)_runtime_vs_avgfits.jld2\"))"))
    eval(Meta.parse("$(ddstr)rng=$(ddsym)[\"rng_$(submtdstr)\"]"))

    # avgfits
    frpx1 = "$(ddstr)_af_$(submtdstr)"
    eval(Meta.parse("$(frpx1)_means=$(ddstr)[\"stat_af_$(submtdstr)\"][1]"))
    eval(Meta.parse("$(frpx1)_stds=$(ddstr)[\"stat_af_$(submtdstr)\"][2]"))
    @eval ($(Symbol("$(frpx1)_upper")) = ($(Symbol("$(frpx1)_means")) + $(z)*$(Symbol("$(frpx1)_stds"))))
    @eval ($(Symbol("$(frpx1)_lower")) = ($(Symbol("$(frpx1)_means")) - $(z)*$(Symbol("$(frpx1)_stds"))))

    ln1 = lines!(ax1, eval(Symbol("$(ddstr)rng"))[plotrng], eval(Symbol("$(frpx1)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd1 = band!(ax1, eval(Symbol("$(ddstr)rng"))[plotrng], eval(Symbol("$(frpx1)_lower"))[plotrng], eval(Symbol("$(frpx1)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns1["$(frpx1)_line"] = ln1; bnds1["$(frpx1)_band"] = bnd1;

    # fxs
    frpx2 = "$(ddstr)_fx_$(submtdstr)"
    eval(Meta.parse("$(frpx2)_means=$(ddstr)[\"stat_fx_$(submtdstr)\"][1]"))
    eval(Meta.parse("$(frpx2)_stds=$(ddstr)[\"stat_fx_$(submtdstr)\"][2]"))
    @eval ($(Symbol("$(frpx2)_upper")) = ($(Symbol("$(frpx2)_means")) + $(z)*$(Symbol("$(frpx2)_stds"))))
    @eval ($(Symbol("$(frpx2)_lower")) = ($(Symbol("$(frpx2)_means")) - $(z)*$(Symbol("$(frpx2)_stds"))))

    ln2 = lines!(ax2, eval(Symbol("$(ddstr)rng"))[plotrng], eval(Symbol("$(frpx2)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd2 = band!(ax2, eval(Symbol("$(ddstr)rng"))[plotrng], eval(Symbol("$(frpx2)_lower"))[plotrng], eval(Symbol("$(frpx2)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns2["$(frpx2)_line"] = ln2; bnds2["$(frpx2)_band"] = bnd2;
end

axislegend(ax1, labelsize=20, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,tmppath,"fits_a$(α)_all$(maxplottime).png"),fig1,px_per_unit=2)
axislegend(ax2, labelsize=20, position = :rb) # halign = :left, valign = :top
save(joinpath(subworkpath,tmppath,"penalty_a$(α)_all$(maxplottime).png"),fig2,px_per_unit=2)


#=========== Symmetric penalty vs Sparsity penalty ====================#

subdirnames = ["sa10_store_r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50"
              ]
dataset = :fakecells; num_experiments=30
for subdirname in subdirnames
    @show subdirname
itp_time_resol=10000 # if the time resolution is too sparse, increase this.
for α in [0.005]
    for (prefix, optim_methods) in [("pcb_sa1", ["lbfgs", "sgd_injectnoise"])]
        initmethod = "isvd"; tailstr = "sp"
        rt2slbfgs = []; rt2ssgdin = []; rngdic = Dict()
        for optim_method in optim_methods
            rt2_min = Inf; 
            fprex = "$(prefix)_$(dataset)_$(initmethod)_$(tailstr)_$(optim_method)"
            for iter in 1:num_experiments
                fn = joinpath(subworkpath,subdirname,"data","$(fprex)_a$(α)_results$(iter).jld2")
                dd = load(fn,"data")
                rt2s = dd["rt2s"]; #@show initmethod, rt2s[end]
                rt2_min = min(rt2_min,rt2s[end])
                optim_method == "sgd_injectnoise" ? push!(rt2ssgdin,rt2s[end]) : push!(rt2slbfgs,rt2s[end])
            end
            rt2_min = floor(rt2_min, digits=4)
            rngdic[optim_method] = range(0,stop=rt2_min,length=itp_time_resol)
        end
        rt2slbfgs_mean = isempty(rt2slbfgs) ? NaN : mean(rt2slbfgs)
        rt2ssgdin_mean = isempty(rt2ssgdin) ? NaN : mean(rt2ssgdin)
        @show α, rt2slbfgs_mean, rt2ssgdin_mean
        
        stat_sym_lbfgs=[]; stat_sym_sgdin=[]
        stat_sparW_lbfgs=[]; stat_sparW_sgdin=[]
        stat_sparH_lbfgs=[]; stat_sparH_sgdin=[]
        for optim_method in optim_methods
            sps=[]; sws=[]; shs=[]
            fprex = "$(prefix)_$(dataset)_$(initmethod)_$(tailstr)_$(optim_method)"
            for iter in 1:num_experiments
                @show optim_method, iter
                dd = load(joinpath(subworkpath,subdirname,"data","$(fprex)_a$(α)_results$(iter).jld2"))
                rt2s = dd["data"]["rt2s"]
                # sym
                sympens = dd["data"]["sympens"]
                lr = length(rt2s); la = length(sympens)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; sympens=sympens[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, sympens, Gridded(Linear()))
                push!(sps,itp(rngdic[optim_method]))
                # sparW
                sparWpens = dd["data"]["sparWpens"]
                lr = length(rt2s); la = length(sparWpens)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; sparWpens=sparWpens[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, sparWpens, Gridded(Linear()))
                push!(sws,itp(rngdic[optim_method]))
                # sparH
                sparHpens = dd["data"]["sparHpens"]
                lr = length(rt2s); la = length(sparHpens)
                lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; sparHpens=sparHpens[1:l])
                nodes = (rt2s,)
                itp = Interpolations.interpolate(nodes, sparHpens, Gridded(Linear()))
                push!(shs,itp(rngdic[optim_method]))
            end
            # sym
            sympens = hcat(sps...)
            means = isempty(sympens) ? NaN : dropdims(mean(sympens,dims=2),dims=2)
            stds = isempty(sympens) ? NaN : dropdims(std(sympens,dims=2),dims=2)
            optim_method == "lbfgs" && (push!(stat_sym_lbfgs,means); push!(stat_sym_lbfgs,stds))
            optim_method == "sgd_injectnoise" && (push!(stat_sym_sgdin,means); push!(stat_sym_sgdin,stds))
            # sparW
            sparWs = hcat(sws...)
            means = isempty(sparWs) ? NaN : dropdims(mean(sparWs,dims=2),dims=2)
            stds = isempty(sparWs) ? NaN : dropdims(std(sparWs,dims=2),dims=2)
            optim_method == "lbfgs" && (push!(stat_sparW_lbfgs,means); push!(stat_sparW_lbfgs,stds))
            optim_method == "sgd_injectnoise" && (push!(stat_sparW_sgdin,means); push!(stat_sparW_sgdin,stds))
            # sparH
            sparHs = hcat(shs...)
            means = isempty(sparHs) ? NaN : dropdims(mean(sparHs,dims=2),dims=2)
            stds = isempty(sparHs) ? NaN : dropdims(std(sparHs,dims=2),dims=2)
            optim_method == "lbfgs" && (push!(stat_sparH_lbfgs,means); push!(stat_sparH_lbfgs,stds))
            optim_method == "sgd_injectnoise" && (push!(stat_sparH_sgdin,means); push!(stat_sparH_sgdin,stds))
        end
        fprex="$(prefix)"
        save(joinpath(subworkpath,subdirname,"$(fprex)_a$(α)_runtime_vs_avgfits.jld2"),
            "rng_lbfgs",rngdic["lbfgs"], "rng_sgdin", rngdic["sgd_injectnoise"],
            "stat_sym_lbfgs", stat_sym_lbfgs, "stat_sym_sgdin", stat_sym_sgdin,
            "stat_sparW_lbfgs", stat_sparW_lbfgs, "stat_sparW_sgdin", stat_sparW_sgdin,
            "stat_sparH_lbfgs", stat_sparH_lbfgs, "stat_sparH_sgdin", stat_sparH_sgdin)
    end # for (prefix, initmethods, tailstrs)
end # for α
end # for subdirname

alpha = 0.2; cls = distinguishable_colors(10); clbs = convert.(RGBA,cls,alpha)
plotrng = Colon(); z = 0.5
(α, maxplottime, ybtm) = (0.005,0.5,0.0); ylimits=(ybtm,1.1)
# Sym pen, Sparseity W, Sparsity H
lns1 = Dict(); bnds1=Dict(); mtdstr = "pcb_sa1"
for (i,(ddstr, subdirname, submtdstr, pen, lbl, clridx, linestyle)) in enumerate(
        [("ddsgdsym","sa10_store_r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50","lbfgs","sym","Sym_pen",2,nothing),
        ("ddsgdsparW","sa10_store_r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50","lbfgs","sparW","Spar_W",6,nothing),
        ("ddsgdsparH","sa10_store_r0.3_ur0.003_nr0.01_itol1.0e-6_iit1000_sit200_it50","lbfgs","sparH","Spar_H",3,nothing)]) # all
    fig1 = Figure(size=(600,450))
    ax1 = AMakie.Axis(fig1[1, 1], limits = ((0,maxplottime#=min(maxplottimes[idx],plottime)=#), nothing),
                    xlabel = "time(sec)", ylabel = "penalty", xlabelsize=20, ylabelsize=20,
                    xticklabelsize=20, yticklabelsize=20)#, title = "Average Fit Value vs. Running Time")
    ddsym = Symbol(ddstr)
#    @eval (($ddsym)=(load("$(mtdstr)_runtime_vs_avgfits.jld2"))) # this doens't work 'mtdstr' refer global variable
    # dirname = eval(Meta.parse("joinpath(subworkpath,\"$(subdirname)\")"))
    # @show dirname
    eval(Meta.parse("$(ddstr)=load(joinpath(subworkpath,tmppath,\"$(subdirname)\",\"$(mtdstr)_a$(α)_runtime_vs_avgfits.jld2\"))"))
    eval(Meta.parse("$(ddstr)rng=$(ddsym)[\"rng_$(submtdstr)\"]"))

    # plot each penalty
    frpx1 = "$(ddstr)_sym_$(submtdstr)"
    eval(Meta.parse("$(frpx1)_means=$(ddstr)[\"stat_$(pen)_$(submtdstr)\"][1]"))
    eval(Meta.parse("$(frpx1)_stds=$(ddstr)[\"stat_$(pen)_$(submtdstr)\"][2]"))
    @eval ($(Symbol("$(frpx1)_upper")) = ($(Symbol("$(frpx1)_means")) + $(z)*$(Symbol("$(frpx1)_stds"))))
    @eval ($(Symbol("$(frpx1)_lower")) = ($(Symbol("$(frpx1)_means")) - $(z)*$(Symbol("$(frpx1)_stds"))))

    ln1 = lines!(ax1, eval(Symbol("$(ddstr)rng"))[plotrng], eval(Symbol("$(frpx1)_means"))[plotrng], color=mtdcolors[clridx], label=lbl, linestyle=linestyle)
    bnd1 = band!(ax1, eval(Symbol("$(ddstr)rng"))[plotrng], eval(Symbol("$(frpx1)_lower"))[plotrng], eval(Symbol("$(frpx1)_upper"))[plotrng], color=mtdcoloras[clridx])
    lns1["$(frpx1)_line"] = ln1; bnds1["$(frpx1)_band"] = bnd1;

    axislegend(ax1, labelsize=20, position = :rb) # halign = :left, valign = :top
    save(joinpath(subworkpath,tmppath,"$(submtdstr)_$(pen)_a$(α)_all$(maxplottime).png"),fig1,px_per_unit=2)
end

