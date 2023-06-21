using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

include("setup.jl")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include("dataset.jl")
include("utils.jl")

using ProfileView, BenchmarkTools

dataset = :fakecells; SNR=0; inhibitidx=0; bias=0.1; initmethod=:lowrank_nndsvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitidx)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitidx=inhibitidx, issave=true, isload=true, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells); dataset == :fakecells && (gtW = datadic["gtW"]; gtH = datadic["gtH"])
X = noisefilter(filter,X)
save_gtH = false; save_gtH && (close("all"); plot(gtH); savefig("gtH.png"))

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitidx]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end
rt1sca = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar)
save_init_figure = false; save_init_figure && begin
    normalizeW!(Wp,Hp); W2,H2 = copy(Wp), copy(Hp) # sortWHslices(Wp,Hp)
    imsave_data(dataset,"$(initmethod)$(datastr)$(filterstr)_rti$(rt1sca)",W2,H2,imgsz,100; signedcolors=dgwm(), saveH=false)
end

mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 0; β = 0; usennc=false
useRelaxedL1=true; s=10*0.3^0; σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=1e-8; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter=2000; scamaxiter = 500; halsmaxiter = 50; spcamaxiter=maxiter; inner_maxiter = 200; ls_maxiter = 500
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = (reg == :WH1 && sd_group != :whole) ? :balance : :none

maxiterstr = mfmethod == :HALS ? "it$(halsmaxiter)" : "it$(scamaxiter)"
initmethodstr = mfmethod == :HALS ? "_nndsvd" : mfmethod == :SPCA ? "" : "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
optimmethodstr = (optimmethod ∈ [:SB_newton, :SB_lbfgs, :SB_cg] || !useRelaxedL1) ? "$optimmethod" : "$(optimmethod)_σ$(s)r$(r)"
β1 = β2 = β; α1 = α; α2 = α; # α2 = 0
scaparamstr = "_$(reg)_$(sd_group)_$(optimmethodstr)_$(maxiterstr)_init$(inner_maxiter)_αw$(α1)_αh$(α2)_β$(β)"
uselogscale = true

figi, axi = plt.subplots(1,1, figsize=(5,4))
#ax.set_title(yaxisstr)
axi.set_ylabel("log10(penalty)",fontsize = 12)
axi.set_xlabel("iteration",fontsize = 12)

figt, axt = plt.subplots(1,1, figsize=(5,4))
#ax.set_title(yaxisstr)
axt.set_ylabel("log10(penalty)",fontsize = 12)
axt.set_xlabel("time",fontsize = 12)

#for α in [0.1,1,10,100,1000]
for (optimmethod,useRelaxedL1,useprecond,α,innermaxiter,usennc) in [#=(:optim_lbfgs,true,true,100,50,false),=#
                                                            (:sca_admm,false,false,0,100,true),
                                                            (:sca_admm_optim_lbfgs,false,false,0,100,true)]
    α1=α2=α; β1=β2=β
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
    lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=scamaxiter, inner_maxiter=inner_maxiter,
        store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    Mw, Mh = copy(Mw0), copy(Mh0);
    rt2 = @elapsed W1, H1, objvals, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh; penmetric=penmetric, stparams=stparams,
                                        lsparams=lsparams, cparams=cparams);
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    makepositive && flip2makepos!(W3,H3)
    fxs = getdata(trs,:f_x)
    imsave_data(dataset,"$(fprex)$(optimmethod)_a$(α1)_b$(β1)_r$(r)_it$(niters)_rt$(rt2)",W3,H3,imgsz,100; saveH=false)

    iterrng = 0:niters; trng = iterrng/niters*rt2
    # fxs ./= fx0
    uselogscale && (fxs=log10.(fxs))
    length(fxs) != 0 && begin
        axi.plot(iterrng[2:end],fxs[2:end]); axt.plot(trng[2:end],fxs[2:end])
    end
end
title = "w_sep_vs_wo_sep_nn"
for withlegend = [false, true]
    for ax in [axi,axt]
        withlegend && ax.legend([#="SCA",=# "With W H seperation","W/O  W H seperation"], fontsize = 12)
    end
    figi.savefig("$(title)_$(initmethod)_$(optimmethod)_it$(maxiter)_iter_$(withlegend).png")
    figt.savefig("$(title)_$(initmethod)_$(optimmethod)_it$(maxiter)_time_$(withlegend).png")
end
close("all")

# SCA

mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 100; β = 0; usennc=false
useRelaxedL1=true; s=10*0.3^0; σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=1e-6; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter=200; scamaxiter = 200; halsmaxiter = 50; spcamaxiter=maxiter; inner_maxiter = 50; ls_maxiter = 100
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = (reg == :WH1 && sd_group != :whole) ? :balance : :none

maxiterstr = mfmethod == :HALS ? "it$(halsmaxiter)" : "it$(scamaxiter)"
initmethodstr = mfmethod == :HALS ? "_nndsvd" : mfmethod == :SPCA ? "" : "_$(initmethod)"
fprex = "$(mfmethod)$(datastr)$(filterstr)$(initmethodstr)"
optimmethodstr = (optimmethod ∈ [:SB_newton, :SB_lbfgs, :SB_cg] || !useRelaxedL1) ? "$optimmethod" : "$(optimmethod)_σ$(s)r$(r)"
β1 = β2 = β; α1 = α; α2 = α; # α2 = 0
scaparamstr = "_$(reg)_$(sd_group)_$(optimmethodstr)_$(maxiterstr)_init$(inner_maxiter)_αw$(α1)_αh$(α2)_β$(β)"

rrng = 0.1:0.1:0.9
for α in [0]
    numiters = []; rtsscas = []; mssds = []; ; mssdWs = []; mssdHs = []; fxsss=[]
    for (optimmethod,ls_method) = [(:sca_admm,:none)]#,(:sca_lbfgs,:ls_Exact)]
        α1= α2=α
        stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
            reg=reg, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust, useprecond=useprecond, uselv=uselv)
        lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=0, maxiter=scamaxiter, inner_maxiter=inner_maxiter, store_trace=true,
            store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
        Mw, Mh = copy(Mw0), copy(Mh0);
        W1, H1, objvals, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh; penmetric=penmetric, stparams=stparams,
                                                lsparams=lsparams, cparams=cparams);

        Mw, Mh = copy(Mw0), copy(Mh0);
        cparams.store_trace = false; cparams.store_inner_trace = false;
        cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
        rt2 = @elapsed scasolve!(X, W0, H0, D, Mw, Mh; penmetric=penmetric, stparams=stparams,
                                                    lsparams=lsparams, cparams=cparams);
        normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
        makepositive && flip2makepos!(W3,H3)
        omstr = stparams.optimmethod == :sca_lbfgs ? "_lbfgs" : chop(String(stparams.optimmethod),head=5,tail=0)
        fxs = getdata(trs,:f_x)
        fxss = getdata(trs,:fxs) # inner_trace
        push!(fxsss,fxss)
        lfxss, totaliters = length.(fxss), sum(length.(fxss))
        @show lfxss, totaliters
        imsave_data(dataset,"$(fprex)$(omstr)_a$(α1)_r$(r)_it$(niters)_tit$(totaliters)_rt$(rt2)",W3,H3,imgsz,100; saveH=false)
        # Mw, Mh = copy(Mw0), copy(Mh0);
        # rt = @belapsed scasolve!($X, $W0, $H0, $d, $Mw, $Mh; penmetric=:SCA, stparams=stparams, lsparams=lsparams, cparams=cparams)

        # rts=0
        # Mw, Mh = copy(Mw0), copy(Mh0);
        # scasolve!(X, W0, H0, d, Mw, Mh; penmetric=:SCA, stparams=stparams, lsparams=lsparams, cparams=cparams)
        # for i in 1:10
        #     Mw, Mh = copy(Mw0), copy(Mh0);
        #     rts += @elapsed scasolve!(X, W0, H0, d, Mw, Mh; penmetric=:SCA, stparams=stparams, lsparams=lsparams, cparams=cparams)
        # end
        # rtssca = rts/10
        # push!(rtsscas,rtssca)

        # push!(numiters,totaliters)

        # if dataset == :fakecells
        #     mssd, ml, ssds = matchednssda(gtW,W3)
        #     mssdH = ssdH(ml, gtH,H3')
        #     push!(mssdWs,mssd); push!(mssdHs, mssdH); push!(mssds,mssd*mssdH)
        # else
        #     sparseness = norm(W1,1)
        #     push!(mssds,sparseness)
        # end
    end
    # if dataset == :fakecells
    #     close("all"); plot(rrng,mssds); title("ratio vs. mssd"); savefig("$(fprex)_$(SNR)dB_$(omstr)_a$(α1)_mssd.png")
    # else
    #     close("all"); plot(rrng,mssds); title("ratio vs. W sparseness"); savefig("$(fprex)_$(SNR)dB_$(omstr)_a$(α1)_W_sparseness.png")
    # end
    # close("all"); plot(rrng,numiters); title("ratio vs. iterations"); savefig("$(fprex)_$(SNR)dB_$(omstr)_a$(α1)_iter.png")
end
for i in 1:10
    plot(fxsss[1][i+1]);plot(fxsss[2][i+1])
    legend(["Back tracking","Step length minimizer"])
    xlabel("iteration"); ylabel(("penalty"))
    title("outer iteration = $i")
    savefig("linesearch_vs_step_minimizer_tol1e-1_$(i).png")
    close("all")
end

# NMF
rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rt50s=0; rt100s=0
for i in 1:100
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt50s += @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=50, α=0.1, l₁ratio=1,
                    tol=tol), X, Wcd, Hcd)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt100s += @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=100, α=0.1, l₁ratio=1,
                    tol=tol), X, Wcd, Hcd)
end
rtscd50 = rt50s/100; rtscd100 = rt100s/100

@show niters, rtssca, rtscd50, rtscd100


Wcd, Hcd = copy(Wcd0), copy(Hcd0);
rtcd100 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=100, α=0.1, l₁ratio=1,
                tol=tol), $X, $Wcd, $Hcd)

normalizeW!(Wcd,Hcd); W3,H3 = sortWHslices(Wcd,Hcd)
imsave_data(dataset,"HALS$(omstr)_$(SNR)dB_rt$(rtscd100)",W3,H3,imgsz,100; saveH=true)
                
# Profile View
Mw, Mh = copy(Mw0), copy(Mh0)
W1, H1, objvals, trs, niters = scasolve!(X, W0, H0, d, Mw, Mh; penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
Mw, Mh = copy(Mw0), copy(Mh0)
@profview scasolve!(X, W0, H0, d, Mw, Mh; penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);

# Cthulhu
@descend scasolve!(X, W0, H0, d, Mw, Mh; penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams)



# save images
normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
makepositive && flip2makepos!(W3,H3)
omstr = stparams.optimmethod == :sca_lbfgs ? "_lbfgs" : chop(String(stparams.optimmethod),head=5,tail=0)
imsave_data(dataset,"$(mfmethod)$(omstr)_$(SNR)dB_a$(α1)_it$(niters)_rt$(rt2)",W3,H3,imgsz,100; saveH=true)
fdts=getdata(trs,:sympen); fsps=getdata(trs,:sparseW)+getdata(trs,:sparseH)
xdiffss=getdata(trs,:x_abs); fxss = fdts + fsps
plot_convergence("$(mfmethod)$(omstr)_$(SNR)dB_a$(α1)_it$(niters)_rt$(rt2)",xdiffss,fxss; title="convergence (SCA)")
plotH_data(dataset,"$(mfmethod)$(omstr)_$(SNR)dB_a$(α1)_it$(niters)_rt$(rt2)",H3)

fxss=getdata(trs,:fxs)


using BoxTrees, TileTrees, TileTreesGUI, ImageView

## Generate some data to visualize (if you already have a TileTree, this step is
## not necessary )

# Create a tile tree with two rectangular tiles, each "active" only a
# small portion of the time
t1, t2 = zeros(200), zeros(200)
t1[rand(1:200, 30)] = rand(30)   # active for 30/200 time points
t2[rand(1:200, 30)] = rand(30)
ttree = TileTree([Tile((Span(16,50), Span(1,80)), ones(35,80), t1),
                  Tile((Span(1,40), Span(71,100)), ones(40,30), t2)])

## Visualize the TileTree

# Create the rendering object
ttc = TileTreeColor(ttree, (0,1), (0,1))
# View the rendering
imshow(ttc)

tileaction = initialize_triage_action(ttree)
tdct = triage_gui(ttree, tileaction, Array(ttree), ttree)
