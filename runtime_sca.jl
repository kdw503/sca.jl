using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

include("setup_light.jl")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include("dataset.jl")
include("utils.jl")

using ProfileView, BenchmarkTools

dataset = :fakecells; SNR=0; inhibitidx=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitidx)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

# testdic = Dict()
# A=rand(40,20,1000); img_nl = AxisArray(colorview(Gray, A), :y, :x, :time)
# gtW = rand(40,20,7); gtH = rand(1000,7)

# testdic["gt_ncells"] = 7
# testdic["imgrs"] = Matrix(reshape(img_nl,800,1000))
# testdic["img_nl"] = img_nl
# testdic["gtW"] = gtW
# testdic["gtH"] = gtH
# JLD.save("C:\\Users\\kdw76\\WUSTL\\Work\\Data\\fakecells\\fakecells0_calcium_sz(40, 20)_lengthT1000_J0_SNR0_bias0.1.jld",testdic)
# JLD.load("C:\\Users\\kdw76\\WUSTL\\Work\\Data\\fakecells\\fakecells0_calcium_sz(40, 20)_lengthT1000_J0_SNR0_bias0.1.jld")

data_sca_nn = Dict[]; data_sca_sp = Dict[]; data_sca_sp_nn = Dict[]
num_experiments = 30

for iter in 1:num_experiments
    @show iter
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitidx=inhibitidx, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells); dataset == :fakecells && (gtW = datadic["gtW"]; gtH = datadic["gtH"])
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitidx]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# SCA
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 100; β = 1000
useRelaxedL1=true; useRelaxedNN=true; s=10*0.3^0; σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter=100; inner_maxiter = 50; ls_maxiter = 100
# Result demonstration parameters
makepositive = true; poweradjust = :none

for (tailstr,initmethod,α,β) in [("_nn",:nndsvd,0.,1000.), ("_sp",:isvd,100.,0.), ("_sp_nn",:nndsvd,100.,1000.)]
    @show tailstr
    dd = Dict()
    rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        reg=reg, useRelaxedL1=useRelaxedL1, useRelaxedNN=useRelaxedNN, σ0=σ0, r=r, poweradjust=poweradjust,
        useprecond=useprecond, uselv=uselv)
    lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    Mw, Mh = copy(Mw0), copy(Mh0);
    rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; penmetric=penmetric,
                                            stparams=stparams, lsparams=lsparams, cparams=cparams);
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; penmetric=penmetric,
                                            stparams=stparams, lsparams=lsparams, cparams=cparams);
    normalizeW!(W1,H1); W3,H3 = sortWHslices(W1,H1)
    makepositive && flip2makepos!(W3,H3)
    f_xs = getdata(trs,:f_x); niters = getdata(trs,:niter); totalniters = sum(niters)
    Mwss = getdata(trs,:Mws); Mhss = getdata(trs,:Mhs); fxss = getdata(trs,:fxs)
    Mwsall = []; Mhsall = []; fxsall = Float64[]; rt2sall = Float64[]
    for (iter,(Mwsi,Mhsi,fxsi)) in enumerate(zip(Mwss,Mhss, fxss))
        isempty(Mwsi) && continue
        for (inner_iter,(Mw, Mh, fx)) in enumerate(zip(Mwsi,Mhsi,fxsi))
            push!(Mwsall,Mw); push!(Mhsall,Mh); push!(fxall,fx)
        end
        if iter == 1
            rt2 = 0.
        else
            rt2 = collect(range(start=laps[iter-1],stop=laps[iter],length=length(Mwsi)+1))[2:end].-laps[1]
        end
        append!(rt2sall,rt2)
    end
    @show length(rt2sall)
    for (i,(Mw,Ms,fx,rt2)) in enumerate(zip(Mwsall,Mhsall,fxall,rt2sall))
        i%2 != 1 && continue
        @show i
        W3=W0*Mw; H3 = Mh*H0
        normalizeW!(W3,H3)
        avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
#            avgfit = fitdold(X,W3*H3)
        push!(avgfits,avgfit); push!(inner_fxs,fx); push!(rt2s,rt2)
    end

    dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"]=avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    tailstr == "_nn" && push!(data_sca_nn, dd)
    tailstr == "_sp" && push!(data_sca_sp, dd)
    tailstr == "_sp_nn" && push!(data_sca_sp_nn, dd)
    if iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
        metadata["alpha"] = σ; metadata["beta"] = β; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
        tailstr == "_nn" && push!(data_sca_nn, metadata)
        tailstr == "_sp" && push!(data_sca_sp, metadata)
        tailstr == "_sp_nn" && push!(data_sca_sp_nn, metadata)
    end
end
end
save("sca_results.jld","data_sca_nn",data_sca_nn,"data_sca_sp",data_sca_sp,"data_sca_sp_nn",data_sca_sp_nn)
