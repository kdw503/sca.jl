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

dataset = :fakecells; SNR=0; inhibitidx=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitidx)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 100
    sca_maxiter = 400; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 200
else
    num_experiments = 2
    sca_maxiter = 2; sca_inner_maxiter = 2; sca_ls_maxiter = 2
    admm_maxiter = 2; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 2
end

for iter in 1:num_experiments
    @show iter; flush(stdout)
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitidx=inhibitidx, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitidx]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# ADMM
@show "ADMM"
mfmethod = :ADMM; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 10; β = 0; usennc=true
useRelaxedL1=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# Result demonstration parameters
makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
poweradjust = :none

for (tailstr,initmethod,α,usennc) in [("_nn",:lowrank_nndsvd,0.,true), ("_sp",:lowrank,10.,false), ("_sp_nn",:lowrank_nndsvd,10.,true)]
    @show tailstr; flush(stdout)
    dd = Dict()
    rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    σ0=s*std(W0) #=10*std(W0)=#
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    α1=α2=α; β1=β2=β
    stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
        store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
    Mw, Mh = copy(Mw0), copy(Mh0);
    rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                        penmetric=penmetric, stparams=stparams, cparams=cparams);
    Mw, Mh = copy(Mw0), copy(Mh0);
    cparams.store_trace = false; cparams.store_inner_trace = false;
    cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
    rt2 = @elapsed  W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                                        penmetric=penmetric, stparams=stparams, cparams=cparams);
    f_xs = getdata(trs,:f_x)
    avgfitss = getdata(trs,:avgfits)
    avgfits = Float64[]
    for (iter,afs) in enumerate(avgfitss)
        isempty(afs) && continue
        append!(avgfits,afs)
    end
    rt2s = collect(range(start=0,stop=rt2,length=length(avgfits)))

    dd["niters"] = niters; dd["totalniters"] = niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = Float64[]
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = 0; metadata["r"] = 0; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
        metadata["alpha"] = α; metadata["beta"] = 0; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
    end
    save("admm$(tailstr)_results$(iter).jld","metadata",metadata,"data",dd)
end

end
