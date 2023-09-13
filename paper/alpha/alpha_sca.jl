using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 50
    sca_maxiter = 400; sca_inner_maxiter = 50; sca_ls_maxiter = 100
    admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 200
    αrng = 0:2:500
else
    num_experiments = 2
    sca_maxiter = 2; sca_inner_maxiter = 2; sca_ls_maxiter = 2
    admm_maxiter = 2; admm_inner_maxiter = 0; admm_ls_maxiter = 0
    hals_maxiter = 2
    αrng = 0:1:2
end

for iter in 26:num_experiments
    @show iter; flush(stdout)
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
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)

# SCA
@show "SCA"
mfmethod = :SCA; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 100; β = 1000
useRelaxedL1=true; useRelaxedNN=true; s=10*0.3^0; 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased
# Optimization parameters
tol=-1; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking; useprecond=true; uselv=false
maxiter = sca_maxiter; inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter
# Result demonstration parameters
makepositive = true; poweradjust = :none
try
for (tailstr,initmethod,β) in [("_sp",:isvd,0.), ("_sp_nn",:nndsvd,1000.)]
    @show tailstr; flush(stdout)
    dd = Dict()
    rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    σ0=s*std(W0) #=10*std(W0)=#
    β1=β2=β
    avgfits = Float64[]
    for α in αrng
        α1 = α2 = α
        stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
            reg=reg, useRelaxedL1=useRelaxedL1, useRelaxedNN=useRelaxedNN, σ0=σ0, r=r, poweradjust=poweradjust,
            useprecond=useprecond, uselv=uselv)
        lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
            x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
            store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
        Mw, Mh = copy(Mw0), copy(Mh0);
        cparams.store_trace = false; cparams.store_inner_trace = false;
        cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
        rt2 = @elapsed W1, H1, objvals, laps, _ = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
                                        penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
        avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
        push!(avgfits,avgfit)
    end
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter; metadata["inner_maxiter"] = inner_maxiter;
        metadata["αrng"] = αrng; metadata["beta"] = β; metadata["optimmethod"] = optimmethod; metadata["initmethod"] = initmethod
    end
    save(joinpath(workpath,"paper","alpha","sca$(tailstr)_alpha_results$(iter).jld2"),"metadata",metadata,"avgfits",avgfits)
    GC.gc()
end
catch e
    save("sca_error_$(iter).jld2",datadic)
    @warn e
    iter -= 1
end
end

#=
using Interpolations

num_expriments=3
dd = load(joinpath(workpath,"paper","alpha","sca_sp_nn_alpha_results1.jld2"))
αrng = dd["metadata"]["αrng"]
rng = range(0,stop=αrng[end],length=100)

stat_sp=[]; stat_sp_nn=[]
for tailstr in ["_sp","_sp_nn"]
    afs=[]
    for iter in 1:num_expriments
        dd = load(joinpath(workpath,"paper","alpha","sca$(tailstr)_alpha_results$(iter).jld2"))
        avgfits = dd["avgfits"]
        nodes = (αrng,)
        itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
        push!(afs,itp(rng))
    end
    avgfits = hcat(afs...)
    means = dropdims(mean(avgfits,dims=2),dims=2)
    stds = dropdims(std(avgfits,dims=2),dims=2)
    tailstr == "_sp" && (push!(stat_sp,means); push!(stat_sp,stds))
    tailstr == "_sp_nn" && (push!(stat_sp_nn,means); push!(stat_sp_nn,stds))
end
save(joinpath(workpath,"paper","alpha","sca_alpha_vs_avgfits.jld2"),"rng", rng, "stat_sp", stat_sp, "stat_sp_nn", stat_sp_nn)
=#