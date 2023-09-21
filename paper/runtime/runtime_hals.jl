using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","runtime")

include(joinpath(workpath,"setup_light.jl"))

dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

if true
    num_experiments = 50
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
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);

# HALS
@show "HALS"
W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize)
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1
for (tailstr,α) in [("_nn",0.),("_sp_nn",0.1)]
    @show tailstr; flush(stdout)
    dd = Dict()
    Wcd, Hcd = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
    result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=true), X, Wcd, Hcd; W0=W0, H0=H0, d=diag(D), gtW=gtW, gtH=gtH)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, Wcd, Hcd)
    rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
    if true#iter == num_experiments
        metadata = Dict()
        metadata["maxiter"] = maxiter; metadata["alpha"] = α
    end
    save(joinpath(subworkpath,"hals","hals$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
end

end

using Interpolations

num_expriments=50
rt2_min = Inf
for tailstr in ["_nn","_sp_nn"]
    for iter in 1:num_expriments
        dd = load(joinpath(subworkpath,"hals","hals$(tailstr)_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_min = floor(rt2_min, digits=2)
rng = range(0,stop=rt2_min,length=100)

stat_nn=[]; stat_sp_nn=[]
for tailstr in ["_nn","_sp_nn"]
    afs=[]
    for iter in 1:num_expriments
        dd = load(joinpath(subworkpath,"hals","hals$(tailstr)_results$(iter).jld2"))
        rt2s = dd["data"]["rt2s"]; avgfits = dd["data"]["avgfits"]
        nodes = (rt2s,)
        itp = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
        push!(afs,itp(rng))
    end
    avgfits = hcat(afs...)
    means = dropdims(mean(avgfits,dims=2),dims=2)
    stds = dropdims(std(avgfits,dims=2),dims=2)
    tailstr == "_nn" && (push!(stat_nn,means); push!(stat_nn,stds))
    tailstr == "_sp_nn" && (push!(stat_sp_nn,means); push!(stat_sp_nn,stds))
end
save(joinpath(subworkpath,"hals_runtime_vs_avgfits.jld2"),"rng",rng, "stat_nn", stat_nn, "stat_sp_nn", stat_sp_nn)
