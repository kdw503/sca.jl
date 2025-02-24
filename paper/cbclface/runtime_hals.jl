using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","cbclface")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))

dataset = :cbclface
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"

if true
    num_experiments = 50; lcsvd_maxiter = 80; compnmf_maxiter = 500; hals_maxiter = 400
else
    num_experiments = 2; lcsvd_maxiter = 2; compnmf_maxiter = 2; hals_maxiter = 2
end

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = LCSVD.noisefilter(filter,X)

subtract_bg=false; sbgstr = subtract_bg ? "sbg" : "nosbg"

if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
    plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
    bg = W*fill(mean(H),1,n); X .-= bg
end
W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=:nndsvd, svdmethod=:isvd)
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1

# HALS
prefix="hals"; @show prefix
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; αhals=0.1; maxiter = hals_maxiter; tol=-1 # αhals=0.6 
αrng = 0:0.1:5
for (iter, α) in enumerate(αrng)
@show iter; flush(stdout)
for (tailstr,) in [("_sp_nn",)]
    @show tailstr; flush(stdout)
    dd = Dict()
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=true, SCA_penmetric=:SPARSE_W), X, Wcd, Hcd; W0=W0, H0=H0, d=diag(D), gtW=gtW, gtH=gtH)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, Wcd, Hcd)
    rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
    dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = result.avgfits; dd["sparseWs"] = result.sparsevalues; dd["f_xs"] = result.objvalues;
    if true#iter == num_experiments
        metadata = Dict()
        metadata["maxiter"] = maxiter; metadata["alpha"] = α
    end
    save(joinpath(subworkpath,prefix,"$(prefix)$(tailstr)_alpha_results$(iter).jld2"),"metadata",metadata,"data",dd)
end

end

using Interpolations

num_expriments=50
rt2_min = Inf
for tailstr in ["_sp_nn"]
    for iter in 1:num_expriments
        dd = load(joinpath(subworkpath,prefix,"$(prefix)$(tailstr)_alpha_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_minf = floor(rt2_min, digits=2)
rng = range(0,stop=rt2_minf,length=100)

stat_nn1=[]; stat_sp1=[]; stat_sp_nn1=[] # fits
stat_nn2=[]; stat_sp2=[]; stat_sp_nn2=[] # sparse W
for tailstr in ["_sp_nn"]
    afs=[]; sws=[]
    for iter in 1:num_expriments
        @show tailstr, iter
        dd = load(joinpath(subworkpath,prefix,"$(prefix)$(tailstr)_alpha_results$(iter).jld2"))
        rt2s = dd["data"]["rt2s"]; avgfits = dd["data"]["avgfits"]; sparseWs = dd["data"]["sparseWs"]
        lr = length(rt2s); la = length(avgfits)
        lr != la && (l=min(lr,la); rt2s=rt2s[1:l]; avgfits=avgfits[1:l])
        nodes = (rt2s,)
        itp1 = Interpolations.interpolate(nodes, avgfits, Gridded(Linear()))
        itp2 = Interpolations.interpolate(nodes, sparseWs, Gridded(Linear()))
        push!(afs,itp1(rng)); push!(sws,itp2(rng))
    end
    avgfits = hcat(afs...); sparseWs = hcat(sws...)
    means1 = dropdims(mean(avgfits,dims=2),dims=2)
    stds1 = dropdims(std(avgfits,dims=2),dims=2)
    means2 = dropdims(mean(sparseWs,dims=2),dims=2)
    stds2 = dropdims(std(sparseWs,dims=2),dims=2)
    tailstr == "_nn" && (push!(stat_nn1,means1); push!(stat_nn1,stds1);
                        push!(stat_nn2,means2); push!(stat_nn2,stds2))
    tailstr == "_sp" && (push!(stat_sp1,means1); push!(stat_sp1,stds1);
                        push!(stat_sp2,means2); push!(stat_sp2,stds2))
    tailstr == "_sp_nn" && (push!(stat_sp_nn1,means1); push!(stat_sp_nn1,stds1);
                        push!(stat_sp_nn2,means2); push!(stat_sp_nn2,stds2))
end
save(joinpath(subworkpath,"hals_cbcl_alpha_runtime_vs_fits.jld2"),"rng",rng,
        "stat_nn1", stat_nn1, "stat_sp1", stat_sp1, "stat_sp_nn1", stat_sp_nn1,
        "stat_nn2", stat_nn2, "stat_sp2", stat_sp2, "stat_sp_nn2", stat_sp_nn2)
