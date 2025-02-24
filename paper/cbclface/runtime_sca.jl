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

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)

mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; tol=-1 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
  # if this is too big iteration number would be increased

usedenoiseW0H0 = false; makepositive = true
denoiseW0H0str = usedenoiseW0H0 ? "_udnW0H0" : ""
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,.0) # ("_sp_nn",:isvd,0.005,5.0),("_nn",:nndsvd,0.,5.0)

β1=β; β2=β; α1 = α2 = α
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
  # if this is too big iteration number would be increased

αrng = 0.0001:0.0002:0.01
for (iter, α) in enumerate(αrng)
    @show iter; flush(stdout)

for (tailstr,initmethod,β) in [("_sp",:isvd,0.)]
    @show tailstr; flush(stdout)
    dd = Dict()
    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; sparseWs=Float64[]; inner_fxs=Float64[]

    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=true, usedenoiseW0H0=usedenoiseW0H0,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = lcsvd_maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0, store_sparsity_nneg=true)
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst = LCSVD.solve!(alg, X, W0, H0, D, M, N);
    alg.store_trace = false; alg.store_inner_trace = false; alg.store_sparsity_nneg = false
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
    Wlc, Hlc = rst0.W, rst0.H
    # avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
    normalizeW!(Wlc,Hlc); fitval = LCSVD.fitd(X,Wlc*Hlc)
    flip2makepos!(Wlc,Hlc)
    fname = joinpath(subworkpath,"$(prefix)_$(initmethod)_a$(α)_b$(β)_f$(fitval)_it$(rst0.niters)_rt$(rt2)")
#    imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false)
 
    f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niter); totalniters = sum(niters)
    avgfitss = LCSVD.getdata(rst.traces,:avgfits); sparseWss = LCSVD.getdata(rst.traces,:sparseWs); fxss = LCSVD.getdata(rst.traces,:fxs)
    avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
    for (iter,(afs,sws, fxs)) in enumerate(zip(avgfitss, sparseWss, fxss))
        isempty(afs) && continue
        append!(avgfits,afs); append!(sparseWs,sws); append!(inner_fxs,fxs)
        if iter == 1
            rt2i = 0.
        else
            rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
        end
        append!(rt2s,rt2i)
    end
    dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["sparseWs"] = sparseWs; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter
        metadata["alpha"] = α; metadata["beta"] = β; metadata["initmethod"] = initmethod
    end
    save(joinpath(subworkpath,prefix,"$(prefix)$(tailstr)_alpha_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
end
# catch e
#     save(joinpath(subworkpath,"sca_error_$(iter).jld2"),datadic)
#     @warn e
#     iter -= 1
# end
end


using Interpolations

num_expriments=50
rt2_min = Inf
for tailstr in ["_sp"]
    for iter in 1:num_expriments
        dd = load(joinpath(subworkpath,prefix,"$(prefix)$(tailstr)_alpha_results$(iter).jld2"),"data")
        rt2s = dd["rt2s"]; @show rt2s[end]
        rt2_min = min(rt2_min,rt2s[end])
    end
end
rt2_minf = floor(rt2_min, digits=2)
rng = range(0,stop=rt2_minf,length=100)

stat_nn1=[]; stat_sp1=[]; stat_sp_nn1=[] # fits
stat_nn2=[]; stat_sp2=[]; stat_sp_nn2=[] # sparse W
for tailstr in ["_sp"]
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
save(joinpath(subworkpath,"lcsvd_cbcl_alpha_runtime_vs_fits.jld2"),"rng",rng,
        "stat_nn1", stat_nn1, "stat_sp1", stat_sp1, "stat_sp_nn1", stat_sp_nn1,
        "stat_nn2", stat_nn2, "stat_sp2", stat_sp2, "stat_sp_nn2", stat_sp_nn2)
