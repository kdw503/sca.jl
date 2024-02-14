using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","size")

include(joinpath(workpath,"setup_light.jl"))

using LCSVD

# ARGS =  ["0","20","10","150","120","800"]
# ARGS =  ["0","20","15","100","100","600"]
# ARGS =  ["0","20","20","50","50","400"]
# ARGS = ["0","2","15","2"]
# julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\size\runtime_all.jl 0 20 150 120 800
SNR = eval(Meta.parse(ARGS[1])); num_experiments = eval(Meta.parse(ARGS[2]));
ncells = eval(Meta.parse(ARGS[3]));
factor=1
lcsvd_maxiter = eval(Meta.parse(ARGS[4]));

# for factor == 10
# lcsvd_maxiter = 50; hals_maxiter = 50; compnmf_maxiter = 400

# for factor == 1
# lcsvd_maxiter = 150; hals_maxiter = 120; compnmf_maxiter = 800

# for testing
# lcsvd_maxiter = 1; sca_inner_maxiter = 2; sca_ls_maxiter = 2
# hals_maxiter = 1; maskth=0.25
# compnmf_maxiter = 1; admm_inner_maxiter = 0; admm_ls_maxiter = 0

dataset = :fakecells; inhibitindices=0; bias=0.1; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; maskth=0.25
makepositive = true; tol = -1.0;

imgsz0 = (40,20); iter=1
for iter in 1:num_experiments
    @show iter; flush(stdout)
    sqfactor = Int(floor(sqrt(factor)))
    imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncells, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
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
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

# LCSVD
prefix = "lcsvdisvd"
@show prefix; flush(stdout)
# penmetric = :SCA; sd_group=:whole; regSpar = :WH1M2; regNN=:WH2M2; α = 0.005; β = 5.0
# useRelaxedL1=true; optimmethod = :optim_lbfgs; ls_method = :ls_BackTracking;
# inner_maxiter = sca_inner_maxiter; ls_maxiter = sca_ls_maxiter; poweradjust = :none

mfmethod = :LCSVD; initmethod=:lowrank; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
      # if this is too big iteration number would be increased

# Result demonstration parameters
for (tailstr,initmethod,α,β) in [("_sp",:isvd,0.005,0.)]# 
try
    dd = Dict(); tol=-1
    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
    σ0=s*std(W0) #=10*std(W0)=#
    r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
          # if this is too big iteration number would be increased
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, uselv=false, maskW=maskW, maskH=maskH,
        maxiter = maxiter, store_trace = true, store_inner_trace = true, show_trace = false, allow_f_increases = true,
        f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, successive_f_converge=0)
    W, H = copy(Wp), copy(Hp); M, N = copy(M0), copy(N0)
    rst = LCSVD.solve!(alg, X, W0, H0, D, M, N, W, H; gtW=gtW, gtH=gtH);
    alg.store_trace = false; alg.store_inner_trace = false
    W, H = copy(Wp), copy(Hp); M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N, W, H; gtW=gtW, gtH=gtH);
    # 0.391060 seconds (197.17 k allocations: 135.467 MiB)
    
    avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W, H; clamp=false)
    normalizeW!(W,H); W3,H3 = sortWHslices(W,H)
    fprex = "$(prefix)$(SNR)db$(ncells)s"
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(rst.niters)_rt$(rt2)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", W3,H3,gtW,gtH,imgsz,100; saveH=false, verbose=false)
    
    f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niter); totalniters = sum(niters)
    avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
    avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
    for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
        isempty(afs) && continue
        append!(avgfits,afs); append!(inner_fxs,fxs)
        if iter == 1
            rt2i = 0.
        else
            rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
        end
        append!(rt2s,rt2i)
    end

    dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
    dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
    if true#iter == num_experiments
        metadata = Dict()
        metadata["alpha0"] = σ0; metadata["r"] = r; metadata["maxiter"] = maxiter
        metadata["alpha"] = α; metadata["beta"] = β; metadata["initmethod"] = initmethod
    end
    save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
catch e
    fprex = "$(prefix)$(SNR)db$(ncells)s"
    save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_error_$(iter).jld2"),datadic)
    @warn e
    iter -= 1
end
end
end # for iter


exit() # want to execute batch job until this point

#========= some tests =================#
include(joinpath(workpath,"setup_plot.jl"))

ncells = 15; lcsvd_maxiter = 20;
dataset = :fakecells; SNR=0; inhibitindices=0; bias=0.1; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; maskth=0.25
makepositive = true; tol = -1.0;

imgsz0 = (40,20); iter=1; factor=1
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncells, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
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
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

prefix = "lcsvd"
mfmethod = :LCSVD; initmethod=:lowrank; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter 
r=(0.3)^1
(tailstr,initmethod,α,β) = ("_sp_nn",:isvd,0.005,5.0)
tol=-1
α1=α2=α; β1=β2=β
avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
        # if this is too big iteration number would be increased
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, uselv=false, maskW=maskW, maskH=maskH,
    maxiter = maxiter, store_trace = true, store_inner_trace = true, show_trace = false, allow_f_increases = true,
    f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, successive_f_converge=0)
alg.store_sparsity_nneg = true
W, H = copy(Wp), copy(Hp); M, N = copy(M0), copy(N0); use_σ2_cal_pen = true
rst = LCSVD.solve!(alg, X, W0, H0, D, M, N, W, H; gtW=gtW, gtH=gtH, use_σ2_cal_pen=use_σ2_cal_pen);
alg.store_trace = false; alg.store_inner_trace = false
W, H = copy(Wp), copy(Hp); M, N = copy(M0), copy(N0)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N, W, H; gtW=gtW, gtH=gtH);
# 0.391060 seconds (197.17 k allocations: 135.467 MiB)

f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niter); totalniters = sum(niters)
avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
for (iter,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
    isempty(afs) && continue
    append!(avgfits,afs); append!(inner_fxs,fxs)
    if iter == 1
        rt2i = 0.
    else
        rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
    end
    append!(rt2s,rt2i)
end
invss = LCSVD.getdata(rst.traces,:invs)
sparseWss = LCSVD.getdata(rst.traces,:sparseWs); sparseHss = LCSVD.getdata(rst.traces,:sparseHs)
nnWss = LCSVD.getdata(rst.traces,:nnWs); nnHss = LCSVD.getdata(rst.traces,:nnHs)
invs = Float64[]; sparseWs = Float64[]; sparseHs = Float64[]; nnWs = Float64[]; nnHs = Float64[]
for (iter,(ivs,sws,shs,nws,nhs)) in enumerate(zip(invss, sparseWss, sparseHss, nnWss, nnHss))
    isempty(ivs) && continue
    append!(invs,ivs); append!(sparseWs,sws); append!(sparseHs,shs); append!(nnWs,nws); append!(nnHs,nhs)
end

# plot
fig = Figure(resolution=(600,300))
ax = AMakie.Axis(fig[1, 1], yscale=log10, limits = ((0,0.4), nothing), xlabel = "time(sec)", ylabel = "penalty")#, title = "Average Fit Value vs. Running Time")
ax2 = AMakie.Axis(fig[1, 1], yaxisposition = :right, limits = ((0,0.4), nothing), ylabel = "average fit" #= yticklabelcolor = :red =# )
lns = []; lbls = []; σstr = use_σ2_cal_pen ? "σ=σ" : "σ=0"
push!(lns, lines!(ax2, rt2s, avgfits, color=mtdcolors[1])); push!(lbls,"avgfits")
push!(lns, lines!(ax, rt2s, inner_fxs, color=mtdcolors[7])); push!(lbls,"fxs(σ=σ)")
push!(lns, lines!(ax, rt2s, invs, color=mtdcolors[2])); push!(lbls,"invs")
push!(lns, lines!(ax, rt2s, sparseWs, color=mtdcolors[3])); push!(lbls,"parseWs($(σstr))")
push!(lns, lines!(ax, rt2s, sparseHs, color=mtdcolors[4])); push!(lbls,"sparseHs($(σstr))")
push!(lns, lines!(ax, rt2s, nnWs, color=mtdcolors[5])); push!(lbls,"nnWs")
push!(lns, lines!(ax, rt2s, nnHs, color=mtdcolors[6])); push!(lbls,"nnHs")
fig[1,2] = Legend(fig, lns, lbls, "", framevisible=false) # halign = :left, valign = :top
save(joinpath(subworkpath,"avgfits$(SNR)db$(ncells)s_isvd_vs_penalties$(σstr).png"),fig,px_per_unit=2)
