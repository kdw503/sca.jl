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

# linux prompt & batch file> julia $MYSTORAGE/Work/julia/sca/paper/runtime_all.jl \"SNR\" [\"lcsvd_precon\",\"hals\",\"compnmf\"] 50 -10 1 15 150 120 0.1 800
# powershell prompt> julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\runtime_all.jl '\"SNR\"' '[\"lcsvd_precon\",\"hals\",\"compnmf\"]' 50 -10 1 15 150 120 0.1 800
# in batchfile> julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\runtime_all.jl \"SNR\" [\"lcsvd_precon\",\"hals\",\"compnmf\"] 50 -10 1 15 150 120 0.1 800
# to run the batch file in powershell> Start-Process -FilePath "C:\Users\kdw76\WUSTL\work\julia\sca\expr.bat -Wait
# in julia REPL> ARGS = ["\"Denoise\"","[\"lcsvd_precon\"]", "50","-10","1","15","150",":meanT","3","3","120"]
subdir = eval(Meta.parse(ARGS[1]))
subworkpath = joinpath(workpath,"paper",subdir)
methods=eval(Meta.parse(ARGS[2]));
num_experiments = eval(Meta.parse(ARGS[3]));
SNR = eval(Meta.parse(ARGS[4]))
factor = eval(Meta.parse(ARGS[5]));
ncells = eval(Meta.parse(ARGS[6]));
lcsvd_maxiter = eval(Meta.parse(ARGS[7]));
# useprecond = eval(Meta.parse(ARGS[8]));
# usedenoiseW0H0 = eval(Meta.parse(ARGS[9]));
filter = eval(Meta.parse(ARGS[8])); # :meanS, :meanT, :meanST
flS = eval(Meta.parse(ARGS[9]));
flT = eval(Meta.parse(ARGS[10]));
hals_maxiter = eval(Meta.parse(ARGS[11]));
# subdir="size_test"; SNR=0; ncells=15; factor=10; lcsvd_maxiter=80; hals_maxiter=80; compnmf_maxiter=800; iter=1;

using InteractiveUtils
sysdir = joinpath(subworkpath,"sysinfo")
isdir(sysdir) || mkdir(sysdir)
sysfn = joinpath(sysdir,"sysinfo$(SNR)db$(factor)f$(ncells)s.txt")
isfile(sysfn) && rm(sysfn)
open(sysfn,"w") do file
    versioninfo(file;verbose=true)
end

dataset = :fakecells; inhibitindices=0; bias=0.1
filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; maskth=0.25; makepositive = true; tol=-1
imgsz0 = (40,20)
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

for iter in 1:num_experiments
@show iter; flush(stdout)
X, imsz, lhT, ncs, gtncells, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

for (filter,flS,flT) in [(:meanST,3,3),(:none,0,0)]
if filter == :meanST 
    Xlpf = LCSVD.noisefilter(:meanS,X;filterlength=flS)
    Xlpf = LCSVD.noisefilter(:meanT,Xlpf;filterlength=flT)
elseif filter ∈ [:meanS]
    Xlpf = LCSVD.noisefilter(:meanS,X;filterlength=flS)
elseif filter == :meanT
    Xlpf = LCSVD.noisefilter(:meanT,X;filterlength=flT)
elseif filter == :none
    Xlpf = X
end

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(Xlpf, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), Xlpf, Wcd, Hcd)
    LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); Xlpf .-= bg
end

prefix = "lcsvd_precon"
# LCSVD
useprecond = prefix ∈ ["lcsvd_precon","lcsvd_precon_LPF"] ? true : false
usedenoiseW0H0 = false
uselv=false; s=10; maxiter = lcsvd_maxiter
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
#    try
for (tailstr,initmethod,α,β) in [("_sp",:isvd,0.005,0.),("_nn",:isvd,0,5.),("_sp_nn",:isvd,0.005,5.0)]#
     @show tailstr
    dd = Dict()
    α1=α2=α; β1=β2=β
    avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
    # rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
    # stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
    #     regSpar=regSpar, regNN=regNN, useRelaxedL1=useRelaxedL1, σ0=σ0, r=r, poweradjust=poweradjust,
    #     useprecond=useprecond, uselv=uselv, maskW=maskW, maskH=maskH)
    # lsparams = LineSearchParams(method=ls_method, α0=1.0, c_1=1e-4, maxiter=ls_maxiter)
    # cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
    #     x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
    #     store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=499:500)
    # Mw, Mh = copy(Mw0), copy(Mh0);
    # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
    #                                 penmetric=penmetric, stparams=stparams, lsparams=lsparams, cparams=cparams);
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(Xlpf, ncells; initmethod=initmethod, svdmethod=:isvd)
    σ0=s*std(W0) #=10*std(W0)=#
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond, usedenoiseW0H0=usedenoiseW0H0,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    M, N = copy(M0), copy(N0)
    rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
    alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, Xlpf, W0, H0, D, M, N);
    Wlc, Hlc = rst0.W, rst0.H
    avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wlc, Hlc; clamp=false)
    LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
    fprex = "$(prefix)$(SNR)db$(filter)"
    # precondstr = useprecond ? "_precond" : ""
    # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
    fname = joinpath(subworkpath,prefix,"$(filter)_$(flS)_$(flT)","$(fprex)_a$(α)_b$(β)_it$(rst0.niters)_rt$(rt2)_af$(avgfit)")
    #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
    TestData.imsave_data_gt(dataset,fname*"_gt", Wlc,Hlc,gtW,gtH,imgsz,100; saveH=false, verbose=false)

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
        metadata["sigma0"] = σ0; metadata["r"] = r; metadata["initmethod"] = initmethod
        metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
        metadata["usedenoiseW0H0"] = usedenoiseW0H0; metadata["denoisefilter"] = alg.denoisefilter; 
        metadata["alpha"] = α; metadata["beta"] = β
    end
    save(joinpath(subworkpath,prefix,"$(filter)_$(flS)_$(flT)","$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    GC.gc()
end # tailstr
end # filter
end # iter


if prefix == "hals"
    # HALS
    rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar); hals_α = 0.1
    maxiter = hals_maxiter
    for (tailstr,initmethod,α) in [("_nn",:nndsvd,0.),("_sp_nn",:nndsvd,hals_α)]#
        dd = Dict()
        W1, H1 = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
        result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                        tol=tol, verbose=true), X, W1, H1; gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
        W1, H1 = copy(Wcd0), copy(Hcd0);
        rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                        tol=tol, verbose=false), X, W1, H1)
        avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
        LCSVD.normalizeW!(W1,H1)#; W1,H1 = LCSVD.sortWHslices(W1,H1)
        fprex = "$(prefix)$(SNR)db$(factor)f$(ncells)s$(initmethod)"
        fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
        #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
        TestData.imsave_data_gt(dataset,fname*"_gt", W1,H1,gtW,gtH,imgsz,100; saveH=false, verbose=false)
        rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
        dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1cd; dd["rt2s"] = rt2s
        dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
        if true#iter == num_experiments
            metadata = Dict()
            metadata["maxiter"] = maxiter; metadata["alpha"] = α
        end
        save(joinpath(subworkpath,"$(filter)_$(flS)_$(flT)",prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end
end # for methods
end # for iter

# Q = qr(randn(8, 8))
# Q = Q.Q
# Q*Q'
# Q = Q[:,1:3]
# D = Diagonal([10, 1, 0.1])
# F = svd(X)
# FN = svd(X .+ 0.1 * randn(8, 8))
# i = 1; dot(FN.U[:,i], F.U[:,i])
# i = 2; dot(FN.U[:,i], F.U[:,i])
# i = 3; dot(FN.U[:,i], F.U[:,i])

