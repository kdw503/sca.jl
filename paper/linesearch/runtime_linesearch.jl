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

# linux prompt & batch file> julia $MYSTORAGE/Work/julia/sca/paper/runtime_all.jl \"SGD\" [:lbfgs,:slbfgs,:sgd] 1 50 -10 1 15 0 150
# powershell prompt> julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\runtime_all.jl '\"SGD\"' '[:lbfgs,:slbfgs,:sgd]' 1 50 -10 1 15 0 150
# in batchfile> julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\runtime_all.jl \"SGD\" [:lbfgs,:slbfgs,:sgd] 1 50 -10 1 15 0 150
# to run the batch file in powershell> Start-Process -FilePath "C:\Users\kdw76\WUSTL\work\julia\sca\expr.bat -Wait
# in julia REPL> ARGS = ["\"linesearch\"","[:backtracking,:morethuente,:hagerzhang,:none]", "1", "50","-10","1","15","0","30"]
subdir = eval(Meta.parse(ARGS[1]))
subworkpath = joinpath(workpath,"paper",subdir)
methods=eval(Meta.parse(ARGS[2]));
num_experistrt = eval(Meta.parse(ARGS[3]));
num_experiments = eval(Meta.parse(ARGS[4]));
SNR = eval(Meta.parse(ARGS[5]))
factor = eval(Meta.parse(ARGS[6]));
noc = eval(Meta.parse(ARGS[7]));
nac = eval(Meta.parse(ARGS[8]));
ncells = noc+nac
pcb_maxiter = eval(Meta.parse(ARGS[9]));

dataset = :fakecells; inhibitindices=0; bias=0.1
filter = dataset ∈ [:neurofinder] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; maskth=0.25; makepositive = true; tol=-1
imgsz0 = (40,20)
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

for iter in num_experistrt:num_experiments
@show iter; flush(stdout)
X, imsz, lhT, ncs, gtncells, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

#(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = LCSVD.noisefilter(filter,X,imgsz)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

for prefix in methods
@show prefix; flush(stdout)
    maxiter = pcb_maxiter == 0 ? Int(ceil(log(eps(eltype(X)))/log(r))) : pcb_maxiter
    useprecond = false; uselv=false; smaxiter=500; inner_maxiter = 1000
    r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
        # if this is too big iteration number would be increased
#    try
    for (tailstr,initmethod,α,β) in [("_sp_nn",:isvd,0.005,5.0),("_nn",:nndsvd,0.,5.0), ("_sp",:isvd,0.005,0.)]#
        @show tailstr
        dd = Dict()
        α1=α2=α; β1=β2=β
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
        H0t, N0t = copy(H0'), copy(N0')
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=useprecond, optim_method = :lbfgs,
            ls_method=prefix, denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, 
            inner_maxiter = inner_maxiter, smaxiter = smaxiter, store_trace = true, store_inner_trace = true, show_trace = false,
            allow_f_increases = true, f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol,
            successive_f_converge=0)
        M, Nt = copy(M0), copy(N0t)
        rst = LCSVD.solve!(alg, X, W0, H0t, D, M, Nt; gtW=gtW, gtH=gtH);
        alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        M, Nt = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0t, D, M, Nt);
        Wlc, Hlc = rst0.W, rst0.Ht'
        avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wlc, Hlc; clamp=false)
        LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
        fprex = "$(prefix)$(SNR)db$(factor)f$(ncells)s$(initmethod)"
        # precondstr = useprecond ? "_precond" : ""
        # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
        fname = joinpath(subworkpath,"$(prefix)","$(fprex)_a$(α)_b$(β)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
        #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
        TestData.imsave_data_gt(dataset,fname*"_gt", Wlc,Hlc,gtW,gtH,imgsz,100; saveH=false, verbose=false)

        f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niters); totalniters = sum(niters)
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
            metadata["r"] = r; metadata["initmethod"] = initmethod
            metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
            metadata["denoisefilter"] = alg.denoisefilter; 
            metadata["alpha"] = α; metadata["beta"] = β
        end
        save(joinpath(subworkpath,"$(prefix)","$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
        GC.gc()
    end
    # catch e
    #     fprex = "$(prefix)$(SNR)db$(factor)f$(ncells)s"
    #     save(joinpath(subworkpath,prefix,"$(fprex)_error_$(iter).jld2"),datadic)
    #     @warn e
    # end
end # for methods
end # for iter
