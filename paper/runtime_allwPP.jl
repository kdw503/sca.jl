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

# linux prompt & batch file> julia $MYSTORAGE/Work/julia/sca/paper/runtime_all.jl \"SNR\" [\"pcb_precon\",\"hals\",\"compnmf\"] 50 -10 1 15 150 120 0.1 800
# powershell prompt> julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\runtime_all.jl '\"SNR\"' '[\"pcb_precon\",\"hals\",\"compnmf\"]'  50 -10 1 15 150 120 0.1 800
# in batchfile> julia C:\Users\kdw76\WUSTL\Work\julia\sca\paper\runtime_all.jl \"SNR\" [\"pcb_precon\",\"hals\",\"compnmf\"] 50 -10 1 15 150 120 0.1 800
# to run the batch file in powershell> Start-Process -FilePath "C:\Users\kdw76\WUSTL\work\julia\sca\expr.bat -Wait
# in julia REPL> ARGS = ["\"SNRwPP\"","[\"pcb\",\"hals\",\"compnmf\"]", "1", "2","0","1","15","150","120","0.1","800"]
# in julia REPL> ARGS = ["\"tsvd_test\"","[\"pcb_tsvd\"]", "1", "2","0","1","15","150","120","0.1","800"]
subdir = eval(Meta.parse(ARGS[1]))
@show subdir; flush(stdout)
subworkpath = joinpath(workpath,"paper",subdir)
methods=eval(Meta.parse(ARGS[2]));
num_experistrt = eval(Meta.parse(ARGS[3]));
num_experiments = eval(Meta.parse(ARGS[4]));
SNR = eval(Meta.parse(ARGS[5]))
factor = eval(Meta.parse(ARGS[6]));
noc = eval(Meta.parse(ARGS[7]));
nac = 0
pcb_maxiter = eval(Meta.parse(ARGS[8]));
# useprecond = eval(Meta.parse(ARGS[8]));
# usedenoiseW0H0 = eval(Meta.parse(ARGS[9]));
hals_maxiter = eval(Meta.parse(ARGS[9]));
hals_α = eval(Meta.parse(ARGS[10]));
compnmf_maxiter = eval(Meta.parse(ARGS[11]));
# subdir="size_test"; SNR=0; noc=15; factor=10; pcb_maxiter=80; hals_maxiter=80; compnmf_maxiter=800; iter=1;

# using InteractiveUtils
# try
#     sysdir = joinpath(subworkpath,"sysinfo")
#     isdir(sysdir) || mkdir(sysdir, 0o700) # why error here
#     sysfn = joinpath(sysdir,"sysinfo$(SNR)db$(factor)f$(noc)s.txt")
#     isfile(sysfn) && rm(sysfn)
#     open(sysfn,"w") do file
#         versioninfo(file;verbose=true)
#     end
# catch e
#     @warn e
# end

dataset = :fakecells; inhibitindices=0; bias=0.1
lpfilter = :meanST; filterstr = "_$(lpfilter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; maskth=0.25; makepositive = true; tol=-1
imgsz0 = (40,20)
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

#initisvd(X,noc) = ((U,s)=IncrementalSVD.isvd(X,noc); H = U'*X ; (U, H, copy(U), copy(H))) # H isn't normalized one
for iter in num_experistrt:num_experiments
@show iter; flush(stdout)
X, imsz, lhT, ncs, gtnoc, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,noc)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
if lpfilter == :meanST
    X = LCSVD.noisefilter(:meanS,X,imgsz)
    X = LCSVD.noisefilter(:meanT,X,imgsz)
else
    X = LCSVD.noisefilter(lpfilter,X,imgsz)
end
# maxindices = argmax.(eachcol(gtH))
# maxSNR_X = X[:,[maxindices...]]
# TestData.imsaveW(joinpath(subworkpath,"X_SNR$(SNR)_LPF_maxSNR_W.png"), maxSNR_X, imgsz, borderwidth=1,colors=TestData.bbw())

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

for prefix in methods
@show prefix; flush(stdout)

if prefix in ["pcb_precon","pcb_precon_LPF","pcb","pcb_LPF","pcb_precon_tsvd","pcb_tsvd"]
    # LCSVD
    useprecond = prefix ∈ ["pcb_precon","pcb_precon_LPF","pcb_precon_tsvd"] ? true : false
    usedenoiseW0H0 = prefix ∈ ["pcb_precon_LPF","pcb_LPF"] ? true : false
    uselv=false; maxiter = pcb_maxiter
    r=0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
        # if this is too big iteration number would be increased
#    try
    for (tailstr,α,β) in [("_sp_nn",0.005,5.0), ("_sp",0.005,0.)]#,("_nn",0.,5.0)
        initmethod = tailstr == "_nn" ? :nndsvd : prefix ∈ ["pcb_tsvd","pcb_precon_tsvd"] ? :tsvd : :isvd
        dd = Dict()
        β1 = β2= β; α1 = α2 = α
        β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
        α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]

        rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
        V = copy(H0'); N0t = copy(N0')
        inner_tol = 1e-6; inner_maxiter = 50#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseW0H0=usedenoiseW0H0, maskW=maskW, maskH = maskH,
            denoisefilter=:avg, uselv=uselv, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
            store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
        M1, N1t = copy(M0), copy(N0t)
        rst = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
        alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

        @show tailstr, rt2
        W1, H1 = rst0.W, rst0.Ht'
        LCSVD.normalizeW!(W1,H1);
        dataset == :fakecells && begin
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            W1, H1 = W1[:,nodr], H1[nodr,:]
        end
        dataset != :fakecells && (fv = LCSVD.fitd(X,W1*H1))
        fprex = "$(prefix)$(SNR)db$(factor)f$(noc)s$(initmethod)"
        # precondstr = useprecond ? "_precond" : ""
        # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
        fname = joinpath(subworkpath,"$(prefix)","$(fprex)_a$(α)_b$(β)_f$(fv)_it$(rst0.niters)_rt$(rt2)")
        #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
        imsave_data(dataset,fname,W1,H1,imgsz,100; saveH=false, verbose=false)

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
            metadata["usedenoiseW0H0"] = usedenoiseW0H0; metadata["denoisefilter"] = alg.denoisefilter; 
            metadata["alpha"] = α; metadata["beta"] = β
        end
        save(joinpath(subworkpath,"$(prefix)","$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
        GC.gc()
    end
    # catch e
    #     fprex = "$(prefix)$(SNR)db$(factor)f$(noc)s"
    #     save(joinpath(subworkpath,prefix,"$(fprex)_error_$(iter).jld2"),datadic)
    #     @warn e
    # end
end

if prefix == "hals"
    # HALS
    rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, noc, variant=:ar)
    maxiter = hals_maxiter
    for (tailstr,initmethod,α) in [("_nn",:nndsvd,0.),("_sp_nn",:nndsvd,hals_α)]#
        dd = Dict()
        W1, H1 = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
        result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                        tol=tol, verbose=true), X, W1, H1; gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)
        W1, H1 = copy(Wcd0), copy(Hcd0);
        rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                        tol=tol, verbose=false), X, W1, H1)

        @show tailstr, rt2
        avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
        LCSVD.normalizeW!(W1,H1)#; W1,H1 = LCSVD.sortWHslices(W1,H1)
        fprex = "$(prefix)$(SNR)db$(factor)f$(noc)s$(initmethod)"
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
        save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end

if prefix == "compnmf"
    # COMPNMF
    mfmethod = :COMPNMF; maxiter = compnmf_maxiter
    for (tailstr,initmethod) in [("_nn",:lowrank_nndsvd)]
        dd = Dict()
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        # rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, noc, initmethod=initmethod,poweradjust=initpwradj)
        # stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
        #     regSpar=regSpar, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc,
        #     uselv=uselv, maskW=maskW, maskH=maskH)
        # cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        #     x_abstol=tol, successive_f_converge=0, maxiter=maxiter, inner_maxiter=inner_maxiter,
        #     store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
        # Mw, Mh = copy(Mw0), copy(Mh0);
        # rt2 = @elapsed W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
        #                                                     penmetric=penmetric, stparams=stparams, cparams=cparams);
        rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, noc, variant=:ar);
        Wcn, Hcn = copy(Wcn0), copy(Hcn0);
        result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn;
                            gtU=gtW, gtV=gtH, maskU=maskW, maskV=maskH)
        Wcn, Hcn = copy(Wcn0), copy(Hcn0);
        rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
        rt1 += rst0.inittime # add calculation time for compression matrices L and R
        rt2 -= rst0.inittime

        @show tailstr, rt2
        avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wcn, Hcn; clamp=false)
        LCSVD.normalizeW!(Wcn,Hcn)#; W1,H1 = LCSVD.sortWHslices(Wcn,Hcn)
        fprex = "$(prefix)$(SNR)db$(factor)f$(noc)s$(initmethod)"
        fname = joinpath(subworkpath,prefix,"$(fprex)_af$(avgfit)_it$(rst0.niters)_rt$(rt2)")
        #imsave_data(dataset,fname,W3,H3,imgsz,100; saveH=false)
        TestData.imsave_data_gt(dataset,fname*"_gt", Wcn,Hcn,gtW,gtH,imgsz,100; saveH=false, verbose=false)

        rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
        dd["niters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
        dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;
        if true#iter == num_experiments
            metadata = Dict()
            metadata["maxiter"] = maxiter
        end
        save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
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

