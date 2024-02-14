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

dataset = :cbclface; SNR=0; inhibitindices=0; bias=0.1; initmethod=:lowrank_nndsvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false; tol=-1

if true
    num_experiments = 50
    lcsvd_maxiter = 40; lcsvd_inner_maxiter = 50; lcsvd_ls_maxiter = 100
    compnmf_maxiter = 500; compnmf_inner_maxiter = 0; compnmf_ls_maxiter = 0
    hals_maxiter = 400
else
    num_experiments = 3
    lcsvd_maxiter = 2; lcsvd_inner_maxiter = 2; lcsvd_ls_maxiter = 2
    compnmf_maxiter = 2; compnmf_inner_maxiter = 0; compnmf_ls_maxiter = 0
    hals_maxiter = 2
end

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
# X = noisefilter(filter,X)
maskth = 0.25
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end
# rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)

hals_αrng = range(0.0,0.5,num_experiments); αrng = range(0.001,0.01,num_experiments)
for iter in 1:num_experiments
    @show iter; flush(stdout)

    # LCSVD
    prefix = "lcsvd_precon"
    @show prefix
    useprecond = prefix ∈ ["lcsvd_precon","lcsvd_precon_LPF"] ? true : false
    uselv=false; s=10; maxiter = lcsvd_maxiter
    r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
        # if this is too big iteration number would be increased
    α1=α2=α=αrng[iter]; β1=β2=β=0
    for (tailstr,initmethod) in [("_sp",:isvd)]#
        dd = Dict()
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
        σ0=s*std(W0) #=10*std(W0)=#
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=useprecond, usedenoiseW0H0=false,
            denoisefilter=:avg, uselv=false, imgsz=imgsz, maskW=maskW, maskH=maskH, maxiter = maxiter, store_trace = true,
            store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            store_sparsity_nneg = true, f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
        M, N = copy(M0), copy(N0)
        rst = LCSVD.solve!(alg, X, W0, H0, D, M, N; gtW=gtW, gtH=gtH);
        alg.store_trace = false; alg.store_inner_trace = false; alg.maskW = alg.maskH = Colon()
        M, N = copy(M0), copy(N0)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);

        f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niter); totalniters = sum(niters)
        avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
        sparseWss = LCSVD.getdata(rst.traces,:sparseWs)
        avgfits = Float64[]; inner_fxs = Float64[]; sparseWs = Float64[]; rt2s = Float64[]
        for (iter,(afs,fxs,sws)) in enumerate(zip(avgfitss, fxss, sparseWss))
            isempty(afs) && continue
            append!(avgfits,afs); append!(inner_fxs,fxs); append!(sparseWs,sws)
            if iter == 1
                rt2i = 0.
            else
                rt2i = collect(range(start=rst0.laps[iter-1],stop=rst0.laps[iter],length=length(afs)+1))[1:end-1].-rst0.laps[1]
            end
            append!(rt2s,rt2i)
        end
        dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
        dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs; dd["sparseWs"] = sparseWs
        if true#iter == num_experiments
            metadata = Dict()
            metadata["sigma0"] = σ0; metadata["r"] = r; metadata["initmethod"] = initmethod
            metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
            metadata["usedenoiseW0H0"] = false; metadata["denoisefilter"] = alg.denoisefilter; 
            metadata["alpha"] = α; metadata["beta"] = β
        end
        fprex = "$(prefix)"
        save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end

    # HALS
    prefix = "hals"
    @show prefix
    rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar) # initdata=svd(X)
    maxiter = hals_maxiter
    α = hals_αrng[iter]
    for (tailstr) in [("_sp_nn")]#
        dd = Dict()
        W1, H1 = copy(Wcd0), copy(Hcd0)#; avgfit, _ = NMF.matchedfitval(gtW,gtH, Wcd, Hcd; clamp=false); push!(avgfits,avgfit)
        result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1, tol=tol, verbose=true,
                        SCA_penmetric=:SPARSE_W), X, W1, H1; gtW=gtW, gtH=gtH, maskW=:, maskH=:)
        W1, H1 = copy(Wcd0), copy(Hcd0);
        rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=hals_αrng[iter], l₁ratio=1,
                        tol=tol, verbose=false), X, W1, H1)
        fprex = "$(prefix)"
        rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
        dd["niters"] = result.niters; dd["totalniters"] = result.niters; dd["rt1"] = rt1cd; dd["rt2s"] = rt2s
        dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues;; dd["sparseWs"] = result.sparsevalues
        if true#iter == num_experiments
            metadata = Dict()
            metadata["maxiter"] = maxiter; metadata["alpha"] = α
        end
        save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end

    # COMPNMF
    prefix = "compnmf"
    @show prefix
    maxiter = compnmf_maxiter
    for (tailstr,initmethod) in [("_nn",:lowrank_nndsvd)]
        dd = Dict()
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
        # L, R, X_tilde, Y_tilde, A_tilde = CompNMF.compmat(X, Wcn0, Hcn0; w=4)
        # X_tilde, Y_tilde = CompNMF.compressive_nmf(A_tilde, L, R, ncells; max_iter=1000, ls=0)
        # Wcn = L*X_tilde; Hcn = Y_tilde*R
        Wcn, Hcn = copy(Wcn0), copy(Hcn0);
        result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true, SCA_penmetric=:SPARSE_U),
                            X, Wcn, Hcn; gtU=gtW, gtV=gtH, maskU=:, maskV=:)
        Wcn, Hcn = copy(Wcn0), copy(Hcn0);
        rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
        rt1 += rst0.inittime # add calculation time for compression matrices L and R
        rt2 -= rst0.inittime
        
        rt2s = collect(range(start=0,stop=rt2,length=length(result.avgfits)))
        dd["niters"] = result.niters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
        dd["avgfits"] = result.avgfits; dd["f_xs"] = result.objvalues; dd["sparseWs"] = result.sparsevalues
        if true#iter == num_experiments
            metadata = Dict()
            metadata["maxiter"] = maxiter
        end
        fprex = "$(prefix)"
        save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end
