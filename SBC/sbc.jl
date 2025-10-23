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
subworkpath = joinpath(workpath,"SBC")

dataset = :fakecells; SNR = 0
imgsz = (40,20); lengthT = 1000; prefix = "pcb"; num_experiments=10000
maskth=0.25; maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
subtract_bg = false

for dataset in [:fakecells, :neurofinder_small, :cbclface, :pcrnaseq]
    @show dataset
    X, imsz, lhT, noc, gtnoc, datadic = load_data(dataset; sigma=5.0, imgsz=imgsz, lengthT=lengthT,
            SNR=SNR, bias=0.1, useCalciumT=true, inhibitindices=0, issave=false, isload=false,
            gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false, dataset_name="Baron");
    (m,n,p) = (size(X)...,noc)
    nac =0; nc = noc + nac
    gtW, gtH = dataset ∈ [:fakecells] ? (datadic["gtW"], datadic["gtH"]) : (zeros(0,0), zeros(0,0))

    if subtract_bg
        rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
        NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
        LCSVD.normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
        plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
        bg = W*fill(mean(H),1,n)
#        bg = W*H
        X .-= bg
    end

    useprecond = false; usedenoiseUVt = false; uselv = false
    r=0.3; maxiter = 100; sbc_maxiter = dataset == :pcrnaseq ? 1000 : 100
    for initmethod in [:sbc, :tsvd]
        (tailstr,α,β) = ("_sp",0.005,0.)
        fprex = dataset == :fakecells ? "$(dataset)$(SNR)db_$(initmethod)" : "$(dataset)_$(initmethod)"

        dd = Dict()
        α1=α2=α; β1=β2=β
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]
        rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod,
                                                                svdmethod=:tsvd, sbc_maxiter=sbc_maxiter)
        LCSVD.normalizeW!(Wp,Hp)
        sparsityW = norm(Wp,1)
        LCSVD.flip2makepos!(Wp,Hp)
        if dataset == :fakecells
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, Wp, Hp; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            Wp1, Hp1 = Wp[:,nodr], Hp[nodr,:]
        elseif dataset == :neurofinder_small
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(X,datadic["cells"], Wp, Hp; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            Wp1, Hp1 = Wp[:,nodr], Hp[nodr,:]
        else
            fv = LCSVD.fitd(X,Wp*Hp)
            Wp1, Hp1 = Wp, Hp
        end
        fpstx = initmethod == :sbc ? "iter$(sbc_maxiter)_rt$(rt1)" : "rt$(rt1)"
        fname = dataset ∈ [:fakecells, :neurofinder_small] ? "$(fprex)_af$(fv)_sW$(sparsityW)_$(fpstx)" :
                                                             "$(fprex)_f$(fv)_sW$(sparsityW)_$(fpstx)"
        pathfname = joinpath(subworkpath,fname)
        #imsave_data(dataset,fname,W3,H3,imsz,100; saveH=false)
        imsave_data(dataset,pathfname,Wp1,Hp1,imsz,100; saveH=false, verbose=false, w_limit_factor=0.01, h_limit_factor=0.9)

        V = copy(H0'); N0t = copy(N0')
        tol=1e-6; inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseUVt=usedenoiseUVt, maskW = Colon(), maskH = Colon(), optim_method= :sgd_injectnoise,
            denoisefilter=:avg, uselv=uselv, imgsz=imsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = false,
            store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0, ur=0.002, nr=0.001);
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

        W1, H1 = rst0.W, rst0.Ht'
        LCSVD.normalizeW!(W1,H1);
        sparsityW = norm(W1,1)
        LCSVD.flip2makepos!(W1,H1)
        if dataset == :fakecells
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            W1, H1 = W1[:,nodr], H1[nodr,:]
        elseif dataset == :neurofinder_small
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(X,datadic["cells"], W1, H1; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            W1, H1 = W1[:,nodr], H1[nodr,:]
        else
            fv = LCSVD.fitd(X,W1*H1)
        end
        fpstx = "it$(rst0.niters)_rt$(rt2)"
        fname = dataset ∈ [:fakecells, :neurofinder_small] ? "$(fprex)_a$(α)_b$(β)_af$(fv)_sW$(sparsityW)_$(fpstx)" :
                                                             "$(fprex)_a$(α)_b$(β)_f$(fv)_sW$(sparsityW)_$(fpstx)"
        pathfname = joinpath(subworkpath,fname)
        #imsave_data(dataset,fname,W3,H3,imsz,100; saveH=false)
        imsave_data(dataset,pathfname,W1,H1,imsz,100; saveH=false, verbose=false, w_limit_factor=0.01, h_limit_factor=0.9)
        @show dataset, initmethod, avgfit, rt2
    end
end
