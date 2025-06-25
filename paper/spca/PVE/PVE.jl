using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","spca","PVE")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))
include(joinpath(workpath,"clustering.jl"))

#=============== seqRNA ===============#
##### SMA method (using RCall)
using RCall, StatsBase

R"""
library(magrittr)
library(dplyr)
library(epca)
library(Matrix) # as.matrix
library(ggplot2)
"""
for i in 1:30
    for gtnoc in 2:2:16
        ##### data generation
        # ssize = 100; S = zeros(ssize,16)
        # ysize = 100; Y = zeros(ysize,16)
        # for j in 1:16
        #     S[6j+1:min(6j+7,ssize),j] = rand(min(6j+7,ssize)-6j)
        #     Y[6j+1:min(6j+7,ysize),j] = rand(min(6j+7,ysize)-6j)
        # end
        # E = 1.0*rand(100,100)
        # X = S*Y' + E
        # normX2 = norm(X)^2

        X, imgsz, lengthT, ncs, gtncells, datadic = load_data(:fakecells; sigma=5.0, imgsz=(80,40),
                lengthT=5000, SNR=0., orthogonal=true,  bias=0.1, useCalciumT=true, issave=false, isload=false,
                gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
        normX2 = norm(X)^2
        
        ##### SCA method
        @rput X
        @rput gtnoc
        rtsma = @elapsed R"""
        scar <- sca(X, k = gtnoc, gamma = 10000,
                    center = F, scale = F,
                    epsilon = 1e-3)
        Wsma <- as.matrix(scar$scores) # score
        Htsma <- as.matrix(scar$loadings)
        """
        @rget Wsma
        @rget Htsma
        fv_sma = LCSVD.fitd(X,Wsma*Htsma') # Fit : 0.7412118762276092, 0.9169270245523752(without shrink)
        LCSVD.normalizeWH!(Wsma, Htsma')
        Xy = X*Htsma*inv(Htsma'*Htsma)*Htsma'; pve_sca = norm(Xy)^2/normX2 # PVE : 0.6872496795041968, 0.7700655887190125(without shrink)

        ##### SVD method
        initmethod=:svd; svdmethod=:svd; nac=0
        rtisvd = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, gtnoc, 0; initmethod=initmethod, svdmethod=svdmethod)
        V = copy(H0'); N0t = copy(N0')
        Wtsvd = U*D # cell
        Httsvd = V # gene

        fv_hals = LCSVD.fitd(X,Wtsvd*Httsvd') # Fit : 0.9487231956060997
        LCSVD.normalizeWH!(Wtsvd, Httsvd')
        Xy = X*Httsvd*inv(Httsvd'*Httsvd)*Httsvd'; pve_svd = norm(Xy)^2/normX2 # PVE : 0.8144133301694165

        ##### PCB method
        α=0.005; βw = 0
        β1 = βw; β2= βw; α1 = α2 = α
        r=0.3; tol=1e-7
        T = eltype(U)
        maxiter = Int(ceil(log(eps(T))/log(r))) #lcsvd_maxiter
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            r=r, useprecond=false, usedenoiseW0H0=false, optim_method = :lbfgs,
            uselv=false, maxiter = maxiter, inner_maxiter = 1000, store_trace = false,
            store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = tol, successive_f_converge=0);
        M1, N1t = copy(M0), copy(N0t)

        rtpcb = @elapsed rst1 = LCSVD.solve!(alg, T.(X), U, V, D, M1, N1t);
        Wpcb, Htpcb = rst1.W, rst1.Ht
        normdiffpcb = norm(X-Wpcb*Htpcb') # 0.0
        fv_hals = LCSVD.fitd(X,Wpcb*Htpcb') # Fit : 0.9484673684926179
        LCSVD.normalizeWH!(Wpcb, Htpcb')
        Xy = X*Htpcb*inv(Htpcb'*Htpcb)*Htpcb'; pve_pcb = norm(Xy)^2/normX2 # PVE : 0.8626050278736054

        ##### HALS method
        rtnndsvd = @elapsed Wnnd, Hnnd = NMF.nndsvd(T.(X), gtnoc, variant=:ar);
        mfmethod = :HALS; αhals=0.1; maxiter = 60; tol=-1
        αhals = 0.1
        W, H = copy(Wnnd), copy(Hnnd);
        rthals = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=αhals, l₁ratio=1,
                        tol=tol, verbose=false), T.(X), W, H)
        Whals, Hthals = W, H'
        fv_hals = LCSVD.fitd(X,Whals*Hthals') # Fit : 0.9471974071709811
        LCSVD.normalizeWH!(Whals,Hthals')
        Xy = X*Hthals*inv(Hthals'*Hthals)*Hthals'; pve_hals = norm(Xy)^2/normX2 # PVE : 0.8121568519786516
    end
end

save(joinpath(subworkpath,dataset,"$(dataset)_Result_sp.jld2"), "gtnoc", gtnoc,
    "Wisvd", Wisvd, "Htisvd", Htisvd, "rtisvd", rtisvd,
    "Wtsvd", Wtsvd, "Httsvd", Httsvd, "rttsvd", rttsvd,
    "Wpcb", Wpcb, "Htpcb", Htpcb, "rtpcb", rtpcb, "α", α, "β", β,
    "Wsma", Wsma, "Htsma", Htsma, "rtirlba", rtirlba, "rtsma", rtsma,
    "Whals", Whals, "Hthals", Hthals, "rtnndsvd", rtnndsvd, "rthals", rthals, "αhals", αhals)
