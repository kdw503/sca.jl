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
# in julia REPL> ARGS = ["\"SNR\"",":fakecells","[\"pcb\",\"hals\",\"compnmf\"]", "1", "2","0","1","15","150","120","0.1","800"]
# in julia REPL> ARGS = ["\"naomi/pavg\"",":naomi","[\"pcb_tsvd\",\"hals\",\"compnmf\"]", "1", "2","5.0","1","15","0.0","100","200","0.1","1000","false"]
subdir = eval(Meta.parse(ARGS[1]))
@show subdir; flush(stdout) 
subworkpath = joinpath(workpath,"paper",subdir)
dataset = eval(Meta.parse(ARGS[2]))
methods=eval(Meta.parse(ARGS[3]));
num_experistrt = eval(Meta.parse(ARGS[4]));
num_experiments = eval(Meta.parse(ARGS[5]));
SNR = eval(Meta.parse(ARGS[6])); pavg = SNR
factor = eval(Meta.parse(ARGS[7]));
noc = eval(Meta.parse(ARGS[8])); nac = 0
inh_frac = eval(Meta.parse(ARGS[9]));
pcb_maxiter = eval(Meta.parse(ARGS[10]));
# useprecond = eval(Meta.parse(ARGS[8]));
# usedenoiseW0H0 = eval(Meta.parse(ARGS[9]));
hals_maxiter = eval(Meta.parse(ARGS[11]));
hals_α = eval(Meta.parse(ARGS[12]));
compnmf_maxiter = eval(Meta.parse(ARGS[13]));
subtract_bg = eval(Meta.parse(ARGS[14]));
gridcols=8; gridrows=6 # grid cols and rows for W image saving
numWimg = gridcols*gridrows
# subdir="size_test"; SNR=0; noc=15; factor=10; pcb_maxiter=80; hals_maxiter=80; compnmf_maxiter=800; iter=1;

# using InteractiveUtils
# try
#     sysdir = joinpath(subworkpath,"sysinfo")
#     isdir(sysdir) || mkdir(sysdir, 0o700) # why error here
#     sysfn = joinpath(sysdir,"sysinfo$(noisestr)$(factor)f$(noc)s.txt")
#     isfile(sysfn) && rm(sysfn)
#     open(sysfn,"w") do file
#         versioninfo(file;verbose=true)
#     end
# catch e
#     @warn e
# end

(imgsz0, lengthT0, hplot_space) = dataset == :naomi ?     ((158,158), 5000, -10000) :
                                  dataset == :fakecells ? ((40,20), 1000, -5) :
                                                          ((40,20), 1000, -5)
issaveimg = true; figsize=(900,600)
inhibitindices=0; bias=0.1; orthogonal=false
lpfilter = dataset ∈ [:neurofinder] ? :meanT : :none; filterstr = "_$(lpfilter)"
noisestr = dataset == :fakecells ? "$(SNR)dB" : dataset == :naomi ? "$(pavg)mW" : ""
# datastr = dataset  ∈ [:fakecells, :naomi] ? "_fc$(inhibitindices)_$(noisestr)" : "_$(dataset)"

maskth=0.25; makepositive = true; tol=-1
sqfactor = Int(floor(sqrt(factor)))
vres0 = 1.0; vres = sqfactor*vres0 # v resolution for naomi dataset
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*lengthT0; sigma = sqfactor*5.0
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

#initisvd(X,noc) = ((U,s)=IncrementalSVD.isvd(X,noc); H = U'*X ; (U, H, copy(U), copy(H))) # H isn't normalized one
for iter in num_experistrt:num_experiments
@show iter; flush(stdout)
X, imsz, lhT, ncs, gtnoc, datadic = load_data(dataset; # dpath=joinpath(workpath,"paper", subdir,"test"),
        imgsz=imgsz, lengthT=lengthT,                               # fakecells and naomi common parameters
        sigma=sigma, SNR=SNR, bias=bias, useCalciumT=true, orthogonal=orthogonal, # fakecells-specific parameters
        gtincludebg=false, inhibitindices=inhibitindices,           # fakecells-specific parameters (avg_rad unit is μm)
        seed=iter, pavg=pavg, vres=vres, avg_rad=7.0, inh_frac=inh_frac, psf_NA=0.3, gtncells=500, # naomi-specific parameters
        issave=true, isload=true,
        save_gtimg=true, save_maxSNR_X=false, save_X=false,
        verbose=false);

(m,n,p) = (size(X)...,noc)
gtW, gtH = dataset in [:fakecells, :naomi] ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

if dataset in [:naomi]
    powers, mod_vals = datadic["powers"], datadic["mod_vals"]
    gtH_nobase = (gtH./powers.-mod_vals).*powers
    dt = datadic["dt"]; inh_idx = datadic["inh_idx"]
else
    gtH_nobase = gtH
end

X = LCSVD.noisefilter(lpfilter,X,imgsz)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    LCSVD.normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
#    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitindices]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

for prefix in methods
@show prefix; flush(stdout)

if prefix in ["pcb_precon","pcb","pcb_precon_tsvd","pcb_tsvd"]
    # LCSVD
    useprecond = prefix ∈ ["pcb_precon","pcb_precon_tsvd"] ? true : false
    usedenoiseUVt = false
    uselv=true; maxiter = pcb_maxiter
    r=0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
        # if this is too big iteration number would be increased
#    try
    for (tailstr,α,β) in [("_sp_nn",0.005,5.0), ("_sp",0.005,0.)]# ,("_nn",0.,5.0)
        initmethod = tailstr == "_nn" ? :nndsvd : prefix ∈ ["pcb_tsvd","pcb_precon_tsvd"] ? :tsvd : :isvd
        dd = Dict()
        β1 = β2= β; α1 = α2 = α
        β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
        α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
        avgfits=Float64[]; rt2s=Float64[]; inner_fxs=Float64[]

        rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
        V = copy(H0'); N0t = copy(N0')
        inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseUVt=usedenoiseUVt, maskW=Colon(), maskH = Colon(),
            denoisefilter=:avg, uselv=uselv, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = false,
            store_inner_trace = false, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

        @show tailstr, rt2
        W1, H1 = rst0.W, rst0.Ht'
        powers = norm.(eachcol(W1))
        LCSVD.normalizeW!(W1,H1);
        makepositive && LCSVD.flip2makepos!(W1,H1)
        if dataset in [:fakecells, :naomi]
            if dataset == :naomi
                H1_nobase = subtract_baseline(H1; q=0.01)
            else
                H1_nobase = H1
            end
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH_nobase, W1, H1_nobase; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            W1, H1, H1_nobase = W1[:,nodr], H1[nodr,:], H1_nobase[nodr,:]
        else
            fv = LCSVD.fitd(X,W1*H1)
        end
        fprex = "$(prefix)$(noisestr)$(factor)f$(noc)s$(initmethod)"
        uselvstr = alg.uselv ? "_lv" : ""
        # precondstr = useprecond ? "_precond" : ""
        # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
        fname = joinpath(subworkpath,"$(prefix)","$(fprex)_a$(α)_b$(β)_expr$(iter)_f$(fv)$(uselvstr)_it$(rst0.niters)_rt$(rt2)")
        issaveimg && imsave_data(dataset,fname,W1[:,1:numWimg],H1_nobase[1:numWimg,:],imgsz,100; gridcols=gridcols, saveH=false, verbose=false)
        issaveimg && plot_H_gt_n(fname, H1_nobase[1:numWimg,:], dt; figsize=figsize, verbose=false)
        # plotH_data(fname,H1; space=hplot_space,ylabel="",ytickformat="{:.2f}")

        dd["rt1"] = rt1; dd["rt2"] = rt2; dd["M"] = M1; dd["N"] = N1t'
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
end

if prefix == "hals"
    # HALS
    rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, noc, variant=:ar)
    maxiter = hals_maxiter
    for (tailstr,initmethod,α) in [("_nn",:nndsvd,0.),("_sp_nn",:nndsvd,hals_α)]#
        dd = Dict()
        W1, H1 = copy(Wcd0), copy(Hcd0);
        rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                        tol=tol, verbose=true), X, W1, H1)

        @show tailstr, rt2
        LCSVD.normalizeW!(W1,H1)#; W1,H1 = LCSVD.sortWHslices(W1,H1)
        if dataset in [:fakecells, :naomi]
            if dataset == :naomi
               H1_nobase = subtract_baseline(H1; q=0.01)
            else
                H1_nobase = H1
            end
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH_nobase, W1, H1_nobase; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            W1, H1, H1_nobase = W1[:,nodr], H1[nodr,:], H1_nobase[nodr,:]
        else
            fv = LCSVD.fitd(X,W1*H1)
        end
        fprex = "$(prefix)$(noisestr)$(factor)f$(noc)s$(initmethod)"
        fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_expr$(iter)_f$(fv)_it$(rst0.niters)_rt$(rt2)")
        issaveimg && imsave_data(dataset,fname,W1[:,1:numWimg],H1_nobase[1:numWimg,:],imgsz,100; gridcols=gridcols, saveH=false, verbose=false)
        issaveimg && plot_H_gt_n(fname, H1_nobase[1:numWimg,:], dt; figsize=figsize, verbose=false)
        # issaveimg && plotH_data(fname,H1; space=hplot_space,ylabel="",ytickformat="{:.2f}")
        # TestData.imsave_data_gt(dataset,fname*"_gt", W1,H1,gtW,gtH,imgsz,100; saveH=false, verbose=false)

        dd["rt1"] = rt1cd; dd["rt2"] = rt2; dd["W"] = W1; dd["H"] = H1
        if true#iter == num_experiments
            metadata = Dict()
            metadata["maxiter"] = maxiter; metadata["alpha"] = α
        end
        save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end

if prefix == "compnmf"
    # COMPNMF
    maxiter = compnmf_maxiter
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
        rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, W1, H1)
        rt1 += rst0.inittime # add calculation time for compression matrices L and R
        rt2 -= rst0.inittime

        @show tailstr, rt2
        LCSVD.normalizeW!(W1,H1)#; W1,H1 = LCSVD.sortWHslices(W1,H1)
        if dataset in [:fakecells, :naomi]
            if dataset == :naomi
               H1_nobase = subtract_baseline(H1; q=0.01)
            else
                H1_nobase = H1
            end
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH_nobase, W1, H1_nobase; clamp=false)
            nodr = LCSVD.matchedorder(ml,noc)
            W1, H1, H1_nobase = W1[:,nodr], H1[nodr,:], H1_nobase[nodr,:]
        else
            fv = LCSVD.fitd(X,W1*H1)
        end
        fprex = "$(prefix)$(noisestr)$(factor)f$(noc)s$(initmethod)"
        fname = joinpath(subworkpath,prefix,"$(fprex)_expr$(iter)_f$(fv)_it$(rst0.niters)_rt$(rt2)")
        issaveimg && imsave_data(dataset,fname,W1[:,1:numWimg],H1_nobase[1:numWimg,:],imgsz,100; gridcols=gridcols, saveH=false, verbose=false)
        issaveimg && plot_H_gt_n(fname, H1_nobase[1:numWimg,:], dt; figsize=figsize, verbose=false)
        # issaveimg && plotH_data(fname, H1; space=hplot_space,ylabel="",ytickformat="{:.2f}")
        # TestData.imsave_data_gt(dataset,fname*"_gt", W1,H1,gtW,gtH,imgsz,100; saveH=false, verbose=false)

        dd["rt1"] = rt1; dd["rt2"] = rt2; dd["W"] = W1; dd["H"] = H1
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

