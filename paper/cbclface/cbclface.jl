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
using Printf

subworkpath = joinpath(workpath,"paper","cbclface")
function loadface(n)
    dirpath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
    fname = "face"*@sprintf("%05d",n)*".pgm"
    load(joinpath(dirpath,fname))
end

dataset = :cbclface; SNR=0; inhibitindices=0; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

if true
    num_experiments = 10
    lcsvd_maxiter = 40; lcsvd_inner_maxiter = 50; lcsvd_ls_maxiter = 100
    compnmf_maxiter = 500; compnmf_inner_maxiter = 0; compnmf_ls_maxiter = 0
    hals_maxiter = 400
# sca_maxiter = 400; sca_inner_maxiter = 50; sca_ls_maxiter = 100
# admm_maxiter = 1500; admm_inner_maxiter = 0; admm_ls_maxiter = 0
# hals_maxiter = 200; tol = -1
else
    num_experiments = 3
    lcsvd_maxiter = 2; lcsvd_inner_maxiter = 2; lcsvd_ls_maxiter = 2
    compnmf_maxiter = 2; compnmf_inner_maxiter = 0; compnmf_ls_maxiter = 0
    hals_maxiter = 2
end
tol = -1

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
cls = distinguishable_colors(ncells; lchoices=range(0, stop=50, length=15))

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
maskth = 0.25
maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;

subtract_bg=false; sbgstr = subtract_bg ? "sbg" : "nosbg"

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    series(Hcd); save(joinpath(subworkpath,"Hr1.png"),current_figure())
    series(gtH[:,inhibitindices]'); save(joinpath(subworkpath,"Hr1_gtH.png"),current_figure())
    # bg = fit_background(X);
    # normalizeW!(bg.S,bg.T); imsave_data(dataset,"Wr1",reshape(bg.S,length(bg.S),1),bg.T,imgsz,100; signedcolors=dgwm(), saveH=false)
    # series(bg.T'); save("Hr2.png",current_figure())
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# LCSVD
prefix = "lcsvd_precon"; @show prefix
αrng = range(0.001,0.01,num_experiments); fvs = Float64[]; spws = Float64[]; msess=[]
for iter in 1:num_experiments
    @show iter; flush(stdout)

    useprecond = prefix ∈ ["lcsvd_precon","lcsvd_precon_LPF"] ? true : false
    uselv=false; s=10; maxiter = lcsvd_maxiter
    r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
        # if this is too big iteration number would be increased
    α1=α2=α=0.01    #αrng[iter]; 
    β1=β2=β=0
    (tailstr,initmethod) = ("_sp",:isvd)

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
    Wlc, Hlc = rst0.W, rst0.H
    fitval = LCSVD.fitd(X,Wlc*Hlc); push!(fvs,fitval); push!(spws,norm(Wlc,1))
    LCSVD.normalizeW!(Wlc,Hlc)#; Wlc,Hlc = LCSVD.sortWHslices(Wlc,Hlc)
    fprex = "$(prefix)"
    # precondstr = useprecond ? "_precond" : ""
    # useLPFstr = usedenoiseW0H0 ? "_$(alg.denoisefilter)" : ""
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_b$(β)_fv$(fitval)_it$(rst0.niters)_rt$(rt2)")
    if false
        imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false)
        imsave_reconstruct(fname,X,Wlc,Hlc,imgsz; index=100, orgimg=loadface(100), gridcols=7, clamp_level=1.0)
        imsave_reconstruct(fname,X,Wlc,Hlc,imgsz; index=53, orgimg=loadface(53), gridcols=7, clamp_level=1.0)
        imsave_reconstruct(fname,X,Wlc,Hlc,imgsz; index=199,  orgimg=loadface(199), gridcols=7, clamp_level=1.0)
    end
    mses = Float64[]
    for index = 1:2429
        orgimg=loadface(index)
        reconimg = Wlc*Hlc[:,index]
        mse = norm(vec(orgimg)-reconimg)^2/length(reconimg)
        push!(mses,mse)
    end
    push!(msess,mses)
end
msesslc = hcat(msess...); msemeanslc = dropdims(mean(msesslc[1:500,:],dims=2),dims=2)
msemxslc = dropdims(maximum(msesslc,dims=2),dims=2); msemnslc = dropdims(minimum(msesslc,dims=2),dims=2)
save(joinpath(subworkpath,"mselc.jld2"),"msesslc",msesslc,"msemeanslc",msemeanslc,"msemxslc",msemxslc,"msemnslc",msemnslc)

# CompNMF
prefix="compnmf"; @show prefix
maxiter = compnmf_maxiter
@show iter
rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
for iter in 1:num_experiments
    # L, R, X_tilde, Y_tilde, A_tilde = CompNMF.compmat(X, Wcn0, Hcn0; w=4)
    # X_tilde, Y_tilde = CompNMF.compressive_nmf(A_tilde, L, R, ncells; max_iter=1000, ls=0)
    # Wcn = L*X_tilde; Hcn = Y_tilde*R
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn;
                        gtU=gtW, gtV=gtH, maskU=:, maskV=:)
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
    rt1 += rst0.inittime # add calculation time for compression matrices L and R
    rt2 -= rst0.inittime
    fitval = LCSVD.fitd(X,Wcn*Hcn)
    LCSVD.normalizeW!(Wcn,Hcn)
    fprex = "$(prefix)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_fv$(fitval)_it$(maxiter)_rt$(rt2)")
    imsave_data(dataset,fname,Wcn,Hcn,imgsz,100; saveH=false)
    imsave_reconstruct(fname,X,Wcn,Hcn,imgsz; index=100,  orgimg=loadface(100), gridcols=7, clamp_level=1.0)
    imsave_reconstruct(fname,X,Wcn,Hcn,imgsz; index=101,  orgimg=loadface(101), gridcols=7, clamp_level=1.0)
    imsave_reconstruct(fname,X,Wcn,Hcn,imgsz; index=199,  orgimg=loadface(199), gridcols=7, clamp_level=1.0)
    # series([gtH[:,inhibitindices],Hadmm[inhibitindices,:]]; color=cls); save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())
end

# HALS
prefix="hals"; @show prefix
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1 = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
maxiter = hals_maxiter
hals_αrng = range(0,0.5,num_experiments); msess=[]
for iter in 1:num_experiments
    @show iter; flush(stdout)

    α = hals_αrng[iter]
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                tol=tol, verbose=false), X, Wcd, Hcd)
    fitval = LCSVD.fitd(X,Wcd*Hcd)
    LCSVD.normalizeW!(Wcd,Hcd)
    fprex = "$(prefix)"
    fname = joinpath(subworkpath,prefix,"$(fprex)_a$(α)_fv$(fitval)_it$(maxiter)_rt$(rt2)")
    if false
        imsave_data(dataset,fname,Wcd,Hcd,imgsz,100; saveH=false)
        imsave_reconstruct(fname,X,Wcd,Hcd,imgsz; index=100,  orgimg=loadface(100), gridcols=7, clamp_level=1.0)
        imsave_reconstruct(fname,X,Wcd,Hcd,imgsz; index=53,  orgimg=loadface(53), gridcols=7, clamp_level=1.0)
        imsave_reconstruct(fname,X,Wcd,Hcd,imgsz; index=199,  orgimg=loadface(199), gridcols=7, clamp_level=1.0)
    end
    mses = Float64[]
    for index = 1:2429
        orgimg=loadface(index)
        reconimg = Wcd*Hcd[:,index]
        mse = norm(vec(orgimg)-reconimg)^2/length(reconimg)
        push!(mses,mse)
    end
    push!(msess,mses)
end
msesshals = hcat(msess...); msemeanshals = dropdims(mean(msesshals[1:500,:],dims=2),dims=2)
msemxshals = dropdims(maximum(msesshals,dims=2),dims=2); msemnshals = dropdims(minimum(msesshals,dims=2),dims=2)
save(joinpath(subworkpath,"msehals.jld2"),"msesshals",msesshals,"msemeanshals",msemeanshals,"msemxshals",msemxshals,"msemnshals",msemnshals)


fontsize = 20
f = Figure(resolution=(800,400)); lbls = ["LCSVD", "HALS NMF"]; plts = []
msemeanslc = load(joinpath(subworkpath,"mselc.jld2"),"msemeanslc")
msemeanshals = load(joinpath(subworkpath,"msehals.jld2"),"msemeanshals")
ax=AMakie.Axis(f[1,1], ylabel="MSE", ylabelsize=fontsize, yticklabelsize=fontsize,
        xlabel="face number", xlabelsize=fontsize, xticklabelsize=fontsize)
push!(plts,plot!(ax,1:200,msemeanslc[1:200], color=mtdcolors[2], label=lbls[1])) # 3 HALS
#rangebars!(ax,1:500,msemnslc[1:500],msemxslc[1:500], color=mtdcolors[2]) # 3 HALS
push!(plts,plot!(ax,1:200,msemeanshals[1:200], color=mtdcolors[3], label=lbls[2])) # 3 HALS
#rangebars!(ax,1:500,msemnshals[1:500],msemxshals[1:500], color=mtdcolors[3]) # 3 HALS
#axislegend(ax, labelsize=25, position = :lt) # halign = :left, valign = :top
Legend(f[1,2], plts, lbls, #= ["Methods"],=# labelsize=fontsize)
save(joinpath(subworkpath,"face1to200mses.png"),f,px_per_unit=2)

imgfit = load(joinpath(subworkpath,"cbclface_alpha_fits.png")) # this from plot.jl
imgmse = load(joinpath(subworkpath,"face1to200mses.png"))
f = Figure(resolution = (1800,500))
ax11=AMakie.Axis(f[1,1], titlesize=30, aspect = DataAspect())
hidedecorations!(ax11); hidespines!(ax11); image!(ax11, rotr90(imgfit));
ax21=AMakie.Axis(f[1,2], titlesize=30, aspect = DataAspect())
hidedecorations!(ax21); hidespines!(ax21); image!(ax21, rotr90(imgmse));
save(joinpath(subworkpath,"fit_mse.png"),f,px_per_unit=2)


# Figure
mtdcolors = [RGBA{N0f8}(0.00,0.00,0.00,1.0),RGBA{N0f8}(0.00,0.45,0.70,1.0),RGBA{N0f8}(0.90,0.62,0.00,1.0),
             RGBA{N0f8}(0.00,0.62,0.45,1.0),RGBA{N0f8}(0.80,0.47,0.65,1.0),RGBA{N0f8}(0.34,0.71,0.91,1.0),
             RGBA{N0f8}(0.84,0.37,0.00,1.0),RGBA{N0f8}(0.94,0.89,0.26,1.0)]
dtcolors = distinguishable_colors(ncells; lchoices=range(0, stop=50, length=15))
# Input data
# gridcols=Int(ceil(sqrt(size(Wscas[1],2))))
# imgsca1 = mkimgW(Wscas[1],imgsz,gridcols=gridcols); imgsca2 = mkimgW(Wscas[end],imgsz,gridcols=gridcols)
# imgadmm1 = mkimgW(Wadmms[1],imgsz,gridcols=gridcols); imgadmm2 = mkimgW(Wadmms[end],imgsz,gridcols=gridcols)
# imghals1 = mkimgW(Whalss[1],imgsz,gridcols=gridcols); imghals2 = mkimgW(Whalss[end],imgsz,gridcols=gridcols)
imglc = load(joinpath(subworkpath,"lcsvd_precon_a0.005_b0_fv0.9985_it40_rt2.03_W.png"))
imghals = load(joinpath(subworkpath,"hals_a0.1_fv0.9982_it400_rt4.03_W.png"))
imgcn = load(joinpath(subworkpath,"compnmf_fv0.9881_it50_rt0.21_W.png"))

orgimg100 = loadface(100)
#orgimg101 = loadface(101)
orgimg199 = loadface(199)

facelc100_1 = load(joinpath(subworkpath,"lcsvd_precon_a0.005_b0_fv0.9985_it40_rt2.03_recon100_mse0.00150.png"))
facelc100_2 = load(joinpath(subworkpath,"lcsvd_precon_a0.01_b0_fv0.9982_it40_rt3.10_recon100_mse0.001682.png"))
facecn100_1 = load(joinpath(subworkpath,"compnmf_fv0.9881_it50_rt0.21_recon100_mse0.00392.png"))
facecn100_2 = load(joinpath(subworkpath,"compnmf_fv0.9818_it500_rt2.88_recon100_mse0.01158.png"))
facehals100_1 = load(joinpath(subworkpath,"hals_a0_fv0.9983_it400_rt3.04_recon100_mse0.00175.png"))
facehals100_2 = load(joinpath(subworkpath,"hals_a0.1_fv0.9982_it400_rt4.03_recon100_mse0.00184.png"))

# facelc101_1 = load(joinpath(subworkpath,"lcsvd_precon_a0.005_b0_fv0.9985_it40_rt2.03_recon101_mse0.00160.png"))
# facelc101_2 = load(joinpath(subworkpath,"lcsvd_precon_a0.01_b0_fv0.9982_it40_rt3.10_recon101_mse0.00183.png"))
# facecn101_1 = load(joinpath(subworkpath,"compnmf_fv0.9881_it50_rt0.21_recon101_mse0.00534.png"))
# facecn101_2 = load(joinpath(subworkpath,"compnmf_fv0.9818_it500_rt2.88_recon101_mse0.01370.png"))
# facehals101_1 = load(joinpath(subworkpath,"hals_a0.1_fv0.9982_it400_rt4.03_recon101_mse0.00196.png"))
# facehals101_2 = load(joinpath(subworkpath,"hals_a0.5_fv0.9967_it400_rt4.57_recon101_mse0.00289.png"))

facelc199_1 = load(joinpath(subworkpath,"lcsvd_precon_a0.005_b0_fv0.9985_it40_rt6.44_recon199_mse0.00481.png"))
facelc199_2 = load(joinpath(subworkpath,"lcsvd_precon_a0.01_b0_fv0.9981_it40_rt2.32_recon199_mse0.00618.png"))
facecn199_1 = load(joinpath(subworkpath,"compnmf_fv0.9859_it50_rt0.25_recon199_mse0.03361.png"))
facecn199_2 = load(joinpath(subworkpath,"compnmf_fv0.9826_it500_rt2.97_recon199_mse0.04554.png"))
facehals199_1 = load(joinpath(subworkpath,"hals_a0_fv0.9983_it400_rt3.78_recon199_mse0.00580.png"))
facehals199_2 = load(joinpath(subworkpath,"hals_a0.1_fv0.9982_it400_rt3.07_recon199_mse0.00670.png"))

labels = ["LCSVD","Compressed NMF","HALS NMF"]
f = Figure(resolution = (1500,900))

ax123=AMakie.Axis(f[1:2,2:3],title=labels[1], titlesize=30, aspect = DataAspect())
hidedecorations!(ax123); image!(ax123, rotr90(imglc));
ax145=AMakie.Axis(f[1:2,4:5],title=labels[2], titlesize=30, aspect = DataAspect())
hidedecorations!(ax145); image!(ax145, rotr90(imgcn));
ax167=AMakie.Axis(f[1:2,6:7],title=labels[3], titlesize=30, aspect = DataAspect())
hidedecorations!(ax167); image!(ax167, rotr90(imghals))

ax31=AMakie.Axis(f[3,1], title="100th image", titlesize=25, aspect = DataAspect())
hidedecorations!(ax31); image!(ax31, rotr90(orgimg100));
ax32=AMakie.Axis(f[3,2], aspect = DataAspect(), titlesize=25, title="""α=0.005, β=0
MSE=0.0015"""); hidedecorations!(ax32); image!(ax32, rotr90(facelc100_1));
ax33=AMakie.Axis(f[3,3], aspect = DataAspect(), titlesize=25, title="""α=0.01, β=0
MSE=0.0017"""); hidedecorations!(ax33); image!(ax33, rotr90(facelc100_2));
ax34=AMakie.Axis(f[3,4], aspect = DataAspect(), titlesize=25, title="""iteration=50
MSE=0.0039"""); hidedecorations!(ax34); image!(ax34, rotr90(facecn100_1));
ax35=AMakie.Axis(f[3,5], aspect = DataAspect(), titlesize=25, title="""iteration=500
MSE=0.0116"""); hidedecorations!(ax35); image!(ax35, rotr90(facecn100_2));
ax36=AMakie.Axis(f[3,6], aspect = DataAspect(), titlesize=25, title="""α=0
MSE=0.0018"""); hidedecorations!(ax36); image!(ax36, rotr90(facehals100_1));
ax37=AMakie.Axis(f[3,7], aspect = DataAspect(), titlesize=25, title="""α=0.1
MSE=0.0018"""); hidedecorations!(ax37); image!(ax37, rotr90(facehals100_2));

ax41=AMakie.Axis(f[4,1], title="199th image", titlesize=25, aspect = DataAspect())
hidedecorations!(ax41); image!(ax41, rotr90(orgimg199));
ax42=AMakie.Axis(f[4,2], aspect = DataAspect(), titlesize=25, title="""α=0.005, β=0
MSE=0.0048"""); hidedecorations!(ax42); image!(ax42, rotr90(facelc199_1));
ax43=AMakie.Axis(f[4,3], aspect = DataAspect(), titlesize=25, title="""α=0.01, β=0
MSE=0.0062"""); hidedecorations!(ax43); image!(ax43, rotr90(facelc199_2));
ax44=AMakie.Axis(f[4,4], aspect = DataAspect(), titlesize=25, title="""iteration=50
MSE=0.0336"""); hidedecorations!(ax44); image!(ax44, rotr90(facecn199_1));
ax45=AMakie.Axis(f[4,5], aspect = DataAspect(), titlesize=25, title="""iteration=500
MSE=0.0455"""); hidedecorations!(ax45); image!(ax45, rotr90(facecn199_2));
ax46=AMakie.Axis(f[4,6], aspect = DataAspect(), titlesize=25, title="""α=0
MSE=0.0058"""); hidedecorations!(ax46); image!(ax46, rotr90(facehals199_1));
ax47=AMakie.Axis(f[4,7], aspect = DataAspect(), titlesize=25, title="""α=0.1
MSE=0.0067"""); hidedecorations!(ax47); image!(ax47, rotr90(facehals199_2));

save(joinpath(subworkpath,"cbclface.png"),f)

fontsize = 30
f = Figure(resolution=(1500,1400))
gt = f[1,1] = GridLayout()
gb = f[2,1] = GridLayout()

fname = joinpath(subworkpath, "cbclface_alpha_fits.png")
axi=AMakie.Axis(gt[1,1], title="(a)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axi, label=false); hidespines!(axi); image!(axi, rotr90(load(fname)))
fname = joinpath(subworkpath, "face1to200mses.png")
axj=AMakie.Axis(gt[1,2], title="(b)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axj, label=false); hidespines!(axj); image!(axj, rotr90(load(fname)));
rowsize!(gt,1,400); rowgap!(gt,0)
fname = joinpath(subworkpath, "cbclface.png")
axk=AMakie.Axis(gb[1,1], title="(c)", titlesize=fontsize, aspect = DataAspect())
hidedecorations!(axk, label=false); hidespines!(axk); image!(axk, rotr90(load(fname)))

save(joinpath(subworkpath,"cbclface_all_figures.png"),f,px_per_unit=2)
