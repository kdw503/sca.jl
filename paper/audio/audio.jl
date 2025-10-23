using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","audio")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))

dataset = :audio
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"

lcsvd_maxiter = 150
compnmf_maxiter = 1000
hals_maxiter = 150

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset);

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = LCSVD.noisefilter(filter,X,(0,0))

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

mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter; tol=-1 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
  # if this is too big iteration number would be increased

usedenoiseUVt = true; makepositive = true
denoiseW0H0str = usedenoiseUVt ? "_udnW0H0" : ""
(tailstr,initmethod,α,β) = ("_sp",:nndsvd,0.005,0.0) # ("_sp_nn",:isvd,0.005,5.0),("_nn",:nndsvd,0.,5.0)

β1=β; β2=β; α1 = α2 = α
rt1 = @elapsed U, Vt, M0, N0, Wp, Hp, D  = LCSVD.initpcb(X, ncells, 0; initmethod=initmethod, svdmethod=:isvd)
V = copy(Vt'); N0t = copy(N0')
r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
  # if this is too big iteration number would be increased
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseUVt=usedenoiseUVt,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, store_trace = false,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
M, Nt = copy(M0), copy(N0t)
rt0 = @elapsed rst1 = LCSVD.solve!(alg, X, U, V, D, M, Nt);
M, Nt = copy(M0), copy(N0t)
rt2 = @elapsed LCSVD.solve!(alg, X, U, V, D, M, Nt);
Wlc, Hlc = rst1.W, rst1.Ht'
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
LCSVD.normalizeW!(Wlc,Hlc); fitval = LCSVD.fitd(X,Wlc*Hlc)
Wlc .*= 10; Hlc ./=10
Wlc,Hlc = (nhs = map(a->norm(a[1:42]),eachrow(Hlc)); orderindices = sortperm(nhs, rev=true); (Wlc[:,orderindices],Hlc[orderindices,:]))
fname = joinpath(subworkpath,"$(prefix)_$(initmethod)$(denoiseW0H0str)_a$(α)_b$(β)_f$(fitval)_it$(rst0.niters)_rt$(rt2)")
fig = plotWH_data(dataset,fname,Wlc,Hlc; space=10, issave=true)

# method="admm"; @show method
# mfmethod = :ADMM; initmethod=:lowrank_nndsvd; penmetric = :SCA; sd_group=:whole; reg = :WH1; α = 10; β = 0; usennc=true
# useRelaxedL1=true; s=10*0.3^0; 
# r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
#       # if this is too big iteration number would be increased
# # Optimization parameters
# tol=-1; optimmethod = :sca_admm; ls_method = :ls_BackTracking; useprecond=false; uselv=false
# maxiter = admm_maxiter; inner_maxiter = admm_inner_maxiter; ls_maxiter = admm_ls_maxiter
# # Result demonstration parameters
# makepositive = true; save_figure = true; uselogscale=true; isplotxandg = false; plotnum = isplotxandg ? 3 : 1
# poweradjust = :none
# rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=initmethod,poweradjust=initpwradj)
# stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=true, α1=α1, α2=α2, β1=β1, β2=β2,
#     reg=reg, useRelaxedL1=false, σ0=σ0, r=r, poweradjust=:none, useprecond=useprecond, usennc=usennc, uselv=uselv)
#     α1=α2=α; β1=β2=β
# cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
#     x_abstol=tol, successive_f_converge=0, maxiter=admm_maxiter, inner_maxiter=inner_maxiter,
#     store_trace=true, store_inner_trace=false, show_trace=false,plotiterrng=1:0, plotinneriterrng=1:0)
# Mw, Mh = copy(Mw0), copy(Mh0);
# cparams.store_trace = false; cparams.store_inner_trace = false;
# cparams.show_trace=false; cparams.show_inner_trace=false; cparams.plotiterrng=1:0
# rt2 = @elapsed  W1, H1, objvals, laps, trs, niters = scasolve!(X, W0, H0, D, Mw, Mh, Wp, Hp; gtW=gtW, gtH=gtH,
#                                                     penmetric=penmetric, stparams=stparams, cparams=cparams);
# fitval = SCA.fitd(X,W1*H1)
# makepositive && flip2makepos!(W1,H1)
# normalizeW!(W1,H1); W1 .*= 10; H1 ./=10; W3,H3 = sortWHslices(W1,H1)
# fprx = "$(mfmethod)$(dataset)_$(initmethod)_a$(α)_it$(admm_maxiter)_fv$(fitval)_rt$(rt2)"
# fig = plotWH_data(dataset,joinpath(subworkpath,fprx),W3,H3; space=10, issave=true)

prefix="compnmf"; @show prefix
maxiter = compnmf_maxiter; initmethod=:nndsvd
for iter in 1:100
    @show iter
    rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);

    # Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    # result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn;
    #                     gtU=gtW, gtV=gtH, maskU=:, maskV=:)
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
    rt1 += rst0.inittime # add calculation time for compression matrices L and R
    rt2 -= rst0.inittime
    fitval = LCSVD.fitd(X,Wcn*Hcn)
    LCSVD.normalizeW!(Wcn,Hcn)
    Wcn .*= 10; Hcn ./=10; W3,H3 = LCSVD.sortWHslices(Wcn,Hcn)
    fprex = "$(prefix)_$(initmethod)"
    fname = joinpath(subworkpath,"$(fprex)_fv$(fitval)_it$(maxiter)_rt$(rt2)")
    fig = TestData.plotWH_data(dataset,fname,W3,H3; space=10, issave=true)
end

# HALS
prefix="hals"; @show prefix
mfmethod = :HALS; 
initmethod = :rsvd
if initmethod == :rsvd
    rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, ncells, variant=:ar);
else
    rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, ncells, variant=:ar, initdata=svd(X));
end
mfmethod = :HALS; αhals=0.1; maxiter = hals_maxiter; tol=-1
Whals, Hhals = copy(Whals0), copy(Hhals0);
rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=αhals, l₁ratio=1,
            tol=tol, verbose=false), X, Whals, Hhals)
LCSVD.normalizeW!(Whals,Hhals); fitval = LCSVD.fitd(X,Whals*Hhals)
Whals .*= 10; Hhals ./=10; Whals,Hhals = LCSVD.sortWHslices(Whals,Hhals)
fname = joinpath(subworkpath,"$(prefix)_a$(αhals)_f$(fitval)_it$(maxiter)_rt$(rt2)")
fig = plotWH_data(dataset,fname,Whals,Hhals; space=10, issave=true)



# COMPNMF
prefix = "compnmf"
@show prefix; flush(stdout)
mfmethod = :COMPNMF; maxiter = compnmf_maxiter
(tailstr,initmethod) = ("_nn",:lowrank_nndsvd)
dd = Dict(); tol=-1
rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn)
Wcn, Hcn = copy(Wcn0), copy(Hcn0);
rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, Wcn, Hcn)
rt1 += rst0.inittime # add calculation time for compression matrices L and R
rt2 -= rst0.inittime
normalizeW!(Wcn,Hcn); fitval = LCSVD.fitd(X,Wcn*Hcn)
Wcn .*= 10; Hcn ./=10
Wcn,Hcn = (nhs = map(a->norm(a[1:42]),eachrow(Hcn)); orderindices = sortperm(nhs, rev=true); (Wcn[:,orderindices],Hcn[orderindices,:]))
fname = joinpath(subworkpath,"$(prefix)_f$(fitval)_it$(maxiter)_rt$(rt2)")
fig = plotWH_data(dataset,fname,Wcn,Hcn; space=10, issave=true)


# Figure
imglcsvd1 = load(joinpath(subworkpath,"lcsvd_isvd_udnW0H0_a0.005_b5.0_f0.9920515228911607_it150_rt0.05575102_plot_WH.png"))
#imglcsvd2 = load(joinpath(subworkpath,"SCAaudio_nndsvd_a100_b0_it100_fv0.992302949908781_rt0.0268503_plot_WH.png"))
imgcn1 = load(joinpath(subworkpath,"compnmf_f0.988533712367913_it1000_rt0.027901287925659178_plot_WH.png"))
#imgcn2 = load(joinpath(subworkpath,"ADMMaudio_lowrank_nndsvd_a10_it1500_fv0.9882295945463136_rt0.0619883_plot_WH.png"))
imghals = load(joinpath(subworkpath,"hals_a0.1_f0.9921729737463991_it150_rt0.021844562_plot_WH.png"))

f = Figure(resolution = (1000,1600))
ax11=AMakie.Axis(f[1,1],title="(a) LCSVD", titlesize=20, aspect = DataAspect()); hidespines!(ax11)
hidedecorations!(ax11)
# ax12=AMakie.Axis(f[1,2],title="(b) LCSVD (NNDSVD init.)", titlesize=20, aspect = DataAspect()); hidespines!(ax12)
# hidedecorations!(ax12)
ax21=AMakie.Axis(f[2,1],title="(b) Compressed NMF", titlesize=20, aspect = DataAspect()); hidespines!(ax21)
hidedecorations!(ax21)
# ax22=AMakie.Axis(f[2,2],title="(e) Compressed NMF", titlesize=20, aspect = DataAspect()); hidespines!(ax22)
# hidedecorations!(ax22)
ax31=AMakie.Axis(f[3,1],title="(c) HALS", titlesize=20, aspect = DataAspect()); hidespines!(ax31)
hidedecorations!(ax31)
image!(ax11, rotr90(imglcsvd1))#; image!(ax12, rotr90(imglcsvd2));
image!(ax21, rotr90(imgcn1))#; image!(ax22, rotr90(imgcn2))
image!(ax31, rotr90(imghals))
save(joinpath(subworkpath,"audio.png"),f)
