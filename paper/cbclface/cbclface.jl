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

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))

dataset = :cbclface
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"

lcsvd_maxiter = 400
compnmf_maxiter = 1500
hals_maxiter = 200

X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset)

(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
X = LCSVD.noisefilter(filter,X)

subtract_bg=false; sbgstr = subtract_bg ? "sbg" : "nosbg"

if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    LCSVD.normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
    plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
    bg = W*fill(mean(H),1,n); X .-= bg
end

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)

mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter; tol=-1 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
  # if this is too big iteration number would be increased

usedenoiseW0H0 = false; makepositive = true
denoiseW0H0str = usedenoiseW0H0 ? "_udnW0H0" : ""
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,.0) # ("_sp_nn",:isvd,0.005,5.0),("_nn",:nndsvd,0.,5.0)

β1=β; β2=β; α1 = α2 = α
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
  # if this is too big iteration number would be increased
Wlcs = []; rtlcs=[]; lcsvdmaxiterrng = 4:2:20
for lcsvdmaxiter in lcsvdmaxiterrng
# lcsvdmaxiter=20; for α in 0.001:0.001:0.01
    for α in 0.006:0.001:0.014
     α1 = α2 = 0.005
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=true, usedenoiseW0H0=usedenoiseW0H0,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = lcsvdmaxiter, store_trace = false,
        store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    M, N = copy(M0), copy(N0)
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
    alg.α1=alg.α2=α
    rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
    Wlc, Hlc = rst0.W, rst0.H
    sparsity = norm(Wlc,1)
    # avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
    LCSVD.normalizeW!(Wlc,Hlc); fitval = LCSVD.fitd(X,Wlc*Hlc)
    LCSVD.flip2makepos!(Wlc,Hlc)
    fname = joinpath(subworkpath,"$(prefix)_$(initmethod)$(denoiseW0H0str)_a$(α)_b$(β)_f$(fitval)_sp$(sparsity)_it$(rst0.niters)_rt$(rt2)")
    imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false)
    end
    # imsave_reconstruct(fname,X,Wlc,Hlc,imgsz; index=100, gridcols=7, clamp_level=1.0)
    push!(Wlcs,Wlc); push!(rtlcs,rt2)
end
msesslc = hcat(msess...); msemeanslc = dropdims(mean(msesslc[1:500,:],dims=2),dims=2)
msemxslc = dropdims(maximum(msesslc,dims=2),dims=2); msemnslc = dropdims(minimum(msesslc,dims=2),dims=2)
save(joinpath(subworkpath,"mselc.jld2"),"msesslc",msesslc,"msemeanslc",msemeanslc,"msemxslc",msemxslc,"msemnslc",msemnslc)

# COMPNMF
prefix = "compnmf"
@show prefix; flush(stdout)
mfmethod = :COMPNMF; maxiter = compnmf_maxiter
(tailstr,initmethod) = ("_nn",:lowrank_nndsvd)
dd = Dict(); tol=-1
rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
Wcns=[]; rtcns=[]; cnmaxiterrng = 50:50:500
for cnmaxiter = cnmaxiterrng
    Wcn, Hcn = copy(Wcn0), copy(Hcn0);
    rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=cnmaxiter, tol=tol, verbose=false), X, Wcn, Hcn)
    rt1 += rst0.inittime # add calculation time for compression matrices L and R
    rt2 -= rst0.inittime
    LCSVD.normalizeW!(Wcn,Hcn); fitval = LCSVD.fitd(X,Wcn*Hcn)
    fname = joinpath(subworkpath,"$(prefix)_f$(fitval)_it$(cnmaxiter)_rt$(rt2)")

    # imsave_data(dataset,fname,Wcn,Hcn,imgsz,100; saveH=false)
    # imsave_reconstruct(fname,X,Wcn,Hcn,imgsz; index=100, gridcols=7, clamp_level=1.0)
    push!(Wcns,Wcn); push!(rtcns,rt2)
    # series([gtH[:,inhibitindices],Hcn[inhibitindices,:]]; color=cls); save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())
end

# HALS
prefix="hals"; @show prefix
# W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; αhals=0.1; maxiter = hals_maxiter; tol=-1 # αhals=0.6 
Whalss=[]; rthalss=[]; halsmaxiterrng = 20:20:100
for halsmaxiter = halsmaxiterrng
    Whals, Hhals = copy(Whals0), copy(Hhals0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=halsmaxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, Whals, Hhals)
    LCSVD.normalizeW!(Whals,Hhals); fitval = LCSVD.fitd(X,Whals*Hhals)
    fname = joinpath(subworkpath,"$(prefix)_a$(αhals)_f$(fitval)_it$(halsmaxiter)_rt$(rt2)")

    # imsave_data(dataset,fname,Whals,Hhals,imgsz,100; saveH=false)
    # imsave_reconstruct(fname,X,Whals,Hhals,imgsz; index=100, gridcols=7, clamp_level=1.0)
    push!(Whalss,Whals); push!(rthalss,rt2)
    # series([gtH[:,inhibitindices],Hhals[inhibitindices,:]]; color=cls); save(joinpath(subworkpath,"$(fprx)_H.png"),current_figure())
end

# Figure

# Input data
factor = 2; titlefontsize=30*factor; subtitlefontsize=25*factor
gridcols=Int(ceil(sqrt(size(Wlcs[1],2))))
imglc1 = mkimgW(Wlcs[1],imgsz,gridcols=gridcols); imglc2 = mkimgW(Wlcs[end],imgsz,gridcols=gridcols)
imgcn1 = mkimgW(Wcns[1],imgsz,gridcols=gridcols); imgcn2 = mkimgW(Wcns[end],imgsz,gridcols=gridcols)
imghals1 = mkimgW(Whalss[1],imgsz,gridcols=gridcols); imghals2 = mkimgW(Whalss[end],imgsz,gridcols=gridcols)
labels = ["LCSVD","Compressed NMF","HALS NMF"]
f = Figure(resolution = (900*factor,1500*factor))
ax11=AMakie.Axis(f[1,1],title=labels[1], subtitle="maxiter=$(lcsvdmaxiterrng[1]), runtime=$(round(rtlcs[1],digits=2))sec",
        titlesize=titlefontsize, subtitlesize=subtitlefontsize, aspect = DataAspect())
hidedecorations!(ax11)
ax12=AMakie.Axis(f[1,2],title=labels[1], subtitle="maxiter=$(lcsvdmaxiterrng[end]), runtime=$(round(rtlcs[end],digits=2))sec",
        titlesize=titlefontsize, subtitlesize=subtitlefontsize, aspect = DataAspect())
hidedecorations!(ax12)
ax21=AMakie.Axis(f[2,1],title=labels[2], subtitle="maxiter=$(cnmaxiterrng[1]), runtime=$(round(rtcns[1],digits=2))sec",
        titlesize=titlefontsize, subtitlesize=subtitlefontsize, aspect = DataAspect())
hidedecorations!(ax21)
ax22=AMakie.Axis(f[2,2],title=labels[2], subtitle="maxiter=$(cnmaxiterrng[end]), runtime=$(round(rtcns[end],digits=2))sec",
        titlesize=titlefontsize, subtitlesize=subtitlefontsize, aspect = DataAspect())
hidedecorations!(ax22)
ax31=AMakie.Axis(f[3,1],title=labels[3], subtitle="maxiter=$(halsmaxiterrng[1]), runtime=$(round(rthalss[1],digits=2))sec",
        titlesize=titlefontsize, subtitlesize=subtitlefontsize, aspect = DataAspect())
hidedecorations!(ax31)
ax32=AMakie.Axis(f[3,2],title=labels[3], subtitle="maxiter=$(halsmaxiterrng[end]), runtime=$(round(rthalss[end],digits=2))sec",
        titlesize=titlefontsize, subtitlesize=subtitlefontsize, aspect = DataAspect())
hidedecorations!(ax32)
image!(ax11, rotr90(imglc1)); image!(ax12, rotr90(imglc2))
image!(ax21, rotr90(imgcn1)); image!(ax22, rotr90(imgcn2))
image!(ax31, rotr90(imghals1)); image!(ax32, rotr90(imghals2))
save(joinpath(subworkpath,"cbclface.png"),f)

#=== combine all ===#
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
