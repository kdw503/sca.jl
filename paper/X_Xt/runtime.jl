using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","X_Xt")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.5
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

lcsvd_maxiter = 150
compnmf_maxiter = 1000
hals_maxiter = 150

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncells, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
ncells = noc = 15 # ncs
(m,n,p) = (size(X)...,ncells)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
gtfname = "fakecells$(inhibitindices)_calcium_sz$(imgsz)_lengthT$(lengthT)_SNR$(SNR)_bias$(bias)"
imsave_data(dataset,joinpath(subworkpath,gtfname),gtW,gtH',imgsz,100; saveH=false)
plotH_data(joinpath(subworkpath,gtfname),gtH'; space=0.,ylabel="",ytickformat="{:.2f}")
X = LCSVD.noisefilter(filter,X,size)

subtract_bg = false
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    LCSVD.normalizeW!(W,H)
    bg = W*fill(mean(H),1,n)
#        bg = W*H
    X .-= bg
end

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)

mfmethod = :LCSVD; useprecond=false; uselv=false
r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
usedenoiseW0H0 = false; makepositive = true

r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, ncells, 0; initmethod=initmethod, svdmethod=:svd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)

maxiter = 10 #Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
tol = -1e-6
inner_tol = -1e-6; inner_maxiter = 100 #Int(ceil(2.5*ncells+350))# Int(ceil(0.75*ncells+100)) # 

β1 = β2= β; α1 = α2 = α
β1vec = fill(β1,noc); β2vec = fill(β2,noc);# β1vec[1:2] .= 0.; β2vec[1:2] .= 0.
α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1:1] .= 0.; α2vec[1:1] .= 0.

# PCB(ISVD(X))
σ0=std(W0*M0) # std(N0*H0t) #=10*std(W0)=#
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, store_sparsity_nneg = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M1, N1t = copy(M0), copy(N0t)
rst1 = LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);
alg.store_trace = false; alg.store_inner_trace = false
M1, N1t = copy(M0), copy(N0t)
rt = @elapsed LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);
W1, H1 = copy(rst1.W), copy(rst1.Ht')
LCSVD.normalizeW!(W1,H1);
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
nodr = LCSVD.matchedorder(ml,ncells); Wlc1, Hlc1 = W1, H1 # W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
#LCSVD.flip2makepos!(Wlc1,Hlc1)
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_a$(α)_b$(β)"
fname = joinpath(subworkpath,"fakecells1$(regstr)_af$(avgfit)_it$(rst1.niters)_rt$(rt)")
imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)

M02 = M0.+(rand(size(M0)...).-0.5)*eps(Float64) 
N0t2 = N0t.+(rand(size(N0t)...).-0.5)*eps(Float64) 
# M02 = M0; M02[1] += eps(Float64) 
# N0t2 = N0t; N0t1 += eps(Float64) 
# σ0=std(W0*M02) # std(N0*H0t) #=10*std(W0)=#
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    # α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, store_sparsity_nneg = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M2, N2t = copy(M02), copy(N0t2)
rst2 = LCSVD.solve!(alg, X, W0, H0t, D, M2, N2t);
W2, H2 = copy(rst2.W), copy(rst2.Ht')
LCSVD.normalizeW!(W2,H2);
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W2, H2; clamp=false)
nodr = LCSVD.matchedorder(ml,ncells); Wlc2, Hlc2 = W2, H2 #W2[:,nodr], H2[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
#LCSVD.flip2makepos!(Wlc1,Hlc1)
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_a$(α)_b$(β)"
fname = joinpath(subworkpath,"fakecells2$(regstr)_af$(avgfit)_it$(rst2.niters)")
imsave_data(dataset,fname,Wlc2,Hlc2,imgsz,100; saveH=false)

# # PCB(ISVD(X)')
# # σ0= std(M0t*W0t)# std(W0t*M0t) #=10*std(W0)=#
# alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
#     denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
#     store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
#     f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
# M2, Nt2 = copy(N0t), copy(M0)
# rst2 = LCSVD.solve!(alg, X', H0t, W0, D', M2, Nt2);

# norm(M1-N2')
# norm(N1-M2')
# for i in 2:length(rst1.traces)
#     print("$(rst1.traces[i].niters), ")
#     println()
# end
# for i in 2:length(rst2.traces)
#     print("$(rst2.traces[i].niters), ")
#     println()
# end

# nc = ncells
# fdiffs = Float64[]; ngdiffs = Float64[]; xdiffs = Float64[]
# nrmWs = Float64[]; nrmHs = Float64[]; diffMs = Float64[]; diffNs = Float64[];
# for (tr1,tr2) in zip(rst1.traces[2:9], rst2.traces[2:9])
#     for (fx1, fx2) in zip(tr1.fxs,tr2.fxs)
#         push!(fdiffs, abs(fx1-fx2))
#     end
#     for (gxs1, gxs2) in zip(tr1.gxs,tr2.gxs)
#         gxs2t = [gxs2[nc*p+1:2*nc*p]...,gxs2[1:nc*p]...]
#         push!(ngdiffs, norm(gxs1-gxs2t))
#     end
#     for (x1, x2) in zip(tr1.xs,tr2.xs)
#         M1, N1t = reshape(x1[1:nc*p],p,nc)', reshape(x1[nc*p+1:2nc*p],p,nc)'
#         M2, N2t = reshape(x2[1:nc*p],p,nc)', reshape(x2[nc*p+1:2nc*p],p,nc)'
#         x2t = [x2[nc*p+1:2*nc*p]...,x2[1:nc*p]...]
#         push!(nrmWs, norm(M1-N2t)); push!(nrmHs, norm(N1t'-M2'))
#         um = M1\N2t; un = M2'/N1t'; umtumIdiff = norm(um'um-I); ununtIdiff = norm(un*un'-I)
#         push!(diffMs, umtumIdiff); push!(diffNs, ununtIdiff); push!(xdiffs, norm(x1-x2t))
#     end
# end

# f = Figure(size=(350,250))
# ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
# lines!(ax,nrmWs[3:end],label="norm(M1-N2')")
# lines!(ax,nrmHs[3:end],label="norm(N1_M2')")
# @show nrmWs[3:end]
# axislegend(ax; position = :rb)
# save(joinpath(subworkpath,"MNnormdiff_im$(inner_maxiter).png"),f)

# f = Figure(size=(350,250))
# ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
# lines!(ax,diffMs[3:end],label="M_diff")
# lines!(ax,diffNs[3:end],label="N_diff")
# axislegend(ax; position = :rb)
# save(joinpath(subworkpath,"MNdiff_im$(inner_maxiter).png"),f)

# f = Figure(size=(350,250))
# ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
# lines!(ax,xdiffs[3:end],linestyle=:dot,label="norm(x1-x2)")
# lines!(ax,ngdiffs[3:end],linestyle=:dot,label="norm(g1-g2)")
# @show ngdiffs[3:end]
# axislegend(ax; position = :rb)
# save(joinpath(subworkpath,"xgdiff_im$(inner_maxiter).png"),f)

# f = Figure(size=(350,250))
# ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
# lines!(ax,fdiffs[2:end],label="abs(fx1-fx2)")
# axislegend(ax; position = :rt)
# save(joinpath(subworkpath,"fxdiff_im$(inner_maxiter).png"),f)

# f = Figure(size=(350,250))
# ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
# lines!(ax,ngdiffs[2:end],label="norm(g1-g2)")
# axislegend(ax; position = :rt)
# save(joinpath(subworkpath,"ngdiff_im$(inner_maxiter).png"),f)


nc = ncells
fdiffs = Float64[]; fmdiffs = Float64[];
fswdiffs = Float64[]; fshdiffs = Float64[]; fnwdiffs = Float64[]; fnhdiffs = Float64[]; 
ngdiffs = Float64[]; xdiffs = Float64[]; mndiffs = Float64[]
nrmWs = Float64[]; nrmHs = Float64[]; diffMs = Float64[]; diffNs = Float64[];
for (tr1,tr2) in zip(rst1.traces[2:end], rst2.traces[2:end])
    for (fx1, fx2) in zip(tr1.fxs,tr2.fxs)
        push!(fdiffs, abs(fx1-fx2)/abs(fx1+fx2))
    end
    for (fx1, fx2) in zip(tr1.invs,tr2.invs)
        push!(fmdiffs, abs(fx1-fx2)/abs(fx1+fx2))
    end
    for (fx1, fx2) in zip(tr1.sparseWs,tr2.sparseWs)
        push!(fswdiffs, abs(fx1-fx2)/abs(fx1+fx2))
    end
    for (fx1, fx2) in zip(tr1.sparseHs,tr2.sparseHs)
        push!(fshdiffs, abs(fx1-fx2)/abs(fx1+fx2))
    end
    for (fx1, fx2) in zip(tr1.nnWs,tr2.nnWs)
        push!(fnwdiffs, abs(fx1-fx2)/abs(fx1+fx2))
    end
    for (fx1, fx2) in zip(tr1.nnHs,tr2.nnHs)
        push!(fnhdiffs, abs(fx1-fx2)/abs(fx1+fx2))
    end
    for (gxs1, gxs2) in zip(tr1.gxs,tr2.gxs)
        push!(ngdiffs, norm(gxs1-gxs2))
    end
    @show length(tr1.xs), length(tr2.xs)
    i=0
    for (x1, x2) in zip(tr1.xs,tr2.xs)
        i+=1
        M1, N1t = reshape(x1[1:nc*p],p,nc)', reshape(x1[nc*p+1:2nc*p],p,nc)'
        M2, N2t = reshape(x2[1:nc*p],p,nc)', reshape(x2[nc*p+1:2nc*p],p,nc)'
        push!(nrmWs, norm(M1-M2)); push!(nrmHs, norm(N1t-N2t))
        push!(mndiffs,norm(M1*N1t'-M2*N2t')/norm(M1*N1t'+M2*N2t'))
        um = M1\M2; un = N2t'/N1t'; umtumIdiff = norm(um'um-I); ununtIdiff = norm(un*un'-I)
        push!(diffMs, umtumIdiff); push!(diffNs, ununtIdiff); push!(xdiffs, norm(x1-x2))
    end
    @show i
end


f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,mndiffs[3:end],label="norm(M₁N₁-M₂N₂)/norm(M₁N₁+M₂N₂)")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"Dnormdiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,nrmWs[3:end],label="norm(M1-M2)")
lines!(ax,nrmHs[3:end],label="norm(N1_N2)")
text!( (Float64(length(nrmWs)),nrmWs[end]),
    text = "$(round(nrmWs[end],sigdigits=2))",
    rotation = 0,
    align = (:right, :top),
    color = :black
)
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"MNnormdiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,diffMs[3:end],label="M_diff")
lines!(ax,diffNs[3:end],label="N_diff")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"MNdiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,xdiffs[3:end],linestyle=:dot,label="norm(x1-x2)")
lines!(ax,ngdiffs[3:end],linestyle=:dot,label="norm(g1-g2)")
@show ngdiffs[3:end]
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"xgdiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fdiffs[2:end],label="abs(fx1-fx2)/abs(fx1+fx2)")
# text!(
#     circlepoints,
#     text = "this is point " .* string.(1:15),
#     rotation = LinRange(0, 2pi, 16)[1:end-1],
#     align = (:right, :baseline),
#     color = cgrad(:Spectral)[LinRange(0, 1, 15)]
# )
text!( (length(fdiffs),0),
    text = "$(round(fdiffs[end],sigdigits=2))",
    rotation = 0,
    align = (:right, :bottom),
    color = :black
)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"fxreldiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fmdiffs[2:end],label="abs(finv1-finv2)/abs(finv1+finv2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"fmxreldiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fswdiffs[2:end],label="abs(fsw1-fsw2)/abs(fsw1+fsw2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"fswxreldiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fshdiffs[2:end],label="abs(fsh1-fsh2)/abs(fsh1+fsh2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"fshxreldiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fnwdiffs[2:end],label="abs(fnw1-fnw2)/abs(fnw1+fnw2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"fnwxreldiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fnhdiffs[2:end],label="abs(fnh1-fnh2)/abs(fnh1+fnh2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"fnhxreldiff_$(regstr).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,ngdiffs[2:end],label="norm(g1-g2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"ngdiff_$(regstr).png"),f)


# f(x) = |M*N-D|^2 test
using Optim, LineSearches

M = copy(M0); Nt = copy(N0t)
x1 = vcat(vec(M'),vec(Nt'))
# x1 = rand(length(M)+length(Nt))
x2 = x1 + 10*(rand(length(x1)).-0.5)*eps(Float64)

function prepare_fg_MNmD(D,x0)
    nc = size(D,1); p = size(M,2)
    MNmD = Matrix(undef,nc,nc)
    grad = Vector(undef,length(x0))
    function fg!(F,G,x)
        M = reshape(x[1:p*nc],p,nc)'; Nt = reshape(x[p*nc+1:end],p,nc)'
        LCSVD.mmul!(MNmD,M,Nt';uselv=false)
        MNmD -= D
        if G !== nothing
            gradEiw = reshape(view(grad,1:p*nc),p,nc)
            gradEih = reshape(view(grad,p*nc+1:2*p*nc),p,nc)
            LCSVD.mmul!(gradEiw,Nt',MNmD',uselv=uselv); LCSVD.mmul!(gradEih,M',MNmD,uselv=uselv)
            grad .*= 2
            copyto!(G,grad)
        end
        if F !== nothing
            LCSVD.fvalEi(MNmD)
        end
    end
    fg!, nothing
end

function minimize_MNmD(D, x0; tol=-1e-7, maxiter = 1000)
    options = Optim.Options(x_abstol=-1, x_reltol=tol, f_abstol=-1,
                f_reltol=tol, g_abstol=-1, iterations=maxiter,
                store_trace=true, show_trace=false, extended_trace=true,
                callback=nothing, allow_f_increases=true, successive_f_tol=1)
    alphaguess=LineSearches.InitialStatic(alpha=1.0)
    #linesearch = LineSearches.BackTracking(iterations=100, c_1=1e-4)
    linesearch = LineSearches.MoreThuente()
    fgh!, P = prepare_fg_MNmD(D, x0)
    result = optimize(Optim.only_fg!(fgh!),x0,
                LBFGS(m=10, alphaguess=alphaguess,linesearch=linesearch, P=nothing),
                #ConjugateGradient(alphaguess=alphaguess,linesearch=linesearch, P=nothing),
                options)
    xsol = Optim.minimizer(result)
    Eval = result.minimum
    xs = map(tr->tr.metadata["x"], result.trace[1:end])
    fxs = map(tr->tr.value, result.trace[1:end])
    xsol, Eval, xs, fxs
end

xsol1, Eval1, xs1, fxs1 = minimize_MNmD(D, x1)
xsol2, Eval2, xs2, fxs2 = minimize_MNmD(D, x2)

fdiffs = Float64[]; freldiffs = Float64[]; xdiffs = Float64[]; xreldiffs = Float64[]
for (x1, x2) in zip(xs1,xs2)
    push!(xdiffs, norm(x1-x2))
    push!(xreldiffs, norm(x1-x2)/norm(x1+x2))
end
for (fx1, fx2) in zip(fxs1, fxs2)
    push!(fdiffs, abs(fx1-fx2))
    push!(freldiffs, abs(fx1-fx2)/abs(fx1+fx2))
end

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,xdiffs,label="norm(x1-x2)")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"MNmD_LBFGS_xdiff.png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,xreldiffs,label="norm(x1-x2)/norm(x1+x2)")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"MNmD_LBFGS_xreldiff.png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fdiffs,label="abs(fx1-fx2)")
text!( (length(freldiffs),0),
    text = "$(round(fdiffs[end],sigdigits=2))",
    rotation = 0,
    align = (:right, :bottom),
    color = :black
)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"MNmD_LBFGS_fxdiff.png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,freldiffs,label="abs(fx1-fx2)/abs(fx1+fx2)")
text!( (length(freldiffs),0),
    text = "$(round(freldiffs[end],sigdigits=2))",
    rotation = 0,
    align = (:right, :bottom),
    color = :black
)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"MNmD_LBFGS_fxreldiff.png"),f)

# f(x) = (x-1)'A(x-1) test
using Optim, LineSearches
subworkpath = joinpath(workpath,"paper","X_Xt")

A = rand(100,100); A = A'A

function prepare_fg(A,x0)
    os = ones(length(x0))
    function fg!(F,G,x)
        xmos = x-os
        if G !== nothing
            grad = 2A*xmos
            copyto!(G,grad)
        end
        if F !== nothing
            fval = xmos'A*xmos
            fval
        end
    end
    fg!, nothing
end

function minimize(A, x0; optim_mtd=:LBFGS, tol=-1e-7, maxiter = 1000)
    options = Optim.Options(x_abstol=-1, x_reltol=tol, f_abstol=-1,
                f_reltol=tol, g_abstol=-1, iterations=maxiter,
                store_trace=true, show_trace=false, extended_trace=true,
                callback=nothing, allow_f_increases=true, successive_f_tol=1)
    alphaguess=LineSearches.InitialStatic(alpha=1.0)
    #linesearch = LineSearches.BackTracking(iterations=100, c_1=1e-4)
    linesearch = LineSearches.MoreThuente()
    fgh!, P = prepare_fg(A, x0)
    result = optimize(Optim.only_fg!(fgh!),x0,
                optim_mtd == :LBFGS ? LBFGS(m=10, alphaguess=alphaguess,linesearch=linesearch, P=nothing) :
                             ConjugateGradient(alphaguess=alphaguess,linesearch=linesearch, P=nothing),
                options)
    xsol = Optim.minimizer(result)
    Eval = result.minimum
    xs = map(tr->tr.metadata["x"], result.trace[1:end])
    fxs = map(tr->tr.value, result.trace[1:end])
    xsol, Eval, xs, fxs
end

x1 = 2ones(100)
x2 = x1 + 10*(rand(100).-0.5)*eps(Float64)
x3 = x1 + (rand(100).-0.5)*1e-2

maxiter = 100000; optim_mtd = :LBFGS; tol=-1e-9
xsol1, Eval1, xs1, fxs1 = minimize(A, x1; optim_mtd = optim_mtd, tol = tol, maxiter = maxiter)
xsol2, Eval2, xs2, fxs2 = minimize(A, x2; optim_mtd = optim_mtd, tol = tol, maxiter = maxiter)
xsol3, Eval3, xs3, fxs3 = minimize(A, x3; optim_mtd = optim_mtd, tol = tol, maxiter = maxiter)

f12diffs = Float64[]; f12reldiffs = Float64[]; x12diffs = Float64[]; x12reldiffs = Float64[]
for (x1, x2) in zip(xs1,xs2)
    push!(x12diffs, norm(x1-x2))
    push!(x12reldiffs, norm(x1-x2)/norm(x1+x2))
end
for (fx1, fx2) in zip(fxs1, fxs2)
    push!(f12diffs, abs(fx1-fx2))
    push!(f12reldiffs, abs(fx1-fx2)/abs(fx1+fx2))
end
f13diffs = Float64[]; f13reldiffs = Float64[]; x13diffs = Float64[]; x13reldiffs = Float64[]
for (x1, x3) in zip(xs1,xs3)
    push!(x13diffs, norm(x1-x3))
    push!(x13reldiffs, norm(x1-x3)/norm(x1+x3))
end
for (fx1, fx3) in zip(fxs1, fxs3)
    push!(f13diffs, abs(fx1-fx3))
    push!(f13reldiffs, abs(fx1-fx3)/abs(fx1+fx3))
end

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,(nothing)), yscale=log10)
lines!(ax,x12diffs,label="norm(x1-x2)")
lines!(ax,x13diffs,label="norm(x1-x3)")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"xtAx_$(optim_mtd)_xdiff_$(tol)_$(maxiter).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,x12reldiffs,label="norm(x1-x2)/norm(x1+x2)")
lines!(ax,x13reldiffs,label="norm(x1-x3)/norm(x1+x3)")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"xtAx_$(optim_mtd)_xreldiff_$(tol)_$(maxiter).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,f12diffs,label="abs(fx1-fx2)")
lines!(ax,f13diffs,label="abs(fx1-fx3)")
text!( (length(f12reldiffs),0),
    text = "$(round(f12diffs[end],sigdigits=2))",
    rotation = 0,
    align = (:right, :bottom),
    color = :black
)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"xtAx_$(optim_mtd)_fxdiff_$(tol)_$(maxiter).png"),f)

f = Figure(size=(350,250)); ylimits = (1e-24,1e-20)
ax = AMakie.Axis(f[1,1],limits=(nothing,ylimits), yscale=log10)
lines!(ax,f12diffs,label="abs(fx1-fx2)")
lines!(ax,f13diffs,label="abs(fx1-fx3)")
text!( (length(f12reldiffs),ylimits[1]),
    text = "$(round(f12diffs[end],sigdigits=2))",
    rotation = 0,
    align = (:right, :bottom),
    color = :black
)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"xtAx_$(optim_mtd)_fxdiff_log10_$(tol)_$(maxiter).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,f12reldiffs,label="abs(fx1-fx2)/abs(fx1+fx2)")
lines!(ax,f13reldiffs,label="abs(fx1-fx3)/abs(fx1+fx3)")
text!( (length(f12reldiffs),0),
    text = "$(round(f12reldiffs[end],sigdigits=2))",
    rotation = 0,
    align = (:right, :bottom),
    color = :black
)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"xtAx_$(optim_mtd)_fxreldiff_$(tol)_$(maxiter).png"),f)
