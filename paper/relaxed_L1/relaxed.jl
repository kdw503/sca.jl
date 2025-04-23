using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","relaxed_L1")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.1
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

lcsvd_maxiter = 150
compnmf_maxiter = 1000
hals_maxiter = 150

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))
bias = 0.1

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtnoc, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
noc = 500 # ncs
(m,n,p) = (size(X)...,noc)
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
    LCSVD.normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
    plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
    bg = W*fill(mean(H),1,n)
#        bg = W*H
    X .-= bg
end

#================= plot linesearch ===========#
prefix="pcb"

noc=15; initmethod=:isvd; svdmethod=:isvd
rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, 0; initmethod=initmethod, svdmethod=svdmethod)
V = copy(H0'); N0t = copy(N0')

useprecond=false; uselv=false; tol=1e-5; plot_fig = false; fsrdiff_conv = true; fsrdiff_tol = 1e7
r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
maxiter = Int(ceil(log(eps(eltype(X)))/log(r)/2)) #lcsvd_maxiter
usedenoiseW0H0 = false; inner_maxiter=5; makepositive = true
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005, 5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)
β1 = β2= β; α1 = α2 = α
inner_tol = 1e-6; inner_maxiter = 1000# Int(ceil(0.75*noc+100))
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false, imgsz=imgsz,
    maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true, fsrdiff_convergence = fsrdiff_conv,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, plot_figure = plot_fig, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, fsrdiff_tol = fsrdiff_tol, successive_f_converge=0)
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);
W1, H1 = rst0.W, rst0.Ht'
LCSVD.normalizeW!(W1,H1);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
fitval = LCSVD.fitd(X,W1*H1)
fv = dataset == :fakecells ? avgfit : fitval
nodr = LCSVD.matchedorder(ml,noc); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
LCSVD.flip2makepos!(Wlc1,Hlc1)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)_$(initmethod)"
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_a$(α)_b$(β)"
fname = alg.fsrdiff_convergence ? joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_fsrtol$(fsrdiff_tol)_f$(fv)_tit$(rst0.totalniters)_rt$(rt2)") :
                            joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_tit$(rst0.totalniters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)
println("avgfit = $(round(avgfit,sigdigits=4)), runtime = $(round(rt2,sigdigits=4))sec, total iter = $(rst0.totalniters)")

#=================================================#
first_inner_iters = Int[]; total_inner_iters = Int[]; inner_iters=[]
for noc in 20:20:500
    @show noc
    # LCSVD
    prefix = "lcsvd"
    @show prefix; flush(stdout)

    mfmethod = :PCB; useprecond=false; uselv=false; tol=1e-5
    r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
        # if this is too big iteration number would be increased
    maxiter = 150#Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter

    usedenoiseW0H0 = false; makepositive = true

    (tailstr,initmethod,α,β) = ("_sp",:svd,0.005,.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

    β1 = β2= β; α1 = α2 = α
    rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, 0; initmethod=initmethod, svdmethod=:isvd)
    V = copy(H0'); N0t = copy(N0')
    inner_tol = 1e-6; inner_maxiter = 1000# Int(ceil(0.75*noc+100))
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false, imgsz=imgsz,
        maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
        store_inner_trace = true, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
    M1, N1t = copy(M0), copy(N0t)
    rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);
    alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
    M1, N1t = copy(M0), copy(N0t)
    rt2 = @elapsed LCSVD.solve!(alg, X, U, V, D, M1, N1t);

    W, H = rst0.W, rst0.Ht'
    LCSVD.normalizeW!(W,H);
    # avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
    avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W, H; clamp=false)
    fitval = LCSVD.fitd(X,W*H)
    fv = dataset == :fakecells ? avgfit : fitval
    nodr = LCSVD.matchedorder(ml,noc); Wlc, Hlc = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
    makepositive && LCSVD.flip2makepos!(Wlc,Hlc)
    fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)"
    fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_r$(r))_intol$(inner_tol)_inmiter$(inner_maxiter)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst0.niters)_rt$(rt2)")
    imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false)
    # plotH_data(fname*"_Hinhibit",Hlc[inhibitindices,:]; space=0.,ylabel="",ytickformat="{:.2f}")
    #plotH_data(fname*"_H",Hlc[1:8,:]; space=0.,ylabel="",ytickformat="{:.2f}")
    #continue


    # inner iter
    total_inner_iter = 0; niters = Int[]
    for inner_iter = 2:rst0.niters+1
        niter = rst0.traces[inner_iter].niter
    #    @show niter
        push!(niters,niter)
        total_inner_iter += niter
    end
    @show total_inner_iter
    push!(first_inner_iters,niters[1])
    push!(total_inner_iters,total_inner_iter)
    push!(inner_iters,niters)
    save(joinpath(subworkpath,"inner_iter$(noc)_tol$(inner_tol).jld2"),"first_inner_iter",niters[1],"total_inner_iter",total_inner_iter,"inner_iter",niters)
end
save(joinpath(subworkpath,"inner_iters500_tol1e-5.jld2"),"first_inner_iters",first_inner_iters,"total_inner_iters",total_inner_iters,"inner_iters",inner_iters)

nocs = 20:20:500
f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,(0,2000)))
lines!(ax,nocs,total_inner_iters,label="total_inner_iters")
lines!(ax,nocs,first_inner_iters,label="first_inner_iters")
axislegend(ax; position = :lt)

using LinearRegression
# total_inner_iters
lr = linregress(nocs, total_inner_iters)
tis = LinearRegression.slope(lr)[1]
tib = LinearRegression.bias(lr)
tilr(x) = tis*x+tib
tiregress = tilr.(collect(nocs))
lines!(ax,nocs,tiregress, linestyle=:dash, linewidth=1)
text!(100, 1000, text = "$(round(tis,digits=4))*x+$(round(tib,digits=4))", align = (:left,:top),fontsize=10)
# total_inner_iters
flr = linregress(nocs, first_inner_iters)
fis = LinearRegression.slope(flr)[1]
fib = LinearRegression.bias(flr)
filr(x) = fis*x+fib
firegress = filr.(collect(nocs))
lines!(ax,nocs,firegress,linestyle=:dash, linewidth=1)
text!(100, 200, text = "$(round(fis,digits=4))*x+$(round(fib,digits=4))", align = (:left,:top),fontsize=10)
save(joinpath(subworkpath,"inner_iters_ft$(factor)_nc$(noc)_r$(r)_intol$(inner_tol).png"),f)

dd5 = load(joinpath(subworkpath,"inner_iters500_r0.3_tol1e-5.jld2"))
dd6 = load(joinpath(subworkpath,"inner_iters720_r0.3_tol1e-6.jld2"))
avgfits5 = dd5["avgfits"]
avgfits6 = dd6["avgfits"][1:25]
nocs = 20:20:500
f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,(0.96,0.99)))
lines!(ax,nocs,avgfits6,label="tol=1e-6")
lines!(ax,nocs,avgfits5,label="tol=1e-5")
axislegend(ax; position = :lb)
save(joinpath(subworkpath,"tol_vs_avgfits_ft$(factor)_nc$(noc)_r$(r).png"),f)

for inner_iter = 2:length(rst0.traces)
    xs = rst0.traces[inner_iter].xs
    @show length(xs)
    fxs = rst0.traces[inner_iter].fxs
    ngxs = rst0.traces[inner_iter].ngss
    xreldiffs = Float64[]; xrelmaxdiffs = Float64[]; xmaxdiffs = Float64[]
    fabsdiffs = Float64[]; frelabsdiffs = Float64[]
    append!(xreldiffs,map((x,y)->norm(x-y)/norm(x+y),xs[2:end], xs[1:end-1]))
    append!(xrelmaxdiffs,map((x,y)->maximum(abs.(x-y))/maximum(x),xs[2:end], xs[1:end-1]))
    append!(xmaxdiffs,map((x,y)->maximum(abs.(x-y)),xs[2:end], xs[1:end-1]))
    append!(fabsdiffs,map((fx,fy)->abs(fx - fy), fxs[2:end], fxs[1:end-1]))
    append!(frelabsdiffs,map((fx,fy)->abs(fx - fy)/abs(fx), fxs[2:end], fxs[1:end-1]))

    f = Figure(size=(350,250))
    ax = AMakie.Axis(f[1,1],limits=(nothing,(1e-6,0.1)), yscale=log10)
    lines!(ax,xreldiffs,label="xreldiffs")
    lines!(ax,xrelmaxdiffs,label="xrelmaxdiffs")
    # lines!(ax,xmaxdiffs,label="xmaxdiffs")
    # lines!(ax,fabsdiffs,label="fabsdiffs")
    lines!(ax,frelabsdiffs,label="frelabsdiffs")
    # lines!(ax,ngxs,label="maximum(abs, g)") # maximum(abs, g)
    axislegend(ax; position = :lb)
    save(joinpath(subworkpath,"xdiffs_ft$(factor)_nc$(noc)_r$(r)_intol$(inner_tol)_inmiter$(inner_maxiter)_$(inner_iter).png"),f)
end

# outer iter
x_reldiffs = Float64[]; x_relabsmaxdiffs = Float64[]; f_x_relabsdiffs = Float64[]
for inner_iter = 2:length(rst0.traces)
    push!(x_reldiffs,rst0.traces[inner_iter].x_reldiff)
    push!(x_relabsmaxdiffs,rst0.traces[inner_iter].x_relabsmaxdiff)
    push!(f_x_relabsdiffs,rst0.traces[inner_iter].f_x_relabsdiff)
end
f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=((1,rst0.niters),(1e-7,1)), yscale=log10)
lines!(ax,x_reldiffs,label="xreldiffs")
lines!(ax,x_relabsmaxdiffs,label="xrelmaxdiffs")
# lines!(ax,xmaxdiffs,label="xmaxdiffs")
# lines!(ax,fabsdiffs,label="fabsdiffs")
lines!(ax,f_x_relabsdiffs,label="frelabsdiffs")
# lines!(ax,ngxs,label="maximum(abs, g)") # maximum(abs, g)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"xdiffs_ft$(factor)_nc$(noc)_r$(r)_intol$(inner_tol)_inmiter$(inner_maxiter)_tol$(tol)_miter$(maxiter).png"),f)


mfmethod = :PCB; useprecond=false; uselv=false; tol=1e-5
r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
maxiter = 150#Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter
usedenoiseW0H0 = false; makepositive = true
(tailstr,initmethod,α,β) = ("_sp",:svd,0.005,.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

β1 = β2= 5.0; α1 = α2 = 0.005
rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, 0; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false, imgsz=imgsz,
        maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M1, N1t = copy(M0), copy(N0t)
rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

invs = Float64[]; sws = Float64[]; shs = Float64[]; nws = Float64[]; nhs = Float64[]; nhs = Float64[]; pns = Float64[]
for tr in rst0.traces
    append!(invs, tr.invs)
    append!(sws, tr.sparseWs)
    append!(shs, tr.sparseHs)
    append!(nws, tr.nnWs)
    append!(nhs, tr.nnHs)
    pens = tr.invs .+ tr.sparseWs .+ tr.sparseHs .+ tr.nnWs .+ tr.nnHs
    append!(pns,pens)
end

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=((0,450),(0,5000)), yscale=identity)
lines!(ax,pns,label="penalty")
lines!(ax,invs,label="invert.")
lines!(ax,sws,label="sparsity w")
lines!(ax,shs,label="sparsity h")
lines!(ax,nws,label="nneg w")
lines!(ax,nhs,label="nneg h")
# lines!(ax,ngxs,label="maximum(abs, g)") # maximum(abs, g)
#axislegend(ax; position = :rt)
save(joinpath(subworkpath,"penalties_nolegends.png"),f)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"penalties.png"),f)


β1 = β2= 5.0; α1 = α2 = 0.005
rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X', noc, 0; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
inner_tol = 1e-6; inner_maxiter = Int(ceil(2.5*noc+350))# Int(ceil(0.75*noc+100))
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false, imgsz=imgsz,
        maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M1, N1t = copy(M0), copy(N0t)
rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

invs = Float64[]; sws = Float64[]; shs = Float64[]; nws = Float64[]; nhs = Float64[]; nhs = Float64[]; pns = Float64[]
for tr in rst0.traces
    append!(invs, tr.invs)
    append!(sws, tr.sparseWs)
    append!(shs, tr.sparseHs)
    append!(nws, tr.nnWs)
    append!(nhs, tr.nnHs)
    pens = tr.invs .+ tr.sparseWs .+ tr.sparseHs .+ tr.nnWs .+ tr.nnHs
    append!(pns,pens)
end

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=((0,450),(0,5000)), yscale=identity)
lines!(ax,pns,label="penalty")
lines!(ax,invs,label="invert.")
lines!(ax,sws,label="sparsity w")
lines!(ax,shs,label="sparsity h")
lines!(ax,nws,label="nneg w")
lines!(ax,nhs,label="nneg h")
# lines!(ax,ngxs,label="maximum(abs, g)") # maximum(abs, g)
#axislegend(ax; position = :rt)
save(joinpath(subworkpath,"penalties_Xt_nolegends.png"),f)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"penalties_Xt.png"),f)


β1 = β2= 5.0; α1 = α2 = 0.005
rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, 0; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
inner_tol = 1e-6; inner_maxiter = 1000# Int(ceil(0.75*noc+100))
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false, imgsz=imgsz,
        maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
        store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M1, N1t = copy(M0), copy(N0t)
rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

invs = Float64[]; sws = Float64[]; shs = Float64[]; nws = Float64[]; nhs = Float64[]; nhs = Float64[]; pns = Float64[]
for tr in rst0.traces
    append!(invs, tr.invs)
    append!(sws, tr.sparseWs)
    append!(shs, tr.sparseHs)
    append!(nws, tr.nnWs)
    append!(nhs, tr.nnHs)
    pens = tr.invs .+ tr.sparseWs .+ tr.sparseHs .+ tr.nnWs .+ tr.nnHs
    append!(pns,pens)
end

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=((0,450),(0,5000)), yscale=identity)
lines!(ax,pns,label="penalty")
lines!(ax,invs,label="invert.")
lines!(ax,sws,label="sparsity w")
lines!(ax,shs,label="sparsity h")
lines!(ax,nws,label="nneg w")
lines!(ax,nhs,label="nneg h")
# lines!(ax,ngxs,label="maximum(abs, g)") # maximum(abs, g)
#axislegend(ax; position = :rt)
save(joinpath(subworkpath,"penalties_SVD(X)t_nolegends.png"),f)
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"penalties_SVD(X)t.png"),f)

