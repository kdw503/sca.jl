using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"StochasticLM")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using StochasticLM
using LinearAlgebra

# ARGS = [":randcolX","5e-4","1","0.5","200","100","100"]
initmethod = eval(Meta.parse(ARGS[1])) # :randcolX
αh_user = eval(Meta.parse(ARGS[2])) # 5e-4
sd = eval(Meta.parse(ARGS[3])) # 0.03
batch_rate = eval(Meta.parse(ARGS[4])) # 0.5
outer_maxiter = eval(Meta.parse(ARGS[5])) # 200
maxiter = eval(Meta.parse(ARGS[6])) # 10
inner_maxiter = eval(Meta.parse(ARGS[7])) # 100
@show initmethod, αh_user, sd, batch_rate, outer_maxiter, maxiter, inner_maxiter
flush(stdout) 

#========= Natural dataset (sparse coding) ==========#
include(joinpath(subworkpath,"pcb_slm.jl"))

dataset = :natural
imgsz = (12,12); lengthT = 100000; noc = ncs = 72; nac = 0
patch_size = imgsz[1]

# dd = load(joinpath(subworkpath, "X_whitened_Hspar","natural_SC_l3.0_iter50.jld2"))
# sD = dd["D"]; αs = dd["αs"]; X_whitened = dd["X_whitened"]

prefix = "pcb"
dataset = :natural; p = 72; nac = 0; k = p+nac; imgsz=(12,12); lengthT = 100000; (m,n) = (*(imgsz...), lengthT)

dd = load(joinpath(subworkpath,"allinit.jld2"))
X_whitened = dd["X_whitened"][1]
U, Vt, D = dd["SVD"]; V = Vt'
(m,n,p) = (size(X_whitened)...,ncs); imgsz = (12,12)
gtW, gtH = (Matrix{eltype(X_whitened)}(undef,0,0),Matrix{eltype(X_whitened)}(undef,0,0))
optim_method = :stochasticLM2
β = 0

#======== Relaxed L1 ==========#

Winit, Hinit, Minit, Ninitt, _ = dd[String(initmethod)]
fvinit = LCSVD.fitd(X_whitened,Winit*Hinit)
σ0 = 1; r=0.5; varH = var(Hinit)
M, N = copy(Minit), copy(Ninitt')
θ0 = vcat(vec(copy(M)), vec(copy(N)))
W = U*M; H = N*Vt

# sc2
# for (sd,batch_rate) in [(1.0,1.0)]
    βw=βh=β; αw=0
    σh = σ0*sqrt(varH)
    M .= copy(Minit)
    N .= copy(Ninitt')
    W0 = U*M; H0 = N*Vt # norm(H0,1) = 15953.009989164451
    αh = αh_user*norm(D)^2/norm(H0,1)/norm(M)
    objval = 0
    rt2 = 0
    objvals = Float64[]; f_xs = Float64[]; Ms = []; Ns = []; sympens = Float64[]; sparhs = Float64[]
    rt2 = @elapsed for  iter in 1:outer_maxiter
        global σh, objval
        @show σh, iter 
        θ0 = vcat(vec(copy(M)), vec(copy(N)))
        W = U*M; H = N*Vt
        rtαh = sqrt(αh); Ph = g(H, σh)*rtαh; Gh = dg(H, σh)*rtαh
        nM = Ref(norm(M))
        myminibatcher = makescminibather(k,p,n; sd=sd)
        rjop = SLMCache(makesc2op(M, N, Vt, Gh, Ph, nM); Hd = ones(Float64,2k*p), minibatcher=myminibatcher) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
        nmf_rjop! = make_sc2_fcache_op!(M, N, Vt, D, Gh, Ph, nM, rtαh, σh)
        nbatch = calculate_sc_hbatchsize(Int((n*p)*batch_rate),p)*p + k^2
        θopt, objval = stochasticlm(nmf_rjop!, θ0, nbatch, rjop; itermax=maxiter,
                                                solver_kwargs=(; itmax=inner_maxiter),
                                                # convergence=ConvergenceParams(; dprime=1e-3),
                                                verbose=false) # length(rjop.r) -> full batch
        M .= reshape(θopt[1:k*p], k, p)
        N .= reshape(θopt[k*p+1:end], p, k)
        # sum(abs2, M0 * N0 - D) / 2
        # sum(abs2, M * N - D) / 2
        σh *= r
        f_x, sympen, sparw, sparh, _ = LCSVD.penaltyMN_wholeparams(U, Vt', D, M, N', 0, αh, 0, 0, 0, σh^2; uselv=true)
        push!(objvals, objval); push!(f_xs, f_x); push!(Ms, M); push!(Ns, N); push!(sympens, sympen); push!(sparhs, sparh)
    end
    W = U*M; H = N*Vt
    normWHmX = norm(W*H-X_whitened)
    L1h = norm(H,1)
    LCSVD.normalizeWH!(W,H)
    norm1nH = norm(H,1) # (αh, normWHmX, norm1nH)
                        # (init, 3054.20, 15953.01)
                        # (0.01, 2580.88, 15936.18)
                        # (0.1 , 2611.99, 15929.63)
                        # (1e0,  3717.00, 15921.01)
                        # (1e1,  3717.00, 15940.81)
                        # (1e2,  3717.00, 15935.08)
                        # (1e3,  3717.00, 15866.77)
                        # (1e4,  3716.26, 15902.75)
                        # (1e5,  3717.69, 15952.95)
                        # (1,  3717.00, 15929.31)# sd = 0, batch_rate = 1
                        # (1,  3717.00, 15973.58)# sd = 1, batch_rate = 1
    @show normWHmX, norm1nH; flush(stdout)
    fprex = "$(prefix)_$(dataset)_$(initmethod)_$(optim_method)_sd$(sd)_br$(batch_rate)_oi$(outer_maxiter)_mi$(maxiter)_ii$(inner_maxiter)"
    #fprex = "$(prefix)_BPDN"
    αh_userp =  round(αh_user; sigdigits=4); normWHmXp = round(normWHmX; sigdigits=4); norm1nHp = round(norm1nH; sigdigits=4)
    regstr = "_ah$(αh_userp)_nWHX$(normWHmXp)_nH1$(norm1nHp)"
    fv = LCSVD.fitd(X_whitened,W*H)
    objvalp = round(objval; sigdigits=4); fvp = round(fv; sigdigits=4); rt2p = round(rt2; sigdigits=4)
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_objval$(objvalp)_f$(fvp)_rt$(rt2p)")
    imsave_data(dataset,fname,W,H,imgsz,lengthT; saveH=false)
    save(fname*".jld2","f_xs", f_xs,"objvals", objvals,"Ms", Ms,"Ns", Ns,"sympens", sympens,"sparhs", sparhs, "αh", αh)
# end

#=
experiments = [ (200,200,10, 0.0072, 1, "pcb_natural_randcolX_stochasticLM2_sd0.03_br0.0072_oi200_mi200_ii10_ah0.0005_nWHX2581.0_nH115930.0_objval18.19_f0.007814_rt61520.0"),
                (200,20000,10, 0.0072, 2, "pcb_natural_randcolX_stochasticLM2_sd0.03_br0.0072_oi200_mi20000_ii10_ah0.0005_nWHX2581.0_nH115930.0_objval18.62_f0.007815_rt35430.0"),
                (200,100,10, 0.0144, 3, "pcb_natural_randcolX_stochasticLM2_sd0.03_br0.0144_oi200_mi100_ii10_ah0.0005_nWHX2581.0_nH115920.0_objval35.59_f0.007814_rt59220.0"),
                (200,1000,10, 0.00144, 4, "pcb_natural_randcolX_stochasticLM2_sd0.03_br0.00144_oi200_mi1000_ii10_ah0.0005_nWHX2581.0_nH115910.0_objval3.481_f0.007811_rt41580.0"),
                (400,100,10, 0.0072, 5, "pcb_natural_randcolX_stochasticLM2_sd0.03_br0.0072_oi400_mi100_ii10_ah0.0005_nWHX2581.0_nH115810.0_objval18.04_f0.007813_rt65090.0"),
                (400,100,10, 0.0144, 6,"pcb_natural_randcolX_stochasticLM2_sd0.03_br0.0144_oi400_mi100_ii10_ah0.0005_nWHX2581.0_nH115910.0_objval35.41_f0.007814_rt71230.0")]
# dd = load(joinpath(subworkpath,"pcb_natural_randcolX_stochasticLM2_sd0.03_br0.5_oi200_mi10_ii10_ah0.0005_nWHX2581.0_nH115650.0_objval1220.0_f0.007814_rt48220.0.jld2"))
# f_xs = dd["f_xs"]

resol=(800,600); fntsize1 = 20; fntsize2 = 30

fig = Figure(size=resol)
ymax = 5000
ax = AMakie.Axis(fig[1, 1], limits = ((0,400), (4930, ymax)),
                xlabel = "iteration", ylabel = "penalty", xlabelsize=fntsize2, ylabelsize=fntsize2,
                xticklabelsize=fntsize2, yticklabelsize=fntsize2, yscale = log10)#, title = "Average Fit Value vs. Running Time")
for (oiter,iter,iiter,br,cidx,fname) in experiments
    ddd = load(joinpath(subworkpath,"$(fname).jld2"))
    f_xs = ddd["f_xs"]
    @show minimum(f_xs), maximum(f_xs)
    lines!(ax, f_xs, color=mtdcolors[cidx], label="br=$(br), maxiter=$(iter)", linestyle=nothing)
end
axislegend(ax, labelsize=fntsize1, position = :rt)
save(joinpath(subworkpath,"br_vs_penalty$(ymax).png"),fig, px_per_unit=2)
=#

# #======== L1 ========#
# for initmethod in [:randcolX, :isvd]
#     @show initmethod
#     Winit, Hinit, Minit, Ninitt, _ = dd[String(initmethod)]
#     fvinit = LCSVD.fitd(X_whitened,Winit*Hinit)
#     for αpow in αpows
#         @show αpow
#         maxiter = 1000
#         inner_maxiter = 100
#         M0, N0 = copy(Minit), copy(Ninitt')
# θ0 = vcat(vec(copy(M0)), vec(copy(N0)))
# # θ1 = vcat(vec(copy(Minit)), vec(Ninitt'))
# # norm(θ0-θ1)

# batch_rate = 0.5 # batchsize divide rate
# σw = σh = σ = 1e-16
# W0 = U*M0; H0 = N0*Vt
# @show batch_rate
# # βw=βh=β; αw=αh=50 # 10.0^(αpow); 
# # rtβw = sqrt(βw); rtαw = sqrt(αw); Sw = (W0.<0)*rtβw; Gw = dg(W0, σw)*rtαw
# # rtβh = sqrt(βh); rtαh = sqrt(αh); Sh = (H0.<0)*rtβh; Gh = dg(H0, σh)*rtαh
# # myminibatcher = makeminibather(k,p,m,n)
# # rjop = SLMCache(makeop(M0, N0, U, Vt, Sw, Sh, Gw, Gh); Hd = ones(Float64,2k*p), minibatcher=myminibatcher) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
# # nmf_rjop! = make_fcache_op!(M0, N0, U, Vt, D, Sw, Sh, Gw, Gh, rtβw, rtβh, rtαw, rtαh, σw, σh)
# # nbatch = sum(calculate_whbatchsize(Int((2(m+n)*p)*batch_rate), m/(m+n), p).*p)*2 + k^2
# # rt2 = @elapsed θopt, objval = stochasticlm(nmf_rjop!, θ0, nbatch, rjop; itermax=maxiter, solver_kwargs=(; itmax=100), verbose=true) # length(rjop.r) -> full batch

# # # sc1
# # βw=βh=β; αw=0; αh=10.0^(αpow)
# # rtαh = sqrt(αh); Gh = dg(H0, σh)*rtαh
# # myminibatcher = makescminibather(k,p,n; sd=0.1)
# # rjop = SLMCache(makescop(M0, N0, Vt, Gh); Hd = ones(Float64,2k*p), minibatcher=myminibatcher) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
# # nmf_rjop! = make_sc_fcache_op!(M0, N0, Vt, D, Gh, rtαh, σh)
# # nbatch = calculate_sc_hbatchsize(Int((n*p)*batch_rate),p)*p + k^2
# # rt2 = @elapsed θopt, objval = stochasticlm(nmf_rjop!, θ0, nbatch, rjop; itermax=maxiter, solver_kwargs=(; itmax=inner_maxiter), verbose=false) # length(rjop.r) -> full batch

# # sc2
# βw=βh=β; αw=0; αh=10.0^(αpow)
# rtαh = sqrt(αh); Ph = g(H0, σh)*rtαh; Gh = dg(H0, σh)*rtαh
# nM = Ref(norm(M0))
# myminibatcher = makescminibather(k,p,n; sd=0.1)
# rjop = SLMCache(makesc2op(M0, N0, Vt, Gh, Ph, nM); Hd = ones(Float64,2k*p), minibatcher=myminibatcher) # J = makeop(Wcache, Hcache) jacobian operator; rjop = SLMCache(J)
# nmf_rjop! = make_sc2_fcache_op!(M0, N0, Vt, D, Gh, Ph, nM, rtαh, σh)
# nbatch = calculate_sc_hbatchsize(Int((n*p)*batch_rate),p)*p + k^2
# rt2 = @elapsed θopt, objval = stochasticlm(nmf_rjop!, θ0, nbatch, rjop; itermax=maxiter,
#                                         solver_kwargs=(; itmax=inner_maxiter),
#                                         # convergence=ConvergenceParams(; dprime=1e-3),
#                                         verbose=true) # length(rjop.r) -> full batch

# @show objval
# M = reshape(θopt[1:k*p], k, p)
# N = reshape(θopt[k*p+1:end], p, k)
# # sum(abs2, M0 * N0 - D) / 2
# # sum(abs2, M * N - D) / 2
# W1 = U*M; H1 = N*Vt

#         L1h = norm(H1,1)
#         fv = LCSVD.fitd(X_whitened,W1*H1)
#         LCSVD.normalizeWH!(W1,H1); norm1nH = norm(H1,1)
#         fprex = "$(prefix)_$(dataset)_$(initmethod)_$(optim_method)_br$(batch_rate)_initer$(inner_maxiter)"
#         #fprex = "$(prefix)_BPDN"
#         regstr = "_aw$(αw)_ah$(αh)_b$(β)"
#         fname = joinpath(subworkpath,"$(fprex)$(regstr)_objval$(objval)_f$(fv)_rt$(rt2)")
#         imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
#         # imsave_data(dataset,fname*"_c",W1,H1,imgsz,200; saveH=true, signedcolors=TestData.g1wm())
#         # Xest = W1*H1; mse = norm(X_whitened[:,1:72]-Xest[:,1:72])^2/length(X_whitened[:,1:72])
#         # imsave_data(dataset,joinpath(subworkpath,"$(fprex)$(regstr)_mse$(mse)_Xest1to72.png"),Xest[:,1:72],Xest[1:72,:],imgsz,lengthT; saveH=false)
#     end
# end

# # Winit = LCSVD.init_dictionary(X_whitened,noc+nac)
# # Hinit = Winit\X_whitened
# # # Hinit = rand(noc+nac,100000)
# # # Winit = X_whitened/Hinit
# # M0 = U\Winit; N0 = Hinit/Vt
# # LCSVD.balanceWH!(M0, N0)
# # Winit, Hinit = U*M0, N0*Vt
# # imsave_data(dataset,joinpath(subworkpath,"pcb_natural_randH"),Winit,Hinit,imgsz,lengthT; saveH=false)

# # fv = LCSVD.fitd(X_whitened,Winit*Hinit)
# # Esym = norm(M0*N0-D)^2
# # Esw = norm(Winit,1); Esh = norm(Hinit,1) # Esh includes balanced power
# # LCSVD.normalizeW!(Winit,Hinit)
# # Sw = norm(Winit,1); Sh = norm(Hinit,1) # Sh includes whole power
# # N0t = copy(N0')
# # dd[String(initmethod)] = (Winit, Hinit, M0, N0t, fv, Esym, Esw, Esh, Sw, Sh)
# # save(joinpath(subworkpath,"allinit.jld2"),dd)
