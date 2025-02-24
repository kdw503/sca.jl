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
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

# sizestep, pcb_maxiter, pcb_inner_maxiter, tol, per_component, bothreg, αrng = 20, 1, 1, 1e-6, false, true, 0.0001:0.0005:0.01
sizestep = eval(Meta.parse(ARGS[1]))
pcb_maxiter = eval(Meta.parse(ARGS[2]))
pcb_inner_maxiter = eval(Meta.parse(ARGS[3]))
tol = eval(Meta.parse(ARGS[4]))
per_component = eval(Meta.parse(ARGS[5]))
bothreg = eval(Meta.parse(ARGS[6]))
αrng = eval(Meta.parse(ARGS[7]))
memsize = Int(Sys.total_memory())/1e9
per_com_str = per_component ? "cw" : ""
subworkpath = joinpath(workpath,"paper","rnaseq","$(sizestep)to1")

#colormap
# Makie.available_gradients()
# Plasma, Inferno, Magma, Cividis, Jet, grays, heat, :Spectral
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

#======== Load Data ===========#
dataset = :rnaseq
#X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)
file_name = feature_name*"-raw"
adata = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad"))
Xraw = adata.X # cell_label(adata.obs_names), gene_identifier(adata.var_names)

#========= Matrix Factorization methods ===========#
noc = 500; nac = 0; nc = noc+nac

X = sqrt.(Xraw[1:sizestep:end,1:sizestep:end])'; Xtpstr = "t"

# LCSVD
prefix = "pcb"
exprstr = bothreg ? "$(prefix) alpha both" : "$(prefix) alpha each" 
@show exprstr, sizestep, αrng; flush(stdout)

r = 0.3
inner_tol = tol; inner_maxiter = pcb_inner_maxiter
optim_method = :lbfgs; smaxiter = 500
maxiter = pcb_maxiter == 0 ? Int(ceil(log(eps(eltype(X)))/log(r))) : pcb_maxiter

fitvals = []; sws = []
β = 3.0
tailstr = "_sp_nn"
initmethod = :isvd
initmtdstr="init$(initmethod)"
fname = joinpath(subworkpath,"$(file_name)$(Xtpstr)_ss$(sizestep)_$(initmtdstr)_nc$(nc).jld2")
if isfile(fname)
    println("reading init.")
    dd = load(fname)
    W0, H0t, M0, N0t, D, rt1 = dd["W0"], dd["H0t"], dd["M0"], dd["N0t"], dd["D"], dd["rt1"]
    # noc = 500, norm(X-W0*M0*N0*H0) = 15375.242f0
    # fitval = LCSVD.fitd(X,W0*M0*N0*H0) # noc = 500, 0.92099863f0
elseif initmethod == :sbc
    fn = joinpath(subworkpath,"$(file_name)$(Xtpstr)_ss$(sizestep)_isvd_nc$(nc).jld2")
    if isfile(fn)
        @show "Reading isvd..."
        dd = load(fn)
        U, s, Vt, rt0, memsize0 = dd["U"], dd["s"], dd["Vt"], dd["rt0"], dd["memsize"]
    else
        @show "Calculating isvd..."
        rt0 = @elapsed ((U, s) = isvd(X, nc); Vt = Array(Diagonal(s.^-1)*(U'*X))) # Vt = Array((pinv(U*Diagonal(s))*X))
        save(fn, "U",U,"s",s,"Vt",Vt,"rt0",rt0,"memsize", memsize)
    end
    rt1 = @elapsed begin 
                        W0 = U; H0 = Vt; D = Diagonal(s)
                        M0 = sbc(W0; maxiter=sbc_maxiter)
                        W = W0*M0; H = W\X; N0 = H/H0
                   end
    rt1 += rt0
    save(fname, "W0",W0,"H0t",H0',"M0",M0,"N0t",N0',"D",D,"rt1",rt1,"memsize", memsize)
    H0t, N0t = copy(H0'), copy(N0')
else
    println("calculating init.")
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod)
    H0t, N0t = copy(H0'), copy(N0')
    save(fname, "W0",W0,"H0t",H0t,"Wp",Wp,"Hpt",Hp',"M0",M0,"N0t",N0t,"D",D,"rt1",rt1)
end
# fitval0 = LCSVD.fitd(X,W0*D*H0t')
# sw0 = norm(W0,1)

for (iter, α) in enumerate(αrng)
    @show iter; flush(stdout)

if per_component
    α1vec = fill(α,noc); α2vec = fill(α,noc); β1vec = fill(β,noc); β2vec = fill(β,noc)
    α1vec[1] = α2vec[1] = 0
    alg = LCSVD.LinearCombSVD(α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec, r=r,
        useprecond=false, optim_method = optim_method, uselv=false, maxiter = maxiter,
        smaxiter = smaxiter, inner_maxiter = inner_maxiter, inner_tol = inner_tol,
        store_trace = false, store_inner_trace = false, show_trace = false,
        allow_f_increases = true, f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol,
        x_reltol=tol, successive_f_converge=0)
else
    (α1, β2) = (α, β); (α2, β1) = bothreg ? (α, β) : (0.,0)
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false,
        optim_method = optim_method, maxiter = maxiter, smaxiter = smaxiter,
        inner_maxiter = inner_maxiter, inner_tol = inner_tol, store_trace = false,
        store_inner_trace = false, show_trace = false, allow_f_increases = true,
        f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
end

M, Nt = copy(M0), copy(N0t)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0t, D, M, Nt);
W, H = rst0.W, rst0.Ht'
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
fitval = LCSVD.fitd(X,W*H)
# jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_sbc_it$(sbc_maxiter)"
# fname = joinpath(subworkpath,"$(jldfprex).jld2")
# save(fname, "W",W,"H",H,"M",M0,"N",N0,"rt1",rt1,"iter",sbc_maxiter,"fitval",fitval)
jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb_noc$(noc)_aw$(α)_bh$(β)_it$(rst0.niters)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "W",W,"H",H,"M",M,"N",Nt',"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)
push!(fitvals,fitval)
d=LCSVD.normalizeWH!(W,H)
sw = norm(W,1)
push!(sws,sw)

# dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
# M,Nt,W, H, iter, rt1, rt2, fitval = dd["M"], dd["N"]', dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
# d=LCSVD.normalizeWH!(W,H)
# sw = norm(W,1)
# sh = norm(H,1)
# dn = norm(M*Nt'-D)^2
# # W heatmap
# limit = 0.99
# xlimit = round(quantile(vec(W),limit),sigdigits=3) # cf)  round(123456,digits=-2) = 123500.0
# y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50
# for i in 1:min(1,num_blocks)
#     fig = Figure(size=(xsize,ysize)) # 
#     rowsizeq = size(W,1)÷num_blocks
#     rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
#     ax = AMakie.Axis(fig[1, 1],width=noc,height=length(rows))#,xaxisposition=:top
#     joint_limits = (-xlimit, xlimit) # 0.08(), 0.1(small, 0.15)
#     hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
#     hidedecorations!(ax) # , ticks = false
#     Colorbar(fig[:, end+1], hm1)                     # These three
#     save(joinpath(subworkpath,"$(jldfprex)_W$(i)_sw$(sw)_dn$(dn)_limit$(limit).png"),fig)
# end
# # H heatmap
# xlimit = round(quantile(vec(H),limit),sigdigits=3)
# y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
# for i in 1:min(1,num_blocks)
#     fig = Figure(size=(xsize,ysize)) # (2200,600), small(130,1100)
#     rows = noc:-1:1
#     colsizeq = size(H,2)÷num_blocks
#     cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
#     ax = AMakie.Axis(fig[1, 1],width=length(cols),height=noc)
#     joint_limits = (-xlimit, xlimit) # 200(1591), 30(small 171)
#     hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
#     hidedecorations!(ax)
#     Colorbar(fig[:, end+1], hm1)                     # These three
#     save(joinpath(subworkpath,"$(jldfprex)_H$(i)_sh$(sh)_fv$(fitval)_limit$(limit).png"),fig)
# end

end

jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb_noc$(noc)_αwrng"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "αrng", αrng,"fitvals",fitvals,"sws",sws ,"per_component", per_component, "optim_method", optim_method, "β", β, "r",r,"tol",tol)
dd = load(fname)
αrng = dd["αrng"]; fitvals = dd["fitvals"]; sws = dd["sws"]

labels = ["fitval","norm(W,1)"]
fig = Figure(resolution = (700,400))
ax1 = AMakie.Axis(fig[1, 1], limits=(nothing, (0.,1.1) ), xlabel = "αw", ylabel = "fit", title = "Fit Value vs. αw")
ax2 = AMakie.Axis(fig[1, 1], limits=(nothing, (0.,8000) ), yaxisposition = :right, ylabel = "Sparseness of W" #= yticklabelcolor = :red =# )
ln1 = lines!(ax1, αrng, fitvals, color=mtdcolors[1], label=labels[1])
ln2 = lines!(ax2, αrng, sws, color=mtdcolors[1], label=labels[2], linestyle = :dash, linewidth = 2)
fig[:,2] = Legend(fig[:,1],[ln1,ln2],labels)
save(joinpath(subworkpath,"fit_and_sparsity_αwrng_noc$(noc)_b$(β).png"),fig)



# jldfprex = "WMB-10Xv2-HY-rawt_ss10_hals_noc500_a0.1_mit100"
# jldfprex = "WMB-10Xv2-HY-rawt_ss10_pcb_noc500_a0.005_b5.0_tol1.0e-6_it6"
# jldfprex = "WMB-10Xv2-HY-rawt_ss5_pcb_noc500_a0.005_b5.0_tol1.0e-6_it6"
# dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
# W = dd["W"]
# H = dd["H"]
# M = dd["M"]
# N = dd["N"]; Nt = N'
# M = W0\W
# Nt = H0t\H'

# M, Nt = copy(M0), copy(N0t)
# rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0t, D, M, Nt);
# W, H = rst0.W, rst0.Ht'
# # avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
# fitval = LCSVD.fitd(X,W*H)
# jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb$(per_com_str)_from_hals_noc$(noc)_a$(α)_b$(β)_tol$(tol)_it$(rst0.niters)"
# fname = joinpath(subworkpath,"$(jldfprex).jld2")
# save(fname, "W",W,"H",H,"M",M,"N",Nt',"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)

# dd = load(joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss10_pcb_from_hals_noc500_a0.005_b5.0_tol1.0e-6_it7.jld2"))
# W = dd["W"]; H = dd["H"]; M = dd["M"]; Nt = dd["N"]'; rt1 = dd["rt1"]; rt2 = dd["rt2"]; iter = dd["iter"]
# fitval = LCSVD.fitd(X,W*H)
# jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb$(per_com_str)_from_hals_noc$(noc)_a$(α)_b$(β)_tol$(tol)_it$(iter)"
# fname = joinpath(subworkpath,"$(jldfprex).jld2")
# save(fname, "W",W,"H",H,"M",M,"N",Nt',"rt1",rt1,"rt2",rt2,"iter",iter,"fitval",fitval)



