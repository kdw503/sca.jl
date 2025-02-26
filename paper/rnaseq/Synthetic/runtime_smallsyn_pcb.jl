using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","rnaseq")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

# sizestep, pcb_maxiter, pcb_inner_maxiter, tol, per_component = 10, 0, 1000, 1e-6, false
sizestep = eval(Meta.parse(ARGS[1]))
pcb_maxiter = eval(Meta.parse(ARGS[2]))
pcb_inner_maxiter = eval(Meta.parse(ARGS[3]))
tol = eval(Meta.parse(ARGS[4]))
per_component = eval(Meta.parse(ARGS[5]))
memsize = Int(Sys.total_memory())/1e9

#colormap
# Makie.available_gradients()
# Plasma, Inferno, Magma, Cividis, Jet, grays, heat, :Spectral
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

#======== Load Data ===========#
dataset = :rnaseq
#X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
feature_name = "Synthetic" # (100562×32285)
file_name = feature_name*"-rand" # "-rand"


#========= Matrix Factorization methods ===========#
using FakeCells

ncells = 1000
noc = 500; nac = 0; nc = noc+nac
Xtpstr = ""
fn = joinpath(subworkpath,"$(file_name)$(Xtpstr)_ss$(sizestep)_X.jld2")
if isfile(fn)
    dd = load(fn)
    X = dd["X"]
else
    H = rand(Float32,1000,10056); W = makefiringrate(3228, ncells; lambda = 0.01)
    X = W*H
    save(fn, "X",X)
end

# LCSVD
prefix = "pcb"
@show prefix; flush(stdout)

(tailstr,initmethod,α,β) = ("_sp_nn",:isvd,0.0001,5.0)# ("_sp_nn",:custom,0.005,5.0) ,("_nn",:nndsvd,0.,5.0)
sbc_maxiter = 100
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
fitval0 = LCSVD.fitd(X,W0*D*H0t')
sw0 = norm(W0,1)

r = 0.3
inner_tol = tol; inner_maxiter = pcb_inner_maxiter
optim_method = :lbfgs; smaxiter = 500; pcb_maxiter = 0
maxiter = pcb_maxiter == 0 ? Int(ceil(log(eps(eltype(X)))/log(r))) : pcb_maxiter
if per_component
    α1vec = fill(α,noc); α2vec = fill(α,noc); β1vec = fill(β,noc); β2vec = fill(β,noc)
    α1vec[1] = α2vec[1] = 0
    alg = LCSVD.LinearCombSVD(α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec, r=r,
        useprecond=false, optim_method = optim_method, uselv=false, maxiter = maxiter,
        smaxiter = smaxiter, inner_maxiter = inner_maxiter, inner_tol = inner_tol,
        store_trace = false, store_inner_trace = false, show_trace = true,
        allow_f_increases = true, f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol,
        x_reltol=tol, successive_f_converge=0)
    per_com_str = "cw"
else
    α1 = α; α2 = α; β1 = β; β2= β;
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false,
        optim_method = optim_method, maxiter = maxiter, smaxiter = smaxiter,
        inner_maxiter = inner_maxiter, inner_tol = inner_tol, store_trace = false,
        store_inner_trace = false, show_trace = true, allow_f_increases = true,
        f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
    per_com_str = ""
end

M, Nt = copy(M0), copy(N0t)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0t, D, M, Nt);
W, H = rst0.W, rst0.Ht'
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
fitval = LCSVD.fitd(X,W*H)
# jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_sbc_it$(sbc_maxiter)"
# fname = joinpath(subworkpath,"$(jldfprex).jld2")
# save(fname, "W",W,"H",H,"M",M0,"N",N0,"rt1",rt1,"iter",sbc_maxiter,"fitval",fitval)
jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb$(per_com_str)_$(optim_method)_noc$(noc)_a$(α)_b$(β)_r$(r)_tol$(tol)_it$(rst0.niters)_iit$(inner_maxiter)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "W",W,"H",H,"M",M,"N",Nt',"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)

dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
d=LCSVD.normalizeWH!(W,H)
sw = norm(W,1)
sh = norm(H,1)
dn = norm(M*Nt'-D)^2
# W heatmap
xlimit = round(quantile(vec(W),0.9999),sigdigits=3) # cf)  round(123456,digits=-2) = 123500.0
y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50
for i in 1:min(1,num_blocks)
    fig = Figure(size=(xsize,ysize)) # 
    rowsizeq = size(W,1)÷num_blocks
    rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    ax = AMakie.Axis(fig[1, 1],width=noc,height=length(rows))#,xaxisposition=:top
    joint_limits = (-xlimit, xlimit) # 0.08(), 0.1(small, 0.15)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax) # , ticks = false
    Colorbar(fig[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprex)_W$(i)_sw$(sw)_dn$(dn).png"),fig)
end
# H heatmap
xlimit = round(quantile(vec(H),0.9999),sigdigits=3)
y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
for i in 1:min(1,num_blocks)
    fig = Figure(size=(xsize,ysize)) # (2200,600), small(130,1100)
    rows = noc:-1:1
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    ax = AMakie.Axis(fig[1, 1],width=length(cols),height=noc)
    joint_limits = (-xlimit, xlimit) # 200(1591), 30(small 171)
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(fig[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprex)_H$(i)_sh$(sh)_fv$(fitval).png"),fig)
end


jldfprex = "WMB-10Xv2-HY-rawt_ss10_hals_noc500_a0.1_mit100"
jldfprex = "WMB-10Xv2-HY-rawt_ss10_pcb_noc500_a0.005_b5.0_tol1.0e-6_it6"
jldfprex = "WMB-10Xv2-HY-rawt_ss5_pcb_noc500_a0.005_b5.0_tol1.0e-6_it6"
dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
W = dd["W"]
H = dd["H"]
M = dd["M"]
N = dd["N"]; Nt = N'
M = W0\W
Nt = H0t\H'

M, Nt = copy(M0), copy(N0t)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0t, D, M, Nt);
W, H = rst0.W, rst0.Ht'
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
fitval = LCSVD.fitd(X,W*H)
jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb$(per_com_str)_from_hals_noc$(noc)_a$(α)_b$(β)_tol$(tol)_it$(rst0.niters)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "W",W,"H",H,"M",M,"N",Nt',"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)

dd = load(joinpath(subworkpath,"WMB-10Xv2-HY-rawt_ss10_pcb_from_hals_noc500_a0.005_b5.0_tol1.0e-6_it7.jld2"))
W = dd["W"]; H = dd["H"]; M = dd["M"]; Nt = dd["N"]'; rt1 = dd["rt1"]; rt2 = dd["rt2"]; iter = dd["iter"]
fitval = LCSVD.fitd(X,W*H)
jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb$(per_com_str)_from_hals_noc$(noc)_a$(α)_b$(β)_tol$(tol)_it$(iter)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "W",W,"H",H,"M",M,"N",Nt',"rt1",rt1,"rt2",rt2,"iter",iter,"fitval",fitval)



