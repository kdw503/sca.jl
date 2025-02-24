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

# sizestep, hals_maxiter = 20, 1000
sizestep = eval(Meta.parse(ARGS[1]))
hals_maxiter = eval(Meta.parse(ARGS[2]))
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

# HALS
prefix="hals"; @show prefix
@show prefix; flush(stdout)

fname = joinpath(subworkpath,"$(file_name)$(Xtpstr)_ss$(sizestep)_initnndsvd_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    Whals0, Hhals0, rt1h = dd["Whals0"], dd["Hhals0"], dd["rt1"]
else
    @show "calculating nndsvd..."
    rt1h = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar);
    save(fname, "Whals0",Whals0,"Hhals0",Hhals0,"rt1",rt1h)
end
mfmethod = :HALS; αhals=0.1; maxiter = hals_maxiter; tol=1e-6
Wh, Hh = copy(Whals0), copy(Hhals0);
rt2h = @elapsed rst0h = NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=αhals,
                l₁ratio=1, tol=tol, verbose=false), X, Wh, Hh)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
fitvalh = LCSVD.fitd(X,Wh*Hh)
jldfprexh = "$(file_name)$(Xtpstr)_ss$(sizestep)_hals_noc$(noc)_a$(αhals)_mit$(maxiter)"
fnameh = joinpath(subworkpath,"$(jldfprexh).jld2")
save(fnameh, "W",Wh,"H",Hh,"rt1",rt1h,"rt2",rt2h,"iter",rst0h.niters,"fitval",fitvalh)

# jldfprex = "WMB-10Xv2-HY-raw_small25141_hals_noc200_a0.1_mit100"
ddh = load(joinpath(subworkpath,"$(jldfprexh).jld2"))
W, H, iter, rt1, rt2, fitvalh = ddh["W"], ddh["H"], ddh["iter"], ddh["rt1"], ddh["rt2"], ddh["fitval"]
LCSVD.normalizeWH!(W,H)
sw = norm(W,1)
sh = norm(H,1)
# W heatmap
limit = 0.995
xlimit = round(quantile(vec(W),limit),sigdigits=3)
y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50
for i in 1:min(1,num_blocks)
    f = Figure(size=(xsize,ysize)) # 
    rowsizeq = size(W,1)÷num_blocks
    rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    ax = AMakie.Axis(f[1, 1],width=noc,height=length(rows))#,xaxisposition=:top
    joint_limits = (-xlimit, xlimit) # 0.08(), 0.1(small, 0.15)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax) # , ticks = false
    Colorbar(f[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprexh)_W$(i)_sw$(sw)_limit$(limit).png"),f)
end
# H heatmap
xlimit = round(quantile(vec(H),limit),sigdigits=3)
y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
for i in 1:min(1,num_blocks)
    f = Figure(size=(xsize,ysize)) # (2200,600), small(130,1100)
    rows = noc:-1:1
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    ax = AMakie.Axis(f[1, 1],width=length(cols),height=noc)
    joint_limits = (-xlimit, xlimit) # 200(1591), 30(small 171)
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprexh)_H$(i)_sh$(sh)_fv$(fitvalh)_limit$(limit).png"),f)
end
