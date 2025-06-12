using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
allenbrainversion = "20241130"
subworkpath = joinpath(workpath,"paper","rnaseq",allenbrainversion)

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"utils.jl"))

# ARGS = ["\"WMB-10Xv2-HY\"", "\"log2\"", "0", "500","0.1","100"]
feature_name = eval(Meta.parse(ARGS[1]))
pp = eval(Meta.parse(ARGS[2]))
ncells = eval(Meta.parse(ARGS[3]))
noc = eval(Meta.parse(ARGS[4]))
αhals = eval(Meta.parse(ARGS[5]));
hals_maxiter = eval(Meta.parse(ARGS[6]));

@show feature_name, ncells, noc, αhals, hals_maxiter

nac=0
nc = noc+nac
memsize = Int(Sys.total_memory())/1e9
@show memsize; flush(stdout)

# load data
using NRRD
feature_group=first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-"*pp
Xgcraw = load(joinpath(fgpath,file_name*".nhdr")).data
m,nraw = size(Xgcraw)
ncells = ncells == 0 ? nraw : (file_name*="_n$(ncells)"; ncells); @show ncells; flush(stdout)
ncells > nraw && error("ncells must be smaller than $(nraw)")
X = pp == "raw" ? sqrt.(Xgcraw[:,1:ncells]) : view(Xgcraw,:,1:ncells)
n=ncells

# HALS
prefix="hals"; @show prefix
fname = joinpath(subworkpath,"$(file_name)_nndrsvd_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    Whals0, Hhals0, rt1 = dd["Whals0"], dd["Hhals0"], dd["rt1"]
else
    rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar);
    save(fname, "Whals0",Whals0,"Hhals0",Hhals0,"rt1",rt1)
end
mfmethod = :HALS; maxiter = hals_maxiter; tol=-1
W, H = copy(Whals0), copy(Hhals0);
rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, W, H)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X,W*H)
fname = joinpath(subworkpath,"$(file_name)_hals_noc$(noc)_a$(αhals)_iter$(maxiter).jld2")
save(fname, "W",W,"H",H,"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)


# W heatmap
LCSVD.normalizeWH!(W,H)
sw = norm(W,1)
sh = norm(H,1)
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)
y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50
xlimit = round(quantile(vec(W),0.9999),sigdigits=3) # cf)  round(123456,digits=-2) = 123500.0
for i in 1:min(2,num_blocks) # som many blocks, so now just plot the first twos.
    f = Figure(size=(xsize,ysize))
    rowsizeq = size(W,1)÷num_blocks
    rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    ax = CairoMakie.Axis(f[1, 1],width=noc,height=length(rows))
    joint_limits = (-xlimit, xlimit)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(jldfprex)_W$(i)_sw$(sw).png"),f)
end
# H heatmap
y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
xlimit = round(quantile(vec(H),0.9999),sigdigits=3)
for i in 1:min(2,num_blocks)
    f = Figure(size=(xsize,ysize))
    rows = noc:-1:1
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    ax = CairoMakie.Axis(f[1, 1],width=length(cols),height=noc)
    joint_limits = (-xlimit, xlimit)
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(jldfprex)_H$(i)_sh$(sh)_fv$(fitval).png"),f)
end
