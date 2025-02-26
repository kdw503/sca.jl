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

feature_name = eval(Meta.parse(ARGS[1]))
ncells = eval(Meta.parse(ARGS[2]))
noc = eval(Meta.parse(ARGS[3]))
αhals = eval(Meta.parse(ARGS[4]));
hals_maxiter = eval(Meta.parse(ARGS[5]));

@show feature_name, ncells, noc, αhals, hals_maxiter
nc = noc
memsize = Int(Sys.total_memory())/1e9
@show memsize

# load data
using NRRD
feature_group=first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-raw"
Xraw = load(joinpath(fgpath,file_name*".nhdr")).data
m,nraw = size(Xraw)
ncells = ncells == 0 ? nraw : (file_name*="_n$(ncells)"; ncells); @show ncells; flush(stdout)
ncells < nraw && error("ncells must be greater than $(nraw)")
X = view(Xraw,:,1:ncells); n=ncells

# HALS
method="hals"
@show method; flush(stdout)
# Initialization
initmtd = :nndrsvd; initstr = "init$(initmtd)"
fname = joinpath(subworkpath,"$(file_name)_$(initstr)_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    Whals0, Hhals0, rt1 = dd["Whals0"], dd["Hhals0"], dd["rt1"]
else
    @show "Calculating initialization..."
    memsize = Int(Sys.total_memory())/1e9
    if initmtd == :nndrsvd
        rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar);
    elseif initmtd == :nndisvd
        fn = joinpath(subworkpath,"$(file_name)_isvd_nc$(nc).jld2")
        if isfile(fn)
            @show "Reading isvd..."
            dd = load(fn)
            U, s, Vt, rt0, memsize0 = dd["U"], dd["s"], dd["Vt"], dd["rt0"], dd["memsize0"]
        else
            @show "Calculating isvd..."
            rt0 = @elapsed ((U, s) = isvd(X, nc); Vt = Array(Diagonal(s.^-1)*(U'*X))) # Vt = Array((pinv(U*Diagonal(s))*X))
            save(fn, "U",U,"s",s,"Vt",Vt,"rt0",rt0, "memsize0", memsize0)
        end
        rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar, initdata=SVD(U,s,Vt));
        rt1 += rt0
    end
    @show memsize, Int(Sys.free_memory())/1e9
    save(fname, "Whals0",Whals0,"Hhals0",Hhals0,"rt1",rt1, "memsize", memsize)
end
# solve
maxiter = hals_maxiter; tol=3e-3
W, H = copy(Whals0), copy(Hhals0);
@show "solve"
rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, W, H)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
LCSVD.normalizeW!(W,H)
# fitval = LCSVD.fitd(X,W*H)
fitval = 0
# @show "s3"
fname = joinpath(subworkpath,"$(file_name)_n$(ncells)_$(initstr)_$(method)_noc$(noc)_a$(αhals)_tol$(tol)_mit$(maxiter)_nrrd.jld2")
save(fname, "W",W,"H",H,"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval, "rst", rst0, "memsize", memsize)

jldfprex = "$(file_name)_n$(ncells)_$(initstr)_$(method)_noc$(noc)_a$(αhals)_tol$(tol)_mit$(maxiter)_nrrd"
dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
# W, H, rt1, rt2, iter, fitval = dd["W"], dd["H"], dd["rt1"], dd["rt2"], dd["iter"], dd["fitval"]



# W heatmap
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
    save(joinpath(subworkpath,"$(jldfprex)_W$i.png"),f)
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
    save(joinpath(subworkpath,"$(jldfprex)_H$i.png"),f)
end
