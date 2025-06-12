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

using AllenBrain, FileIO, LCSVD, Statistics, CairoMakie, LinearAlgebra

#====== Load data from AllenBrain site ============#
version = "20230630"
manifest = awsmanifest(version)

using FileIO, IncrementalSVD
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)s
file_name = feature_name*"-log2"
X = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X

#====== Save data as NRRD memory mapped file and read it =============#
using AxisArrays, NRRD

# Write whole WMB-10XV3 datasets
sizexs = []; sizey = 32285
download_base = joinpath(datapath,"AllenBrain")
fgpath = joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv3")
expression_matrices = manifest.file_listing["WMB-10Xv3"]["expression_matrices"]
open(joinpath(fgpath,"WMB-10Xv3-log2.nrrd"),"w") do io   # write data
    for fm_label in ["WMB-10Xv3-CB","WMB-10Xv3-CTXsp","WMB-10Xv3-HPF","WMB-10Xv3-HY"
                    ,"WMB-10Xv3-Isocortex-1","WMB-10Xv3-Isocortex-2","WMB-10Xv3-MB","WMB-10Xv3-MY"
                    ,"WMB-10Xv3-OLF","WMB-10Xv3-P","WMB-10Xv3-PAL","WMB-10Xv3-STR", "WMB-10Xv3-TH"
                    ]
        feature_matrix_label = fm_label; scale="log2"
        rpath = expression_matrices[feature_matrix_label][scale]["files"]["h5ad"]["relative_path"]
        local_path = joinpath(download_base, split(rpath,"/")... )
        AllenBrain.download_dir(manifest, rpath, local_path)
        # Load .h5ad file
        adata = load(local_path)
        X = sqrt.(adata.X')
        sizey = size(X,1)
        push!(sizexs,size(X,2))
        write(io,X)
    end
end

axy = AxisArrays.Axis{:y}(1:sizey)
axx = AxisArrays.Axis{:x}(1:sum(sizexs))
header = NRRD.headerinfo(Float32, (axy, axx))
header["datafile"] = "WMB-10Xv3-log2.nrrd"
open(joinpath(fgpath,"WMB-10Xv3-log2.nhdr"),"w") do io # write header
    write(io,magic(format"NRRD"))
    NRRD.write_header(io,"0004",header)
end

# write single Gene expression Matrix
# read from site
feature_name = "WMB-10Xv2-HY"; scale="log2"
feature_group = first(feature_name,9)
download_base = joinpath(datapath,"AllenBrain")
expression_matrices = manifest.file_listing[feature_group]["expression_matrices"]
rpath = expression_matrices[feature_name][scale]["files"]["h5ad"]["relative_path"]
local_path = joinpath(download_base, split(rpath,"/")... )
AllenBrain.download_dir(manifest, rpath, local_path) # download from site
adata = load(local_path) # Load .h5ad file
Xgc = sqrt.(adata.X')
# write data
fgpath = joinpath(download_base,"expression_matrices",feature_group)
fprex = "$(feature_name)-$(scale)"
dfname = fprex*".nrrd"
open(joinpath(fgpath,dfname),"w") do io
    write(io,Xgc)
end
# write header
sizey, sizex = size(Xgc)
axy = AxisArrays.Axis{:y}(1:sizey)
axx = AxisArrays.Axis{:x}(1:sizex)
header = NRRD.headerinfo(Float32, (axy, axx))
header["datafile"] = dfname
hfname = fprex*".nhdr"
open(joinpath(fgpath,hfname),"w") do io
    write(io,magic(format"NRRD"))
    NRRD.write_header(io,"0004",header)
end

#========= load data from NRRD file as memory mapped array ======#
feature_name = "WMB-10Xv3"; scale="raw" # WMB-10Xv3(32285×2349544)
feature_group = first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-raw"
X = load(joinpath(fgpath,file_name*".nhdr")).data

feature_name = "WMB-10Xv3-CB"; scale="raw" # WMB-10Xv3(32285×182026)
feature_group = first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-raw"
X = load(joinpath(fgpath,file_name*".nhdr")).data

feature_name = "WMB-10Xv2-HY"; scale="log2" # WMB-10Xv2-HY(32285×100562)
feature_group = first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-log2"
Xgc = load(joinpath(fgpath,file_name*".nhdr")).data

#========== PCB ===============#
# Initialization
noc = 500 # number of component
nac = 3500 # number of additional component

initmethod = :isvd; initstr = initmethod
fname = joinpath(subworkpath,"$(file_name)_$(initmethod)_noc$(noc)_nac$(nac).jld2")

if !isfile(fname)
    println("calculating init.")
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
    save(fname, "W0",W0,"H0",H0,"Wp",Wp,"Hp",Hp,"M0",M0,"N0",N0,"D",D,"rt1",rt1)
else
    dd = load(fname)
    W0, H0, M0, N0, Wp, Hp, D, rt1 = dd["W0"], dd["H0"], dd["M0"], dd["N0"], dd["Wp"], dd["Hp"], dd["D"], dd["rt1"]
end

if init_hals
    jldfprex = "WMB-10Xv2-HY-raw_hals_noc500_a0.1_mit50"
    dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
    Wp, Hp, iter, rtnnd, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
    LCSVD.balanceWH!(Wp,Hp)
    M0, N0 = (W0'Wp, Hp*H0')
    W, H = W0*M0, N0*H0
    initstr = "inithals"
    fname = joinpath(subworkpath,"$(file_name)_$(initstr)_noc$(noc)_nac$(nac).jld2")
    save(fname, "W0",W0,"H0",H0,"Wp",Wp,"Hp",Hp,"W",W,"H",H,"M0",M0,"N0",N0,"D",D,"rt1",rt1)
end
if init_sbc
    fname = joinpath(subworkpath,"WMB-10Xv2-HY-raw_initsbc_noc500_nac0_Wspar.jld2")
    if !isfile(fname)
        rt11 = @elapsed M0 = sbc(W0)
        rt12 = @elapsed N0 = M0\D
        rt13 = @elapsed LCSVD.balanceWH!(M0, N0)
        rt1 = rt11+rt12+rt12
        # rt11 = @elapsed N0 = sbc(H0')' # this fail to converge
        # rt12 = @elapsed M0 = D/N0
        # rt13 = @elapsed LCSVD.balanceWH!(M0, N0)
        # rt1 = rt11+rt12+rt12
        Wp = W0*M0; Hp = N0*H0
        save(fname, "W0",W0,"H0",H0,"Wp",Wp,"Hp",Hp,"M0",M0,"N0",N0,"D",D,"rt1",rt1)
    else
        dd = load(fname)
        W0, H0, M0, N0, Wp, Hp, D, rt1 = dd["W0"], dd["H0"], dd["M0"], dd["N0"], dd["Wp"], dd["Hp"], dd["D"], dd["rt1"]
    end
    initstr = "initsbc"
end

# PCB solve!
α1 = 0.005; α2 = 0.005  # H sparsity
β1 = 5.0; β2 = 5.0 # W nonnegativity
σ0=std(W0*M0); r=0.3; maxiter = Int(ceil(log(eps(eltype(X)))/log(r))); tol=0 

alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, uselv=false,
    maxiter = maxiter, store_trace = true, store_inner_trace = false, show_trace = true,
    allow_f_increases = true, f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol,
    successive_f_converge=0)
M, N = copy(M0), copy(N0)
rt2 = @elapsed rst = LCSVD.solve!(alg, X, W0, H0, D, M, N);
W, H = rst.W, rst.H
LCSVD.normalizeW!(W,H)
fitval = LCSVD.fitd(X,W*H) # calculate fit value. This takes a while
fname = joinpath(subworkpath,"$(file_name)_$(initstr)_pcb_noc$(noc)_nac$(nac)_r$(r)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(rst.niters).jld2")
save(fname, "W",W,"H",H,"M",M,"N",N,"rt1",rt1,"rt2",rt2,"iter",rst.niters,"fitval",fitval,"traces",rst.traces)

# Plot the result
jldfprexs = ["WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it14",
"WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it13",
"WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac1500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it14",
"WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it12",
"WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac3500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it14"]

jldfprexs = ["WMB-10Xv2-HY-raw_n100562_initnndrsvd_hals_noc500_a0.1_tol0.003_mit100_nrrd",
"WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac3500_aw0.005_ah0.0_bw0.0_bh5.0_tol1.0e-6_it14",
"WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac3500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it14"]

"WMB-10Xv2-HY-raw_initisvd_noc500_nac3500_X",

for jldfprex in jldfprexs
dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
@show norm(X'-W*H)^2, norm(W,1), norm(H,1), rt1, rt1/60/60, rt2, rt2/60/60 
end

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
    save(joinpath(subworkpath,"$(jldfprex)_heatmap_W$i.png"),f)
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
    save(joinpath(subworkpath,"$(jldfprex)_heatmap_H$i.png"),f)
end

α1 = 0.; α2 = 0.005  # H sparsity
β1 = 5.; β2 = 0. # W nonnegativity
σ0=std(W0*M0); r=0.3; maxiter = Int(ceil(log(eps(eltype(X)))/log(r))); tol=0 
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, uselv=false,
    maxiter = maxiter, store_trace = false, store_inner_trace = false, show_trace = true,
    allow_f_increases = true, f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol,
    successive_f_converge=0)

#======== nac vs. W0'W0 =================#
noc = 500
dd = load(joinpath(subworkpath,"old","WMB-10Xv2-HY-raw_hals_noc500_a0.1_mit50.jld2"))
Whals = dd["W"]; m = size(Whals,1)
mxs = Float32[]; nms = Float32[]
for nac = [0,500,1500, 2500, 3500, 5500, 9500]
    nc = noc+nac
    dd = load(joinpath(subworkpath,"old","WMB-10Xv2-HY-raw_isvd_noc500_nac$(nac)_X.jld2"))
    W0 = dd["W0"]
    W0W0T = W0*W0'
    for i = 1:m
        W0W0T[i,i] -= 1.0
    end
    mx = maximum(W0W0T)
    nm = norm(W0W0T)
    push!(mxs,mx)
    push!(nms,nm)
    @show nac, mx, nm
end

#======== How to use Mmap ===============#
using Mmap

# Create a file for mmapping
# (you could alternatively use mmap to do this step, too)
s = open(joinpath(v3path,"X.bin"), "w+")
# We'll write the dimensions of the array as the first two Ints in the file
write(s, size(X,1))
write(s, size(X,2))
# Now write the data
write(s, X)
close(s)

# Test by reading it back in
s = open(joinpath(subworkpath,"X.bin"))   # default is read-only
m = read(s, Int)
n = read(s, Int)
A2 = mmap(s, Matrix{Int}, (m,n))

#========= mmap test by changing memory size ================#
#========= 64GB ===========#
s = open(joinpath(subworkpath,"X.bin"))   # default is read-only
m = read(s, Int)
n = read(s, Int)
Int(Sys.free_memory())/1e9 # 57.58GB
X = mmap(s, Matrix{Float32}, (m,n))
Int(Sys.free_memory())/1e9 # 58.32GB
@time sum(X) # 12.54 sec (1 allocation: 16 bytes)
Int(Sys.free_memory())/1e9 # 54.55GB
@time sum(X) # 1.06 sec (1 allocation: 16 bytes)
X=0 # free memory
GC.gc() 
Int(Sys.free_memory())/1e9 # 55.40GB
X = mmap(s, Matrix{Float32}, (m,n))
Int(Sys.free_memory())/1e9 # 56.45GB
@time sum(X) # 19.45 sec (1 allocation: 16 bytes)
Int(Sys.free_memory())/1e9 # 54.65GB

# Mmap
X=0
GC.gc() 
Int(Sys.free_memory())/1e9 # 67.16GB
X = mmap(s, Matrix{Float32}, (m,n))
W = rand(Float32,size(X,2),500)
Int(Sys.free_memory())/1e9 # 66.77GB
@time X*W # 286.686209 seconds (1.88 M allocations: 188.609 MiB, 0.03% gc time, 0.28% compilation time)
@time X*W # 4.167819 seconds (2 allocations: 61.579 MiB)

# Sparse Array
X=0
GC.gc()
Int(Sys.free_memory())/1e9 # 
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)s
file_name = feature_name*"-raw"
X = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X
Int(Sys.free_memory())/1e9 # 
W = rand(Float32,size(Xraw,1),500)
@time X'*W # 
Int(Sys.free_memory())/1e9 # 

# NRRD
v2path = joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2")
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(32285×100562)
file_name = feature_name*"-raw"
X = load(joinpath(v2path,file_name*".nhdr"))
m,n = size(X)
W = rand(Float32,n,noc)
XW = Matrix{Float32}(undef,m,noc)
@time mul!(XW, X.data, W) # 354.153395 seconds (1.84 M allocations: 124.005 MiB, 0.03% gc time, 0.31% compilation time)
@time mul!(XW, X.data, W) # 4.348448 seconds (1 allocation: 48 bytes)
# For the whole data calculation of @time mul!(XW, X.data, W)
# non-interactive : rt1 = 2096.102594584, rt2 = 1488.236813635
# interactive : rt1 = 6954.527543939, rt2 = 7111.703174235

#============ 4GB ==============#
s = open(joinpath(subworkpath,"X.bin"))   # default is read-only
m = read(s, Int)
n = read(s, Int)
Int(Sys.free_memory())/1e9 # 2.89GB
X = mmap(s, Matrix{Float32}, (m,n))
Int(Sys.free_memory())/1e9 # 2.87GB
@time sum(X) # 15.94 sec (1 allocation: 16 bytes)
Int(Sys.free_memory())/1e9 # 0.0013GB
@time sum(X) # 115.25 sec (1 allocation: 16 bytes)
X=1
GC.gc() 
Int(Sys.free_memory())/1e9 # 0.56GB
X = mmap(s, Matrix{Float32}, (m,n))
Int(Sys.free_memory())/1e9 # 0.98GB
@time sum(X) # 16.58 sec (1 allocation: 16 bytes)
Int(Sys.free_memory())/1e9 # 0.0014GB
@time sum(X) # 225.83 sec (1 allocation: 16 bytes)

# Mmap
X=0
GC.gc()
X = mmap(s, Matrix{Float32}, (m,n))
Int(Sys.free_memory())/1e9 # 0.0014GB
@time X*W # 299.820450 seconds (1.34 M allocations: 152.138 MiB, 0.24% compilation time)
@time X*W # 301.257148 seconds (2 allocations: 61.579 MiB, 0.05% gc time)
W = rand(Float32,n,noc) # 201MB
@time XW = Matrix{Float32}(undef,m,noc) # 64MB, 0.000023 seconds
@time mul!(XW, X, W) # 287.690721 seconds (1.88 M allocations: 126.758 MiB, 0.02% gc time, 0.34% compilation time)
Int(Sys.free_memory())/1e9 # 0.00227GB
@time mul!(XW, X, W) # 285.468311 seconds

# Sparse Array
X=0
GC.gc()
Int(Sys.free_memory())/1e9 # 2.84GB
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)s
file_name = feature_name*"-raw"
Xraw = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X
Int(Sys.free_memory())/1e9 # 0.0002GB
W = rand(Float32,size(Xraw,1),500)
@time sum(Xraw) # 0.11sec
@time Xraw'*W # 424.471320 seconds (675.15 k allocations: 107.040 MiB, 0.02% gc time, 0.05% compilation time)
Int(Sys.free_memory())/1e9 # 0.0014GB

#======== 8GB =====================#
# Sparse Array
noc = 500
GC.gc()
Int(Sys.free_memory())/1e9 # 6.73GB
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)s
file_name = feature_name*"-raw"
Xraw = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X' # 1.4G
m, n = size(Xraw)
Int(Sys.free_memory())/1e9 # 3.94GB
W = rand(Float32,n,noc)
XW = Matrix{Float32}(undef,m,noc)
Int(Sys.free_memory())/1e9 # 3.61GB
@time mul!(XW, Xraw, W)  # 422.428279 seconds (675.15 k allocations: 107.040 MiB, 0.05% compilation time)
Int(Sys.free_memory())/1e9 # 3.62GB
@time mul!(XW, Xraw, W)  # 420.948137 seconds (3 allocations: 61.579 MiB)
Int(Sys.free_memory())/1e9 # 3.55GB

# dense NRRD
Xraw=0
GC.gc()
Int(Sys.free_memory())/1e9 # 6.78GB
v2path = joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2")
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(32285×100562)
file_name = feature_name*"-raw"
X = load(joinpath(v2path,file_name*".nhdr")) # 18GB
Int(Sys.free_memory())/1e9 # 0.2
m,n = size(X)
W = rand(Float32,n,noc)
XW = Matrix{Float32}(undef,m,noc)
Int(Sys.free_memory())/1e9 # 4.5056e-5
@time mul!(XW, X.data, W) # 354.153395 seconds (1.84 M allocations: 124.005 MiB, 0.03% gc time, 0.31% compilation time)
Int(Sys.free_memory())/1e9 # 0.000135168
@time mul!(XW, X.data, W) # 208.565794 seconds (1 allocation: 48 bytes)
Int(Sys.free_memory())/1e9 # 0.001007616

# general sparse
Int(Sys.free_memory())/1e9 #
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)s
file_name = feature_name*"-raw"
Xraw = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad")).X' # 1.4G
Int(Sys.free_memory())/1e9 #
m, n = size(Xraw)
Int(Sys.free_memory())/1e9 #
W = rand(Float32,n,noc)
Int(Sys.free_memory())/1e9 # 
XW = Matrix{Float32}(undef,m,noc)
@time mul!(XW, Xraw, W) # 449.487735291
Int(Sys.free_memory())/1e9 # 
@time mul!(XW, Xraw, W) # 449.487735291
Int(Sys.free_memory())/1e9 # 

Xraw=0
GC.gc()

# general dense NRRD 
Int(Sys.free_memory())/1e9 # 8.59GB
v2path = joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2")
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(32285×100562)
file_name = feature_name*"-raw"
X = load(joinpath(v2path,file_name*".nhdr"))
m,n = size(X)
W = rand(Float32,n,noc)
XW = Matrix{Float32}(undef,m,noc)
@time mul!(XW, X.data, W) # 44.446767994
@time mul!(XW, X.data, W) # 16.782524337


#======== 4GB by changing ncells =============#
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(32285×100562)
file_name = feature_name*"-raw"
rt1s = []; rt2s = []; noc=500; 
for n in 1000:1000:10000
    X=0
    GC.gc()
    X = load(joinpath(v3path,file_name*".nhdr"))
    W = rand(Float32,n,noc)
    m = size(X,1)
    XW = Matrix{Float32}(undef,m,noc)
    rt1 = @elapsed mul!(XW, view(X,:,1:n), W)
    rt2 = @elapsed mul!(XW, view(X,:,1:n), W)
    @show n, rt1, rt2
    push!(rt1s,rt1)
    push!(rt2s,rt2)
end

#========== Required Memory ================#
function RAM_required_PCB(T::Type,m,n,noc,nac; useprecond=false)
    nc = noc+nac
    size = 0
    size += m*nc     # W0
    useprecond && (size += m*nc) # W0spar2
    size += m*noc    # W
    size += nc*n     # H0
    useprecond && (size += nc*n) # H0spar2
    size += noc*n    # H
    size += nc*noc   # Mprev
    size += nc*noc   # M
    size += noc*nc   # Nprev
    size += noc*nc   # N
    size += nc*nc    # MNmD
    size += nc*noc   # W0TW
    size += noc*nc   # bnn
    size += 2*noc*nc # grad
    size += 2*noc*nc # diagH
    size += m*noc*1  # TmpWs[1:1]
    size += noc*n*1  # TmpHs[1:1]
    size += nc*noc*3 # TmpMs[1:3]
    size*sizeof(T) # bytes
end

function RAM_required_HALS(T::Type,m,n,noc)
    noc^2+m*noc+n*noc
end

m,n = size(X)

#===== one iteration =======#
# for whole data
# PCB(nac:2500) : 5074.238936423 sec (for 300 iterations : 17.6days)
# PCB(nac:2500) : 6381.634747722 sec (for 300 iterations : 22.2days)

jldfprex = "WMB-10Xv2-HY-raw_hals_noc500_a0.1_mit50"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
dd = load(fname)
