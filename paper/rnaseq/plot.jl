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
download_base = joinpath(datapath,"AllenBrain")
allenbrainversion = "20241130"

#include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"clustering.jl"))
#include(joinpath(workpath,"utils.jl"))

#colormap
# Makie.available_gradients()
# Plasma, Inferno, Magma, Cividis, Jet, grays, heat, :Spectral
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# X data
f = Figure(size=(120,300))
ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
joint_limits = (-0.0, 10)
Xs = Array(X[1:402:100562,1:323:32285])
hm1 = heatmap!(ax, Xs', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
hidedecorations!(ax)
save(joinpath(subworkpath,"$(file_name)_X[250,100].png"),f)

mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# LCSVD
fprefix = "WMB-10Xv2-HY-raw_lcsvd_noc500_aw0.0_ah0.005_bw5.0_bh0.0_it100"
dd = load(joinpath(subworkpath,"$(fprefix).jld2"))
W, H, rt1, rt2, fitval = dd["W"], dd["H"], dd["rt1"], dd["rt2"], dd["fitval"]
noc = size(W,2)
for i in 1:4
    f = Figure()
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    rowsizeq = size(W,1)÷4
    rows = (i==4 ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    joint_limits = (-0.04, 0.04)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_W$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,1000000)))
ht1 = hist!(ax,vec(W),bin=5)
save(joinpath(subworkpath,"$(fprefix)_histo1000000_W.png"),f)
num_blocks = 16
for i in 1:num_blocks
    f = Figure(size=(2200,600))
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (-200, 200) # 1024
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_H$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
ht1 = hist!(ax,vec(H),bin=5)
save(joinpath(subworkpath,"$(fprefix)_histo100_H.png"),f)



# HALS
fprefix = "WMB-10Xv2-HY-raw_hals_noc500_a0.1_iter100"
dd = load(joinpath(subworkpath,"$(fprefix).jld2"))
W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
now = size(W,2)
for i in 1:4
    f = Figure()
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    rowsizeq = size(W,1)÷4
    rows = (i==4 ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    @show rows
    joint_limits = (-0.08, 0.08)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)                     # These three
    # scatter!(ax, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_W$i.png"),f)
end
num_blocks = 16
for i in 1:num_blocks
    f = Figure(size=(2200,600))
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (-200, 200) # 1591
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)                     # These three
    # scatter!(ax, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
    save(joinpath(subworkpath,"$(fprefix)_heatmap_H$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
ht1 = hist!(ax,vec(H),bin=5)
save(joinpath(subworkpath,"$(fprefix)_histo100_H.png"),f)

# AllenBrain UMAP
cd(subworkpath); Pkg.activate(".")

# WMB-10Xv2-HY data
using AllenBrain, FileIO, Muon
using CSVFiles, CSV, DataFrames
version = "20241130" # "20230630"
manifest = awsmanifest(version)

expression_matrices = manifest.file_listing["WMB-10Xv2"]["expression_matrices"]
feature_matrix_label = "WMB-10Xv2-HY" # "WMB-10Xv2-TH"(131212×32285)
rpath = expression_matrices[feature_matrix_label]["log2"]["files"]["h5ad"]["relative_path"]
local_path = joinpath(download_base, split(rpath,"/")... )
AllenBrain.download_dir(manifest, rpath, local_path)
# Load .h5ad file
adata = load(local_path)

# WMB-10X annotation data
ann_rpath = manifest.file_listing["WMB-10X"]["metadata"]["cell_metadata_with_cluster_annotation"]["files"]["csv"]["relative_path"]
ann_local_path = joinpath(download_base, split(ann_rpath,"/")... )
AllenBrain.download_dir(manifest, ann_rpath, ann_local_path)
ldata = CSV.read(ann_local_path, DataFrame)

# choose only for WMB-10Xv2-HY (version : "20230630" no 'feature_matrix_label' field)
ldata.cell_label # cell_label in cell annotation data
adata.obs_names # cell_label in gene expression data
i = 0; annindices = Int[]; keeping_indices = Int[]
for (i,cl) in enumerate(adata.obs_names)
    if cl in ldata.cell_label
        idx = findfirst(s->s==cl,ldata.cell_label)
        push!(annindices,idx)
#        @show i, ldata.library_method[idx], ldata.anatomical_division_label[idx], ldata.cell_label[idx], ldata.cluster_alias[idx]
        push!(keeping_indices,i)
    else
        @show i, cl
    end
end
ldata.anatomical_division_label[ ldata.anatomical_division_label.=="HY" .&& ldata.library_method .== "10Xv2"]

# check if cell_label in adata and ldata are same
save(joinpath(download_base,version,"WMB-10Xv2-HY_cell_label.jld2"), "annindices", annindices)
annindices = load(joinpath(download_base,version,"WMB-10Xv2-HY_cell_label.jld2"), "annindices")
cell_lable = String.(ldata.cell_label[annindices])
class = String.(ldata.class[annindices]) # "20230630"27, "20241130"
subclass = String.(ldata.subclass[annindices]) # "20230630"205, "20241130"
supertype = String.(ldata.supertype[annindices] )# "20230630"569, "20241130"
cluster_alias = ldata.cluster_alias[annindices] # "20230630"1809, "20241130", class>subclass>supertype>cluster
x = ldata.x[annindices]
y = ldata.y[annindices]
class_color = String.(ldata.class_color[annindices])
subclass_color = String.(ldata.subclass_color[annindices])
supertype_color = String.(ldata.supertype_color[annindices])
cluster_color = String.(ldata.cluster_color[annindices])# ccolors = parse.(RGB, cluster_color)
cls_cpairs = map((ucls,uclr)->ucls=>uclr, unique(class), unique(class_color))
scls_cpairs = map((ucls,uclr)->ucls=>uclr, unique(subclass), unique(subclass_color))
styp_cpairs = map((ucls,uclr)->ucls=>uclr, unique(supertype), unique(supertype_color))
clst_cpairs = map((ucls,uclr)->ucls=>uclr, unique(cluster_alias), unique(cluster_color))
save(joinpath(download_base,allenbrainversion,"WMB-10Xv2-HY_annotation.jld2"),
        "annindices", annindices, "keeping_indices", keeping_indices,
        "class", class, "class_color", class_color, "subclass", subclass,
        "subclass_color", subclass_color, "supertype", supertype, "supertype_color", supertype_color,
        "cluster_alias", cluster_alias, "cluster_color", cluster_color, "cls_cpairs", cls_cpairs,
        "scls_cpairs", scls_cpairs, "styp_cpairs", styp_cpairs, "clst_cpairs", clst_cpairs,
        "cell_lable", cell_lable, "x", x, "y", y)


#========= AllenBrain annotation data ==========#
using AllenBrain, FileIO, Muon

pp="log2"; method = "GT"; clnoc = 28
optionstr = "$(pp)_$(method)_cn$(clnoc)"
fprex = "WMB-10Xv2-HY-$(optionstr)_class"

dd = load(joinpath(download_base,allenbrainversion,"WMB-10Xv2-HY_annotation.jld2"))
annindices = dd["annindices"]; keeping_indices = dd["keeping_indices"]
cell_lable = dd["cell_lable"]; x = dd["x"]; y = dd["y"]
class = dd["class"]; class_color = dd["class_color"]; cls_cpairs = dd["cls_cpairs"]
subclass = dd["subclass"]; subclass_color = dd["subclass_color"]; scls_cpairs = dd["scls_cpairs"]
supertype = dd["supertype"]; supertype_color = dd["supertype_color"]; styp_cpairs = dd["styp_cpairs"]
cluster_alias = dd["cluster_alias"]; cluster_color = dd["cluster_color"]; clst_cpairs = dd["clst_cpairs"]

cm = countmap(class)
dclasses = getindex.(Vec(sort(reverse_dict(cm),order=Base.Reverse)...)[1:14],2)
dcolors = map(k->Dict(cls_cpairs)[k], dclasses)

# UMAP
f = AMakie.Figure(size=(800, 600))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
pcb = scatter!(ax, x, y, color = class_color, markersize = 3, label="class")
elems = map(c->MarkerElement(color = c, marker = :circle, markersize = 15), dcolors) # marker = 'π', points = Point2f[(0.2, 0.2), (0.5, 0.8), (0.8, 0.2)])
Legend(f[1,2], elems, dclasses, rowgap = 5)
# scatter!(ax, [(x, y) for x in sws for y in fits], color=:red, strokecolor=:black, strokewidth=1)
# axislegend(ax, position = :rt) # halign = :left, valign = :top
save(joinpath(subworkpath,allenbrainversion,fprex*"_wLegend.png"),f,px_per_unit=2)

# cell heatmap with HALS data
fname = joinpath(subworkpath,allenbrainversion,"WMB-10Xv2-HY-log2_hals_noc500_a0.1_iter100.jld2")
ddhals = load(fname)
cells = ddhals["H"][:,keeping_indices] # 28×99879
cellsclass, classboundries = permute_cells(class_color,cells)
fprexhm = joinpath(subworkpath,allenbrainversion,fprex*"_cells_hals_heatmap")
clustered_hitmap(cellsclass, classboundries, mycmap, fprexhm; qlevel=0.999)

# cell heatmap with PCB data
fname = joinpath(subworkpath,allenbrainversion,"WMB-10Xv2-HY-log2_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it11.jld2") # sp_nn
ddpcb = load(fname)
cells = ddpcb["H"][:,keeping_indices] # 28×99879
cellsclass, classboundries = permute_cells(class_color,cells)
fprexhm = joinpath(subworkpath,allenbrainversion,fprex*"_cells_pcb_sp_nn_heatmap")
clustered_hitmap(cellsclass, classboundries, mycmap, fprexhm; qlevel=0.999)

#======== PCB annotation data ===========#
pp="log2"; method = "pcb_sp_nn"; clnoc = 28; normalization = :true; linkage = :average
optionstr = "$(pp)_$(method)_cn$(clnoc)_$(normalization)_$(linkage)"
fprex = "WMB-10Xv2-HY-$(optionstr)_class"
clustnpcb = load(joinpath(subworkpath,allenbrainversion,fprex*".jld2"),"clustn")
clust_label_pcb = assign_celltypes(clustnpcb[keeping_indices], class)
d = Dict(cls_cpairs)
clust_label_pcb_color =  map(l->l∈keys(d) ? d[l] : "#505050", clust_label_pcb)

# UMAP
f = AMakie.Figure(size=(600, 600))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
pcb = scatter!(ax, x, y, color = clust_label_pcb_color, markersize = 3, label="class")
# axislegend(ax, position = :rt) # halign = :left, valign = :top
save(joinpath(subworkpath,allenbrainversion,fprex*".png"),f,px_per_unit=2)

# cell heatmap
#fname = joinpath(subworkpath,allenbrainversion,"WMB-10Xv2-HY-log2_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw0.0_bh0.0_tol1.0e-6_it13.jld2") # sp
fname = joinpath(subworkpath,allenbrainversion,"WMB-10Xv2-HY-log2_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it11.jld2") # sp_nn
ddpcb = load(fname)
cells = ddpcb["H"][:,keeping_indices] # 28×99879
cellsclass, classboundries = permute_cells(clustnpcb[keeping_indices],cells)
fprexhm = joinpath(subworkpath,allenbrainversion,fprex*"_cells_heatmap")
clustered_hitmap(cellsclass, classboundries, mycmap, fprexhm; qlevel=0.999)

#========== SVD annotation data ==========#
pp="log2"; method = "svd"; clnoc = 28; normalization = :true; linkage = :average
optionstr = "$(pp)_$(method)_cn$(clnoc)_$(normalization)_$(linkage)"
fprex = "WMB-10Xv2-HY-$(optionstr)_class"
clustnsvd = load(joinpath(subworkpath,allenbrainversion,fprex*".jld2"),"clustn")
clust_label_svd = assign_celltypes(clustnsvd[keeping_indices], class)
d = Dict(cls_cpairs)
clust_label_svd_color =  map(l->l∈keys(d) ? d[l] : "#505050", clust_label_svd)

# UMAP
f = AMakie.Figure(size=(600, 600))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
svd = scatter!(ax, x, y, color = clust_label_svd_color, markersize = 3, label="class")
save(joinpath(subworkpath,allenbrainversion,fprex*".png"),f,px_per_unit=2)

# cell heatmap
fname = joinpath(subworkpath,allenbrainversion,"WMB-10Xv2-HY-log2_initisvd_noc500_X.jld2")
ddsvd = load(fname)
cells = (ddsvd["D"]*ddsvd["V"]')[:,keeping_indices] # 28×99879
cellsclass, classboundries = permute_cells(clustnsvd[keeping_indices],cells)
fprexhm = joinpath(subworkpath,allenbrainversion,fprex*"_cells_heatmap")
clustered_hitmap(cellsclass, classboundries, mycmap, fprexhm; qlevel=0.999)

#============= HALS annotation data ===========#
pp="log2"; method = "hals"; clnoc = 28; normalization = :true; linkage = :average
optionstr = "$(pp)_$(method)_cn$(clnoc)_$(normalization)_$(linkage)"
fprex = "WMB-10Xv2-HY-$(optionstr)_class"
ddhls = load(joinpath(subworkpath,allenbrainversion,fprex*".jld2"))
resultnhls = ddhls["resultn"]
# clustnhls = ddhls["clustn"]
clnoc = 28
optionstr = "$(pp)_$(method)_cn$(clnoc)_$(normalization)_$(linkage)"
fprex = "WMB-10Xv2-HY-$(optionstr)_class"
clustnhls = cutree(resultnhls; k=clnoc, h=nothing)
maxval = maximum(values(countmap(clustnhls)))
clust_label_hls = assign_celltypes(clustnhls[keeping_indices], class)
d = Dict(cls_cpairs)
clust_label_hls_color =  map(l->l∈keys(d) ? d[l] : "#505050", clust_label_hls)

# UMAP
f = AMakie.Figure(size=(600, 600))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
hls = scatter!(ax, x, y, color = clust_label_hls_color, markersize = 3, label="class")
save(joinpath(subworkpath,allenbrainversion,fprex*".png"),f,px_per_unit=2)

# cell heatmap
fname = joinpath(subworkpath,allenbrainversion,"WMB-10Xv2-HY-log2_hals_noc500_a0.1_iter100.jld2")
ddhals = load(fname)
cells = ddhals["H"][:,keeping_indices] # 28×99879
cellsclass, classboundries = permute_cells(clustnhls[keeping_indices],cells)
fprexhm = joinpath(subworkpath,allenbrainversion,fprex*"_cells_heatmap")
clustered_hitmap(cellsclass, classboundries, mycmap, fprexhm; qlevel=0.999)

