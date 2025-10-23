using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","spca")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using UMAP, ColorSchemes
using TSne, Statistics

function cont_colors(name::Symbol, n=256)
    get(ColorSchemes.colorschemes[name], range(0, 1; length=n))
end
# colors = AMakie.wong_colors() # has only 7 colors
colors = cont_colors(:hsv, 9); colors[3]=RGB(0.7,0.7,0.0); colors[4]=RGB(0.0,0.8,0.0); colors[9]=RGB(0.5,0.0,0.2)
push!(colors, RGB(0.5,0.5,0.5)) # gray

rescale(A; dims=2) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())
normalizewhole(A) = A ./ norm.(eachcol(A))'

dataset = "Baron"
dd = load(joinpath(subworkpath,dataset,"Xr_$(dataset).jld2"))
Xt = dd["Xr"] # Xr (genes, cellss)
label = dd["label"] # labels (cellss, 1)
ulabel = unique(label)
class = map(l->findfirst(isequal(l), ulabel), label)
Xr = rescale(Xt)
Xn = normalizewhole(Xt)
X = Array(Xt')
# Normalize the data, this should be done if there are large scale differences in the dataset
resX = rescale(X, dims=1);
# save(joinpath(subworkpath,"Xr.jld2"), "Xt", Xt, "Xr", Xr, "label", label, "ulabel", ulabel, "class", class)

#dd = load(joinpath(subworkpath,"Baron","aftr_nor_bootstrap_p0.001_nr100_ne10.jld2"))
#dd = load(joinpath(subworkpath,"Baron","aftr_nor_hclust_noc11_lkgcomplete_hnothing_ne1.jld2"))
#dd = load(joinpath(subworkpath,"Muraro","aftr_nor_hclust_noc9_lkgaverage_hnothing_ne1.jld2"))
#dd = load(joinpath(subworkpath,"Segerstolpe","aftr_nor_hclust_noc9_lkgaverage_hnothing_ne1.jld2"))
#dd = load(joinpath(subworkpath,"Xin","aftr_nor_hclust_noc6_lkgaverage_hnothing_ne1.jld2"))
dd = load(joinpath(subworkpath,dataset,"aftr_nor_hclust_noc9_lkgaverage_hnothing_ne1_090925.jld2"))
clust_pcb = dd["clustsn"][1]; clust_label_pcb = assign_celltypes(clust_pcb, label)
clust_tsvd = dd["clustsn_tsvd"][1]; clust_label_tsvd = assign_celltypes(clust_tsvd, label)
clust_sma = dd["clustsn_sma"][1]; clust_label_sma = assign_celltypes(clust_sma, label)
clust_hals = dd["clustsn_hals"][1]; clust_label_hals = assign_celltypes(clust_hals, label)
label = dd["label"]
label_counts = dd["label_counts"]

# input X (n_features, n_samples)
n_components = 2; n_neighbors=15
metric=Euclidean(); min_dist=0.1
for n_neighbors in 5:5:15
    for min_dist in 0.1:0.1:0.6
        @show n_neighbors, min_dist
        embedding = umap(Xn, n_components; n_neighbors=n_neighbors, metric=metric, min_dist=min_dist)

        f = AMakie.Figure(size=(500,320))
        ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
        for i in 1:length(ulabel)
            class_indices = findall(class .== i)
            scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
        end
        #pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1,
        #axislegend(ax, position = :rt) # halign = :left, valign = :top
        elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
        push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
        Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
        save(joinpath(subworkpath,dataset,"UMAP_Xn_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).png"),f,px_per_unit=2)
        save(joinpath(subworkpath,dataset,"UMAP_Xn_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).jld2"),"embedding",embedding)
    end
end

# input X (n_features, n_samples)
n_components = 2; n_neighbors=15
metric=Euclidean(); min_dist=0.1
for n_neighbors in 5:5:15
    for min_dist in 0.1:0.1:0.6
        @show n_neighbors, min_dist
        embedding = umap(Xn, n_components; n_neighbors=n_neighbors, metric=metric, min_dist=min_dist)

        f = AMakie.Figure(size=(500,320))
        ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
        for i in 1:length(ulabel)
            class_indices = findall(class .== i)
            scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
        end
        #pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1, 
        #axislegend(ax, position = :rt) # halign = :left, valign = :top
        elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
        push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
        Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
        save(joinpath(subworkpath,dataset,"UMAP_Xn_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).png"),f,px_per_unit=2)
        save(joinpath(subworkpath,dataset,"UMAP_Xn_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).jld2"),"embedding",embedding)
    end
end
#embedding = load(joinpath(subworkpath,"Baron","UMAP_Xn_ncom2_nnbr10_md0.4.jld2"), "embedding")
#embedding = load(joinpath(subworkpath,"Muraro","UMAP_Xn_ncom2_nnbr15_md0.4.jld2"), "embedding")
#embedding = load(joinpath(subworkpath,"Segerstolpe","UMAP_Xn_ncom2_nnbr5_md0.6.jld2"), "embedding")
#embedding = load(joinpath(subworkpath,"Xin","UMAP_Xn_ncom2_nnbr5_md0.6.jld2"), "embedding")
embedding = load(joinpath(subworkpath,dataset,"UMAP_Xn_ncom2_nnbr10_md0.4.jld2"), "embedding")


# PCB
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
class_indices = length.(clust_label_pcb).<2
scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_pcb .== lbl)
    scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "UMAP_$(dataset)_Xn_PCB_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).png"),f,px_per_unit=2)

# TSVD
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
class_indices = length.(clust_label_tsvd).<2
scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_tsvd .== lbl)
    scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "UMAP_$(dataset)_Xn_TSVD_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).png"),f,px_per_unit=2)

# SMA
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
class_indices = length.(clust_label_sma).<2
scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_sma .== lbl)
    scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "UMAP_$(dataset)_Xn_SMA_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).png"),f,px_per_unit=2)

# HALS
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
class_indices = length.(clust_label_hals).<2
scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_hals .== lbl)
    scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel));
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "UMAP_$(dataset)_Xn_HALS_ncom$(n_components)_nnbr$(n_neighbors)_md$(min_dist).png"),f,px_per_unit=2)

#======================= T-SNE ==========================#

# Input X (n_features, n_samples). (X, ndim, reduce_dims, max_iter, perplexit; [keyword arguments])
ndim = 2; max_iter = 1000
reduce_dims = 9; perplexity = 30 # Segerstolpe
for reduce_dims in [7, 8, 9]#8:12
    for perplexity in 25:5:40
        @show reduce_dims, perplexity
        tembedding = tsne(resX, ndim, reduce_dims, max_iter, perplexity; progress=true);

        f = AMakie.Figure(size=(500,320))
        ax = AMakie.Axis(f[1, 1], xlabel = "t-sne1", ylabel = "t-sne2", title = "")
        for i in 1:length(ulabel)
            class_indices = findall(class .== i)
            @show ulabel[i]
            scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
        end
        #pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
        #axislegend(ax, position = :rt) # halign = :left, valign = :top
        elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
        push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
        Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
        save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_GT_rdim$(reduce_dims)_pxty$(perplexity).png"),f,px_per_unit=2)
        save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_GT_rdim$(reduce_dims)_pxty$(perplexity).jld2"),"tembedding",tembedding)
    end
end

reduce_dims = 9; perplexity = 30 # Segerstolpe
for i in 2:10
tembedding = tsne(resX, ndim, reduce_dims, max_iter, perplexity; progress=true);

f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "t-sne1", ylabel = "t-sne2", title = "")
for i in 1:length(ulabel)
    class_indices = findall(class .== i)
    @show ulabel[i]
    scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_GT_rdim$(reduce_dims)_pxty$(perplexity)_$i.png"),f,px_per_unit=2)
save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_GT_rdim$(reduce_dims)_pxty$(perplexity)_$i.jld2"),"tembedding",tembedding)
end
tembedding = load(joinpath(subworkpath, dataset, "TSNE_Segerstolpe_resX_GT_rdim9_pxty30_3.jld2"),"tembedding")

#tembedding = load(joinpath(subworkpath,"Xin","TSNE_Xin_resX_GT_rdim9_pxty30.jld2"), "tembedding")
tembedding = load(joinpath(subworkpath,dataset,"TSNE_Xin_resX_GT_rdim9_pxty30.jld2"), "tembedding")

# ndim = 2; reduce_dims = 11; max_iter = 1000; perplexity = 20.0 # Baron
ndim = 2; reduce_dims = 9; max_iter = 1000; perplexity = 40.0 # Muraro
resX = rescale(X, dims=1);
tembedding = tsne(resX, ndim, reduce_dims, max_iter, perplexity; progress=true);
save(joinpath(subworkpath, dataset, "tsne_all_090925.jld2"), "clust_pcb", clust_pcb, "clust_tsvd", clust_tsvd,
        "clust_sma", clust_sma, "clust_hals", clust_hals, "tembedding", tembedding)
dd = load(joinpath(subworkpath, dataset, "tsne_all.jld2"))
tembedding = dd["tembedding"]
clust_pcb = dd["clust_pcb"]
clust_tsvd = dd["clust_tsvd"]
clust_sma = dd["clust_sma"]
clust_hals = dd["clust_hals"]

f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "t-sne1", ylabel = "t-sne2", title = "")
for i in 1:length(ulabel)
    class_indices = findall(class .== i)
    @show ulabel[i]
    scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_GT_rdim$(reduce_dims)_pxty$(perplexity).png"),f,px_per_unit=2)

# PCB
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "t-sne1", ylabel = "t-sne2", title = "")
class_indices = length.(clust_label_pcb).<3
scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_pcb .== lbl)
    scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_PCB_rdim$(reduce_dims).png"),f,px_per_unit=2)

# TSVD
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "t-sne1", ylabel = "t-sne2", title = "")
class_indices = length.(clust_label_tsvd).<3
scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_tsvd .== lbl)
    scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_TSVD_rdim$(reduce_dims).png"),f,px_per_unit=2)

# SMA
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "t-sne1", ylabel = "t-sne2", title = "")
class_indices = length.(clust_label_sma).<3
scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_sma .== lbl)
    scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel))
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_SMA_rdim$(reduce_dims).png"),f,px_per_unit=2)

# HALS
f = AMakie.Figure(size=(500,320))
ax = AMakie.Axis(f[1, 1], xlabel = "t-sne1", ylabel = "t-sne2", title = "")
class_indices = length.(clust_label_hals).<3
scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[10], markersize = 3, label="N/A")
for (i,lbl) in enumerate(ulabel)
    class_indices = findall(clust_label_hals .== lbl)
    scatter!(ax, tembedding'[1,class_indices], tembedding'[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=10), 1:length(ulabel));
push!(elems,MarkerElement(color=colors[10], marker=:circle, markersize=10))
Legend(f[1, 2], elems, [ulabel...,"N/A"], "Cell type")
save(joinpath(subworkpath, dataset, "TSNE_$(dataset)_resX_HALS_rdim$(reduce_dims).png"),f,px_per_unit=2)
