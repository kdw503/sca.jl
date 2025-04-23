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
colors = AMakie.wong_colors()

using UMAP

dd = load(joinpath(subworkpath,"Xr.jld2"))
label = load(joinpath(subworkpath,"Xr.jld2"),"label")
Xt = dd["Xr"] # Xr (genes, cellss)
label = dd["label"] # labels (cellss, 1)
ulabel = unique(label)
class = map(l->findfirst(isequal(l), ulabel), label)
Xr = rescale(Xt)
save(joinpath(subworkpath,"Xr.jld2"), "Xt", Xt, "Xr", Xr, "label", label, "ulabel", ulabel, "class", class)

rescale(A; dims=2) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())
# X (n_features, n_samples)
n_components = 2; 
n_neighbors=15; metric=Euclidean(); min_dist=0.1
embedding = umap(Xr, n_components; n_neighbors=n_neighbors, metric=metric, min_dist=min_dist)

f = AMakie.Figure(size=(500,500))
ax = AMakie.Axis(f[1, 1], xlabel = "UMAP1", ylabel = "UMAP2", title = "")
for i in 1:length(ulabel)
    class_indices = findall(class .== i)
    @show ulabel[i]
    scatter!(ax, embedding[1,class_indices], embedding[2,class_indices], color = colors[i], markersize = 3, label=ulabel[i])
end
#pcb = scatter!(ax, embedding[1,:], embedding[2,:], color = :green, markersize = 1, label="UMAP") # strokecolor=:black, strokewidth=1, 
#axislegend(ax, position = :rt) # halign = :left, valign = :top
elems = map(i->MarkerElement(color=colors[i], marker=:circle, markersize=5), 1:length(ulabel))
Legend(fig[1, 2], elems, ulabel, "Cell type")
save(joinpath(subworkpath,"UMAP_n$(n_components).png"),f,px_per_unit=2)



using TSne, Statistics

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

alldata, allabels = 
# Normalize the data, this should be done if there are large scale differences in the dataset
X = rescale(data, dims=1);
Y = tsne(X, 2, 50, 1000, 20.0);

f = AMakie.Figure()
ax = AMakie.Axis(f[1, 1], xlabel = "∥W∥₁", ylabel = "Fit", title = "")
pcb = scatter!(ax, Y[:,1], Y[:,2], color = :green, strokecolor=:black, strokewidth=1, markersize = 15, label="PCB")
axislegend(ax, position = :rt) # halign = :left, valign = :top
save(joinpath(subworkpath,"t-sne.png"),f,px_per_unit=2)

