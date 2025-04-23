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

using NeighborhoodClustering, StatsBase, RCall

R"""
library(scRNAseq)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)
library(epca)
library(Matrix) # as.matrix
library(ggplot2)
"""

dd = load(joinpath(subworkpath,"Baron","Result_sp_s0321.jld2"))
X, label, genename = dd["X"], dd["cell_type_label"], dd["gene_name"]
Wpcb, Htpcb, Wsma, Hsma = dd["Wpcb"], dd["Htpcb"], dd["Wsma"], dd["Hsma"]
dd = load(joinpath(subworkpath,"Baron","Result_tsvd.jld2"))
Wtsvd, Httsvd = dd["Wtsvd"], dd["Httsvd"]

dd = load(joinpath(subworkpath,"Baron","Result_sp_nn_s0321.jld2"))
X, label, genename = dd["X"], dd["cell_type_label"], dd["gene_name"]
Wpcb, Htpcb, Wsma, Hsma = dd["Wpcb"], dd["Htpcb"], dd["Wsma"], dd["Hsma"]

dd = load(joinpath(subworkpath,"Baron","Result_hals.jld2"))
Whals, Hthals = dd["Whals"], dd["Hthals"]

label_counts = [958, 2525, 601, 284, 1077, 2326, 255, 252, 173]

# Clustering
pvalue = 0.0001;
clust = cluster(Wpcb', pvalue)

# Bootstrap clustering
pvalue=0.0000001; nresample = 5
for pvalue in [0.000001, 0.0000001]
    for nresample in [3]
        clust = cluster_resample(Wpcb', nresample, pvalue)
        save(joinpath(subworkpath,"Baron","Clustering_p$(pvalue)_n$(nresample)_sp_nn_s0322.jld2"),"clust", clust, "pvalue", pvalue, "nresample", nresample)
        clust_sma = cluster_resample(Wsma', nresample, pvalue)
        @show maximum(clust), maximum(clust_sma)
    end
end
countmap(clust)
countmap(label) # Ground truth

function gridsearch_params(W,label,pvalrng,nsamprng)
    for pvalue in pvalrng
        for nresample in nsamprng
            clust = cluster_resample(W', nresample, pvalue)
            celltypes = []
            for i in 1:maximum(clust)
                indices = clust .== i
                celltype = String(StatsBase.mode(label[indices])) # most frequent cell type
                cmap = countmap(label[indices])
                push!(celltypes, celltype)
            #   @show i, celltype#, cmap
            end
            @show pvalue, nresample, maximum(clust), length(unique(celltypes)) # (1.0e-7, 10, 45, 8)
        end
    end
end

function assign_celltypes_old(clust, label) # needs ground truth
    celltypes = String[]
    maxclust = maximum(clust)
    for i in 1:maxclust
        indices = clust .== i
        celltype = any(indices) ? String(StatsBase.mode(label[indices])) : "0" # most frequent cell type
#        @show i, celltype
        cmap = countmap(label[indices])
        push!(celltypes, celltype)
    #   @show i, celltype#, cmap
    end
    #@show maxclust, length(unique(celltypes))

    clust_label = String[]
    for i in 1:length(clust)
        push!(clust_label, celltypes[clust[i]])
    end
    clust_label
end

function assign_celltypes(clust, label) # needs ground truth
    maxclust = maximum(clust)
    celltypes = fill(string(),maxclust); cellcnts = fill(0,maxclust)
    for i in 1:maxclust
        indices = clust .== i
        ftbl = countmap(label[indices])
        celltype = any(indices) ? String(StatsBase.mode(label[indices])) : "0" # most frequent cell type
        cnt = ftbl[celltype]
        if celltype ∈ celltypes
            fidx = findfirst(==(celltype), celltypes)
            if cnt > cellcnts[fidx]
                cellcnts[fidx] = 0; celltypes[fidx] = "$(i)"
                cellcnts[i] = cnt; celltypes[i] = celltype
            else
                cellcnts[i] = cnt; celltypes[i] = "$(i)"
            end
        else
            cellcnts[i] = cnt; celltypes[i] = celltype
        end
    end
    #@show maxclust, length(unique(celltypes))

    clust_label = String[]
    for i in 1:length(clust)
        push!(clust_label, celltypes[clust[i]])
    end
    clust_label
end

function cal_precision_recall(clust_label, label)
    precisions = [] # tp/(tp+fp)
    recalls = [] # tp/(tp+fn)
    tps = []; fps = []; fns = []; tns = []
    ulabel = unique(label)
    for i in 1:length(ulabel)
        tp = 0; fp = 0; fn = 0; tn = 0
        for j in 1:length(clust_label)
            if clust_label[j] == ulabel[i]
                if clust_label[j] == label[j]
                    tp += 1
                else
                    fp += 1
                end
            elseif label[j] == ulabel[i]
                fn += 1
            else
                tn += 1
            end
        end
        push!(tps, tp); push!(fps, fp); push!(fns, fn); push!(tns, tn)
        precision = (tp+fp) != 0 ? tp/(tp+fp) : 0
        recall = tp/(tp+fn)
        push!(precisions, precision)
        push!(recalls, recall)
    end
    # tp, fp, fn, tn = sum(tps), sum(fps), sum(fns), sum(tns)
    # @show tp, fp, fn, tn
    # precision = (tp+fp) != 0 ? tp/(tp+fp) : 0
    # recall = tp/(tp+fn)
    precision = sum(precisions)/length(ulabel)
    recall = sum(recalls)/length(ulabel)
    return precisions, recalls, precision, recall
end

function plot_qm3(qm, qm_tsvd, qm_sma, qmstd, qmstd_tsvd, qmstd_sma, label; ylabel="", d=0.31)
    ulabel = unique(label)
    colors = Makie.wong_colors()
    f = Figure(size=(600,250)) # f.scene.viewport.val.widths
    ax = AMakie.Axis(f[1, 1], xlabel = "cell type", ylabel = ylabel, xticklabelrotation = pi/8,
                    xticks = (1:length(ulabel), ulabel), title = "")
    v = []; lerrs =[]; herrs =[]
    map(i->(push!(v, qm[i]);push!(v, qm_tsvd[i]);push!(v, qm_sma[i])), 1:length(qm))
#    map(i->(push!(lerrs, -qmstd[i]);push!(lerrs, -qmstd_tsvd[i]);push!(lerrs, -qmstd_sma[i])), 1:length(qmstd))
    map(i->(push!(herrs, qmstd[i]);push!(herrs, qmstd_tsvd[i]);push!(herrs, qmstd_sma[i])), 1:length(qmstd))
    tbl = (cell = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9],
        errx = [-d,0,d,-d,0,d,-d,0,d,-d,0,d,-d,0,d,-d,0,d,-d,0,d,-d,0,d,-d,0,d,],
        value = v,
        lerrors = lerrs,
        herrors = herrs,
        grp = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]
        )
    barplot!(ax,tbl.cell, tbl.value, strokewidth = 0.5, gap=0.1, width=1, # bar_labels = :y,
        dodge = tbl.grp, # stack = tbl.grp,
        color = colors[tbl.grp])
    errorbars!(ax, tbl.cell+tbl.errx, tbl.value, tbl.herrors, whiskerwidth = 6, direction=:y, color=:black)
    # crossbar!(ax,tbl.cell, tbl.value, - tbl.lerrors, + tbl.herrors; dodge = tbl.grp, color = :black)
    labels = ["PCB", "TSVD", "SMA"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:3]
    title = ""
    Legend(f[1,2], elements, labels, title)
    f
end
# f = plot_qm3(rec_means, rec_means_tsvd, rec_means_sma, rec_stds, rec_stds_tsvd, rec_stds_sma, label; ylabel="recalls")

function plot_qm4(qm, qm_tsvd, qm_sma, qm_hals, qmstd, qmstd_tsvd, qmstd_sma, qmstd_hals, label, label_counts; ylabel="", d=0.12)
    ulabel = unique(label)
    xtick_labels = map((s1,s2)->s1*"($s2)",ulabel,label_counts)
    colors = Makie.wong_colors()
    f = Figure(size=(600,250)) # f.scene.viewport.val.widths
    ax = AMakie.Axis(f[1, 1], xlabel = "cell type (cell count)", ylabel = ylabel, xticklabelrotation = pi/8,
                    xticks = (1:length(xtick_labels), xtick_labels), title = "")
    v = []; lerrs =[]; herrs =[]
    map(i->(push!(v, qm[i]);push!(v, qm_tsvd[i]);push!(v, qm_sma[i]);push!(v, qm_hals[i])), 1:length(qm))
#    map(i->(push!(lerrs, -qmstd[i]);push!(lerrs, -qmstd_tsvd[i]);push!(lerrs, -qmstd_sma[i])), 1:length(qmstd))
    map(i->(push!(herrs, qmstd[i]);push!(herrs, qmstd_tsvd[i]);push!(herrs, qmstd_sma[i]);push!(herrs, qmstd_hals[i])), 1:length(qmstd))
    tbl = (cell = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9],
        errx = [-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d],
        value = v,
        lerrors = lerrs,
        herrors = herrs,
        grp = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
        )
    barplot!(ax,tbl.cell, tbl.value, strokewidth = 0.5, gap=0.1, width=1, # bar_labels = :y,
        dodge = tbl.grp, # stack = tbl.grp,
        color = colors[tbl.grp])
    errorbars!(ax, tbl.cell+tbl.errx, tbl.value, tbl.herrors, whiskerwidth = 6, direction=:y, color=:black)
    # crossbar!(ax,tbl.cell, tbl.value, - tbl.lerrors, + tbl.herrors; dodge = tbl.grp, color = :black)
    labels = ["PCB", "TSVD", "SMA", "HALS"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:4]
    title = ""
    Legend(f[1,2], elements, labels, title)
    f
end

function plot_qm4(qm, qm_tsvd, qm_sma, qm_hals, qmstd, qmstd_tsvd, qmstd_sma, qmstd_hals, label, label_counts; ylabel="", d=0.12)
    ulabel = unique(label)
    colors = Makie.wong_colors()
    f = Figure(size=(600,250)) # f.scene.viewport.val.widths
    ax = AMakie.Axis(f[1, 1], xlabel = "cell type", ylabel = ylabel, xticklabelrotation = pi/8,
                    xticks = (1:length(ulabel), ulabel), title = "")
    ax2 = AMakie.Axis(f[1, 1], xticks = (1:1:9, string.(label_counts)), xaxisposition = :top, yticks = (1:3,["","",""]), yticksvisible = false, ygridvisible = false,
                    xgridvisible = false, title = "")
#    hidexdecorations!(ax2)#, ticklabels = false)
    v = []; lerrs =[]; herrs =[]
    map(i->(push!(v, qm[i]);push!(v, qm_tsvd[i]);push!(v, qm_sma[i]);push!(v, qm_hals[i])), 1:length(qm))
#    map(i->(push!(lerrs, -qmstd[i]);push!(lerrs, -qmstd_tsvd[i]);push!(lerrs, -qmstd_sma[i])), 1:length(qmstd))
    map(i->(push!(herrs, qmstd[i]);push!(herrs, qmstd_tsvd[i]);push!(herrs, qmstd_sma[i]);push!(herrs, qmstd_hals[i])), 1:length(qmstd))
    tbl = (cell = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9],
        errx = [-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d,-3d,-d,d,3d],
        value = v,
        lerrors = lerrs,
        herrors = herrs,
        grp = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
        )
    barplot!(ax,tbl.cell, tbl.value, strokewidth = 0.5, gap=0.1, width=1, # bar_labels = :y,
        dodge = tbl.grp, # stack = tbl.grp,
        color = colors[tbl.grp])
    errorbars!(ax, tbl.cell+tbl.errx, tbl.value, tbl.herrors, whiskerwidth = 6, direction=:y, color=:black)
    # crossbar!(ax,tbl.cell, tbl.value, - tbl.lerrors, + tbl.herrors; dodge = tbl.grp, color = :black)
    labels = ["PCB", "TSVD", "SMA", "HALS"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:4]
    title = ""
    Legend(f[1,2], elements, labels, title)
    f
end

function clustring_experi(method, Worg, label, label_counts; normalization=true, noc=9, nepmt=20,
        bs_nresample=10, bs_pvalue=1e-3, km_maxiter=100, ds_radius=40, ds_min_ngbr=3, ds_min_clsize = 3,
        hc_h=300, gc_n_classes=40)
    l = size(Worg,1)
    W = copy(Worg)
    if normalization
        for r in eachrow(W)
            n = norm(r)
            r ./= n
        end
    end
    precisionss=[]; recallss=[]; avg_precs=[]; avg_recs=[]; clusts=[]
    for i in 1:nepmt
        @show i
        if method == :bootstrap
            clust = cluster_resample(W', bs_nresample, bs_pvalue)
        elseif method == :kmeans
            clustering = kmeans(W', noc; init=:kmpp, maxiter=km_maxiter, tol=1e-6, display=:none) # each column of X is a d-dimensional data point) into k clusters.
            clust = clustering.assignments
        elseif method == :dbscan
            clustering = dbscan(W', ds_radius, min_neighbors = ds_min_ngbr, min_cluster_size = ds_min_clsize)
            clust = clustering.assignments .+= 1 # vector of clusters indices, clustering.clusters, clustering.counts
        elseif method == :hclust
            D = zeros(l,l)
            for i in 1:l, j in i:l
                D[i,j] = norm(W[i,:]-W[j,:])
            end
            D += D'
            result = hclust(D, linkage=:average)
            clust = cutree(result; k=noc, h=hc_h)
        elseif method == :gmm
            mod = GaussianMixtureClusterer(n_classes=gc_n_classes) # A Generative Mixture Model (unfitted)
            prob_belong_classes = BetaML.fit!(mod,W)
            clust = getindex.(findmax.(eachrow(prob_belong_classes)),2)
        else
            error("Unknown clustering method : $method")
        end
        clust_label = assign_celltypes(clust, label) # 119, 9
        precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label) #  0.7675450079084044, 0.7679667607042434
        push!(precisionss,precisions); push!(recallss,recalls); push!(clusts, clust)
        #push!(avg_precs, avg_prec); push!(avg_recs, avg_rec)
    end
    pre_means = []; pre_stds = []; rec_means = []; rec_stds = []
    for i in 1:9
        pre_mean = mean(getindex.(precisionss,i))
        pre_std = std(getindex.(precisionss,i))
        rec_mean = mean(getindex.(recallss,i))
        rec_std = std(getindex.(recallss,i))
        push!(pre_means, pre_mean)
        push!(pre_stds, pre_std)
        push!(rec_means, rec_mean)
        push!(rec_stds, rec_std)
    end
    wavg_pre = label_counts'pre_means/sum(label_counts)
    wavg_rec = label_counts'rec_means/sum(label_counts)
    pre_means, pre_stds, rec_means, rec_stds, wavg_pre, wavg_rec, precisionss, recallss, clusts
end

# Bootstrap clustering before normalization
method = :bootstrap; normalization = false; nepmt = 20; bs_pvalue = 1e-3; bs_nresample = 100

# Clustering without normalization (PCB)
#gridsearch_params(Wpcb, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 3, 150, 9)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 5, 144, 9)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 7, 130, 9)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 10, 129, 9) <-----
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 3, 110, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 5, 100, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 7, 96, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 10, 94, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 3, 75, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 5, 69, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 7, 68, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 10, 65, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 3, 52, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 5, 49, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 7, 50, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 10, 43, 7)

pre_means, pre_stds, rec_means, rec_stds, wavg_pre, wavg_rec, precisionss, recallss, clusts =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wavg_pre 0.908852208330317(1e-3,5), 0.9080230579171948(1e-3,10)
#          0.8520882894712503(1e-4,5), 0.8330695895307648(1e-4,10)
# wavg_rec 0.8984498875872678(1e-3,5), 0.8984735534256301(1e-3,10)
#          0.8718317358892438(1e-4,5), 0.8708969352739321(1e-4,10)

# Clustering without normalization (TSVD)
pre_means_tsvd, pre_stds_tsvd, rec_means_tsvd, rec_stds_tsvd, wavg_pre_tsvd, wavg_rec_tsvd,
    precisionss_tsvd, recallss_tsvd, clusts_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wavg_pre_tsvd 0.8709154514410484(1e-3,5), 0.8646032089543317(1e-3,10)
#               0.8028255214113876(1e-4,5), 0.7970858415130366(1e-4,10)
# wavg_rec_tsvd 0.8684238551650693(1e-3,5), 0.8668264110756124(1e-3,10)
#               0.856792095609987(1e-4,5), 0.8565199384688204(1e-4,10)

# Clustering without normalization (SMA)
#gridsearch_params(Wsma, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 3, 180, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 5, 178, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 7, 167, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.001, 10, 152, 7) <-----
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 3, 126, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 5, 114, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 7, 111, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (0.0001, 10, 105, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 3, 83, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 5, 81, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 7, 73, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-5, 10, 75, 8)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 3, 69, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 5, 61, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 7, 57, 7)
# (pvalue, nresample, maximum(clust), length(unique(celltypes))) = (1.0e-6, 10, 56, 7)
pre_means_sma, pre_stds_sma, rec_means_sma, rec_stds_sma, wavg_pre_sma, wavg_rec_sma,
    precisionss_sma, recallss_sma, clusts_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wave_pre_sma  0.8674657204198589(1e-3,5), 0.8581916740089387(1e-3,10)
#               0.8440651122317941(1e-4,5), 0.8400452598882309(1e-4,10)
# wavg_rec_sma  0.881120577446456(1e-3,5),  0.8799491184475211(1e-3,10)
#               0.8680333688320908(1e-4,5), 0.8669210744290617(1e-4,10)

# Clustering without normalization (HALS)
#gridsearch_params(Wsma, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
pre_means_hals, pre_stds_hals, rec_means_hals, rec_stds_hals, wavg_pre_hals, wavg_rec_hals,
    precisionss_hals, recallss_hals, clusts_hals = clustring_experi(method, Whals, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wave_pre_hals 0.8770460237041406(1e-3,5), 0.8657251321128286(1e-3,10)
#               0.8728822101232738(1e-4,5), 0.8516510242170764(1e-4,10)
# wavg_rec_hals 0.881374985208851(1e-3,5),  0.8803691870784522(1e-3,10)
#               0.8765826529404804(1e-4,5), 0.8763459945568571(1e-4,10)

f = plot_qm3(pre_means, pre_means_tsvd, pre_means_sma, pre_stds, pre_stds_tsvd, pre_stds_sma, label; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(pre_means, pre_means_tsvd, pre_means_sma, pre_means_hals, pre_stds, pre_stds_tsvd, pre_stds_sma, pre_stds_hals, label; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
f = plot_qm3(rec_means, rec_means_tsvd, rec_means_sma, rec_stds, rec_stds_tsvd, rec_stds_sma, label; ylabel="recalls")
save(joinpath(subworkpath,"Baron","Rec_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(rec_means, rec_means_tsvd, rec_means_sma, rec_means_hals, rec_stds, rec_stds_tsvd, rec_stds_sma, rec_stds_hals, label; ylabel="recall")
save(joinpath(subworkpath,"Baron","Rec_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
save(joinpath(subworkpath,"Baron","wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).jld2"),
    "pre_means", pre_means, "pre_means_tsvd", pre_means_tsvd, "pre_means_sma", pre_means_sma, "pre_means_hals", pre_means_hals,
    "pre_stds", pre_stds, "pre_stds_tsvd", pre_stds_tsvd, "pre_stds_sma", pre_stds_sma, "pre_stds_hals", pre_stds_hals,
    "rec_means", rec_means, "rec_means_tsvd", rec_means_tsvd, "rec_means_sma", rec_means_sma, "rec_means_hals", rec_means_hals,
    "rec_stds", rec_stds, "rec_stds_tsvd", rec_stds_tsvd, "rec_stds_sma", rec_stds_sma, "rec_stds_hals", rec_stds_hals,
    "clusts", clusts, "clusts_tsvd", clusts_tsvd, "clusts_sma", clusts_sma, "clusts_hals", clusts_hals,
    "label", label, "label_counts", label_counts)

# Bootstrap clustering after normalization
method = :bootstrap; normalization = true; nepmt = 10; bs_pvalue = 1e-4; bs_nresample = 100

# Clustering after normalization (PCB)
#gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wavg_pren 0.8973663024880435(1e-3,5)0, 0.9074112177296969(1e-3,100)
#           0.9103654620378429(1e-4,50), 0.9105321425998115(1e-4,100)
# wavg_recn 0.2995385161519347(1e-3,50), 0.30384569873387773(1e-3,100)
#           0.35142586676133(1e-4,50), 0.3584546207549403(1e-4,100)

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wavg_pren_tsvd 0.8918370868837022(1e-3,50), 0.8921889549728975(1e-3,100)
#                0.8367118995343917(1e-4,50), 0.834441031276884(1e-4,100)
# wavg_recn_tsvd 0.2230623594840847(1e-3,50), 0.2262690805821796(1e-3,100)
#                0.2563601940598746(1e-4,50), 0.2606082120459117(1e-4,100)

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wavg_pre_sma 0.8620521640575513(1e-3,50), 0.8537549009497538(1e-3,100)
#              0.8351879780338481(1e-4,50), 0.8335766275454753(1e-4,100)
# wavg_rec_sma 0.23409064016092773(1e-3,50), 0.2355224233818483(1e-3,100)
#              0.33280085197018106(1e-4,50), 0.333238669979884(1e-4,100)

# Clustering after normalization (HALS)
#gridsearch_params(Wsma, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
pre_meansn_hals, pre_stdsn_hals, rec_meansn_hals, rec_stdsn_hals, wavg_pren_hals, wavg_recn_hals,
    precisionssn_hals, recallssn_hals, clustsn_hals = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)
# wave_pre_hals 0.8604429244812161(1e-3,50), 0.8593044568484652(1e-3,100)
#               0.8390796548435766(1e-4,50), 0.8313692612628543(1e-4,100)
# wavg_rec_hals 0.23207904390013018(1e-3,50), 0.23456395692817422(1e-3,100)
#               0.3314518991835286(1e-4,50), 0.3318187196781446(1e-4,100)

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_meansn_hals, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, pre_stdsn_hals, label, label_counts; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,"Baron","Rec_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_meansn_hals, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, rec_stdsn_hals, label, label_counts; ylabel="recalls")
save(joinpath(subworkpath,"Baron","Rec_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
save(joinpath(subworkpath,"Baron","aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).jld2"),
    "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma, "pre_meansn_hals", pre_meansn_hals,
    "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma, "pre_stdsn_hals", pre_stdsn_hals,
    "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma, "rec_meansn_hals", rec_meansn_hals,
    "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma, "rec_stdsn_hals", rec_stdsn_hals,
    "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma, "clustsn_hals", clustsn_hals,
    "label", label, "label_counts", label_counts)

# kmeans clustering after normalization : hc_h=300, gc_n_classes=40)
method = :kmeans; normalization = true; nepmt = 20; km_maxiter = 100

# Clustering after normalization (PCB)
#gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, km_maxiter = km_maxiter)
# wavg_pren 0.8432044819411357(iter100)
# wavg_recn 0.8844515441959531(iter100)

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, km_maxiter = km_maxiter)
# wavg_pren_tsvd 0.8055573439186274
# wavg_recn_tsvd 0.8458348124482309

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, km_maxiter = km_maxiter)
# wavg_pre_sma 0.8248949167782013
# wavg_rec_sma 0.876363743935629

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_aftr_nor_$(method)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,"Baron","Rec_aftr_nor_$(method)_ne$(nepmt).png"),f,px_per_unit=2)
save(joinpath(subworkpath,"Baron","aftr_nor_$(method)_ne$(nepmt).jld2"),
    "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma,
    "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,
    "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma,
    "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma,
    "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma,
    "label", label, "label_counts", label_counts)


# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) after normalization : ds_radius=40; ds_min_ngbr=3; ds_min_clsize = 3
using Clustering

method = :dbscan; normalization = true; nepmt = 20; ds_radius=40; ds_min_ngbr=3; ds_min_clsize = 3
for ds_radius in [100]
    for ds_min_ngbr in [100]
        for ds_min_clsize in [100]
            println("ds_radius=$(ds_radius), ds_min_ngbr=$(ds_min_ngbr), ds_min_clsize=$(ds_min_clsize)")
            #gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
            pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, _ =
                clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                                nepmt=nepmt, ds_radius=ds_radius, ds_min_ngbr=ds_min_ngbr, ds_min_clsize=ds_min_clsize);
            @show wavg_pren, wavg_recn
        end
    end
end

# Clustering after normalization (PCB)
#gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])

pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, ds_radius=ds_radius, ds_min_ngbr=ds_min_ngbr, ds_min_clsize=ds_min_clsize)
# wavg_pren 0.08927021104531527
# wavg_recn 0.2987812093243403

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, ds_radius=ds_radius, ds_min_ngbr=ds_min_ngbr, ds_min_clsize=ds_min_clsize)
# wavg_pren_tsvd 0.08927021104531527
# wavg_recn_tsvd 0.2987812093243403

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, ds_radius=ds_radius, ds_min_ngbr=ds_min_ngbr, ds_min_clsize=ds_min_clsize)
# wavg_pre_sma 0.08927021104531527
# wavg_rec_sma 0.2987812093243403

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_aftr_nor_$(method)_r$(ds_radius)_mn$(ds_min_ngbr)_mc$(ds_min_clsize)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,"Baron","Rec_aftr_nor_$(method)_r$(ds_radius)_mn$(ds_min_ngbr)_mc$(ds_min_clsize)_ne$(nepmt).png"),f,px_per_unit=2)
save(joinpath(subworkpath,"Baron","aftr_nor_$(method)_r$(ds_radius)_mn$(ds_min_ngbr)_mc$(ds_min_clsize)_ne$(nepmt).jld2"),
    "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma,
    "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,
    "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma,
    "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma,
    "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma,
    "label", label, "label_counts", label_counts)


# Hierarchical clustering after normalization : hc_h=300
method = :hclust; normalization = true; nepmt = 20; hc_h=300

# Clustering after normalization (PCB)
#gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, hc_h=hc_h)
# wavg_pren 0.8532968698165214
# wavg_recn 0.8609631996213466

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, hc_h=hc_h)
# wavg_pren_tsvd 0.8249493965060913
# wavg_recn_tsvd 0.8627381374985211

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, hc_h=hc_h)
# wavg_pre_sma 0.8167916668483604
# wavg_rec_sma 0.8722044728434505

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_aftr_nor_$(method)_h$(hc_h))_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,"Baron","Rec_aftr_nor_$(method)_h$(hc_h)_ne$(nepmt).png"),f,px_per_unit=2)
save(joinpath(subworkpath,"Baron","aftr_nor_$(method)_h$(hc_h)_ne$(nepmt).jld2"),
    "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma,
    "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,
    "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma,
    "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma,
    "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma,
    "label", label, "label_counts", label_counts)


# Gaussian Mixture Models (GMM) clustering after normalization : gc_n_classes=40)
using BetaML

method = :gmm; normalization = true; nepmt = 20; gc_n_classes=40

# Clustering after normalization (PCB)
#gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, gc_n_classes=gc_n_classes)
# wavg_pren 0.8432044819411357(iter100)
# wavg_recn 0.8844515441959531(iter100)

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, gc_n_classes=gc_n_classes)
# wavg_pren_tsvd 0.8055573439186274
# wavg_recn_tsvd 0.8458348124482309

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, gc_n_classes=gc_n_classes)
# wavg_pre_sma 0.8248949167782013
# wavg_rec_sma 0.876363743935629

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,"Baron","Pre_aftr_nor_$(method)_nc$(gc_n_classes))_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,"Baron","Rec_aftr_nor_$(method)_nc$(gc_n_classes)_ne$(nepmt).png"),f,px_per_unit=2)
save(joinpath(subworkpath,"Baron","aftr_nor_$(method)_nc$(gc_n_classes)_ne$(nepmt).jld2"),
    "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma,
    "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,
    "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma,
    "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma,
    "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma,
    "label", label, "label_counts", label_counts)


using Clustering

# K-means Clustering:
clustering = kmeans(Wpcb', 9; init=:kmpp, maxiter=100, tol=1e-6, display=:none) # each column of X is a d-dimensional data point) into k clusters.
            # display = :none, :final, :iter(shows the prograss of each iteration)
clustering.centers # k x d matrix of cluster centers
clust = clustering.assignments
length(unique(clust)) # 9
clust_label = assign_celltypes(clust, label) # 5
precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label) # 0.570714749704325, 0.5097162558746051
clustering = kmeans(Wsma', 9; init=:kmpp, maxiter=100, tol=1e-6, display=:none) # each column of X is a d-dimensional data point) into k clusters.
clust_sma = clustering.assignments
clust_label_sma = assign_celltypes(clust_sma, label)
precisions_sma, recalls_sma, avg_prec_sma, avg_rec_sma = cal_precision_recall(clust_label_sma, label) # 0.3725977058922331, 0.3826861462618154

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
for min_neighbors in 3:1:20
    clustering = dbscan(Wpcb', 40, min_neighbors = min_neighbors, min_cluster_size = 3)
    clust = clustering.assignments .+= 1 # vector of clusters indices, clustering.clusters, clustering.counts
    clust_label = assign_celltypes(clust, label)
    precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label)
    @show min_neighbors, length(unique(clust)), avg_prec, avg_rec
end
radius = 40
clustering = dbscan(Wpcb', radius, min_neighbors = 3, min_cluster_size = 5)
clust = clustering.assignments .+= 1 # vector of clusters indices, clustering.clusters, clustering.counts
length(unique(clust))
clust_label = assign_celltypes(clust, label)
precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label) # 0.5279016196619022, 0.34896524363381953

# Hierarchical Clustering :
# D += D'; # symmetric distance matrix (optional)
l = length(clust)
D = zeros(l,l)
for i in 1:l, j in i:l
    D[i,j] = norm(Wpcb[i,:]-Wpcb[j,:])
end
D += D'

result = hclust(D, linkage=:average) # :single(minimum distance between any of the cluster members)
                                    # :average(average distance between all pairs of cluster members)
                                    # :complete(maximum distance between any of the cluster members)
                                    # :ward(the distance is the increase of the average squared distance of a point to its cluster centroid after merging the two clusters)
result.merges   # ::Matrix{Int}: N×2 N×2 matrix encoding subtree merges:
                # each row specifies the left and right subtrees (referenced by their ids) that are merged
                # negative subtree id denotes the leaf node and corresponds to the data point at position
                # positive id denotes nontrivial subtree (the row merges[id, :] specifies its left and right subtrees)
ncluster = 9;
for height = 1:1:40
    clust = cutree(result; k=ncluster, h=height) # cut the tree into ncluster clusters
    clust_label = assign_celltypes(clust, label) # 7
    precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label)
    @show height, length(unique(clust_label)), avg_prec, avg_rec
end
ncluster = 9; height = 300
clust = cutree(result; k=ncluster, h=height) # 141
clust_label = assign_celltypes(clust, label) # 7
precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label) #  0.6582037481688565, 0.6043386698503651

Dsma = zeros(l,l)
for i in 1:l, j in i:l
    Dsma[i,j] = norm(Wsma[i,:]-Wsma[j,:])
end
Dsma += Dsma'
result_sma = hclust(Dsma, linkage=:average)
clust_sma = cutree(result_sma; k=ncluster, h=height) # 141
clust_sma_label = assign_celltypes(clust_sma, label) # 7
precisions_sma, recalls_sma, avg_prec_sma, avg_rec_sma = cal_precision_recall(clust_sma_label, label) #  0.579392522908481, 0.586728155926277

# Gaussian Mixture Models (GMM): BetaML.jl
using BetaML

X = [1.1 10.1; 0.9 9.8; 10.0 1.1; 12.1 0.8; 0.8 9.8];

mod = GaussianMixtureClusterer(n_classes=9) # A Generative Mixture Model (unfitted)
prob_belong_classes = fit!(mod,X)
# 5×2 Matrix{Float64}:
#  1.0  0.0
#  1.0  0.0
#  0.0  1.0
#  0.0  1.0
#  1.0  0.0
new_probs = fit!(mod,[11 0.9]) # online fitting (new data is added)
#  0.0  1.0
info(mod)
parameters(mod)

mod = GaussianMixtureClusterer(n_classes=40) # A Generative Mixture Model (unfitted)
prob_belong_classes = BetaML.fit!(mod,Wpcb)
clust = getindex.(findmax.(eachrow(prob_belong_classes)),2)
clust_label = assign_celltypes(clust, label) # 4
precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label) #  0.6582037481688565, 0.6043386698503651
