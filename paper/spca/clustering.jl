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
using Clustering

R"""
library(scRNAseq)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)
library(epca)
library(Matrix) # as.matrix
library(ggplot2)
"""

dataset = "Xin" # Baron, Muraro, Segerstolpe, Xin
ddorg = load(joinpath(subworkpath, dataset, "Xr_$(dataset).jld2"))
Xr = ddorg["Xr"]; label = ddorg["label"]; genename = ddorg["genename"]

ddrst = load(joinpath(subworkpath,dataset,"$(dataset)_Result_sp.jld2"))
Wisvd, Htisvd, Wtsvd, Httsvd = ddrst["Wisvd"], ddrst["Htisvd"], ddrst["Wtsvd"], ddrst["Httsvd"]
Wpcb, Htpcb, Wsma, Htsma = ddrst["Wpcb"], ddrst["Htpcb"], ddrst["Wsma"], ddrst["Htsma"]
Whals, Hthals = ddrst["Whals"], ddrst["Hthals"]
# dd = load(joinpath(subworkpath,dataset,"Result_sp_nn_s0321.jld2"))
# X, label, genename = dd["X"], dd["cell_type_label"], dd["gene_name"]
# Wpcb, Htpcb, Wsma, Hsma = dd["Wpcb"], dd["Htpcb"], dd["Wsma"], dd["Hsma"]

label = string.(label)
ulabel = unique(label)
cmap = countmap(label)
label_counts = map(l->cmap[l],ulabel)
gtnoc = length(ulabel)

# Clustering
pvalue = 0.0001;
clust = cluster(Wpcb', pvalue)

# Bootstrap clustering
pvalue=0.0000001; nresample = 5
for pvalue in [0.000001, 0.0000001]
    for nresample in [3]
        clust = cluster_resample(Wpcb', nresample, pvalue)
        save(joinpath(subworkpath,dataset,"Clustering_p$(pvalue)_n$(nresample)_sp_nn_s0322.jld2"),"clust", clust, "pvalue", pvalue, "nresample", nresample)
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
#        @show i, sum(indices), celltype, cnt
        if celltype ∈ celltypes
            fidx = findfirst(==(celltype), celltypes)
            if cnt > cellcnts[fidx]
                cellcnts[fidx] = 0; celltypes[fidx] = "$(i)"
                cellcnts[i] = cnt; celltypes[i] = celltype
#                @show i, celltype, cnt
            else
                cellcnts[i] = 0; celltypes[i] = "$(i)"
            end
        else
            cellcnts[i] = cnt; celltypes[i] = celltype
#            @show i, celltype, cnt
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
    cell = Int[]; errx = Float64[]; grp = Int[]
    map(i->(append!(cell,fill(i,3)); append!(errx,[-d,0,d]); append!(grp,collect(1:3))),1:length(qm))
    tbl = (cell = cell, errx = errx, value = v, lerrors = lerrs, herrors = herrs, grp = grp)
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
    cell = Int[]; errx = Float64[]; grp = Int[]
    map(i->(append!(cell,fill(i,4)); append!(errx,[-3d,-d,d,3d]); append!(grp,collect(1:4))),1:length(qm))
    tbl = (cell = cell, errx = errx, value = v, lerrors = lerrs, herrors = herrs, grp = grp)
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

function plot_qm4_old(qm, qm_tsvd, qm_sma, qm_hals, qmstd, qmstd_tsvd, qmstd_sma, qmstd_hals, label, label_counts; ylabel="", d=0.12)
    ulabel = unique(label)
    colors = Makie.wong_colors()
    f = Figure(size=(600,250)) # f.scene.viewport.val.widths
    ax = AMakie.Axis(f[1, 1], xlabel = "cell type", ylabel = ylabel, xticklabelrotation = pi/8,
                    xticks = (1:length(ulabel), ulabel), title = "")
    ax2 = AMakie.Axis(f[1, 1], xticks = (1:1:length(label_counts), string.(label_counts)), xaxisposition = :top, yticks = (1:3,["","",""]),
                    yticksvisible = false, ygridvisible = false, xgridvisible = false, title = "")
#    hidexdecorations!(ax2)#, ticklabels = false)
    v = []; lerrs =[]; herrs =[]
    map(i->(push!(v, qm[i]);push!(v, qm_tsvd[i]);push!(v, qm_sma[i]);push!(v, qm_hals[i])), 1:length(qm))
#    map(i->(push!(lerrs, -qmstd[i]);push!(lerrs, -qmstd_tsvd[i]);push!(lerrs, -qmstd_sma[i])), 1:length(qmstd))
    map(i->(push!(herrs, qmstd[i]);push!(herrs, qmstd_tsvd[i]);push!(herrs, qmstd_sma[i]);push!(herrs, qmstd_hals[i])), 1:length(qmstd))
    cell = Int[]; errx = Float64[]; grp = Int[]
    map(i->(append!(cell,fill(i,4)); append!(errx,[-3d,-d,d,3d]); append!(grp,collect(1:4))),1:length(qm))
    tbl = (cell = cell, errx = errx, value = v, lerrors = lerrs, herrors = herrs, grp = grp)
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
        hc_linkage=:average, # :single, :average, :complete, :ward, :ward_presquared
        hc_h=nothing, # [optional] nothing or number
        gc_n_classes=40)
    l = size(Worg,1)
    T = eltype(Worg); W = copy(Worg)
    if normalization
        for r in eachrow(W)
            n = norm(r)
            r = n == 0 ? r : r ./= n
        end
    end
    precisionss=[]; recallss=[]; avg_precs=[]; avg_recs=[]; clusts=[]
    for i in 1:nepmt
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
            result = hclust(D, linkage=hc_linkage)
            clust = cutree(result; k=noc, h=hc_h)
        elseif method == :gmm
            mod = GaussianMixtureClusterer(n_classes=gc_n_classes) # A Generative Mixture Model (unfitted)
            prob_belong_classes = BetaML.fit!(mod,W)
            clust = getindex.(findmax.(eachrow(prob_belong_classes)),2)
        else
            error("Unknown clustering method : $method")
        end
        clust_label = assign_celltypes(clust, label) # 119, noc
        precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label) #  0.7675450079084044, 0.7679667607042434
        push!(precisionss,precisions); push!(recallss,recalls); push!(clusts, clust)
        #push!(avg_precs, avg_prec); push!(avg_recs, avg_rec)
    end
    pre_means = T[]; pre_stds = T[]; rec_means = T[]; rec_stds = T[]
    for i in 1:length(unique(label))
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

# Hierarchical clustering after normalization : hc_h=300
method = :hclust; normalization = true; clnoc = gtnoc+2
nepmt = 1 # looks diterministic (no statistic)
hc_h=nothing; hc_linkage=:average

# Wpcbn = Wpcb./norm.(eachrow(Wpcb))
# l = size(Wpcbn, 1) # number of cells
# D = zeros(l,l)
# for i in 1:l, j in i:l
#     D[i,j] = norm(Wpcbn[i,:]-Wpcbn[j,:])
# end
# D += D'
# result = hclust(D, linkage=:average)
# clust_pcb = cutree(result; k=noc, h=hc_h)

for clnoc in [gtnoc, gtnoc+2]
for hc_linkage in [:complete, :average]
#    for hc_h in [nothing, 100, 300, 600, 1000, 2000]
        @show clnoc, hc_linkage, hc_h
        # Clustering after normalization (PCB)
        #gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
        pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
            clustring_experi(method, Wpcb, label, label_counts; noc=clnoc, normalization=normalization,
                            nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (TSVD)
        pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
            precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (SMA)
        pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
            precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (HALS)
        pre_meansn_hals, pre_stdsn_hals, rec_meansn_hals, rec_stdsn_hals, wavg_pren_hals, wavg_recn_hals,
            precisionssn_hals, recallssn_hals, clustsn_hals = clustring_experi(method, Whals, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        @show wavg_pren, wavg_recn
        @show wavg_pren_tsvd, wavg_recn_tsvd
        @show wavg_pren_sma, wavg_recn_sma
        @show wavg_pren_hals, wavg_recn_hals

        f = plot_qm4(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_meansn_hals, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, pre_stdsn_hals, label, label_counts; ylabel="precision")
        save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
        f = plot_qm4(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_meansn_hals, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, rec_stdsn_hals, label, label_counts; ylabel="recalls")
        save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
        save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt).jld2"),
            "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma, "pre_meansn_hals", pre_meansn_hals,
            "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,"pre_stdsn_hals", pre_stdsn_hals,
            "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma, "rec_meansn_hals", rec_meansn_hals,
            "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma, "rec_stdsn_hals", rec_stdsn_hals,
            "wavg_pren", wavg_pren, "wavg_pren_tsvd", wavg_pren_tsvd, "wavg_pren_sma", wavg_pren_sma, "wavg_pren_hals", wavg_pren_hals,
            "wavg_recn", wavg_recn, "wavg_recn_tsvd", wavg_recn_tsvd, "wavg_recn_sma", wavg_recn_sma, "wavg_recn_hals", wavg_recn_hals,
            "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma, "clustsn_hals", clustsn_hals,
            "label", label, "label_counts", label_counts)
#    end
end
end
for clnoc in [gtnoc, gtnoc+2]
    for hc_linkage in [:average, :complete]
        dd = load(joinpath(subworkpath,dataset,"aftr_nor_hclust_noc$(clnoc)_lkg$(hc_linkage)_hnothing_ne1.jld2"))
        pres = map(v->round(v,sigdigits=4), [dd["wavg_pren"], dd["wavg_pren_tsvd"], dd["wavg_pren_sma"], dd["wavg_pren_hals"]])
        recs = map(v->round(v,sigdigits=4), [dd["wavg_recn"], dd["wavg_recn_tsvd"], dd["wavg_recn_sma"], dd["wavg_recn_hals"]])
        @show pres, recs
    end
end

# Bootstrap clustering before normalization
method = :bootstrap; normalization = false; nepmt = 20; bs_pvalue = 1e-3; bs_nresample = 100

# Clustering without normalization (PCB)
#gridsearch_params(Wpcb, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
pre_means, pre_stds, rec_means, rec_stds, wavg_pre, wavg_rec, precisionss, recallss, clusts =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)

# Clustering without normalization (TSVD)
pre_means_tsvd, pre_stds_tsvd, rec_means_tsvd, rec_stds_tsvd, wavg_pre_tsvd, wavg_rec_tsvd,
    precisionss_tsvd, recallss_tsvd, clusts_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)

# Clustering without normalization (SMA)
#gridsearch_params(Wsma, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
pre_means_sma, pre_stds_sma, rec_means_sma, rec_stds_sma, wavg_pre_sma, wavg_rec_sma,
    precisionss_sma, recallss_sma, clusts_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)

# Clustering without normalization (HALS)
#gridsearch_params(Wsma, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
pre_means_hals, pre_stds_hals, rec_means_hals, rec_stds_hals, wavg_pre_hals, wavg_rec_hals,
    precisionss_hals, recallss_hals, clusts_hals = clustring_experi(method, Whals, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)

f = plot_qm3(pre_means, pre_means_tsvd, pre_means_sma, pre_stds, pre_stds_tsvd, pre_stds_sma, label; ylabel="precision")
save(joinpath(subworkpath,dataset,"Pre_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(pre_means, pre_means_tsvd, pre_means_sma, pre_means_hals, pre_stds, pre_stds_tsvd, pre_stds_sma, pre_stds_hals, label; ylabel="precision")
save(joinpath(subworkpath,dataset,"Pre_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
f = plot_qm3(rec_means, rec_means_tsvd, rec_means_sma, rec_stds, rec_stds_tsvd, rec_stds_sma, label; ylabel="recalls")
save(joinpath(subworkpath,dataset,"Rec_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(rec_means, rec_means_tsvd, rec_means_sma, rec_means_hals, rec_stds, rec_stds_tsvd, rec_stds_sma, rec_stds_hals, label; ylabel="recall")
save(joinpath(subworkpath,dataset,"Rec_wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
save(joinpath(subworkpath,dataset,"wo_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).jld2"),
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

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)

# Clustering after normalization (HALS)
#gridsearch_params(Wsma, label, [1e-3, 1e-4, 1e-5, 1e-6], [3,5,7,10])
pre_meansn_hals, pre_stdsn_hals, rec_meansn_hals, rec_stdsn_hals, wavg_pren_hals, wavg_recn_hals,
    precisionssn_hals, recallssn_hals, clustsn_hals = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, bs_nresample=bs_nresample, bs_pvalue=bs_pvalue)

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_meansn_hals, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, pre_stdsn_hals, label, label_counts; ylabel="precision")
save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm4(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_meansn_hals, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, rec_stdsn_hals, label, label_counts; ylabel="recalls")
save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_p$(bs_pvalue)_nr$(bs_nresample)_ne$(nepmt).jld2"),
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

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, km_maxiter = km_maxiter)

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, km_maxiter = km_maxiter)

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_ne$(nepmt).png"),f,px_per_unit=2)
save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_ne$(nepmt).jld2"),
    "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma,
    "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,
    "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma,
    "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma,
    "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma,
    "label", label, "label_counts", label_counts)


# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) after normalization : ds_radius=40; ds_min_ngbr=3; ds_min_clsize = 3
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

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, ds_radius=ds_radius, ds_min_ngbr=ds_min_ngbr, ds_min_clsize=ds_min_clsize)

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, ds_radius=ds_radius, ds_min_ngbr=ds_min_ngbr, ds_min_clsize=ds_min_clsize)

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_r$(ds_radius)_mn$(ds_min_ngbr)_mc$(ds_min_clsize)_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_r$(ds_radius)_mn$(ds_min_ngbr)_mc$(ds_min_clsize)_ne$(nepmt).png"),f,px_per_unit=2)
save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_r$(ds_radius)_mn$(ds_min_ngbr)_mc$(ds_min_clsize)_ne$(nepmt).jld2"),
    "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma,
    "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,
    "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma,
    "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma,
    "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma,
    "label", label, "label_counts", label_counts)


# Hierarchical clustering after normalization : hc_h=300
method = :hclust; normalization = true; clnoc = 11; nepmt = 1 # looks diterministic (no statistic)
hc_h=nothing; hc_linkage=:complete

Wpcbn = Wpcb./norm.(eachrow(Wpcb))
l = size(Wpcbn, 1) # number of cells
D = zeros(l,l)
for i in 1:l, j in i:l
    D[i,j] = norm(Wpcbn[i,:]-Wpcbn[j,:])
end
D += D'
result = hclust(D, linkage=:average)
clust_pcb = cutree(result; k=clnoc, h=hc_h)

for clnoc in [9. 11, 13]
for hc_linkage in [:ward, :complete, :average]
#    for hc_h in [nothing, 100, 300, 600, 1000, 2000]
        @show clnoc, hc_linkage, hc_h
        # Clustering after normalization (PCB)
        #gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
        pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
            clustring_experi(method, Wpcb, label, label_counts; noc=clnoc, normalization=normalization,
                            nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (TSVD)
        pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
            precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (SMA)
        pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
            precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (HALS)
        pre_meansn_hals, pre_stdsn_hals, rec_meansn_hals, rec_stdsn_hals, wavg_pren_hals, wavg_recn_hals,
            precisionssn_hals, recallssn_hals, clustsn_hals = clustring_experi(method, Whals, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        @show wavg_pren, wavg_recn
        @show wavg_pren_tsvd, wavg_recn_tsvd
        @show wavg_pren_sma, wavg_recn_sma
        @show wavg_pren_hals, wavg_recn_hals

        f = plot_qm4(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_meansn_hals, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, pre_stdsn_hals, label, label_counts; ylabel="precision")
        save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
        f = plot_qm4(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_meansn_hals, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, rec_stdsn_hals, label, label_counts; ylabel="recalls")
        save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
        save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt).jld2"),
            "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma, "pre_meansn_hals", pre_meansn_hals,
            "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,"pre_stdsn_hals", pre_stdsn_hals,
            "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma, "rec_meansn_hals", rec_meansn_hals,
            "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma, "rec_stdsn_hals", rec_stdsn_hals,
            "wavg_pren", wavg_pren, "wavg_pren_tsvd", wavg_pren_tsvd, "wavg_pren_sma", wavg_pren_sma, "wavg_pren_hals", wavg_pren_hals,
            "wavg_recn", wavg_recn, "wavg_recn_tsvd", wavg_recn_tsvd, "wavg_recn_sma", wavg_recn_sma, "wavg_recn_hals", wavg_recn_hals,
            "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma, "clustsn_hals", clustsn_hals,
            "label", label, "label_counts", label_counts)
#    end
end
end
for clnoc in [9, 11]
    for hc_linkage in [:average, :complete]
        dd = load(joinpath(subworkpath,dataset,"aftr_nor_hclust_noc$(clnoc)_lkg$(hc_linkage)_hnothing_ne1.jld2"))
        pres = map(v->round(v,sigdigits=4), [dd["wavg_pren"], dd["wavg_pren_tsvd"], dd["wavg_pren_sma"], dd["wavg_pren_hals"]])
        recs = map(v->round(v,sigdigits=4), [dd["wavg_recn"], dd["wavg_recn_tsvd"], dd["wavg_recn_sma"], dd["wavg_recn_hals"]])
        @show pres, recs
    end
end

# Gaussian Mixture Models (GMM) clustering after normalization : gc_n_classes=40)
using BetaML

method = :gmm; normalization = true; nepmt = 20; gc_n_classes=40

# Clustering after normalization (PCB)
#gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, clustsn =
    clustring_experi(method, Wpcb, label, label_counts; normalization=normalization,
                    nepmt=nepmt, gc_n_classes=gc_n_classes)

# Clustering after normalization (TSVD)
pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
    precisionssn_tsvd, recallssn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
    normalization=normalization, nepmt=nepmt, gc_n_classes=gc_n_classes)

# Clustering after normalization (SMA)
pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
    precisionssn_sma, recallssn_sma, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
    normalization=normalization, nepmt=nepmt, gc_n_classes=gc_n_classes)

f = plot_qm3(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, label; ylabel="precision")
save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_nc$(gc_n_classes))_ne$(nepmt).png"),f,px_per_unit=2)
f = plot_qm3(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, label; ylabel="recalls")
save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_nc$(gc_n_classes)_ne$(nepmt).png"),f,px_per_unit=2)
save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_nc$(gc_n_classes)_ne$(nepmt).jld2"),
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
