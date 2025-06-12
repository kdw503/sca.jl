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
include(joinpath(workpath,"clustering.jl"))

using RCall

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
