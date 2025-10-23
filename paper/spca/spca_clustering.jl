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

using RCall, ClusteringBenchmarks

R"""
library(scRNAseq)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)
library(epca)
library(Matrix) # as.matrix
library(ggplot2)
"""

dataset = "Baron" # Baron, Muraro, Segerstolpe, Xin
ddorg = load(joinpath(subworkpath, dataset, "Xr_$(dataset).jld2"))
Xr = ddorg["Xr"]; label = ddorg["label"]; genename = ddorg["genename"]

ddrst = load(joinpath(subworkpath,dataset,"$(dataset)_Result_sp090925.jld2"))
Wtsvd, Httsvd = ddrst["Wtsvd"], ddrst["Httsvd"]
Wpcb, Htpcb, Wsma, Htsma = ddrst["Wpcb"], ddrst["Htpcb"], ddrst["Wsma"], Array(ddrst["Htsma"])
Whals, Hthals = ddrst["Whals"], ddrst["Hthals"]
# dd = load(joinpath(subworkpath,dataset,"Result_sp_nn_s0321.jld2"))
# X, label, genename = dd["X"], dd["cell_type_label"], dd["gene_name"]
# Wpcb, Htpcb, Wsma, Hsma = dd["Wpcb"], dd["Htpcb"], dd["Wsma"], dd["Hsma"]

label = string.(label)
ulabel = unique(label)
ilabel = label2int.(label)
cmap = countmap(label)
label_counts = map(l->cmap[l],ulabel)
gtnoc = length(ulabel)

# Clustering
pvalue = 0.0001
pvalues = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
nclust_pcbs = Float64[]; nclust_smas = Float64[]; nclust_halss = Float64[]
ami_pcbs = Float64[]; ami_smas = Float64[]; ami_halss = Float64[];
for pvalue in pvalues
    @show pvalue
    clust = cluster(Wpcb', pvalue)
    amival = ami(ilabel,clust)
    clust_sma = cluster(Wsma', pvalue)
    amival_sma = ami(ilabel,clust_sma)
    clust_hals = cluster(Whals', pvalue)
    amival_hals = ami(ilabel,clust_hals)
    nclust_pcb, nclust_sma, nclust_hals = maximum(clust), maximum(clust_sma), maximum(clust_hals)
    ami_pcb, ami_sma, ami_hals = round(amival, sigdigits=4), round(amival_sma, sigdigits=4), round(amival_hals, sigdigits=4)
    push!(nclust_pcbs, nclust_pcb); push!(nclust_smas, nclust_sma); push!(nclust_halss, nclust_hals)
    push!(ami_pcbs, ami_pcb); push!(ami_smas, ami_sma); push!(ami_halss, ami_hals)
end
for (i,pvalue) in enumerate(pvalues)
    nclust_pcb = nclust_pcbs[i]; nclust_sma = nclust_smas[i]; nclust_hals = nclust_halss[i]
    ami_pcb = ami_pcbs[i]; ami_sma = ami_smas[i]; ami_hals = ami_halss[i]
    @show pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals
end
f = Figure(size=(500, 400))
ax = AMakie.Axis(f[1, 1], xlabel="p-value", xscale=log10, ylabel="Number of clusters", title="Number of clusters")
lines!(ax, pvalues, nclust_pcbs, label="PCB", color=:blue)
lines!(ax, pvalues, nclust_smas, label="SMA", color=:orange)
lines!(ax, pvalues, nclust_halss, label="HALS", color=:green)
axislegend(ax, position=:lt, title="")
save(joinpath(subworkpath, dataset, "clustering_noc.png"),f)

f = Figure(size=(500, 400))
ax = AMakie.Axis(f[1, 1], xlabel="p-value", xscale=log10, ylabel="Adjusted Mutual Information (AMI)", title="Clustering AMI")
lines!(ax, pvalues, ami_pcbs, label="PCB", color=:blue)
lines!(ax, pvalues, ami_smas, label="SMA", color=:orange)
lines!(ax, pvalues, ami_halss, label="HALS", color=:green)
axislegend(ax, position=:lb, title="")
save(joinpath(subworkpath, dataset, "clustering_ami.png"),f)

# Bootstrap clustering
pvalue=0.0000001; nresample = 50
pvalues = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
nclust_pcbs = Float64[]; nclust_smas = Float64[]; nclust_halss = Float64[]
ami_pcbs = Float64[]; ami_smas = Float64[]; ami_halss = Float64[];
for pvalue in pvalues
    @show pvalue
    clust = cluster_resample(Wpcb', nresample, pvalue)
    amival_pcb = ami(ilabel,clust)
    clust_sma = cluster_resample(Wsma', nresample, pvalue)
    amival_sma = ami(ilabel,clust_sma)
    clust_hals = cluster_resample(Whals', nresample, pvalue)
    amival_hals = ami(ilabel,clust_hals)
    nclust_pcb, nclust_sma, nclust_hals = maximum(clust), maximum(clust_sma), maximum(clust_hals)
    ami_pcb, ami_sma, ami_hals = round(amival_pcb, sigdigits=4), round(amival_sma, sigdigits=4), round(amival_hals, sigdigits=4)
    push!(nclust_pcbs, nclust_pcb); push!(nclust_smas, nclust_sma); push!(nclust_halss, nclust_hals)
    push!(ami_pcbs, ami_pcb); push!(ami_smas, ami_sma); push!(ami_halss, ami_hals)
end
for (i,pvalue) in enumerate(pvalues)
    nclust_pcb = nclust_pcbs[i]; nclust_sma = nclust_smas[i]; nclust_hals = nclust_halss[i]
    ami_pcb = ami_pcbs[i]; ami_sma = ami_smas[i]; ami_hals = ami_halss[i]
    @show pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals
end
f = Figure(size=(500, 400))
ax = AMakie.Axis(f[1, 1], xlabel="p-value", xscale=log10, ylabel="Number of clusters", title="Number of clusters")
lines!(ax, pvalues, nclust_pcbs, label="PCB", color=:blue)
lines!(ax, pvalues, nclust_smas, label="SMA", color=:orange)
lines!(ax, pvalues, nclust_halss, label="HALS", color=:green)
axislegend(ax, position=:lt, title="")
save(joinpath(subworkpath, dataset, "bootstrap_noc.png"),f)

f = Figure(size=(500, 400))
ax = AMakie.Axis(f[1, 1], xlabel="p-value", xscale=log10, ylabel="Adjusted Mutual Information (AMI)", title="Clustering AMI")
lines!(ax, pvalues, ami_pcbs, label="PCB", color=:blue)
lines!(ax, pvalues, ami_smas, label="SMA", color=:orange)
lines!(ax, pvalues, ami_halss, label="HALS", color=:green)
axislegend(ax, position=:lb, title="")
save(joinpath(subworkpath, dataset, "bootstrap_ami.png"),f)

countmap(clust)
countmap(label) # Ground truth
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (0.1, 324, 321, 348, 0.4772, 0.4752, 0.4696)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (0.01, 145, 164, 179, 0.5048, 0.5005, 0.4924)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (0.001, 81, 92, 124, 0.531, 0.5273, 0.5059)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (0.0001, 59, 55, 86, 0.5503, 0.5546, 0.5233)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-5, 52, 48, 62, 0.5548, 0.5549, 0.5513)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-6, 37, 30, 45, 0.5788, 0.5853, 0.5684)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-7, 36, 30, 37, 0.5831, 0.5869, 0.5881)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-8, 31, 29, 35, 0.6017, 0.6072, 0.5827)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-9, 21, 26, 25, 0.6375, 0.6312, 0.6062)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-10, 23, 23, 23, 0.6455, 0.6329, 0.6309)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-11, 18, 20, 17, 0.6529, 0.6229, 0.6426)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-12, 20, 21, 18, 0.6097, 0.6255, 0.6354)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-13, 14, 18, 22, 0.6438, 0.6363, 0.6074)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-14, 15, 14, 16, 0.6384, 0.6403, 0.648)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-15, 15, 15, 16, 0.6324, 0.6585, 0.6532)
# (pvalue, nclust_pcb, nclust_sma, nclust_hals, ami_pcb, ami_sma, ami_hals) = (1.0e-16, 13, 15, 16, 0.6373, 0.6516, 0.6424)


# neighborhood
# Hierarchical clustering after normalization : hc_h=300
method = :bootstrap; normalization = true; clnoc = gtnoc
nepmt = 1 # looks diterministic (no statistic)
pvalue = 1e-11; nresample = 50
pvalues = [1e-11]
for bs_pvalue in pvalues, nresample in [50]
    @show bs_pvalue, nresample
    # Clustering after normalization (PCB)
    #gridsearch_params(Wpcbn, label, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], [1,2,3])
    pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, rstsn, clustsn =
        clustring_experi(method, Wpcb, label, label_counts; noc=clnoc, normalization=normalization, bs_pvalue=bs_pvalue,
                        bs_nresample=nresample, nepmt=nepmt)

    # Clustering after normalization (TSVD)
    pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
        precisionssn_tsvd, recallssn_tsvd, rstsn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
        noc=clnoc, normalization=normalization, bs_pvalue=bs_pvalue, bs_nresample=nresample, nepmt=nepmt)

    # Clustering after normalization (SMA)
    pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
        precisionssn_sma, recallssn_sma, rstsn_tsvd, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
        noc=clnoc, normalization=normalization, bs_pvalue=bs_pvalue, bs_nresample=nresample, nepmt=nepmt)

    # Clustering after normalization (HALS)
    pre_meansn_hals, pre_stdsn_hals, rec_meansn_hals, rec_stdsn_hals, wavg_pren_hals, wavg_recn_hals,
        precisionssn_hals, recallssn_hals, rstsn_tsvd, clustsn_hals = clustring_experi(method, Whals, label, label_counts;
        noc=clnoc, normalization=normalization, bs_pvalue=bs_pvalue, bs_nresample=nresample, nepmt=nepmt)

    @show wavg_pren, wavg_recn
    @show wavg_pren_tsvd, wavg_recn_tsvd
    @show wavg_pren_sma, wavg_recn_sma
    @show wavg_pren_hals, wavg_recn_hals

    f = plot_qm4(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_meansn_hals, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, pre_stdsn_hals, label, label_counts; ylabel="precision")
    save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_noc$(clnoc)_p$(bs_pvalue)_nr$(nresample)_wHALS.png"),f,px_per_unit=2)
    f = plot_qm4(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_meansn_hals, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, rec_stdsn_hals, label, label_counts; ylabel="recalls")
    save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_noc$(clnoc)_p$(bs_pvalue)_nr$(nresample)_wHALS.png"),f,px_per_unit=2)
    save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_noc$(clnoc)_p$(bs_pvalue)_nr$(nresample).jld2"),
        "pre_meansn", pre_meansn, "pre_meansn_tsvd", pre_meansn_tsvd, "pre_meansn_sma", pre_meansn_sma, "pre_meansn_hals", pre_meansn_hals,
        "pre_stdsn", pre_stdsn, "pre_stdsn_tsvd", pre_stdsn_tsvd, "pre_stdsn_sma", pre_stdsn_sma,"pre_stdsn_hals", pre_stdsn_hals,
        "rec_meansn", rec_meansn, "rec_meansn_tsvd", rec_meansn_tsvd, "rec_meansn_sma", rec_meansn_sma, "rec_meansn_hals", rec_meansn_hals,
        "rec_stdsn", rec_stdsn, "rec_stdsn_tsvd", rec_stdsn_tsvd, "rec_stdsn_sma", rec_stdsn_sma, "rec_stdsn_hals", rec_stdsn_hals,
        "wavg_pren", wavg_pren, "wavg_pren_tsvd", wavg_pren_tsvd, "wavg_pren_sma", wavg_pren_sma, "wavg_pren_hals", wavg_pren_hals,
        "wavg_recn", wavg_recn, "wavg_recn_tsvd", wavg_recn_tsvd, "wavg_recn_sma", wavg_recn_sma, "wavg_recn_hals", wavg_recn_hals,
        "clustsn", clustsn, "clustsn_tsvd", clustsn_tsvd, "clustsn_sma", clustsn_sma, "clustsn_hals", clustsn_hals,
        "label", label, "label_counts", label_counts)
end


(bs_pvalue, nresample) = (0.001, 50)
(wavg_pren, wavg_recn) = (0.8257720336854938, 0.2505028990651994)
(wavg_pren_tsvd, wavg_recn_tsvd) = (0.8283180009583834, 0.23133356999171695)
(wavg_pren_sma, wavg_recn_sma) = (0.8286385147072002, 0.2339368122115726)
(wavg_pren_hals, wavg_recn_hals) = (0.8335824557262398, 0.1699207194414862)

(bs_pvalue, nresample) = (0.001, 100)
(wavg_pren, wavg_recn) = (0.8243367630945185, 0.24979292391432967)
(wavg_pren_tsvd, wavg_recn_tsvd) = (0.8126797656462078, 0.23677671281505147)
(wavg_pren_sma, wavg_recn_sma) = (0.8291676203551789, 0.24150988048751626)
(wavg_pren_hals, wavg_recn_hals) = (0.8257319423466086, 0.17702047095018342)

(bs_pvalue, nresample) = (1.0e-11, 50)
(wavg_pren, wavg_recn) = (0.8237112260402019, 0.6751863684771033)
(wavg_pren_tsvd, wavg_recn_tsvd) = (0.812918119688126, 0.6771979647379008)
(wavg_pren_sma, wavg_recn_sma) = (0.8134717363504307, 0.6611051946515205)
(wavg_pren_hals, wavg_recn_hals) = (0.8287488534238123, 0.45142586676133)

(bs_pvalue, nresample) = (1.0e-11, 100)
(wavg_pren, wavg_recn) = (0.8237112260402019, 0.6751863684771033)
(wavg_pren_tsvd, wavg_recn_tsvd) = (0.8137761325828681, 0.6766063187788427)
(wavg_pren_sma, wavg_recn_sma) = (0.8138551670063842, 0.6788545734232635)
(wavg_pren_hals, wavg_recn_hals) = (0.8286141526216019, 0.4490592829

# Hierarchical clustering after normalization : hc_h=300
method = :hclust; normalization = true; clnoc = gtnoc
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
        pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, rstsn, clustsn =
            clustring_experi(method, Wpcb, label, label_counts; noc=clnoc, normalization=normalization,
                            nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (TSVD)
        pre_meansn_tsvd, pre_stdsn_tsvd, rec_meansn_tsvd, rec_stdsn_tsvd, wavg_pren_tsvd, wavg_recn_tsvd,
            precisionssn_tsvd, recallssn_tsvd, rstsn_tsvd, clustsn_tsvd = clustring_experi(method, Wtsvd, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (SMA)
        pre_meansn_sma, pre_stdsn_sma, rec_meansn_sma, rec_stdsn_sma, wavg_pren_sma, wavg_recn_sma,
            precisionssn_sma, recallssn_sma, rstsn_tsvd, clustsn_sma = clustring_experi(method, Wsma, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        # Clustering after normalization (HALS)
        pre_meansn_hals, pre_stdsn_hals, rec_meansn_hals, rec_stdsn_hals, wavg_pren_hals, wavg_recn_hals,
            precisionssn_hals, recallssn_hals, rstsn_tsvd, clustsn_hals = clustring_experi(method, Whals, label, label_counts;
            noc=clnoc, normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)

        ami(ilabel, clustsn[1])
        ami(ilabel, clustsn_tsvd[1])
        ami(ilabel, clustsn_sma[1])
        ami(ilabel, clustsn_hals[1])

        @show wavg_pren, wavg_recn
        @show wavg_pren_tsvd, wavg_recn_tsvd
        @show wavg_pren_sma, wavg_recn_sma
        @show wavg_pren_hals, wavg_recn_hals

        f = plot_qm4(pre_meansn, pre_meansn_tsvd, pre_meansn_sma, pre_meansn_hals, pre_stdsn, pre_stdsn_tsvd, pre_stdsn_sma, pre_stdsn_hals, label, label_counts; ylabel="precision")
        save(joinpath(subworkpath,dataset,"Pre_aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
        f = plot_qm4(rec_meansn, rec_meansn_tsvd, rec_meansn_sma, rec_meansn_hals, rec_stdsn, rec_stdsn_tsvd, rec_stdsn_sma, rec_stdsn_hals, label, label_counts; ylabel="recalls")
        save(joinpath(subworkpath,dataset,"Rec_aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt)_wHALS.png"),f,px_per_unit=2)
        save(joinpath(subworkpath,dataset,"aftr_nor_$(method)_noc$(clnoc)_lkg$(hc_linkage)_h$(hc_h)_ne$(nepmt)_090925.jld2"),
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
    dd = load(joinpath(subworkpath,dataset,"aftr_nor_hclust_noc$(clnoc)_lkg$(hc_linkage)_hnothing_ne1.jld2"))
    pres = map(v->round(v,sigdigits=4), [dd["wavg_pren"], dd["wavg_pren_tsvd"], dd["wavg_pren_sma"], dd["wavg_pren_hals"]])
    recs = map(v->round(v,sigdigits=4), [dd["wavg_recn"], dd["wavg_recn_tsvd"], dd["wavg_recn_sma"], dd["wavg_recn_hals"]])
    @show pres, recs
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
