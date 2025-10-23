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

#=============== seqRNA ===============#
##### SMA method (using RCall)
using RCall, StatsBase, ClusteringBenchmarks

R"""
library(scRNAseq)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)
library(epca)
library(Matrix) # as.matrix
library(ggplot2)
"""

# datasets = ["Baron", "Muraro", "Segerstolpe", "Xin"]
# for dataset in datasets
#     dd = load(joinpath(subworkpath, "raw", dataset, "Xr_$(dataset).jld2"))
#     Xr, label, genename = dd["Xr"], dd["label"], dd["genename"]
#     normXr = norm(Xr)
#     Xr = Xr ./ normXr
#     Xr = log10.(Xr .+ 1)
#     save(joinpath(subworkpath, dataset, "Xr_$(dataset).jld2"), "Xr", Xr, "label", label, "genename", genename)
#     Xr = load(joinpath(subworkpath, dataset, "Xr_$(dataset).jld2"), "Xr")
#     @show size(Xr)
# end
dataset = "Baron" # Baron(Int), Muraro(Float raw?), Segerstolpe(Int), Xin(Float raw?)
@rput subworkpath
@rput dataset
R"""
# cat(sprintf("%s dataset\n", dataset))
# dat <- switch(dataset,
#     "Baron" = {
#         cat("Baron dataset\n")
#         BaronPancreasData()
#     },
#     "Muraro" = {
#         cat("Muraro dataset\n")
#         MuraroPancreasData()
#     },
#     "Segerstolpe" = {
#         cat("Segerstolpe dataset\n")
#         SegerstolpePancreasData()
#     },
#     "Xin" = {
#         cat("Xin dataset\n")
#         dat <- XinPancreasData()
#     },
#     stop("unknown dataset")
# )
# Map dataset names to their corresponding functions
dataset_functions <- list(
  "Muraro" = MuraroPancreasData,
  "Baron" = BaronPancreasData,
  "Segerstolpe" = SegerstolpePancreasData,
  "Xin" = XinPancreasData
)

# Check if dataset exists and call the correct function
if (dataset %in% names(dataset_functions)) {
  cat(dataset, "dataset\n")
  dat <- dataset_functions[[dataset]]()
} else {
  stop("Unknown dataset: ", dataset)
}
dim(dat)
names(assays(dat))
"""
# R"""
#     meta_df <- as.data.frame(colData(dat))
#     colnames(meta_df)
# """

if dataset in ["Baron", "Muraro"]
    R"""
    gene.select <- !!apply(counts(dat), 1, sd) # select genes with high variance
    label.select <- colData(dat) %>% # select labels with more than 100 cells
                    data.frame() %>%
                    dplyr::count(label) %>%
                    filter(n > 100)
    dat1 <- dat[gene.select, colData(dat)$label %in% label.select$label]
    label <- setNames(factor(data.frame(colData(dat1))$label), colnames(dat1)) # cell type label
    count <- counts(dat1)
    genename <- as.matrix(rownames(dat1))                       # gene names
    """
elseif dataset in ["Segerstolpe"]
    R"""
    gene.select <- !!apply(counts(dat), 1, sd) # select genes with high variance
    cell.type.select <- colData(dat) %>% # select labels with more than 100 cells
                    data.frame() %>%
                    dplyr::count(cell.type) %>%
                    filter(n > 100)
    dat1 <- dat[gene.select, data.frame(colData(dat))$cell.type %in% cell.type.select$cell.type]
    label <- setNames(factor(data.frame(colData(dat1))$cell.type), colnames(dat1)) # cell type label
    count <- counts(dat1)
    genename <- as.matrix(rownames(dat1))                       # gene names
    """
elseif dataset in ["Xin"] # data is too small and only alpha, beta and delta are avilable
    # if rkpm() is not installed, install it. In the powershell
    # c:\"Program Files"\R\R-4.3.2\bin\Rscript.exe -e "install.packages('BiocManager', repos='https://cloud.r-project.org')"
    # c:\"Program Files"\R\R-4.3.2\bin\Rscript.exe -e "BiocManager::install('edgeR')"
    R"""
    library(edgeR)
    rpkm_vals <- assay(dat, "rpkm")
    th = 0
    sds <- apply(rpkm_vals, 1, sd) # select genes with high variance
    gene.select <- sds > th
    cell.type.select <- colData(dat) %>% # select labels with more than 100 cells
                    data.frame() %>%
                    dplyr::count(cell.type) %>%
                    filter(n > 10)
    dat1 <- dat[gene.select, data.frame(colData(dat))$cell.type %in% cell.type.select$cell.type]
    label <- setNames(factor(data.frame(colData(dat1))$cell.type), colnames(dat1)) # cell type label
    count <- assay(dat1, "rpkm")
    genename <- as.matrix(rownames(dat1))                       # gene names
    """
end 
@rget count
@rget genename
@rget label
label = string.(Array(label))
gtnoc = length(unique(label))
# save dataset as JLD2 file
R"Xr = unname(count)"
R"count_matrix_dense <- as.matrix(count)"
Xr = rcopy(R"count_matrix_dense") # Float64
save(joinpath(subworkpath, dataset, "Xr_$(dataset).jld2"), "Xr", Xr, "label", label, "genename", genename)
dd = load(joinpath(subworkpath, dataset, "Xr_$(dataset).jld2"))
Xr, label, genename = dd["Xr"], dd["label"], dd["genename"]
ulabel = unique(label); cmap = countmap(label); label_counts = map(l->cmap[l],ulabel)
f = Figure()
ylimit = 200; bins = 100
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,ylimit)))
ht1 = hist!(ax,vec(Xr), bins=bins, strokewidth = 1, strokecolor = :black) # bins is number of bins
save(joinpath(subworkpath, dataset, "Xr_histo_bins$(bins)_ylimit$(ylimit).png"),f)

@rput gtnoc
gma = 10
#for gma in 6.:0.5:20. # 10. is best
@rput gma
rtsma = @elapsed R"""
scar <- sca(t(Xr), k = gtnoc, gamma = gma, # Xr is geneXcell, t(Xr) is cellXgene, so gene is sparsified
               center = F, scale = F,
               epsilon = 1e-3)
n.gene <- apply(!!scar$loadings, 2, sum) # sum of non zero loadings number for each gene
ngene_sma <- n.gene
Wsma <- as.matrix(scar$scores) # score
Htsma <- as.matrix(scar$loadings)
"""
@rget Wsma # cell(&power) 8451X9 (Baron)
@rget Htsma # gene(normalized) 17499X9 (Baron)
@rget ngene_sma
Hsma = Htsma'
fv = LCSVD.fitd(X,Wsma*Hsma)
# method = :hclust; normalization = true # each row of Wsma(cells X noc) is normalized
# nepmt = 1; hc_linkage = :average; hc_h = nothing
# pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren_sma, wavg_recn_sma, precisionssn_sma, recallssn_sma, rst_sma, clustsn_sma =
#     clustring_experi(method, Wsma, label, label_counts; noc=clnoc, normalization=normalization,
#                 nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
# amivalue = ami(ilabel, clustsn_sma[1]) # Baron(0.7843344572922389(gamma default), 0.8059945736818778(gamma=12,rt=76.053647))
# @show gma, rtsma, amivalue
# end

#@rget scar
#W = scar[:scores]; H = scar[:loadings]; label = scar[:label]

# initialization only
rtirlba = @elapsed R"""  ## initialize
  x = scale(x = t(Xr),
            center = F,
            scale = F)
  # s = RSpectra::svds(x, k)
  s = irlba::irlba(x, gtnoc, tol = 1e-10)
  z = s$u
  b = diag(s$d)
  y = s$v
  score = sqrt(sum(s$d ^ 2))
  diff = c(z = Inf, y = Inf)
"""

# Without shrink SMA
rtsmawosh = @elapsed R"""
factors_sma <- sca(t(Xr), k = gtnoc, gamma = Inf,
               center = F, scale = F,
               epsilon = 1e-3)
"""
@rget factors_sma
rW = factors_sma[:scores]; rH = Array(factors_sma[:loadings]')
normX2 = norm(X)^2
fv = LCSVD.fitd(X,rW*rH)
Xy = X*rH'*inv(rH*rH')*rH; pve = norm(Xy)^2/normX2 # PVE : 0.9433741435634796
LCSVD.normalizeW!(rW,rH); sw = norm(rW,1) # Sparsity : 317.95

label = string.(label)
ulabel = unique(label)
ilabel = label2int.(label)
clnocdic = Dict("Baron" => 9, "Muraro" => 9, "Segerstolpe" => 9, "Xin" => 6)
clnoc = clnocdic[dataset]
method = :hclust; normalization = true # each row of Wsma(cells X noc) is normalized
nepmt = 1; hc_linkage = :average; hc_h = nothing
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren_sma, wavg_recn_sma, precisionssn_sma, recallssn_sma, rst_sma, clustsn_sma =
    clustring_experi(method, Wsma, label, label_counts; noc=clnoc, normalization=normalization,
                nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
@show wavg_pren_sma, wavg_recn_sma # Baron(0.8220214208322277, 0.851496864276417), Muraro(0.7133540938457136, 0.6879217273954116)
ami(ilabel, clustsn_sma[1]) # Baron(0.7843344572922389(gamma default), 0.8103206815005907(gamma=10,rt=85.8246761))

# n genes
LCSVD.normalizeW!(Hsma',Wsma') # TODO: no need because Hsma is already normalized
m = 0.1
for i in 1:gtnoc
    @show i#; @show Hsma[i,:][Hsma[i,:].>m]; @show Hsma[i,:][Hsma[i,:].<-m];  @show genename[Hsma[i,:].>m]; @show genename[Hsma[i,:].<-m];
    sindps = sortperm(Hsma[i,:][Hsma[i,:].>m],rev=true); sindps_cut = sindps[1:(min(end,3))]
    sindns = sortperm(Hsma[i,:][Hsma[i,:].<-m]); sindns_cut = sindns[1:(min(end,3))]
    genes_p = genename[Hsma[i,:].>m][sindps_cut]; cnt_p = length(sindps)
    maxval = isempty(sindps_cut) ? nothing : maximum(Hsma[i,:][Hsma[i,:].>m])
    genes_n = genename[Hsma[i,:].<-m][sindns_cut]; cnt_n = length(sindns)
    minval = isempty(sindns_cut) ? nothing : minimum(Hsma[i,:][Hsma[i,:].<-m])
    @show genes_p, maxval, cnt_p
    @show genes_n, minval, cnt_n
end

# SMA Boxplot
@rput Wsma
R"""
if (dataset == "Segerstolpe") {
    ylim = c(-10, 300)
} else if (dataset == "Xin") {
    ylim = c(-10, 400)
} else {
    ylim = c(0, 4)
}
Wsma %>%
  reshape2::melt(varnames = c("cell", "PC"),
                 value.name = "scores") %>%
  mutate(PC = factor(PC), label = label[cell]) %>%
  ggplot(aes(PC, scores / 1000, fill = PC)) +
  geom_boxplot(color = "grey30", outlier.shape = NA,
               show.legend = FALSE) +
  labs(x = "gene PC", y = bquote("scores ("~10^3~")")) +
  scale_x_discrete(labels = 1:9) +
  facet_wrap(~ label, nrow = 3) +
  scale_fill_brewer(palette = "Set3") +
  coord_cartesian(ylim = ylim) +
  theme_classic()
  fullname <- file.path(subworkpath, dataset, paste0(dataset,"_SMA_boxplot.png"))
  ggsave(fullname)
"""

##### PCB method

# Load the data set
initmethod=:tsvd; svdmethod=:nndsvd; nac=0
#initmethod=:svd; svdmethod=:svd
rtisvd = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(Xr, gtnoc, nac; initmethod=initmethod, svdmethod=svdmethod)
V = copy(H0'); N0t = copy(N0')
Wtsvd = H0'*D # cell
Httsvd = U # gene

method = :hclust; normalization = true; nepmt = 1; hc_linkage = :average; hc_h = nothing
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren_tsvd, wavg_recn_tsvd, precisionssn_tsvd, recallssn_tsvd, rsts_tsvd, clustsn_tsvd =
clustring_experi(method, Wtsvd, label, label_counts; noc=clnoc, normalization=normalization,
                nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
@show wavg_pren_tsvd, wavg_recn_tsvd # Baron(0.8220214208322277, 0.851496864276417)
ami(ilabel, clustsn_tsvd[1]) # Baron(0.7843344572922389)

# boxplot for ISVD
@rput Wtsvd
R"""
if (dataset == "Segerstolpe") {
    ylim = c(-200,200)
} else if (dataset == "Xin") {
    ylim = c(-300, 300)
} else {
    ylim = c(-4, 2)
}
Wtsvd %>%
  reshape2::melt(varnames = c("cell", "PC"),
                 value.name = "scores") %>%
  mutate(PC = factor(PC), label = label[cell]) %>%
  ggplot(aes(PC, scores / 1000, fill = PC)) +
  geom_boxplot(color = "grey30", outlier.shape = NA,
               show.legend = FALSE) +
  labs(x = "gene PC", y = bquote("scores ("~10^3~")")) +
  scale_x_discrete(labels = 1:9) +
  facet_wrap(~ label, nrow = 3) +
  scale_fill_brewer(palette = "Set3") +
  coord_cartesian(ylim = ylim) +
  theme_classic()
  fullname <- file.path(subworkpath, dataset, paste0(dataset,"_TSVD_boxplot.png"))
  ggsave(fullname)
"""

m = 0.1
Hisvd = Htisvd'
for i in 1:gtnoc
    @show i#; @show H[i,:][H[i,:].>m]; @show H[i,:][H[i,:].<-m];  @show genename[H[i,:].>m]; @show genename[H[i,:].<-m];
    sindps = sortperm(Hisvd[i,:][Hisvd[i,:].>m],rev=true); sindps_cut = sindps[1:(min(end,3))]
    sindns = sortperm(Hisvd[i,:][Hisvd[i,:].<-m]); sindns_cut = sindns[1:(min(end,3))]
    genes_p = genename[Hisvd[i,:].>m][sindps_cut]; cnt_p = length(sindps)
    maxval = isempty(sindps_cut) ? nothing : maximum(Hisvd[i,:][Hisvd[i,:].>m])
    genes_n = genename[Hisvd[i,:].<-m][sindns_cut]; cnt_n = length(sindns)
    minval = isempty(sindns_cut) ? nothing : minimum(Hisvd[i,:][Hisvd[i,:].<-m])
    @show genes_p, maxval, cnt_p
    @show genes_n, minval, cnt_n
end

α=0.004; βw = 0; gtnoc = 9 # Baron
α=0.001; βw = 0; gtnoc = 9 # Muraro
α=0.05; βw = 0; gtnoc = 7 # Segerstolpe
α=0.003; βw = 0; gtnoc = 6 # Xin
for α in [collect(0.001:0.001:0.01)...,collect(0.02:0.01:0.1)...]
β1 = βw; β2= βw; α1 = α2 = α
r=0.3; tol=1e-7
T = eltype(U)
maxiter = Int(ceil(log(eps(T))/log(r))) #lcsvd_maxiter
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    r=r, useprecond=false, usedenoiseUVt=false, optim_method = :lbfgs,
    uselv=false, maxiter = maxiter, inner_maxiter = 1000, store_trace = false,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)

rtpcb = @elapsed rst1 = LCSVD.solve!(alg, T.(Xr), U, V, D, M1, N1t);
W1, H1 = rst1.W, rst1.Ht' # genes, cells
LCSVD.flip2makepos!(W1,H1)
LCSVD.normalizeW!(W1,H1) # genes(normalized), cells(&power)
Wpcb, Htpcb = Array(H1'), W1 # cells(&power), genes(normalized)

method = :hclust; normalization = true; nepmt = 1; hc_linkage = :average; hc_h = nothing
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, rsts, clustsn =
clustring_experi(method, Wpcb, label, label_counts; noc=clnoc, normalization=normalization,
                nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
ami_pcb = ami(ilabel, clustsn[1]) # Baron(α=0.004, 0.8107586979659679)

@show α, wavg_pren, wavg_recn, ami_pcb # Baron(α=0.004, 0.8356932805866343, 0.8107586979659679) Muraro(0.007, 0.7878366617123049)
end

m = 0.1
Hpcb = Htpcb'
for i in 1:gtnoc
    @show i#; @show Hpcb[i,:][Hpcb[i,:].>m]; @show Hpcb[i,:][Hpcb[i,:].<-m];  @show genename[Hpcb[i,:].>m]; @show genename[Hpcb[i,:].<-m];
    sindps = sortperm(Hpcb[i,:][Hpcb[i,:].>m],rev=true); sindps_cut = sindps[1:(min(end,3))]
    sindns = sortperm(Hpcb[i,:][Hpcb[i,:].<-m]); sindns_cut = sindns[1:(min(end,3))]
    genes_p = genename[Hpcb[i,:].>m][sindps_cut]; cnt_p = length(sindps)
    maxval = isempty(sindps_cut) ? nothing : maximum(Hpcb[i,:][Hpcb[i,:].>m])
    genes_n = genename[Hpcb[i,:].<-m][sindns_cut]; cnt_n = length(sindns)
    minval = isempty(sindns_cut) ? nothing : minimum(Hpcb[i,:][Hpcb[i,:].<-m])
    @show genes_p, maxval, cnt_p
    @show genes_n, minval, cnt_n
end

@rget genename # gene names
genenames = []; genenamenegs = []; ngene = []; nneggene = []
for i in 1:9
    v = Htpcb[:,i][Htpcb[:,i].>0.1]
    push!(ngene,v)
    gv = genename[Htpcb[:,i].>0.1]
    push!(genenames,gv)
    v = Htpcb[:,i][Htpcb[:,i].< -0.1]
    push!(nneggene,v)
    gv = genename[Htpcb[:,i].< -0.1]
    push!(genenamenegs,gv)
end

ngene = [1, # INS
        1,  # SST
        11, # CELA3A,CELA3B,CLPS,CPA1,CTRB1,CTRB2,PLA2G1B,PRSS1,PRSS2,REG1A(0.63),REG1B
        2,  # GCG(0.95),TTR(0.27)
        1,  # PPY
        1,  # IAPP
        9,  # CELA2A,CELA3A(0.33),CELA3B,CLPS,CPA1,CTRB1,PLA2G1B,PRSS1,PRSS2
        17, # ACTG1,EEF1A1(0.29),FTH1,FTL,GAPDH,RPL10,RPL13,RPL13A,RPL37A,RPL41,RPL7A,RPS12,RPS2,TIMP1,TMSB4X,TPT1,TTR
        4]  # COL1A1,CPA1,REG1B,TIMP1(0.29)
nneggene = [0,
        0,
        0,
        0,
        0,
        0,
        3, # REG1A(-0.72),REG1B,REG3A
        0,
        4]  # ACTG1,CTRB1,CTRB2(-0.73),SERPINA3

# boxplot for PCB
@rput Wpcb
R"""
if (dataset == "Segerstolpe") {
    ylim = c(-10, 300)
} else if (dataset == "Xin") {
    ylim = c(-10, 400)
} else {
    ylim = c(-1, 4)
}
Wpcb %>%
  reshape2::melt(varnames = c("cell", "PC"),
                 value.name = "scores") %>%
  mutate(PC = factor(PC), label = label[cell]) %>%
  ggplot(aes(PC, scores / 1000, fill = PC)) +
  geom_boxplot(color = "grey30", outlier.shape = NA,
               show.legend = FALSE) +
  labs(x = "gene PC", y = bquote("scores ("~10^3~")")) +
  scale_x_discrete(labels = 1:9) +
  facet_wrap(~ label, nrow = 3) +
  scale_fill_brewer(palette = "Set3") +
  coord_cartesian(ylim = ylim) +
  theme_classic()
  fullname <- file.path(subworkpath, dataset, paste0(dataset,"_PCB_boxplot0.05.png"))
  ggsave(fullname)
"""

# heatmap for PCB and SMA
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# mtd = "pcb"; Whm = Wpcb; Hhm = Htpcb'
mtd = "sma"; Whm = Wsma; Hhm = Hsma'
noc = size(Whm,2)
Wdivision = 2
sz = (262, 654) # f.scene.viewport.val
for i in 1:Wdivision
    f = Figure(size=sz)
    ax = AMakie.Axis(f[1, 1], xaxisposition=:top, xlabelsize=10, xticklabelsize=10, xgridvisible=false, xticks = collect(1:noc))
    rowsizeq = size(Whm,1)÷Wdivision
    rows = (i==Wdivision ? size(Whm,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    joint_limits = (-maximum(Whm), maximum(Whm))
    hm1 = heatmap!(ax, Whm[rows,:]', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,dataset,"$(mtd)_heatmap_W$i.png"),f)
end
# f = Figure()
# ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,1000000)))
# ht1 = hist!(ax,vec(Whm),bin=5)
# save(joinpath(subworkpath,"pcb_histo1000000_W.png"),f)
Hdivision = 2#16
hr = 0.1
sz = (840, 207) # f.scene.viewport.val
for i in 1:Hdivision
    f = Figure(size=sz)
    ax = AMakie.Axis(f[1, 1], xaxisposition=:top, ylabelsize=10, yticklabelsize=10, ygridvisible=false, yticks = collect(1:noc))
    colsizeq = size(Hhm,2)÷Hdivision
    cols = colsizeq*(i-1)+1:(i==Hdivision ? size(Hhm,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (-maximum(Hhm)*hr, maximum(Hhm)*hr)
    hm1 = heatmap!(ax, Hhm[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidexdecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,dataset,"$(mtd)_heatmap_H$i.png"),f)
end
# f = Figure()
# ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
# ht1 = hist!(ax,vec(Hhm),bin=5)
# save(joinpath(subworkpath,"pcb_histo100_H.png"),f)
#==================== HALS =====================================#

# HALS
prefix="hals"; @show prefix
rtnndsvd = @elapsed Wnnd, Hnnd = NMF.nndsvd(T.(Xr), gtnoc, variant=:ar);
mfmethod = :HALS; αhals=0.1; maxiter = 60; tol=-1
αhals = 0.1
W, H = copy(Wnnd), copy(Hnnd);
rthals = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), T.(Xr), W, H)
LCSVD.normalizeW!(W,H)
Whals, Hthals = Array(H'), W
fv_hals = LCSVD.fitd(Xr,Hthals*Whals') # Fit : 0.9908585679152878
normX2 = norm(X)^2
Xy = X*Hthals*inv(Hthals'*Hthals)*Hthals'; pve_hals = norm(Xy)^2/normX2 # PVE : 0.9642667696323223
score = copy(Whals); loading = copy(Hthals')
LCSVD.normalizeW!(score,loading); sw_hals = norm(score,1) # Sparsity : 312.3017367520704

method = :hclust; normalization = true; nepmt = 1; hc_linkage = :average; hc_h = nothing
pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren_hals, wavg_recn_hals, precisionssn_hals,
recallssn_hals, rsts_hals, clustsn_hals = clustring_experi(method, Whals, label, label_counts; noc=clnoc,
                normalization=normalization, nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
@show αhals, wavg_pren_hals, wavg_recn_hals # Baron(0.1, 0.8208461382044688, 0.8149331440066264), Muraro(αhals=0.1 doesn't work)
ami_hals = ami(ilabel, clustsn_hals[1]) # Baron(αhals=0.1, 0.7791362611425111)

m = 0.1
H = Hthals'
for i in 1:gtnoc
    @show i#; @show H[i,:][H[i,:].>m]; @show H[i,:][H[i,:].<-m];  @show genename[H[i,:].>m]; @show genename[H[i,:].<-m];
    sindps = sortperm(H[i,:][H[i,:].>m],rev=true); sindps_cut = sindps[1:(min(end,3))]
    sindns = sortperm(H[i,:][H[i,:].<-m]); sindns_cut = sindns[1:(min(end,3))]
    genes_p = genename[H[i,:].>m][sindps_cut]; cnt_p = length(sindps)
    maxval = isempty(sindps_cut) ? nothing : maximum(H[i,:][H[i,:].>m])
    genes_n = genename[H[i,:].<-m][sindns_cut]; cnt_n = length(sindns)
    minval = isempty(sindns_cut) ? nothing : minimum(H[i,:][H[i,:].<-m])
    @show genes_p, maxval, cnt_p
    @show genes_n, minval, cnt_n
end

@rget genename # gene names
genenames = []; genenamenegs = []; ngene = []; nneggene = []
th = 0.1
for i in 1:9
    v = Hthals[:,i][Hthals[:,i].>th]
    push!(ngene,v)
    gv = genename[Hthals[:,i].>th]
    push!(genenames,gv)
    v = Hthals[:,i][Hthals[:,i].< -th]
    push!(nneggene,v)
    gv = genename[Hthals[:,i].< -th]
    push!(genenamenegs,gv)
end

# save(joinpath(subworkpath,dataset,"Result_hals.jld2"),"X",X, "cell_type_label", label, "gene_name", genename,
#     "Whals", Whals, "Hthals", Hthals, "ngene", ngene, "genenames",genenames, "fithals", fv_hals, "pve_hals", pve_hals,
#     "sw_hals", sw_hals, "rtsma", rtsma, "rt_nndsvd", rt1_nndsvd,"rt_hals", rt_hals)

# boxplot for HALS
@rput Whals
R"""
if (dataset == "Segerstolpe") {
    ylim = c(-10, 300)
} else if (dataset == "Xin") {
    ylim = c(-10, 400)
} else {
    ylim = c(-1, 4)
}
Whals %>%
  reshape2::melt(varnames = c("cell", "PC"),
                 value.name = "scores") %>%
  mutate(PC = factor(PC), label = label[cell]) %>%
  ggplot(aes(PC, scores / 1000, fill = PC)) +
  geom_boxplot(color = "grey30", outlier.shape = NA,
               show.legend = FALSE) +
  labs(x = "gene PC", y = bquote("scores ("~10^3~")")) +
  scale_x_discrete(labels = 1:9) +
  facet_wrap(~ label, nrow = 3) +
  scale_fill_brewer(palette = "Set3") +
  coord_cartesian(ylim = ylim) +
  theme_classic()
  fullname <- file.path(subworkpath, dataset, paste0(dataset,"_HALS_boxplot.png"))
  ggsave(fullname)
"""

# heatmap for PCB and SMA
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

# mtd = "pcb"; Whm = Wpcb; Hhm = Htpcb'
mtd = "hals"; Whm = Whals; Hhm = Hthals'
noc = size(Whm,2)
Wdivision = 2
sz = (262, 654) # f.scene.viewport.val
for i in 1:Wdivision
    f = Figure(size=sz)
    ax = AMakie.Axis(f[1, 1], xaxisposition=:top, xlabelsize=10, xticklabelsize=10, xgridvisible=false, xticks = collect(1:noc))
    rowsizeq = size(Whm,1)÷Wdivision
    rows = (i==Wdivision ? size(Whm,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    joint_limits = (-maximum(Whm), maximum(Whm))
    hm1 = heatmap!(ax, Whm[rows,:]', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,dataset,"$(mtd)_heatmap_W$i.png"),f)
end
# f = Figure()
# ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,1000000)))
# ht1 = hist!(ax,vec(Whm),bin=5)
# save(joinpath(subworkpath,"pcb_histo1000000_W.png"),f)
Hdivision = 2#16
hr = 0.1
sz = (840, 207) # f.scene.viewport.val
for i in 1:Hdivision
    f = Figure(size=sz)
    ax = AMakie.Axis(f[1, 1], xaxisposition=:top, ylabelsize=10, yticklabelsize=10, ygridvisible=false, yticks = collect(1:noc))
    colsizeq = size(Hhm,2)÷Hdivision
    cols = colsizeq*(i-1)+1:(i==Hdivision ? size(Hhm,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (-maximum(Hhm)*hr, maximum(Hhm)*hr)
    hm1 = heatmap!(ax, Hhm[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidexdecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,dataset,"$(mtd)_heatmap_H$i.png"),f)
end

# ISVD
rtisvd = @elapsed Uisvd, H0isvd, M0, N0, Wp, Hp, Disvd = LCSVD.initpcb(Xr, gtnoc, nac; initmethod=:isvd, svdmethod=:isvd)
Visvd = copy(H0isvd')
LCSVD.normalizeW!(Uisvd,Visvd)
Wisvd, Htisvd = Array(Visvd), Uisvd
# save(joinpath(subworkpath,dataset,"Result_isvd.jld2"),"Xr",Xr, "cell_type_label", label, "Wisvd", Wisvd, "Htisvd", Htisvd)

save(joinpath(subworkpath,dataset,"$(dataset)_Result_sp090925.jld2"), "gtnoc", gtnoc,
    "Wtsvd", Wtsvd, "Httsvd", Httsvd, "rttsvd", rttsvd,
    "Wpcb", Wpcb, "Htpcb", Htpcb, "rtpcb", rtpcb, "α", α, "β", β,
    "Wsma", Wsma, "Htsma", Htsma, "rtirlba", rtirlba, "rtsma", rtsma,
    "Whals", Whals, "Hthals", Hthals, "rtnndsvd", rtnndsvd, "rthals", rthals, "αhals", αhals)

#==================== examine rnaSeq data ================================#
dd = load(joinpath(subworkpath,"Xr.jld2"))
Xr = dd["Xr"] # row: gene, column: cell
dd2 = load(joinpath(subworkpath,dataset,"Result_sp_s0324.jld2"))
genename = dd2["gene_name"]; genename = dropdims(genename, dims=2)

f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
v = sum.(eachrow(Xr))

rng = 1:5
indices =  partialsortperm(v, rng, rev=true)
map((v,gn)->lines!(ax,v,label=gn), eachrow(Xr[indices,:]), genename[indices]) # sum = 2.663743e6~1.201136e6
axislegend(ax; position = :rt)
save(joinpath(subworkpath,dataset,"gene_Top$(rng[1])to$(rng[end])_genes.png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend] # map(ln->delete!(ax,ln), lns1)

rng = 1:5; rank = [3,5]; cellrng = Colon() # 700:1000
indices =  partialsortperm(v, rng, rev=true)
map((v,gn)->lines!(ax,v[cellrng],label=gn), eachrow(Xr[indices[rank],:]), genename[indices[rank]]) # sum = 2.663743e6~1.201136e6
axislegend(ax; position = :rt)
save(joinpath(subworkpath,dataset,"gene_$(rank[1])and$(rank[end])_genes.png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]

rng = 6:10
indices =  partialsortperm(v, rng, rev=true)
map(v->lines!(ax,v), eachrow(Xr[indices,:])) # sum = 779029.0~374835.0
save(joinpath(subworkpath,dataset,"gene_Top$(rng[1])to$(rng[end])_genes.png"),f,px_per_unit=2)
empty!(ax)

indices =  partialsortperm(v, 8100:8110, rev=false) # sum = 400
map(v->lines!(ax,v), eachrow(Xr[indices,:]))
save(joinpath(subworkpath,dataset,"gene_MiddleHigh10_genes.png"),f,px_per_unit=2)
empty!(ax)

indices =  partialsortperm(v, 4100:4110, rev=false) # sum = 44
map(v->lines!(ax,v), eachrow(Xr[indices,:]))
save(joinpath(subworkpath,dataset,"gene_Middle10_genes.png"),f,px_per_unit=2)
empty!(ax)

indices =  partialsortperm(v, 1100:1110, rev=false) # sum = 2
map(v->lines!(ax,v), eachrow(Xr[indices,:]))
save(joinpath(subworkpath,dataset,"gene_MiddleLow10_genes.png"),f,px_per_unit=2)
empty!(ax)

function plot_genes!(ax, gene_names::Vector{String}; cellrng=Colon())
    gname = dropdims(genename,dims=2)
    ngenes = length(gname)
    indices = Int[]
    for (idx, g) in enumerate(gene_names)
        idx = findfirst(gn->gn==g,gname)
        idx !== nothing && push!(indices, idx)
    end
    cellrng = cellrng == Colon() ? range(1,size(Xr,2)) : cellrng
    lns = map((v,gn)->lines!(ax,cellrng,v[cellrng],label=gn), eachrow(Xr[indices,:]), gname[indices])
    axislegend(ax; position = :rt)
    lns
end
cellrng = Colon(); cellrngstr = cellrng == Colon() ? "all" : "$(cellrng[1])to$(cellrng[end])"
gns2find = ["TTR", "CLU", "GNAS"]
plot_genes!(ax, gns2find; cellrng=cellrng)
save(joinpath(subworkpath,dataset,"gene_TTR_CLU_GNAS_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]

cellrng = 1:500; cellrngstr = cellrng == Colon() ? "all" : "$(cellrng[1])to$(cellrng[end])"
f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
gns2find = ["TIMP1","COL1A1","CPA1","REG1B"]
plot_genes!(ax, gns2find; cellrng=cellrng)
save(joinpath(subworkpath,dataset,"gene_COL1A1_CPA1_REG1B_TIMP1_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]
f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
gns2find = ["CTRB2", "ACTG1", "CTRB1", "SERPINA3"]
plot_genes!(ax, gns2find; cellrng=cellrng)
save(joinpath(subworkpath,dataset,"gene_CTRB2_ACTG1_CTRB1_SERPINA3_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]
f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
cellrng = Colon();
cellrngstr = cellrng == Colon() ? "all" : "$(cellrng[1])to$(cellrng[end])"
gns2find = ["CTRB2","TIMP1"]
plot_genes!(ax, gns2find; cellrng=cellrng)
save(joinpath(subworkpath,dataset,"gene_TIMP1_COL1A1_CTRB2_ACTG1_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]
