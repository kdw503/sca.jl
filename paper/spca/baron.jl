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

#=============== seqRNA ===============#
##### SMA method (using Rcall)

R"""
library(scRNAseq)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)
library(epca)
library(Matrix) # as.matrix
library(ggplot2)
"""

R"""
dat <- BaronPancreasData()
dim(dat)
gene.select <- !!apply(counts(dat), 1, sd) # select genes with high variance
label.select <- colData(dat) %>% # select labels with more than 100 cells
                data.frame() %>%
                dplyr::count(label) %>%
                filter(n > 100)
dat1 <- dat[gene.select, colData(dat)$label %in% label.select$label]
"""

R"""
count <- counts(dat1)
label <- setNames(factor(dat1$label), colnames(dat1)) # cell type label
genename <- as.matrix(rownames(dat1))                       # gene names
scar <- sca(t(count), k = 9, gamma = 12,
               center = F, scale = F,
               epsilon = 1e-3)
n.gene <- apply(!!scar$loadings, 2, sum) # sum of non zero loadings number for each gene
ngene_sma <- n.gene
Wsma <- as.matrix(scar$scores)
Hsma <- as.matrix(scar$loadings)
"""
@rget Wsma
@rget Hsma
@rget ngene_sma
@rget label
label = Array(label)
#@rget scar
#W = scar[:scores]; H = scar[:loadings]; label = scar[:label]
# save Baron data set as JLD2 file
save(joinpath(subworkpath,"Xr.jld2"),"Xr", count, "label", label)
dd = load(joinpath(subworkpath,"Xr.jld2"))
Xr = dd["Xr"]

# initialization only
rtirlba = @elapsed R"""  ## initialize
  x = scale(x = t(count),
            center = F,
            scale = F)
  # s = RSpectra::svds(x, k)
  s = irlba::irlba(x, 9, tol = 1e-10)
  z = s$u
  b = diag(s$d)
  y = s$v
  score = sqrt(sum(s$d ^ 2))
  diff = c(z = Inf, y = Inf)
"""

# Without shrink SMA
rtsma = @elapsed R"""
factors_sma <- sca(t(count), k = 9, gamma = Inf,
               center = F, scale = F,
               epsilon = 1e-3)
"""
@rget factors_sma
rW = factors_sma[:scores]; rH = Array(factors_sma[:loadings]')
normX2 = norm(X)^2
fv = LCSVD.fitd(X,rW*rH)
Xy = X*rH'*inv(rH*rH')*rH; pve = norm(Xy)^2/normX2 # PVE : 0.9433741435634796
LCSVD.normalizeW!(rW,rH); sw = norm(rW,1) # Sparsity : 317.95

# n genes
i=1; @show H[i,:][H[i,:].>0.1]; @show H[i,:][H[i,:].<-0.1];  @show genename[H[i,:].>0.1]; @show genename[H[i,:].<-0.1];

# SMA Boxplot
R"""
scar$scores %>%
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
  theme_classic()
"""

##### PCB method
R"library(ggplot2)"

# Load the data set
ddseq = load(joinpath(subworkpath,"Xr.jld2"))
Xr = ddseq["Xr"]; X = Array(Xr')
label = ddseq["label"]

initmethod=:isvd; svdmethod=:isvd
#initmethod=:svd; svdmethod=:svd
rtisvd = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(Xr, noc, nac; initmethod=initmethod, svdmethod=svdmethod)
V = copy(H0'); N0t = copy(N0'); Wisvd = V*D

# boxplot for ISVD
@rput Wisvd
R"""
Wisvd %>%
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
  theme_classic()
"""

β1 = β2= 5.; α1 = α2 = 0.005
r=0.3; tol=1e-6
maxiter = Int(ceil(log(eps(eltype(Xr)))/log(r))) #lcsvd_maxiter
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    r=r, useprecond=false, usedenoiseW0H0=false, optim_method = :lbfgs,
    uselv=false, maxiter = maxiter, inner_maxiter = 1000, store_trace = false,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rtpcb = @elapsed rst1 = LCSVD.solve!(alg, Xr, U, V, D, M1, N1t);
W1, H1 = rst1.W, rst1.Ht'
LCSVD.flip2makepos!(W1,H1)
LCSVD.normalizeW!(W1,H1)
Wpcb, Htpcb = Array(H1'), W1

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

save(joinpath(subworkpath,"Baron","Result_sp_s0324.jld2"),"X",X, "cell_type_label", label, "gene_name", genename,
    "Wpcb", Wpcb, "Htpcb", Htpcb, "ngene", ngene, "nneggene", nneggene, "rtisvd", rtisvd, "rtpcb", rtpcb,
    "Wsma", Wsma, "Hsma", Hsma, "ngene_sma",ngene_sma, "rtsma", rtsma)

save(joinpath(subworkpath,"Baron","Result_sp_nn_s0322.jld2"),"X",X, "cell_type_label", label, "gene_name", genename,
    "Wpcb", Wpcb, "Htpcb", Htpcb, "ngene", ngene, "nneggene", nneggene, "genenames",genenames, "genenamenegs", genenamenegs, "rtisvd", rtisvd,"rtpcb", rtpcb,
    "Wsma", Wsma, "Hsma", Hsma, "ngene_sma",ngene_sma, "rtsma", rtsma)

# boxplot for PCB
@rput Wpcb
R"""
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
  theme_classic()
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
    save(joinpath(subworkpath,"Baron","$(mtd)_heatmap_W$i.png"),f)
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
    save(joinpath(subworkpath,"Baron","$(mtd)_heatmap_H$i.png"),f)
end
# f = Figure()
# ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
# ht1 = hist!(ax,vec(Hhm),bin=5)
# save(joinpath(subworkpath,"pcb_histo100_H.png"),f)
#==================== HALS =====================================#

# HALS
prefix="hals"; @show prefix
rt1_hals = @elapsed Whals0, Hhals0 = NMF.nndsvd(Xr, noc, variant=:ar);
mfmethod = :HALS; αhals=0.1; maxiter = 60; tol=-1
αhals = 0.1
W, H = copy(Whals0), copy(Hhals0);
rt_hals = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), Xr, W, H)
LCSVD.normalizeW!(W,H)
Whals, Hthals = Array(H'), W
normX2 = norm(Xr)^2
fv_hals = LCSVD.fitd(X,Whals*Hthals') # Fit : 0.9908585679152878
Xy = X*Hthals*inv(Hthals'*Hthals)*Hthals'; pve_hals = norm(Xy)^2/normX2 # PVE : 0.9642667696323223
score = copy(Whals); loading = copy(Hthals')
LCSVD.normalizeW!(score,loading); sw_hals = norm(score,1) # Sparsity : 312.3017367520704

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

save(joinpath(subworkpath,"Baron","Result_hals.jld2"),"X",X, "cell_type_label", label, "gene_name", genename,
    "Whals", Whals, "Hthals", Hthals, "ngene", ngene, "genenames",genenames, "fithals", fv_hals, "pve_hals", pve_hals,
    "sw_hals", sw_hals, "rtsma", rtsma, "rt_nndsvd", rt1_hals,"rt_hals", rt_hals)

# boxplot for PCB
@rput Whals
R"""
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
  theme_classic()
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
    save(joinpath(subworkpath,"Baron","$(mtd)_heatmap_W$i.png"),f)
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
    save(joinpath(subworkpath,"Baron","$(mtd)_heatmap_H$i.png"),f)
end

# TSVD
Xr = Array(X'); noc = 9
rttsvd = @elapsed Utsvd, H0tsvd, M0, N0, Wp, Hp, Dtsvd = LCSVD.initpcb(Xr, noc, nac; initmethod=:tsvd, svdmethod=:tsvd)
Vtsvd = copy(H0tsvd')
LCSVD.normalizeW!(Utsvd,Vtsvd)
Wtsvd, Httsvd = Array(Vtsvd), Utsvd
save(joinpath(subworkpath,"Baron","Result_tsvd.jld2"),"Xr",Xr, "cell_type_label", label, "Wtsvd", Wtsvd, "Httsvd", Httsvd)

#==================== examine rnaSeq data ================================#
dd = load(joinpath(subworkpath,"Xr.jld2"))
Xr = dd["Xr"] # row: gene, column: cell
dd2 = load(joinpath(subworkpath,"Baron","Result_sp_s0324.jld2"))
genename = dd2["gene_name"]; genename = dropdims(genename, dims=2)

f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
v = sum.(eachrow(Xr))

rng = 1:5
indices =  partialsortperm(v, rng, rev=true)
map((v,gn)->lines!(ax,v,label=gn), eachrow(Xr[indices,:]), genename[indices]) # sum = 2.663743e6~1.201136e6
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"Baron","gene_Top$(rng[1])to$(rng[end])_genes.png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend] # map(ln->delete!(ax,ln), lns1)

rng = 1:5; rank = [3,5]; cellrng = Colon() # 700:1000
indices =  partialsortperm(v, rng, rev=true)
map((v,gn)->lines!(ax,v[cellrng],label=gn), eachrow(Xr[indices[rank],:]), genename[indices[rank]]) # sum = 2.663743e6~1.201136e6
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"Baron","gene_$(rank[1])and$(rank[end])_genes.png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]

rng = 6:10
indices =  partialsortperm(v, rng, rev=true)
map(v->lines!(ax,v), eachrow(Xr[indices,:])) # sum = 779029.0~374835.0
save(joinpath(subworkpath,"Baron","gene_Top$(rng[1])to$(rng[end])_genes.png"),f,px_per_unit=2)
empty!(ax)

indices =  partialsortperm(v, 8100:8110, rev=false) # sum = 400
map(v->lines!(ax,v), eachrow(Xr[indices,:]))
save(joinpath(subworkpath,"Baron","gene_MiddleHigh10_genes.png"),f,px_per_unit=2)
empty!(ax)

indices =  partialsortperm(v, 4100:4110, rev=false) # sum = 44
map(v->lines!(ax,v), eachrow(Xr[indices,:]))
save(joinpath(subworkpath,"Baron","gene_Middle10_genes.png"),f,px_per_unit=2)
empty!(ax)

indices =  partialsortperm(v, 1100:1110, rev=false) # sum = 2
map(v->lines!(ax,v), eachrow(Xr[indices,:]))
save(joinpath(subworkpath,"Baron","gene_MiddleLow10_genes.png"),f,px_per_unit=2)
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
save(joinpath(subworkpath,"Baron","gene_TTR_CLU_GNAS_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]

cellrng = 1:500; cellrngstr = cellrng == Colon() ? "all" : "$(cellrng[1])to$(cellrng[end])"
f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
gns2find = ["TIMP1","COL1A1","CPA1","REG1B"]
plot_genes!(ax, gns2find; cellrng=cellrng)
save(joinpath(subworkpath,"Baron","gene_COL1A1_CPA1_REG1B_TIMP1_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]
f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
gns2find = ["CTRB2", "ACTG1", "CTRB1", "SERPINA3"]
plot_genes!(ax, gns2find; cellrng=cellrng)
save(joinpath(subworkpath,"Baron","gene_CTRB2_ACTG1_CTRB1_SERPINA3_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]
f = Figure(size=(956,318)) # f.scene.viewport.val.widths
ax = AMakie.Axis(f[1, 1], xlabel = "cell", ylabel = "expression level", title = "")
cellrng = Colon();
cellrngstr = cellrng == Colon() ? "all" : "$(cellrng[1])to$(cellrng[end])"
gns2find = ["CTRB2","TIMP1"]
plot_genes!(ax, gns2find; cellrng=cellrng)
save(joinpath(subworkpath,"Baron","gene_TIMP1_COL1A1_CTRB2_ACTG1_$(cellrngstr).png"),f,px_per_unit=2)
empty!(ax); [delete!(leg) for leg in f.content if leg isa Legend]

