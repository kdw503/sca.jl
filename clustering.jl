using NeighborhoodClustering, StatsBase, Clustering

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
    uclust = unique(clust)
    celltypes = fill("",length(uclust)); cellcnts = fill(0,length(uclust))
    for (i,cl) in enumerate(uclust)
        indices = clust .== cl
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
        push!(clust_label, celltypes[findfirst(==(clust[i]),uclust)])
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

function clustring_experi(method, Worg, label, label_counts; D = nothing, normalization=true, noc=9, nepmt=20,
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
    precisionss=[]; recallss=[]; avg_precs=Float64[]; avg_recs=Float64[]
    clusts=[]; results=[]
    for i in 1:nepmt
        if method == :neighborhood
            clust = cluster(W', bs_pvalue)
        elseif method == :bootstrap
            clust = cluster_resample(W', bs_nresample, bs_pvalue)
        elseif method == :kmeans
            clustering = kmeans(W', noc; init=:kmpp, maxiter=km_maxiter, tol=1e-6, display=:none) # each column of X is a d-dimensional data point) into k clusters.
            clust = clustering.assignments
        elseif method == :dbscan
            clustering = dbscan(W', ds_radius, min_neighbors = ds_min_ngbr, min_cluster_size = ds_min_clsize)
            clust = clustering.assignments .+= 1 # vector of clusters indices, clustering.clusters, clustering.counts
        elseif method == :hclust
            # D = zeros(l,l)
            # for i in 1:l, j in i:l
            #     D[i,j] = norm(W[i,:]-W[j,:])
            # end
            # D += D'
            D = D === nothing ? pairwise(Euclidean(), W, dims=1) : D
            result = hclust(D, linkage=hc_linkage)
            clust = cutree(result; k=noc, h=hc_h)
            push!(results, result); push!(clusts, clust)
        elseif method == :gmm
            mod = GaussianMixtureClusterer(n_classes=gc_n_classes) # A Generative Mixture Model (unfitted)
            prob_belong_classes = BetaML.fit!(mod,W)
            clust = getindex.(findmax.(eachrow(prob_belong_classes)),2)
        else
            error("Unknown clustering method : $method")
        end
        clust_label = assign_celltypes(clust, label) # 119, noc
        precisions, recalls, avg_prec, avg_rec = cal_precision_recall(clust_label, label) #  0.7675450079084044, 0.7679667607042434
        push!(precisionss,precisions); push!(recallss,recalls)
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
    pre_means, pre_stds, rec_means, rec_stds, wavg_pre, wavg_rec, precisionss, recallss, results, clusts
end

# for UMAP and clustered heatmap
function reverse_dict(d::AbstractDict)
    Dict(v => k for (k, v) in d)
end
function permute_cells(clust, cells)
    cm = countmap(clust)
    dclasses = getindex.(Vec(sort(reverse_dict(cm),order=Base.Reverse)...),2)
    indices = Int[]; classboundries = Int[]
    for c in dclasses
        append!(indices, findall(s->s==c,clust))
        push!(classboundries, length(indices))
    end
    cells[:,indices], classboundries
end
function clustered_hitmap(cellsclass, classboundries, mycmap, fprex;
            num_blocks = 1, xdecimation = 100, ydecimation = 1, qlevel = 0.99)
    y,x = size(cellsclass); noc = y
    ysize = y÷ydecimation + 20; xsize = x÷(num_blocks*xdecimation) + 100
    xlimit = round(quantile(vec(cellsclass),qlevel),sigdigits=3)
    for i in 1:min(2,num_blocks)
        f = Figure(size=(xsize,ysize))
        rows = noc:-ydecimation:1
        colsizeq = x÷num_blocks
        cols = colsizeq*(i-1)+1:xdecimation:(i==num_blocks ? x : colsizeq*i)
        ax = CairoMakie.Axis(f[1, 1],width=length(cols),height=length(rows))
        joint_limits = (-xlimit, xlimit) 
        hm1 = heatmap!(ax, cellsclass[rows,cols]', colormap = mycmap, colorrange = joint_limits)
        hidedecorations!(ax)
        Colorbar(f[:, end+1], hm1)
        save(fprex*"_$(i).png",f)
        for cb in classboundries
            lnx = cb ÷ xdecimation
            if lnx >= colsizeq*(i-1)+1 && lnx <= (i==num_blocks ? x : colsizeq*i)
                lines!(ax, [lnx,lnx], [0, length(rows)], color = :black, linewidth = 0.5)
            end
        end
        save(fprex*"_wb_$(i).png",f)
    end
end
function label2int(lab)
    return findfirst(ulabel .== lab)
end
