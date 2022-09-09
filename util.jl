
function makegt(S₂,centers,H,imgsz,ncells)
    gtimg = Array{eltype(S₂)}(undef,imgsz[1],imgsz[2],ncells)
    gtH = Base.similar(H)
    for i = 1:ncells
        T = zeros(1,ncells)
        T[i] = 1.0
        img0 = makeimages(imgsz..., S₂, centers, T)
        nrm = sqrt(sum(img0.^2))
        img0 ./= nrm
        gtH[:,i] = H[:,i] .* nrm 
        gtimg[:,:,i] = dropdims(img0,dims=3)
    end
#    gtimg[:,:,end] = dropdims(mean(bg,dims=3),dims=3)
    gtimgrs = reshape(gtimg  , prod(imgsz), size(gtimg)[end])
    mxabs = maximum(abs, gtimg)
    fsc = scalesigned(mxabs)
    fcol = colorsigned()
    gtimgrs,gtH, mappedarray(fcol ∘ fsc, reshape(gtimg, Val(2)))
end

function gaussian2D(sigma, imgsz::NTuple{2}, lengthT, revent=10; fovsz::NTuple{2}=imgsz,
        jitter=0, drift = 0, bias=0.1, SNR = 10, orthogonal=true, overlaplevel=1) # should lengthT > 50
    S₂ = gaussiantemplate(Float64, sigma)
    centers = spreadcells(fovsz, sigma)
    imagecenter = fovsz.÷2
    overlap_cell_center = [imagecenter[1]-min(overlaplevel,imagecenter[1])÷1.5,imagecenter[2]+min(overlaplevel,imagecenter[2])] # to add an overlapped cell
    !orthogonal && (centers = [centers overlap_cell_center])
    ncells = size(centers, 2)
    nevents = revent*ncells*lengthT/100
    T₂ = makeintensity(lengthT, ncells, nevents) # nevents=10*ncells
    dx = rand(-jitter:jitter,2,lengthT)   # simulate some registration jitter
    dx = map(x->round(Int, x), dx .+ range(0, stop=drift, length=lengthT)')
    img₂ = makeimages(imgsz..., S₂, centers, T₂, dx)
    signalpwr = sum(img₂.^2)/length(img₂)
    bg = randn(size(img₂)).*sqrt(signalpwr/10^(SNR/10)) .+ bias
    img₂ = img₂ + bg# add noise and bg
    img₂a = AxisArray(img₂, :x, :y, :time)
    gtW, gtH, gtWimgc = makegt(S₂,centers,T₂,imgsz,ncells)
    gtbg = copy(bg)
    imgrs = Matrix(reshape(img₂a, prod(imgsz), nimages(img₂a)))
    ncells, imgrs, img₂, gtW, gtH, gtWimgc, gtbg
end

function mkimgW(W::Matrix{T},imgsz; gridcols=size(W,2), borderwidth=0, borderval=0.7,
        colors=(colorant"green1", colorant"white", colorant"magenta")) where T
    ncells = size(W,2)
    mxabs = maximum(abs, W)
    fsc = scalesigned(mxabs)
    fcol = colorsigned(colors...)
    Wcolor = Array(mappedarray(fcol ∘ fsc, reshape(W, Val(2))))

    gridsz = ((ncells-1)÷gridcols+1,gridcols)
    add_dim_sz = ntuple(i->1,Val(length(imgsz)-length(gridsz)))
    bordersz = ntuple(i->borderwidth,Val(2))
    bimgsz = imgsz.+(bordersz..., add_dim_sz...)
    gimgsz = bimgsz.*(gridsz..., add_dim_sz...).+bordersz
    fill_val = eltype(Wcolor)(borderval)
    Wrs = fill(fill_val, gimgsz...)
    for i in 1:ncells
        gi = (i-1)÷gridsz[2]+1
        gj = i-(gi-1)*gridsz[2]
        gindices = (gi-1,gj-1, (add_dim_sz.-add_dim_sz)...)
        offset = gindices.*bimgsz .+ bordersz
        rngs = ntuple(i->offset[i]+1:offset[i]+imgsz[i], length(imgsz))
        Wrs[rngs...] = reshape(Wcolor[:,i], imgsz...)
    end
    Wrs
end

function imsaveW(fname,W,imgsz; kwargs...)
    wimg = mkimgW(W,imgsz; kwargs...)
    Images.save(fname, wimg)
    nothing
end

function sortWHslices(W,H)
    pwrs = norm.(eachcol(W)).*norm.(eachrow(H))
    orderindices = sortperm(pwrs, rev=true)
    W[:,orderindices], H[orderindices,:]
end

function normalizeWH!(W,H)
    for i = 1:size(W,2)
        nrm = norm(W[:,i])
        if nrm != 0.
            W[:,i] ./= nrm
            H[i,:] .*= nrm
        end
    end
    W,H
end

function plotW(W::AbstractMatrix, fn=""; scale=:linear, title="", xlbl="index", ylbl="intensity",
        legendstrs=string.(collect(1:size(W,2))), issave=true, legendloc=1, separate_win=true)
    # tickxs = [-π, -π/2, 0, π/2, π]
    # tickxlabels = ["-π", "-π/2", "0", "π/2", "π"]
    m, p = size(W); xs = 1:m
    Wscale = scale == :log10 ? log10.(W) : W
    @show legendstrs
    if separate_win == true
        row_num = p; col_num = 1
        fig, axs = plt.subplots(row_num, 1, figsize=(4*col_num, 2),
                        gridspec_kw=Dict("width_ratios"=>ones(col_num)))
        axs[1].set_title(title)
        for i in 1:row_num*col_num
            ax = axs[i]
            ax.plot(xs, Wscale[:,i])
    #        ax.plot([minx,minx], vrng, color="blue")
    #        ax.axis([-pi,pi,vrng...])
            ax.legend(legendstrs[i], fontsize = 12, loc=legendloc)
            ax.set_ylabel(ylbl)
            # ax.set_title("W$(i)")
        end
        ax.set_xlabel(xlbl)
    else
        fig, ax = plt.subplots(1,1, figsize=(5,4))
        Wscale = scale == :log10 ? log10.(W) : W
        ax.plot(xs, Wscale)
        ax.set_title(title)
        ax.legend(legendstrs, fontsize = 12, loc=legendloc)
        xlabel(xlbl,fontsize = 12)
        ylabel(ylbl,fontsize = 12)
    end
    issave && savefig(fn)
    ax
end

function loadfakecell(fname; svd_method=:isvd, gt_ncells=7, ncells=0, lengthT=100, imgsz=(40,20),
                        fovsz=imgsz, SNR=10, jitter=0, save=true)
    if isfile(fname)
        fakecells_dic = load(fname)
        gt_ncells = fakecells_dic["gt_ncells"]
        imgrs = fakecells_dic["imgrs"]
        img_nl = fakecells_dic["img_nl"]
        gtW = fakecells_dic["gtW"]
        gtH = fakecells_dic["gtH"]
        gtWimgc = fakecells_dic["gtWimgc"]
        gtbg = fakecells_dic["gtbg"]
        imgsz = fakecells_dic["imgsz"]
    else
        @warn "$fname not found. Generating fakecell data..."
        sigma = 5
        imgsz = imgsz
        lengthT = lengthT
        revent = 10
        gt_ncells, imgrs, img_nl, gtW, gtH, gtWimgc, gtbg = gaussian2D(sigma, imgsz, lengthT, revent,
                                                                jitter=jitter, fovsz=fovsz, SNR=SNR, orthogonal=false)
        if save
            Images.save(fname, "gt_ncells", gt_ncells, "imgrs", imgrs, "img_nl", img_nl, "gtW",
                gtW, "gtH", gtH, "gtWimgc", Array(gtWimgc), "gtbg", gtbg, "imgsz", imgsz, "SNR", SNR)
            fakecells_dic = load(fname)
        else
            fakecells_dic = Dict()
            fakecells_dic["gt_ncells"] = gt_ncells
            fakecells_dic["imgrs"] = imgrs
            fakecells_dic["img_nl"] = img_nl
            fakecells_dic["gtW"] = gtW
            fakecells_dic["gtH"] = gtH
            fakecells_dic["gtWimgc"] = Array(gtWimgc)
            fakecells_dic["gtbg"] = gtbg
            fakecells_dic["imgsz"] = imgsz
            fakecells_dic["SNR"] = SNR
        end
    end
    X = imgrs
    ncells == 0 && (ncells = gt_ncells+5)
    maxindices = argmax.(eachcol(gtH))
    maxSNR_X = X[:,[maxindices...]]

    return X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X
end
