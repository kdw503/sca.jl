
using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    #cd("/home/daewoo/work/julia/sca")
end

Pkg.activate(".")

using Images, Convex, SCS, LinearAlgebra
using FakeCells, AxisArrays, ImageCore, MappedArrays
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using SymmetricComponentAnalysis

Images.save("dummy.png",rand(2,2)) # If we didn't call this line before `using PyPlot`,
                            # ImageMagick is used when saving image. That doesn't
                            # properly close the file after saving.
using PyPlot

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
function plotW(W::AbstractMatrix, fn; scale=:linear, title="", xlbl="index", ylbl="intensity",
        legendstrs=string.(collect(1:size(W,2))), legendloc=1, separate_win=true)
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
    savefig(fn)
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

penaltyL1(Mw,Mh,W0,H0,λ,β1,β2) = norm(I-Mw*Mh)^2 + λ*(sca2(W0*Mw)+sca2(Mh*H0)) + β1*norm(W0*Mw,1) + β2*norm(Mh*H0,1)
penaltyL2(Mw,Mh,W0,H0,λ,β1,β2) = norm(I-Mw*Mh)^2 + λ*(sca2(W0*Mw)+sca2(Mh*H0)) + β1*norm(W0*Mw)^2 + β2*norm(Mh*H0)^2

function minpix(Mw,Mh,W0,H0,i,k,λ,β,order)
    p = size(W0,2)
    Eprev = I-Mw*Mh
    Eprevi = Eprev[i,:]; w0i = W0[:,i]
    mwk = Mw[:,k]; mwik=mwk[i]; mhk = Mh[k,:]
    Ei = Eprevi + mwik*mhk
    x = Variable(1)
    w0mwk = W0*mwk; w0mwik = w0mwk-w0i*mwik; womwikx = w0mwik+w0i*x
    sparsity = order == 1 ? norm(womwikx, 1) : sumsquares(womwikx)
    problem = minimize(sumsquares(Ei-x*mhk) + λ*sumsquares(max(0,-womwikx)) + β*sparsity)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    xsol = x.value
    # errprev = norm(Eprevi)^2; err = norm(Ei-xsol*mhk)^2
    # sparsprev = order == 1 ? β*norm(w0mwk,1) : β*norm(w0mwk)^2
    # spars = order == 1 ? β*norm(w0mwik+w0i*xsol,1) : β*norm(w0mwik+w0i*xsol)^2 
    # nnegprev = λ*norm(min.(0,w0mwk))^2
    # nneg = λ*norm(min.(0,w0mwik+w0i*xsol))^2 
    # errprev+sparsprev+nnegprev < err+spars+nneg && @show problem.status
    xsol
end

function minpix(Mw,Mh,W0,H0,l,k,λ,β,order)
    p = size(W0,2)
    x = Variable(1)
    # invertibility
    mwk = Mw[:,k]; mwlk=mwk[l]; mhl = Mh[l,:]
    Eprev = I-Mw*Mh; Eprevl = Eprev[l,:]; El = Eprevl + mwlk*mhl
    invertibility = sumsquares(El-x*mhl)
    # sparsity and non-negativity
    w0l = W0[:,l]; w0mwk = W0*mwk; w0mwlk = w0mwk-w0l*mwlk; w0mwlkx = w0mwlk+w0l*x
    sparsity = order == 1 ? norm(w0mwlkx, 1) : sumsquares(w0mwlkx)
    nnegativity = sumsquares(max(0,-w0mwlkx))
    # solve
    problem = minimize(invertibility + λ*nnegativity + β*sparsity)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    xsol = x.value
    # errprev = norm(Eprevl)^2; err = norm(El-xsol*mhk)^2
    # sparsprev = order == 1 ? β*norm(w0mwk,1) : β*norm(w0mwk)^2
    # spars = order == 1 ? β*norm(w0mwlk+w0l*xsol,1) : β*norm(w0mwlk+w0l*xsol)^2
    # nnegprev = λ*norm(min.(0,w0mwk))^2
    # nneg = λ*norm(min.(0,w0mwlk+w0l*xsol))^2
    # errprev+sparsprev+nnegprev < err+spars+nneg && @show problem.status
    xsol
end

function mincol(Mw,Mh,W0,H0,k,λ,β,order)
    p = size(W0,2)
    Eprev = I-Mw*Mh
    E = Eprev + Mw[:,k]*Mh[k,:]'
    x = Variable(p)
    mhk = Mh[k,:]'
    sparsity = order == 1 ? norm(W0*x, 1) : sumsquares(W0*x)
    problem = minimize(sumsquares(E-x*mhk) + λ*sumsquares(max(0,-W0*x)) + β*sparsity)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    xsol = x.value
    errprev = norm(Eprev)^2; err = norm(E-xsol*mhk)^2
    sparsprev = order == 1 ? β*norm(W0*Mw[:,k],1) : β*norm(W0*Mw[:,k])^2
    spars = order == 1 ? β*norm(W0*xsol,1) : β*norm(W0*xsol)^2
    nnegprev = λ*norm(min.(0,W0*Mw[:,k]))^2
    nneg = λ*norm(min.(0,W0*xsol))^2
    errprev+sparsprev+nnegprev < err+spars+nneg && @show problem.status
    xsol
end

function minMw_pixel!(Mw,Mh,W0,H0,λ,β,order)
    p = size(Mw,2)
    for k in 1:p
        for l in 1:p
            Mw[l,k] = minpix(Mw,Mh,W0,H0,l,k,λ,β,order)
        end
    end
end

function minMw_cbyc!(Mw,Mh,W0,H0,λ,β,order)
    p = size(Mw,2)
    for k in 1:p
        Mw[:,k] = mincol(Mw,Mh,W0,H0,k,λ,β,order)
    end
end

function minMw_ac!(Mw,Mh,W0,H0,λ,β,order)
    m, p = size(W0)
    x = Variable(p^2)
    Ivec = vec(Matrix(1.0I,p,p))
    A = zeros(p^2,p^2)
    SCA.directMw!(A, Mh) # vec(I-reshape(x,p,p)*Mh) == Ivec-A*x
    Aw = zeros(m*p,p^2); bw = zeros(m*p)
    SCA.direct!(Aw, bw, W0; allcomp = false) # vec(W*reshape(x,p,p)) == (Aw*x)
    spars = order == 1 ? norm(Aw*x, 1) : sumsquares(Aw*x)
    problem = minimize(sumsquares(Ivec-A*x)+ λ*sumsquares(max(0,-A*x)) + β*spars)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    Mw[:,:] .= reshape(x.value,p,p)
end

function minMwMh(Mw,Mh,W0,H0,λ,β1,β2,maxiter,order; cd_group=:pixel, imgsz=(40,20), SNR=60)
    f_xs=[]; x_abss=[]
    iter = 0
    while iter <= maxiter
        iter += 1
        Mwprev, Mhprev = copy(Mw), copy(Mh)
        if cd_group == :column
            minMw_cbyc!(Mw,Mh,W0,H0,λ,β1,order)
            minMw_cbyc!(Mh',Mw',H0',W0',λ,β2,order)
        elseif cd_group == :WH
            minMw_ac!(Mw,Mh,W0,H0,λ,β1,order)
            minMw_ac!(Mh',Mw',H0',W0',λ,β2,order)
        elseif cd_group == :pixel
            minMw_pixel!(Mw,Mh,W0,H0,λ,β1,order)
            minMw_pixel!(Mh',Mw',H0',W0',λ,β2,order)
        else
            error("Unsupproted cd_group")
        end            
        pensum = order == 1 ? penaltyL1(Mw,Mh,W0,H0,λ,β1,β2) : penaltyL2(Mw,Mh,W0,H0,λ,β1,β2)
        x_abs = norm(Mwprev-Mw)^2*norm(Mhprev-Mh)^2
        @show iter, x_abs, pensum
        if isnan(x_abs)
            Mw, Mh = copy(Mwprev), copy(Mhprev)
            iter -= 1
            break
        end
        push!(f_xs, pensum)
        push!(x_abss, x_abs)
        W2,H2 = copy(W0*Mw), copy(Mh*H0)
        normalizeWH!(W2,H2)
        if iter%10 == 0
            @show iter
            if cd_group == :column
                imsaveW("W2_SNR$(SNR)_Convex_cbyc_L$(order)_bw$(β1)_bh$(β2)_iter$(iter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            elseif cd_group == :WH
                imsaveW("W2_SNR$(SNR)_Convex_ac_L$(order)_bw$(β1)_bh$(β2)_iter$(iter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            elseif cd_group == :pixel
                imsaveW("W2_SNR$(SNR)_Convex_pbyp_L$(order)_bw$(β1)_bh$(β2)_iter$(iter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
            else
                error("Unsupproted cd_group")
            end
        end
    end
    Mw, Mh, f_xs, x_abss, iter
end

plt.ioff()

SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4])); cd_group = eval(Meta.parse(ARGS[5]))
λ = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
@show SNRs, maxiter
@show order, Wonly, cd_group
@show λ, βs
imgsize = (40,20); lengthT=1000; jitter=0
for SNR in SNRs
    @show SNR
    X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell("fakecells_sz$(imgsize)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld",
                                                fovsz=imgsize, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
    for β in βs
        β1 = β; β2 = Wonly ? 0. : β
        W0, H0, Mw, Mh = initsemisca(X, ncells, balance=true)
        normalizeWH!(W0,H0); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
        imsaveW("W0_SNR$(SNR).png",sortWHslices(W0,H0)[1],imgsz,borderwidth=1)
        Mw0, Mh0 = copy(Mw), copy(Mh);
        rt2 = @elapsed Mw, Mh, f_xs, x_abss, iter = minMwMh(Mw,Mh,W0,H0,λ,β1,β2,maxiter,order,cd_group=cd_group,SNR=SNR)
        if cd_group == :column
            fprefix = "W2_SNR$(SNR)_Convex_cbyc_L$(order)_bw$(β1)_bh$(β2)_iter$(iter)_rt$(rt2)"
        elseif cd_group == :WH
            fprefix = "W2_SNR$(SNR)_Convex_ac_L$(order)_bw$(β1)_bh$(β2)_iter$(iter)_rt$(rt2)"
        elseif cd_group == :pixel
            fprefix = "W2_SNR$(SNR)_Convex_pbyp_L$(order)_bw$(β1)_bh$(β2)_iter$(iter)_rt$(rt2)"
        else
            error("Unsupproted cd_group")
        end               
        save(fprefix*".jld", "f_xs", f_xs, "x_abss", x_abss, "SNR", SNR, "order", order, "β1", β1, "β2", β2, "rt2", rt2)
        W2,H2 = copy(W0*Mw), copy(Mh*H0)
        normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
        imsaveW(fprefix*".png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
        ax = plotW([log10.(x_abss) log10.(f_xs)], fprefix*"_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
            legendstrs = ["log(x_abs)", "log(f_x)"], legendloc=1, separate_win=false)
    end
end

# plt.show()