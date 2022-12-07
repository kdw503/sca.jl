using Pkg
import Base:pathof

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

Pkg.activate(".")

#using MultivariateStats # for ICA
using Images, Convex, SCS, LinearAlgebra, Printf, Colors, MAT, NMF
using FakeCells, AxisArrays, ImageCore, MappedArrays
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using SymmetricComponentAnalysis
SCA = SymmetricComponentAnalysis

Images.save("dummy.png",rand(2,2)) # If we didn't call this line before `using PyPlot`,
                            # ImageMagick is used when saving image. That doesn't
                            # properly close the file after saving.
using PyPlot
scapath = joinpath(dirname(pathof(SymmetricComponentAnalysis)),"..")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))

plt.ioff()

# ARGS =  [":natural","50","[0,100]","[5,1,0.1,10,0.01]",":balance",":balance3",":nndsvd"]
# julia C:\Users\kdw76\WUSTL\Work\julia\sca\orlexpr.jl :natural 50 [0,100] [5,1,0.1,10,0.01] :balance :balance3 [:nndsvd]
# julia $MYSTORAGE/work/julia/sca/orlexpr.jl :natural 50 [0,100] [5,1,0.1,10,0.01] :balance :balance3 [:nndsvd]
SNR = dataset = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]));
λs = eval(Meta.parse(ARGS[3])); βs = eval(Meta.parse(ARGS[4])) # be careful spaces in the argument
initpwradj = eval(Meta.parse(ARGS[5])); pwradj = eval(Meta.parse(ARGS[6]));
initmethod = eval(Meta.parse(ARGS[7])); initfn=SCA.nndsvd2; weighted = :none; 
Msparse = false; order = 1; Wonly = true; sd_group = :column; identityM=false
store_trace = true; store_trace = false
if dataset == :cbclface
    filepath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
    nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
    X = zeros(nRow*nCol,nFace)
    for i in 1:nFace
        fname = "face"*@sprintf("%05d",i)*".pgm"
        img = load(joinpath(filepath,fname))
        X[:,i] = vec(img)
    end
    ncls = ncells = 49; gridcols=Int(ceil(sqrt(ncells))); borderwidth=1
    signedcolors = (colorant"green1", colorant"white", colorant"magenta")
elseif dataset == :orlface # :orlface : 112X92, lengthT = 400, ncomponents = 49; 1iter:10min(convex)
    dirpath = joinpath(datapath,"ORL_Face")
    nRow = 112; nCol = 92; nPeople = 40; nFace = 10; imgsz = (nRow, nCol); lengthT = nPeople*nFace
    X = zeros(nRow*nCol,lengthT)
    for i in 1:nPeople
        subdirpath = "s$i"
        for j in 1:nFace
            fname = "$j.bmp"
            img = load(joinpath(dirpath,subdirpath,fname))
            X[:,i] = vec(Gray.(img))
        end
    end
    ncls=ncells = 25; gridcols=Int(ceil(sqrt(ncells))); borderwidth=1
    signedcolors = (colorant"green1", colorant"white", colorant"magenta")
elseif dataset == :natural # :natural : 12X12, lengthT=100000, ncomponents = 64; 1iter : 2hr(convex)
    frn = "IMAGES.mat"
    dd = matread(frn); img = dd["IMAGES"]
    
    # TODO: need to implement On/Off-contrast filter
    num_trials=1000; batch_size=100; lengthT = num_trials*batch_size
    sz=12; imgsz = (sz, sz); l=sz^2
    num_images=size(img,3); image_size=512
    BUFF=4; # border
    X=zeros(2*l,lengthT);
    for t=1:num_trials
        # choose an image for this batch
        i=Int(ceil(num_images*rand()));
        this_image=img[:,:,i] #reshape(IMAGES(:,i),image_size,image_size)';
        # extract subimages at random from this image to make data vector X(64 X batch_size)
        for i=1:batch_size
            r=Int(BUFF+ceil((image_size-sz-2*BUFF)*rand()));
            c=Int(BUFF+ceil((image_size-sz-2*BUFF)*rand()));
            patchp = this_image[r:r+sz-1,c:c+sz-1]
            patchm = -this_image[r:r+sz-1,c:c+sz-1]
            patchp[patchp.<0].=0; patchm[patchm.<0].=0
            X[:,(t-1)*batch_size+i]=[reshape(patchp,l,1); reshape(patchm,l,1)]
        end
    end
    ncls=ncells=72; gridcols=12; borderwidth=1 # gridcols=Int(ceil(sqrt(ncells))); 
    signedcolors = (colorant"green1", colorant"white", colorant"magenta")
else
    if false # 7cells
        gtncells = 7; imgsz = (40,20); ncls = 15
    else
        gtncells = 2; imgsz = (20,30); ncls = 6
    end
    lengthT=1000; jitter=0
    initpwradj=:balance; identityM=false
    fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"

    X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell(fname, gt_ncells=gtncells, fovsz=imgsz,
            imgsz=imgsz, ncells=ncls, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtW = fakecells_dic["gtW"]; gtH = fakecells_dic["gtH"]
    W3,H3 = copy(gtW), copy(gtH')
    fprefixgt = "GT_$(SNR)B_n$(ncls)"
    imsaveW(fprefixgt*"_W.png", W3, imgsz, borderwidth=1)
    imsaveH(fprefixgt*"_H.png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
    if identityM
        rt1 = @elapsed W0n, H0n, Mwn, Mhn, Wpn, Hpn = initsemisca(X, ncells, poweradjust=:normalize,
                            initmethod=initmethod, initfn=initfn) # initfn works only when initmethod==:custom
        Mwn0, Mhn0 = copy(Mwn), copy(Mhn);
    end
    gridcols=ncells; borderwidth=1
    signedcolors = (colorant"green1", colorant"white", colorant"magenta")
end
rt1 = @elapsed W0b, H0b, Mwb, Mhb, Wpb, Hpb = initsemisca(X, ncells, poweradjust=initpwradj,
                            initmethod=initmethod, initfn=initfn)
fprefix0 = "SCA_$(dataset)_nc$(ncells)_Convex_$(initpwradj)"
Mwb0, Mhb0 = copy(Mwb), copy(Mhb)
W1,H1 = copy(W0b), copy(H0b); normalizeWH!(W1,H1)
imsaveW(fprefix0*"_$(initmethod)_rt$(rt1).png",W1,imgsz, gridcols=gridcols, colors=signedcolors, borderval=0.5,
                borderwidth=borderwidth)

method = :sparseNMF
if method == :ssca
for  β in  βs
    @show β
    maxiter = 50; pwradj = :balance2; weighted = :none
    Msparse = false; order = 1; sd_group = :column; 
    λ=λ1=λ2=λs[1]; β1=1.5; β2=βs[1]
    initpwradjstr = initpwradj==:balance ? "blnc" : "nmlz"
    pwradjstr = pwradj==:balance ? "blnc" :
                pwradj==:normalize ? "nmlz" :
                pwradj==:balance2 ? "blnc2" :
                pwradj==:balance3 ? "blnc3" : "none"
    sparpwrstr = pwradj==:balance2 ? "M2" : pwradj==:balance3 ? "M$(order)" : ""
    sdgroupstr = sd_group==:column ? "col" : sd_group==:component ? "comp" : "pix"
    if identityM  
        fprefix0 = "SCA_$(dataset)B_n$(ncls)_Cnvx_$(sdgroupstr)_$(initmethod)_nmlzWH_idntyM_$(pwradjstr)"
    else
        fprefix0 = "SCA_$(dataset)B_n$(ncls)_Cnvx_$(sdgroupstr)_$(initmethod)_$(initpwradjstr)_$(pwradjstr)"
    end
    paramstr="_wgt$(weighted)_WH$(order)$(sparpwrstr)_lm$(λ)_bw$(β1)_bh$(β2)"
    fprefix = fprefix0*paramstr

    # Mwn, Mhn = copy(Mwn0), copy(Mhn0)
    Mwb, Mhb = copy(Mwb0), copy(Mhb0)
    if identityM
        Mwi = Matrix(1.0I,ncells,ncells); Mhi = Matrix(1.0I,ncells,ncells)
        rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mwi,Mhi,W0n,H0n,λ1,λ2,β1,β2,maxiter,Msparse,order;
            poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, SNR=SNR, store_trace=store_trace, weighted=weighted, decifactor=4)
        @show norm(I-Mw*Mh)^2, norm(W0n*Mw,1), norm(W0n*(I-Mw*Mh)*H0n)^2
        W2,H2 = copy(W0n*Mw), copy(Mh*H0n)
    else
        rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mwb,Mhb,W0b,H0b,λ1,λ2,β1,β2,maxiter,Msparse,order;
        poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, SNR=SNR, store_trace=store_trace, weighted=weighted, decifactor=4)
        @show norm(I-Mwb0*Mhb0)^2, norm(W0b*Mwb0,1)
        @show norm(I-Mw*Mh)^2, norm(W0b*Mw,1), norm(W0b*(I-Mw*Mh)*H0b)^2
        W2,H2 = copy(W0b*Mw), copy(Mh*H0b)
    end
    # match with ground truth
    normalizeWH!(W2,H2); W3,H3 = sortWHslices(W2,H2)
    dataset == :natural && (W3 = W3[1:l,:]-W3[l+1:2*l,:])
    imsaveW(fprefix*"_iter$(iter)_rt$(rt2)_W.png", W3, imgsz, gridcols=gridcols, colors=signedcolors, borderval=0.5, borderwidth=1)
    imsaveH(fprefix*"_iter$(iter)_rt$(rt2)_H.png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
    ax = plotW([log10.(f_xs) log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_rt$(rt2)_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
        legendstrs = ["log(f_x)","log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
    ax = plotW(log10.(f_xs), fprefix*"_rt$(rt2)_fxplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(f_x)"], legendloc=1, separate_win=false)
    ax = plotW([log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_rt$(rt2)_xplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
        legendstrs = ["log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
    ax = plotW(H3[1:8,1:100]', fprefix*"_rt$(rt2)_Hplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
        legendstrs = ["log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)

#=
    # make W components be positive mostly
    for i in 1:ncells
        (w,h) = view(W2,:,i), view(H2,i,:)
        psum = sum(w[w.>0]); nsum = -sum(w[w.<0])
        psum < nsum && (w .*= -1; h .*= -1) # just '*=' doesn't work
    end
    # match with ground truth
    mssd, ml, ssds = matchednssd(gtW,W2)
    mssdH = ssdH(ml,gtH,H2')

    neworder = zeros(Int,length(ml))
    for (gti, i) in ml
        neworder[gti]=i
    end
    for i in 1:ncells
        i ∉ neworder && push!(neworder,i)
    end
    W3,H3 = copy(W2[:,neworder]), copy(H2[neworder,:])
    imsaveW(fprefix*"_iter$(iter)_rt$(rt2)_W_matched_ssd$(mssd).png", W3, imgsz, borderwidth=1)
    imsaveH(fprefix*"_iter$(iter)_rt$(rt2)_H_matched_ssd$(mssdH).png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
=#
end

elseif method == :cd

# CD Test
datart1scd = []; datart2scd = []; mssdscd = []
println("ncells=$ncells")
runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells, variant=:ar)
datart1cd=[]; datart2cd=[]; mssdcd=[]
for α in [0.1] # best α = 0.1
    println("α=$(α)")
    runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=50, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
    normalizeWH!(W88,H88)
    # mssd88, ml88, ssds = matchednssd(gtW,W88)
    # mssdH88 = ssdH(ml88,gtH,H88')
    # push!(datart1cd, runtime1)
    # push!(datart2cd, runtime2)
    # push!(mssdcd, mssd88)
    dataset == :natural && (W88 = W88[1:l,:]-W88[l+1:2*l,:])
    imsaveW("CD_$(SNR)_nc$(ncells)_a$(α)_rt2$(runtime2)_W.png", W88, imgsz, gridcols=gridcols, colors=signedcolors, borderwidth=1)
    imsaveH("CD_$(SNR)_nc$(ncells)_a$(α)_rt2$(runtime2)H.png", H88, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
    # neworder = zeros(Int,length(ml88))
    # for (gti, i) in ml88
    #     neworder[gti]=i
    # end
    # for i in 1:ncells
    #     i ∉ neworder && push!(neworder,i)
    # end
    # W3cd,H3cd = copy(W88[:,neworder]), copy(H88[neworder,:])
    # imsaveW("size_CD_nc$(ncells)_a$(α)_W_matched.png", W3cd, imgsz, borderwidth=1)
    # imsaveH("size_CD_nc$(ncells)_a$(α)_H_matched.png", H3cd, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
end
# push!(datart1scd, datart1cd)
# push!(datart2scd, datart2cd)
# push!(mssdscd, mssdcd)

# datart1cd = getindex.(datart1scd,1)
# datart2cd = getindex.(datart2scd,1)
# datartcd = datart1cd+datart2cd
elseif method == :sparseNMF
    #runtime1 = @elapsed Wsn, Hsn = NMF.nndsvd(X, ncells, variant=:ar)
    m,n = size(X)
    Wsn = rand(m,ncells); Hsn=rand(ncells,n)
    sparseW=0.0; sparseH=0.85
    runtime2 = @elapsed pens = SparseNMF!(X,Wsn,Hsn; sparseW=sparseW, sparseH=sparseH, maxiter=500)
    dataset == :natural && (Wsn = Wsn[1:l,:]-Wsn[l+1:2*l,:])
    # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
    signedcolors = (colorant"black", colorant"gray", colorant"white")
    imsaveW("SparseNMF_$(SNR)_nc$(ncells)_sw$(sparseW)_sh$(sparseH)_rt$(runtime2)_W.png", Wsn, imgsz, gridcols=gridcols, colors=signedcolors, borderwidth=1)
    ax = plotW(log10.(pens), "SparseNMF_rt$(runtime2)_fxplot.png"; title="convergence (Sparse NMF)", xlbl = "iteration", ylbl = "log(penalty)",
        legendstrs = ["log(f_x)"], legendloc=1, separate_win=false)
end

