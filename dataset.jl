using MAT

function load_cbcl()
    filepath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
    nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
    X = zeros(nRow*nCol,nFace)
    for i in 1:nFace
        fname = "face"*@sprintf("%05d",i)*".pgm"
        img = load(joinpath(filepath,fname))
        X[:,i] = vec(img)
    end
    ncells = 49
    X, imgsz, nFace, ncells, 0, Dict()
end

 # :orlface : 112X92, lengthT = 400, ncomponents = 49; 1iter:10min(convex)
function load_orl()
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
    ncells = 25
    X, imgsz, lengthT, ncells, 0, Dict()
end

# :natural : 12X12, lengthT=100000, ncomponents = 72
function load_natural(;num_trials=1000, batch_size=100, sz=12, BUFF=4) # BUFF(border margin)
    frn = "IMAGES.mat"
    dd = matread(frn); img = dd["IMAGES"]
    
    lengthT = num_trials*batch_size
    imgsz = (sz, sz); l=sz^2
    num_images=size(img,3); image_size=512
    X=zeros(l,lengthT);
    for t=1:num_trials
        # choose an image for this batch
        i=Int(ceil(num_images*rand()));
        this_image=img[:,:,i] #reshape(IMAGES(:,i),image_size,image_size)';
        # extract subimages at random from this image to make data vector X(64 X batch_size)
        for i=1:batch_size
            r=Int(BUFF+ceil((image_size-sz-2*BUFF)*rand()));
            c=Int(BUFF+ceil((image_size-sz-2*BUFF)*rand()));
            patchp = this_image[r:r+sz-1,c:c+sz-1]
            X[:,(t-1)*batch_size+i]=reshape(patchp,l,1)
        end
    end
    ncells=72
    X, imgsz, lengthT, ncells, 0, Dict()
end

# :natural : 12X12, lengthT=100000, ncomponents = 72
function load_onoffnatural(;num_trials=1000, batch_size=100, sz=12, BUFF=4) # BUFF(border margin)
    frn = "IMAGES.mat"
    dd = matread(frn); img = dd["IMAGES"]
    
    lengthT = num_trials*batch_size
    imgsz = (sz, sz); l=sz^2
    num_images=size(img,3); image_size=512
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
    ncells=72
    X, imgsz, lengthT, ncells, 0, Dict()
end

function load_neurofinder()
    roi = (71:170, 31:130)
    dataname = "neurofinder.02.00.cut100250_sqrt"
    fullorgimg = load(dataname*".tif")
    orgimg = fullorgimg[roi...,:]
    # save("neurofinder.02.00.cut.gif",orgimg.*2)
    imgsz = size(orgimg)[1:end-1]; lengthT = size(orgimg)[end]
    X = Matrix{Float64}(reshape(orgimg, prod(imgsz), lengthT))
    ncells = 40; gtncells = 32
    X, imgsz, lengthT, ncells, gtncells, Dict()
end

function load_urban()
    filepath = joinpath(datapath,"Hyperspectral-Urban")
    vars = matread(joinpath(filepath,"Urban_R162.mat"))
    Y=vars["Y"]; nRow = Int(vars["nRow"]); nCol = Int(vars["nCol"]); nBand = 162; imgsz = (nRow, nCol)
    maxvalue = maximum(Y)
    X = Array(Y')./maxvalue;  # reinterpret(N0f8,UInt8(255))=1.0N0f8 but vars["Y"] has maximum 1000
    # img = reshape(X, (nRow, nCol, nBand));
    gtncells = 7; ncells = gtncells + 5
    X, imgsz, nBand, ncells, gtncells, Dict()
end

function load_fakecell(;SNR=10, user_ncells=0, imgsz=(40,20), lengthT=1000, jitter=0, only2cells=false, save_gtimg=false)
    fprefix = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR)"
    X, imgsz, fakecells_dic, img_nl, maxSNR_X = loadfakecell(Float64, fprefix*".jld", only2cells=only2cells,
            fovsz=imgsz, imgsz=imgsz, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
    gtncells = fakecells_dic["gt_ncells"]
    if save_gtimg
        gtW = fakecells_dic["gtW"]; gtH = fakecells_dic["gtH"]
        W3,H3 = copy(gtW), copy(gtH')
        imsaveW(fprefix*"_GT_W.png", W3, imgsz, borderwidth=1)
        imsaveH(fprefix*"_GT_H.png", H3, 100, colors=g1wm())
    end
    ncells = user_ncells==0 ? gtncells + 8 : user_ncells
    X, imgsz, lengthT, ncells, gtncells, fakecells_dic
end

function load_data(dataset; SNR=10, user_ncells=0, imgsz=(40,20), lengthT=1000, jitter=0, save_gtimg=false)
    if dataset == :cbclface
        println("loading CBCL face dataset")
        load_cbcl()
    elseif dataset == :orlface
        println("loading ORL face dataset")
        load_orl()
    elseif dataset == :natural
        println("loading natural image")
        load_natural()
    elseif dataset == :onoffnatural
        println("loading On/OFF-contrast filtered natural image")
        load_onoffnatural()
    elseif dataset == :urban
        println("loading hyperspectral urban dataset")
        load_urban()
    elseif dataset == :neurofinder
        println("loading Neurofinder dataset")
        load_neurofinder()
    elseif dataset == :fakecell
        println("loading fakecells")
        load_fakecell(only2cells=false, SNR=SNR, user_ncells=user_ncells, imgsz=imgsz, lengthT=lengthT,
                jitter=jitter, save_gtimg=save_gtimg)
    elseif dataset == :fakecellsmall
        println("loading fakecells with only two cells")
        load_fakecell(only2cells=true, SNR=SNR, user_ncells=6, imgsz=(20,30), lengthT=lengthT,
                jitter=jitter, save_gtimg=save_gtimg)
    end
end

#======== Image Save ==========================================================#

function imsave_data(dataset,fprefix,W,H,imgsz,lengthT; mssdwstr="", mssdhstr="", saveH=true)
    if dataset == :cbclface
        println("Saving image of CBCL face dataset")
        imsave_cbcl(fprefix,W,H,imgsz,lengthT)
    elseif dataset == :orlface
        println("Saving image of ORL face dataset")
        imsave_orl(fprefix,W,H,imgsz,lengthT)
    elseif dataset == :natural
        println("Saving image of natural image")
        imsave_natural(fprefix,W,H,imgsz,lengthT)
    elseif dataset == :onoffnatural
        println("Saving image of On/OFF-contrast filtered natural image")
        imsave_onoffnatural(fprefix,W,H,imgsz,lengthT)
    elseif dataset == :urban
        println("loading hyperspectral urban dataset")
        imsave_urban(fprefix,W[:,2:7],H[2:7,:],imgsz,lengthT)
    elseif dataset == :neurofinder
        println("Saving image of Neurofinder dataset")
        imsave_neurofinder(fprefix,W,H,imgsz,lengthT, saveH=saveH)
    elseif dataset == :fakecell
        println("Saving image of fakecells")
        imsave_fakecell(fprefix,W,H,imgsz,lengthT; mssdwstr=mssdwstr, mssdhstr=mssdhstr, saveH=saveH)
    elseif dataset == :fakecellsmall
        println("loading fakecells with only two cells")
        imsave_fakecell(fprefix,W,H,imgsz,lengthT; mssdwstr=mssdwstr, mssdhstr=mssdhstr, saveH=saveH)
    end
end

g1wm() = (colorant"green1", colorant"white", colorant"magenta")
bbw() = (colorant"black", colorant"black", colorant"white")
bgw() = (colorant"black", colorant"gray", colorant"white")
wwb() = (colorant"white", colorant"white", colorant"black") # cdcl-face NMF

function interpolate(img,n)
    imginter = zeros(size(img).*(n+1))
    offset = CartesianIndex(1,1)
    for i in CartesianIndices(img)
        ii = (i-offset)*n+i
        for j in CartesianIndices((n+1,n+1))
            imginter[ii+j-offset] = img[i]
        end
    end
    imginter
end

function imsave_cbcl(fprefix,W,H,imgsz,tlength; gridcols=Int(ceil(sqrt(size(W,2)))), borderwidth=1, signedcolors=g1wm())
    # clamp_level=0.5; W3_max = maximum(abs,W3)*clamp_level; W3_clamped = clamp.(W3,0.,W3_max)
    # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
    # imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W3, imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
    imsaveW(fprefix*"_W.png", W, imgsz; gridcols=gridcols, borderwidth=borderwidth, colors=signedcolors)
end

function imsave_reconstruct(fprefix,X,W,H,imgsz; index=100, gridcols=7, clamp_level=1.0) # 0.2 for urban
    # TODO: save all face pictures
    # imsave("cbcl_faces.png", X[:,1:49], imgsz;; gridcols=7, borderwidth=1, colors=bbw())
    intplimg = interpolate(reshape(H[:,index],gridcols,gridcols),3)
    imsaveW(fprefix*"_H$(index).png",reshape(intplimg,length(intplimg),1),size(intplimg))
    reconimg = W*H[:,index]
    W_max = maximum(abs,reconimg)*clamp_level; W_clamped = clamp.(reconimg,0.,W_max)
    #save(fprefix*"_Org$index.png", map(clamp01nan, reshape(X[:,index],imgsz...)))
    save(fprefix*"_recon$index.png", map(clamp01nan, reshape(W_clamped,imgsz...)))
end

function imsave_orl(fprefix,W,H,imgsz,tlength; gridcols=Int(ceil(sqrt(size(W,2)))), borderwidth=1, signedcolors=g1wm())
    imsaveW(fprefix*"_W.png", W, imgsz; gridcols=gridcols, borderwidth=borderwidth, colors=signedcolors)
end

function imsave_natural(fprefix,W,H,imgsz,tlength; gridcols=12, borderwidth=1, signedcolors=bgw())
    imsaveW(fprefix*"_W.png", W, imgsz; gridcols=gridcols, borderwidth=borderwidth, colors=signedcolors)
 end
 
function imsave_onoffnatural(fprefix,W,H,imgsz,tlength; gridcols=12, borderwidth=1, signedcolors=bgw())
   l = imgsz[1]^2; W = W[1:l,:]-W[l+1:2*l,:]
   imsaveW(fprefix*"_W.png", W, imgsz; gridcols=gridcols, borderwidth=borderwidth, colors=signedcolors)
end

function imsave_urban(fprefix,W,H,imgsz,tlength; gridcols=2, borderwidth=5, signedcolors=bbw(), clamp_level=0.2)
    W_max = maximum(abs,W)*clamp_level; W_clamped = clamp.(W,0.,W_max)
    imsaveW(fprefix*"_W.png", W_clamped, imgsz; gridcols=gridcols, borderwidth=borderwidth, colors=signedcolors)
end

function imsave_reconstruct_urban(fprefix,X,W,H,imgsz; index=100, clamp_level=0.2)
    reconimg = W*H[:,index]
    W_max = maximum(abs,reconimg)*clamp_level; W_clamped = clamp.(reconimg,0.,W_max)
    save(fprefix*"_Org$index.png", map(clamp01nan, reshape(X[:,index],imgsz...)))
    save(fprefix*"_CG$index.png", map(clamp01nan, reshape(W_clamped,imgsz...)))
    #save(fprefix*"_CG_clamped$index.png", map(clamp01nan, reshape(clamp.(W,0.,Inf)*clamp.(H[:,index],0.,Inf),imgsz...)))
end

function imsave_neurofinder(fprefix,W,H,imgsz,tlength; gridcols=5, borderwidth=1, signedcolors=g1wm(), saveH=true)
    #@show ncells, gridcols
    ncells = size(W,2)
    for i = 1:ncells÷gridcols
        imsaveW(fprefix*"_W_$(i).png",W[:,((i-1)*gridcols+1):(i*gridcols)],imgsz;
                gridcols=gridcols,borderwidth=borderwidth)
    end
    saveH && imsaveH(fprefix*"_H.png", H, tlength; colors=signedcolors)
end

function imsave_fakecell(fprefix,W,H,imgsz,tlength; mssdwstr="", mssdhstr="",
        gridcols=size(W,2), borderwidth=1, signedcolors=g1wm(), saveH=true)
    imsaveW(fprefix*"_W$(mssdwstr).png", W, imgsz; gridcols=gridcols,
            borderwidth=borderwidth, colors=signedcolors)
    saveH && imsaveH(fprefix*"_H$(mssdhstr).png", H, tlength; colors=signedcolors)
end

#=========== Plot Convergence ====================================================#
function plot_convergence(fprefix, x_abss, xw_abss, xh_abss, f_xs; title="")
    xlbl = "iteration"; ylbl = "log10(penalty)"
    lstrs = ["log10(f_x)", "log10(x_abs)", "log10(xw_abs)", "log10(xh_abs)"]
    logfx = log10.(f_xs); logxabs = log10.(x_abss);
    logxw = log10.(xw_abss); logxh = log10.(xh_abss)
    ax = plotW([logfx logxabs logxw logxh], fprefix*"_plot_all.png"; title=title,
            xlbl=xlbl, ylbl="", legendloc=1, arrange=:combine, legendstrs=lstrs)
            #,axis=[480,1000,-0.32,-0.28])
    ax = plotW(logfx, fprefix*"_plot_fx.png"; title=title, xlbl=xlbl, ylbl=ylbl,
            legendstrs = [lstrs[1]], legendloc=1)
    ax = plotW([logxabs logxw logxh], fprefix*"_plot_xabs.png"; title=title, xlbl=xlbl,
            ylbl="", legendloc=1, arrange=:combine, legendstrs = lstrs[2:4])
    ax
end

function plot_convergence(fprefix, x_abss, f_xs; title="")
    xlbl = "iteration"; ylbl = "log10(penalty)"
    lstrs = ["log10(f_x)", "log10(x_abs)"]
    logfx = log10.(f_xs); logxabs = log10.(x_abss);
    ax = plotW([logfx logxabs], fprefix*"_plot_all.png"; title=title,
            xlbl=xlbl, ylbl="", legendloc=1, arrange=:combine, legendstrs=lstrs)
            #,axis=[480,1000,-0.32,-0.28])
    ax = plotW(logfx, fprefix*"_plot_fx.png"; title=title, xlbl=xlbl, ylbl=ylbl,
            legendstrs = [lstrs[1]], legendloc=1)
    ax = plotW(logxabs, fprefix*"_plot_xabs.png"; title=title, xlbl=xlbl, ylbl="",
            legendstrs = [lstrs[2]], legendloc=1)
    ax
end

#=========== Plot H ====================================================#
function plotH_data(dataset,fprefix,H)
    if dataset == :urban
        plotH_data(fprefix,H[2:7,:]; arrange = (3,2), legendloc=0,
            titles = ["2","3","4","5","6","7"])
    else
        plotH_data(fprefix,H; arrange=:combine)
    end
end

function plotH_data(fprefix,H; title="", titles=fill("",size(H,1)),
        arrange=:combine, legendloc=1)
    plotW(H', fprefix*"_plot_H.png"; title=title, titles=titles, xlbl="", ylbl="",
    legendloc=legendloc, arrange=arrange)
end