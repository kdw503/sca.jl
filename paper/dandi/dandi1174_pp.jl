using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","dandi")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

fprefix = "dandi1174_sub-Q_ophys"

using NPZ

# reshape data
full_path = joinpath(subworkpath, fprefix*".npy")
datanpz = NPZ.npzread(full_path)
t,y,x = size(datanpz)
dataraw = Array{Float32}(undef,y,x,t)
for i in 1:t
    dataraw[:,:,i] = datanpz[i,:,:]
end

# reshape data_ann (segmentation results)
full_path = joinpath(subworkpath, fprefix*"_segmentation.npy")
datanpz_ann = NPZ.npzread(full_path)
dataraw_ann_mean = dropdims(mean(datanpz_ann,dims=1),dims=1)
n,yann,xann = size(datanpz_ann)
yann = Int(floor(yann/2)*2); xann = Int(floor(xann/2)*2)
dataraw_ann = Array{Float32}(undef,yann,xann,n)
for i in 1:n
    dataraw_ann[:,:,i] = datanpz_ann[i,1:yann,1:xann]
end

# cut data to match the segmentation and remove black frames
# find matched region
sumdata = dropdims(sum(dataraw, dims=3), dims=3)
Dy, Dx = y-yann, x-xann
sums = zeros(Float32, Dy, Dx); maxd = 0; maxdy = 0; maxdx = 0
for dy in 0:Dy-1
    for dx in 0:Dx-1
        sumd = 0.
        for i in 1:yann
            for j in 1:xann
                sumd += sumdata[i+dy,j+dx]*dataraw_ann_mean[i,j]
            end
        end
        if sumd > maxd
            maxd = sumd; maxdy = dy; maxdx = dx
        end
        sums[dy+1,dx+1] = sumd
    end
end # maxdy, maxdx = 19, 20
datacrop  = dataraw[1+maxdy:yann+maxdy, 1+maxdx:xann+maxdx, :]
ycrop,xcrop,t = size(datacrop); imgsz = (ycrop,xcrop)
# remove black frames
gtW = reshape(dataraw_ann, xann*yann, size(dataraw_ann,3))
Xcrop = sqrt.(Array(reshape(datacrop,xcrop*ycrop,t)))
gtHcrop = Xcrop'/(gtW')
fprex = joinpath(subworkpath,"$(fprefix)_crop_GT") 
plotH_data(fprex, gtHcrop',figsize=(1400,400), space=0)
gtHmean = dropdims(mean(gtHcrop, dims=1), dims=1)
maxgtH, maxgtHind = maximum(gtHmean), argmax(gtHmean)
black_indices = []
for i in 1:size(gtH, 1)
    if gtHcrop[i,maxgtHind].<0.95*maxgtH
        push!(black_indices, i)
    end
end
all_inds = collect(1:size(datacrop, 3))
keep_inds = setdiff(all_inds, black_indices)
datathin = datacrop[:, :, keep_inds]
t = size(datathin,3)
# ground truth gtH
X = sqrt.(Array(reshape(datathin,xcrop*ycrop,t)))
gtHthin = X'/(gtW')
fprex = joinpath(subworkpath,"$(fprefix)_thin_GT") 
imsave_data(:ocpi,fprex,gtW,gtHthin',imgsz,1; gridcols=8, borderwidth=4, saveH=false, verbose=false)
plotH_data(fprex, gtHthin',figsize=(1400,400), space=0)

# save data
save(joinpath(subworkpath,"$(fprefix).jld2"),"X",X,"dataraw",dataraw, "datacrop",datacrop, "data", datathin,
    "data_ann", dataraw_ann[1:ycrop,1:xcrop,1:n], "data_ann_mean", dataraw_ann_mean[1:ycrop,1:xcrop],
    "gtW", gtW, "gtHcrop", gtHcrop, "gtH", gtHthin)

# Background subtraction
rt1cd = @elapsed bgW, bgH = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
NMF.solve!(NMF.CoordinateDescent{eltype(bgW)}(maxiter=60, α=0), X, bgW, bgH)
LCSVD.normalizeW!(bgW,bgH)
wmin = minimum(bgW); bgWs = bgW.-wmin; wmax = maximum(bgWs); bgWs ./= wmax
bgWimg = reshape(bgWs, imgsz)
save(joinpath(subworkpath,"$(fprefix)_thin_bg.png"),bgWimg)
plotH_data(joinpath(subworkpath,"$(fprefix)_thin_bg"),bgH,figsize=(1400,400))
bgWmeanH = bgW*fill(mean(bgH),1,t)
bgWH = bgW*bgH
Xsbgmh = X .- bgWmeanH
Xsbgh = X .- bgWH
save(joinpath(subworkpath,"$(fprefix)_thin_X.jld2"),"Xsbgmh",Xsbgmh,"Xsbgh",Xsbgh,"Xwbg",X)
gtHsbgmh = gtW\Xsbgmh
gtHsbgh = gtW\Xsbgh
gtHwbg = gtW\Xwbg
# plotH_data(joinpath(subworkpath,"$(fprefix)_thin_sbgmh_GT_nospace"),gtHsbgmh',figsize=(1400,400))
# plotH_data(joinpath(subworkpath,"$(fprefix)_thin_sbgh_GT_nospace"),gtHsbgh',figsize=(1400,400))
plotH_data(joinpath(subworkpath,"$(fprefix)_thin_sbgmh_GT"),gtHsbgmh,figsize=(1400,1095), space=-2,
    legend_position=:rc, ylabelvisible = false, ygridvisible=false, yticksvisible=false, yticklabelsvisible=false)
plotH_data(joinpath(subworkpath,"$(fprefix)_thin_sbgh_GT"),gtHsbgh,figsize=(1400,1095), space=-2,
    legend_position=:rc, ylabelvisible = false, ygridvisible=false, yticksvisible=false, yticklabelsvisible=false)
plotH_data(joinpath(subworkpath,"$(fprefix)_thin_wbg_GT"),gtHwbg,figsize=(1400,1095), space=-2,
    legend_position=:rc, ylabelvisible = false, ygridvisible=false, yticksvisible=false, yticklabelsvisible=false)




# mp4
using VideoIO

dat = load(joinpath(subworkpath,"$(fprefix).jld2"),"dataraw")
encoder_options = (crf=23, preset="medium")
clamp_level=1.0; base_level=0.5
dat_max = maximum(abs,dat)*clamp_level; dat_min=minimum(abs,dat); dat.-=min(dat_min,dat_max*base_level)
datnor = dat./dat_max; dat_clamped = clamp.(datnor,0.,1.)
datuint8 = UInt8.(round.(map(clamp01nan, dat_clamped)*255))

fprefixvid = joinpath(subworkpath,"$(fprefix)_thin")
VideoIO.save("$(fprefixvid).mp4", eachslice(datuint8, dims=3), framerate=30, encoder_options=encoder_options) # compatible with ppt (best)
# VideoIO.save("$(fprefixvid).mpg", reshape.(eachcol(Xuint8),imgsz...), framerate=30) # compatible with ppt
# VideoIO.save("$(fprefixvid).avi", reshape.(eachcol(Xuint8),imgsz...), framerate=30) # compatible with ppt
