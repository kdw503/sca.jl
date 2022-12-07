
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
using Images, Convex, SCS, LinearAlgebra, Printf, Colors
using FakeCells, AxisArrays, ImageCore, MappedArrays
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using MAT, KSVD
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

# ARGS = [":ksvd","10000"]
dr_method = eval(Meta.parse(ARGS[1])); num_trials = eval(Meta.parse(ARGS[2]))
dr_method = eval(Meta.parse(ARGS[3]));
@show dr_method, num_trials

frn = "IMAGES.mat"
dd = matread(frn)
img = dd["IMAGES"]
szh,szv,numimgs = size(img)
batch_size = 100; ncomps = 64; subimgsz = 8
L = subimgsz^2
BUFF = 4 # border pixels buffer
W = zeros(L,ncomps)
X = zeros(L,batch_size*num_trials)
# for i=1:numimgs
for t=1:num_trials
    i=Int(ceil(numimgs*rand()))
    this_image=img[:,:,i]
    for j=1:batch_size
        r=Int(BUFF+ceil((szv-subimgsz-2*BUFF)*rand()));
        c=Int(BUFF+ceil((szh-subimgsz-2*BUFF)*rand()));
        X[:,(i-1)*batch_size+j]=reshape(this_image[r:r+subimgsz-1,c:c+subimgsz-1],L,1);
    end
end
if dr_method == :ksvd
    W, H = ksvd(
        X,
        ncomps,  # the number of atoms in D
        max_iter = 200,  # max iterations of K-SVD
        max_iter_mp = 40,  # max iterations of matching pursuit called in K-SVD
        sparsity_allowance = 0.96  # stop iteration when more than 96% of elements in W become zeros
    )
else
    error("Not supported Dimensionality Reduction method")
end
normalizeWH!(W,H)
imgsz = (subimgsz,subimgsz)
clamp_level=0.5; W_max = maximum(abs,W)*clamp_level; W_clamped = clamp.(W,0.,W_max)
signedcolors = (colorant"green1", colorant"white", colorant"magenta")
imsaveW("onoff_natural_$(dr_method)_nc$(ncomps)_nt$(num_trials).png", W, imgsz, gridcols=8, colors=signedcolors, borderval=W_max, borderwidth=1)

