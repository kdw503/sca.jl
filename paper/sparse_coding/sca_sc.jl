
using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","sparse_coding")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :natural
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"
X, imgsz, lengthT, ncs, _ = load_data(dataset)
patch_size = imgsz[1]
(m,n,p) = (size(X)...,ncs)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

# --- Whitening ---
X_mean = mean(X, dims=2)
X_centered = X .- X_mean
covariance = cov(X_centered')
U, S, _ = svd(covariance)
epsilon = 1e-5
X_whitened = U * Diagonal(1 ./ sqrt.(S .+ epsilon)) * U' * X_centered

# SCA : Sparse Component Analysis (sparsity is applied to only H(Y')) + maximize(∥Z'XY∥₂)
using RCall

R"library(epca)"

@rput X_whitened

prefix = "sca"
rt = @elapsed R"factors_sca <-sca(t(X_whitened), k=1)" # default gamma = sqrt(p*k)=sqrt(100000*72)
@rget factors_sca
fname = joinpath(subworkpath,"natural_$(prefix)_rt$(rt)")
save(fname*".jld2", "factors_sca", factors_sca)
rW = factors_sca[:y]'; rH = Array(factors_sca[:x])
LCSVD.normalizeW!(rW,rH); fitval = LCSVD.fitd(X,rW*rH)
imsave_data(dataset,fname,rW,rH,imgsz,lengthT; saveH=false)
