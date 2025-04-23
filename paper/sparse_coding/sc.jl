
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

using Images, Statistics, LinearAlgebra, FileIO, ImageView, Random, TestImages

# Load grayscale image and convert to array
# img = testimage("lena_gray_256.tif")
# img_array = Float64.(channelview(img))

# --- Extract Patches ---
function extract_patches(img, patch_size)
    patches = []
    for i in 1:patch_size:(size(img, 2) - patch_size)
        for j in 1:patch_size:(size(img, 1) - patch_size)
            patch = img[j:j+patch_size-1, i:i+patch_size-1]
            push!(patches, vec(patch))
        end
    end
    hcat(patches...)  # each column is a patch
end
function extract_patches(img, patch_size)
    patches = []
    for i in 1:patch_size:(size(img, 2) - patch_size)
        for j in 1:patch_size:(size(img, 1) - patch_size)
            patch = img[j:j+patch_size-1, i:i+patch_size-1]
            push!(patches, vec(patch))  # <--- FLATTEN PATCH!
        end
    end
    return hcat(patches...)  # Each column is a patch
end

# patch_size = 12
# X = extract_patches(img_array, patch_size)

# --- Whitening ---
X_mean = mean(X, dims=2)
X_centered = X .- X_mean
covariance = cov(X_centered')
U, S, _ = svd(covariance)
epsilon = 1e-5
X_whitened = U * Diagonal(1 ./ sqrt.(S .+ epsilon)) * U' * X_centered

# --- Visualize Sample Whitened Patches ---
function show_patches(X, patch_size, num_patches=25)
    cols = trunc(Int, sqrt(num_patches))
    rows = ceil(Int, num_patches / cols)
    canvas = fill(0.5, patch_size*rows, patch_size*cols)

    for idx in 1:num_patches
        i = div(idx - 1, cols)
        j = (idx - 1)%cols
        patch = reshape(X[:, idx], patch_size, patch_size)
        canvas[i*patch_size+1:(i+1)*patch_size, j*patch_size+1:(j+1)*patch_size] = patch
    end
    imdic, imshow(canvas, name="Whitened Patches")
    canvas, imdic 
end

# Show 25 whitened patches
show_patches(X_whitened, patch_size, 25)

function soft_threshold(x, λ)
    return sign.(x) .* max.(abs.(x) .- λ, 0)
end

function ista(D, x, λ; max_iter=100, lr=1e-1) # lr is 1/L, where L < 1 is Lipschitz constant
    α = zeros(size(D, 2))
    for _ in 1:max_iter
        grad = D' * (D * α - x)
        α = soft_threshold(α - lr * grad, λ * lr)
    end
    return α
end
function init_dictionary(X, K)
    D = X[:, rand(1:end, K)]
    D ./= sqrt.(sum(D .^ 2, dims=1))  # Normalize columns
    return D
end
function sparse_coding(X, K, λ; n_iter=10)
    D = init_dictionary(X, K)
    N = size(X, 2)
    αs = zeros(K, N) # H

    for iter in 1:n_iter
        println("Iteration $iter")

        # Sparse coding step (fix D, update α) (H)
        for i in 1:N
            αs[:, i] = ista(D, X[:, i], λ)
        end

        # Dictionary update step (fix α, update D) (W)
        for k in 1:K
            idx = findall(!=(0), αs[k, :])
            if !isempty(idx)
                Rk = X[:, idx] - D * αs[:, idx] + D[:, k] * αs[k, idx]'
                D[:, k] = Rk * αs[k, idx]
                D[:, k] /= norm(D[:, k])
            end
        end
    end
    return D, αs
end
K = ncs  # Number of atoms
λ = 1.0 # Sparsity regularization
for λ in [3.0]
    for maxiter in [20,50,100]
        @show λ, maxiter
        rt = @elapsed D, αs = sparse_coding(X_whitened, K, λ, n_iter=maxiter)
        # img, _ = show_patches(D, patch_size, 64);
        fprefix = joinpath(subworkpath, "natural_SC_l$(λ)_iter$(maxiter)")
        imsave_data(dataset,fprefix,D,αs,imgsz,lengthT)
        save(fprefix*".jld2", "D", D, "αs", αs, "imgsz", imgsz, "lengthT", lengthT, "rt", rt)
    end
end

#================== Other Sparsity Methods =========================#
# PCB
prefix = "pcb"
noc = ncs; nac = 0
(initmethod,α,β) = (:isvd,0.005,0.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(H0'); N0t = copy(N0')
for α in [1e-3, 1e-4]
β1 = β2= β; α1 = α; α2 = 0.005 # sparse coding
β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
r=0.3; useprecond=false; uselv=false; tol=1e-6
maxiter = Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
    #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    r=r, useprecond=useprecond, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = false,
    store_inner_trace = false, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rst1 = LCSVD.solve!(alg, X_whitened, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed LCSVD.solve!(alg, X, U, V, D, M1, N1t);

W1, H1 = rst1.W, rst1.Ht'
LCSVD.normalizeW!(W1,H1);
fv = LCSVD.fitd(X,W1*H1)
fprex = "$(prefix)_$(initmethod)"
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_aw$(α1)_ah$(α2)_b$(β)"
fname = joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
end
imsave_data(dataset,joinpath(subworkpath,"ISVD.png"),U,H0,imgsz,lengthT; saveH=false)

# SCA : Sparse Component Analysis (sparsity is applied to only H(Y')) + maximize(∥Z'XY∥₂)
using RCall

R"library(epca)"

@rput X_whitened

prefix = "sca"
rt2 = @elapsed R"factors_sca <-sca(t(X_whitened), k=72)" # default gamma = sqrt(p*k)=sqrt(100000*72)
@rget factors_sca
rW = factors_sca[:y]'; rH = Array(factors_sca[:x])
LCSVD.normalizeW!(rW,rH); fitval = LCSVD.fitd(X,rW*rH)
fprex = "$(prefix)"
fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_rt$(rt2)")
imsave_data(dataset,fname,rW,rH,imgsz,lengthT; saveH=false)
save(fname*".jld2", "factors_sca", factors_sca, "rt", rt2)
