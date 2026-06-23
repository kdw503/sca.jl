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

#=
# Load grayscale image and convert to array
# img = testimage("lena_gray_256.tif")
# img_array = Float64.(channelview(img))

# --- Extract Patches ---
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
=#

# --- Whitening ---
X_mean = mean(X, dims=2)
X_centered = X .- X_mean
covariance = cov(X_centered')
U, S, _ = svd(covariance)
epsilon = 1e-5
X_whitened = U * Diagonal(1 ./ sqrt.(S .+ epsilon)) * U' * X_centered
# imsave_data(dataset,joinpath(subworkpath,"X_whitened1to72.png"),X_whitened[:,1:72],X_whitened[1:72,:],imgsz,lengthT; saveH=false)

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

#================== Sparse Coding via ISTA =========================#
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
λ = 3.0 # Sparsity regularization
for λ in [3.0]
    for maxiter in [50]
        @show λ, maxiter
        rt = @elapsed D, αs = sparse_coding(X_whitened, K, λ, n_iter=maxiter)
        # img, _ = show_patches(D, patch_size, 64);
        fprefix = joinpath(subworkpath,"X_whitened_Hspar", "natural_SC_l$(λ)_iter$(maxiter)")
        imsave_data(dataset,fprefix,D,αs,imgsz,lengthT)
        save(fprefix*".jld2", "X_whitened", X_whitened, "D", D, "αs", αs, "imgsz", imgsz, "lengthT", lengthT, "rt", rt)
    end
end
