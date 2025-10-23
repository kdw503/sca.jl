
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
        fprefix = joinpath(subworkpath, "natural_SC_dotinit_l$(λ)_iter$(maxiter)")
        imsave_data(dataset,fprefix,D,αs,imgsz,lengthT)
        save(fprefix*".jld2", "D", D, "αs", αs, "imgsz", imgsz, "lengthT", lengthT, "rt", rt)
    end
end

#================== Other Sparsity Methods =========================#
# PCB
prefix = "pcb"
noc = ncs; nac = 0

dd = load(joinpath(subworkpath, "X_whitened_Hspar","natural_SC_l3.0_iter50.jld2"))
sD = dd["D"]; αs = dd["αs"]

dallinit = Dict{String,Tuple}()
for initmethod in [:isvd, :BPDN]
    @show initmethod
    initmtd = initmethod == :nndsvd ? initmethod : :isvd
    rt1 = @elapsed U, Vt, M0, N0, Wp, Hp, D = LCSVD.initpcb(X_whitened, noc, nac; initmethod=initmtd, svdmethod=:isvd)
    V = copy(Vt'); N0t = copy(N0')
    if initmethod == :BPDN
        LCSVD.balanceWH!(sD,αs)
        M0, N0 = (U'sD, αs*Vt')
    elseif initmethod == :sbc
        try
            rt11 = @elapsed M0 = sbc(U)
        catch e
            save(joinpath(resultpath,"sbc_error$(iter).jld2"),"U",W0)
            error("SBC failed with error: $(e)")
        end
        rt12 = @elapsed N0 = M0\D
        rt13 = @elapsed LCSVD.balanceWH!(M0, N0)
    end
    Winit, Hinit = U*M0, N0*Vt
    fv = LCSVD.fitd(X_whitened,Winit*Hinit)
    Esym = norm(M0*N0t'-D)^2
    Esw = norm(Winit,1); Esh = norm(Hinit,1) # Esh includes balanced power
    LCSVD.normalizeW!(Winit,Hinit)
    Sw = norm(Winit,1); Sh = norm(Hinit,1) # Sh includes whole power
    V = copy(Vt'); N0t = copy(N0')
    initmethod == :isvd && (dallinit["SVD"] = (U, Vt, D))
    dallinit[String(initmethod)] = (Winit, Hinit, M0, N0t, fv, Esym, Esw, Esh, Sw, Sh)
    imsave_data(dataset,joinpath(subworkpath,"$(initmethod)_f$(fv)_Esw$(Esw)_Esh$(Esh).png"),Winit,Hinit,imgsz,lengthT; saveH=false)
end
save(joinpath(subworkpath,"allinit.jld2"),dallinit)

dd = load(joinpath(subworkpath,"allinit.jld2"))
U, Vt, D = dd["SVD"]; V = Vt'
for initmethod in [:BPDN, :isvd]
    Winit, Hinit, M0, N0t, fv, Esym, Esw, Esh, Sw, Sh = dd[String(initmethod)]
    @show initmethod, fv, Esym, Esh, Sh, norm(M0,2)
    for α in [1e-5,0.005]
        alg = LCSVD.LinearCombSVD(α1=0., α2=α, β1=0., β2=0.)
        state = LCSVD.prepare_state(U, V, M0, N0t, alg)
        updater = LCSVD.LinearCombSVDUpd{eltype(D)}(D, state, noc, alg)
        nD22_EshNN12 = norm(D)^2/norm(V*N0t,1)/norm(M0)
        @show initmethod, α, updater.αh, α*nD22_EshNN12, norm(V*N0t,1), norm(M0)
    end
end

for initmethod in [:BPDN, :isvd]
    @show initmethod
    Winit, Hinit, M0, N0t, _ = dd[String(initmethod)]
    Es = []; Esyms = []; L1hs = []; Eshs = []
    for αpow in [-7:1:5] # [1e-5]
        @show αpow
        α = 10.0^αpow
        β1 = β2 = β = 0; α1 = 0; α2 = α# sparse coding
        β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
        α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
        r=0.3; useprecond=false; uselv=false; optim_method = :sgd_injectnoise # :lbfgs
        maxiter = 100#Int(ceil(log(eps(eltype(X_whitened)))/log(r))) #lcsvd_maxiter # 
        tol=1e-6; inner_tol = 1e-7; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2,
            #α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
            r=r, useprecond=useprecond, usedenoiseUVt=false, optim_method=optim_method,
            denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
            store_inner_trace = false, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0, ur=0.00001, nr=0.001);
        M1, N1t = copy(M0), copy(N0t)
        rst1 = LCSVD.solve!(alg, X_whitened, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
        alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed LCSVD.solve!(alg, X_whitened, U, V, D, M1, N1t);

        W1, H1 = rst1.W, rst1.Ht'
        Esh = norm(H1,1)
        LCSVD.normalizeW!(W1,H1); Sh = norm(W1,1)
        fv = LCSVD.fitd(X_whitened,W1*H1)
        Einit = rst1.traces[1].f_x; Eend = rst1.traces[end].f_x
        Esym=norm(M1*N1t'-D)^2
        fprex = "$(prefix)_$(initmethod)_$(optim_method)"
        #fprex = "$(prefix)_BPDN"
        regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_aw$(α1)_ah$(α2)_b$(β)"
        fname = joinpath(subworkpath,"$(fprex)$(regstr)_Einit$(Einit)_Eend$(Eend)_Esy,$(Esym)_Esh$(Esh)_f$(fv)_Sh$(Sh)_it$(rst1.niters)_rt$(rt2)")
        imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
        Xest = W1*H1; mse = norm(X_whitened[:,1:72]-Xest[:,1:72])^2/length(X_whitened[:,1:72])
        imsave_data(dataset,joinpath(subworkpath,"$(fprex)$(regstr)_mse$(mse)_Xest1to72.png"),Xest[:,1:72],Xest[1:72,:],imgsz,lengthT; saveH=false)
    end
end

# SCA : Sparse Component Analysis (sparsity is applied to only H(Y')) + maximize(∥Z'XY∥₂)
using RCall

R"library(epca)"

@rput X_whitened

prefix = "sca"
rt2 = @elapsed R"factors_sca <-sca(t(X_whitened), k=72)" # default gamma = sqrt(p*k)=sqrt(100000*72)
@rget factors_sca
rW = factors_sca[:y]'; rH = Array(factors_sca[:x])
LCSVD.normalizeW!(rW,rH); fitval = LCSVD.fitd(X_whitened,rW*rH)
fprex = "$(prefix)"
fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_rt$(rt2)")
imsave_data(dataset,fname,rW,rH,imgsz,lengthT; saveH=false)
save(fname*".jld2", "factors_sca", factors_sca, "rt", rt2)


#================== PCB style Sparse Coding =========================#

using ForwardDiff

# Check the gradient gradft = 2*M'*(M*Nt'-D) of f(Nt) = norm(M*Nt'-D)^2 with ForwardDiff
M = rand(4,4); Nt = rand(4,4); D = M*Nt'
Nt = rand(4,4)
f(Nt) = norm(M*Nt'-D)^2
fgradf = ForwardDiff.gradient(f, Nt)
gradft = 2*M'*(M*Nt'-D)
gradf = 2*(Nt*M'-D')*M
@show norm(fgradf-gradft')
@show norm(fgradf-gradf)

dd = load(joinpath(subworkpath,"allinit.jld2"))
U, Vt, D = dd["SVD"]; V = Vt'
noc = ncs; λ=0.9; max_iter = 1000; lr=0.02; initmethod = :DICT
for initmethod in [:DICT, :BPDN, :isvd, :nndsvd]
    if initmethod == :DICT
        Winit = init_dictionary(X_whitened, noc)
        M0, N0t = U'Winit, zeros(noc,noc)
    else
        Winit, Hinit, M0, N0t, _ = dd[String(initmethod)]
    end
    for λ in [2.0], lr in [0.1], max_iter in [50, 100, 200]
        M, Nt = copy(M0), copy(N0t)
        rt = @elapsed M1, N1t = LCSVD.sparse_coding_pcb(U, V, D, M, Nt, λ; max_iter=max_iter, lr=lr)
        @show initmethod, λ, rt

        W1, H1 = U*M1, N1t'*Vt
        Esw = norm(W1,1) 
        LCSVD.normalizeW!(W1,H1); Sh = norm(H1,1)
        fv = LCSVD.fitd(X_whitened,W1*H1)
        Es=norm(M1*N1t'-D)^2
        fprex = "SC_PCB_$(initmethod)"
        regstr = "_l$(λ)_lr$(lr)_it$(max_iter)"
        fname = joinpath(subworkpath,"$(fprex)$(regstr)_f$(fv)_Es$(Es)_Sh$(Sh)_rt$(rt)")
        imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
    end
end
for λ in [0.9], lr in [0.02], max_iter in [4000]
    M, Nt = copy(M0), copy(N0t)
    rt = @elapsed M1, N1t = LCSVD.sparse_coding_pcb(U, V, D, M, Nt, λ; max_iter=max_iter, lr=lr)
    @show initmethod, λ, rt

    W1, H1 = U*M1, N1t'*Vt
    Esw = norm(W1,1) 
    LCSVD.normalizeW!(W1,H1); Sh = norm(H1,1)
    fv = LCSVD.fitd(X_whitened,W1*H1)
    Es=norm(M1*N1t'-D)^2
    fprex = "SC_PCB_$(initmethod)"
    regstr = "_l$(λ)_lr$(lr)_it$(max_iter)"
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_f$(fv)_Es$(Es)_Sh$(Sh)_rt$(rt)")
    imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
end
for λ in [2.0], lr in [0.03], max_iter in [50, 100, 200]
    M, Nt = copy(M0), copy(N0t)
    rt = @elapsed M1, N1t = LCSVD.sparse_coding_pcb(U, V, D, M, Nt, λ; max_iter=max_iter, lr=lr)
    @show initmethod, λ, rt

    W1, H1 = U*M1, N1t'*Vt
    Esw = norm(W1,1) 
    LCSVD.normalizeW!(W1,H1); Sh = norm(H1,1)
    fv = LCSVD.fitd(X_whitened,W1*H1)
    Es=norm(M1*N1t'-D)^2
    fprex = "SC_PCB_$(initmethod)"
    regstr = "_l$(λ)_lr$(lr)_it$(max_iter)"
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_f$(fv)_Es$(Es)_Sh$(Sh)_rt$(rt)")
    imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
end

max_iter = 50
for λ in [3.0,0.01,0.1,1.0,10.0], lr in [1e-1,1e-2,1e-3]
    initmethod = :RAND
    rt = @elapsed W1, H1 = LCSVD.sparse_coding(X_whitened, noc, λ; max_iter=max_iter, lr=lr)
    @show initmethod, λ, rt

    LCSVD.normalizeW!(W1,H1); Sh = norm(H1,1)
    fv = LCSVD.fitd(X_whitened,W1*H1)
    Es=norm(W1*H1-X_whitened)^2
    fprex = "SC_$(initmethod)"
    regstr = "_l$(λ)_lr$(lr)_it$(max_iter)"
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_f$(fv)_Es$(Es)_Sh$(Sh)_rt$(rt)")
    imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
end


#================== PCB using constraint instead of regularization =========================#
# need to set α1 = α2 = 0 to nullify the regularization effect
# Then, need to set alg.ur for Ht and alg.nr for W as the threshold rate
# And, need to set tol=0, inner_tol=1e-6 and maxiter = 50(actual iteration number)

# test with fakecells first
dataset = :fakecells # :cbclface # 
tailstr = "_sp"
SNR=0; factor=1; nc=15
imgsz0 = (40,20)
sqfactor = Int(floor(sqrt(factor)))
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
maskth=0.25; maskW=rand(imgsz...).<maskth; maskW = vec(maskW); maskH=rand(lengthT).<maskth;
subtract_bg = false
tol=-1; inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
for iter in 1:2
    X, imsz, lhT, noc, gtnoc, datadic = load_data(dataset; sigma=5.0, imgsz=imgsz, lengthT=lengthT,
            SNR=SNR, bias=0.1, useCalciumT=true, inhibitindices=0, issave=false, isload=false,
            gtincludebg=false, save_gtimg=false, save_maxSNR_X=false, save_X=false, dataset_name="Baron");
    nac = 0
    # (m,n,p) = (size(X)...,nc-nac)
    gtW, gtH = dataset ∈ [:fakecells] ? (datadic["gtW"], datadic["gtH"]) : (zeros(0,0), zeros(0,0))

    if subtract_bg
        rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
        NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
        LCSVD.normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
        plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
        bg = W*fill(mean(H),1,n)
    #        bg = W*H
        X .-= bg
    end
    initmethod = :tsvd
    rt1 = @elapsed U, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, nc-nac, nac; initmethod=initmethod, svdmethod=:tsvd)
    V = copy(H0'); N0t = copy(N0')

    for optim_method in [:lbfgs, :lbfgs_constraint]
        prefix = "$(optim_method)"
        @show optim_method
        if optim_method == :lbfgs_constraint
            maxiter = 400 # Int(ceil(log(eps(eltype(X)))/log(r)))
            α = 0; wthr = 0.01; hthr = 0.01
            regstr0 = "_wthr$(wthr)_hthr$(hthr)_a$(α)"
        else
            maxiter = 100 # Int(ceil(log(eps(eltype(X)))/log(r)))
            α = 0.005; wthr = 0; hthr = 0
            regstr0 = "_a$(α)"
        end
        β1 = β2 = β = 0;  α1 = α2 = α
        useprecond=false; usedenoiseUVt=false; uselv=false; r=0.3
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r,
            useprecond=useprecond, usedenoiseUVt=usedenoiseUVt, optim_method=optim_method,
            denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
            store_inner_trace = true, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0, nr=wthr, ur=hthr);
        M1, N1t = copy(M0), copy(N0t)
        rst = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
        alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
        M1, N1t = copy(M0), copy(N0t)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, U, V, D, M1, N1t);

        W1, H1 = rst.W, rst.Ht'
        Esh = norm(H1,1) 
        LCSVD.normalizeW!(W1,H1); Sh = norm(W1,1)
        LCSVD.flip2makepos!(W1,H1)
        if dataset == :fakecells
            fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
            nodr = LCSVD.matchedorder(ml,nc-nac)
            W1, H1 = W1[:,nodr], H1[nodr,:]
            regstr = "$(regstr0)_b$(β)_af$(fv)_Sh$(Sh)"
        else
            fv = LCSVD.fitd(X,W1*H1)
            regstr = "$(regstr0)_b$(β)_f$(fv)_Sh$(Sh)"
        end
        fprex = "$(prefix)$(SNR)db$(factor)f$(nc-nac)s$(initmethod)"
        fname = joinpath(subworkpath,prefix,"$(fprex)$(regstr)_it$(rst.niters)_rt$(rt2)")
        imsave_data(dataset,fname,W1,H1,imsz,lengthT; saveH=false)

        f_xs = LCSVD.getdata(rst.traces,:f_x); niters = LCSVD.getdata(rst.traces,:niters); totalniters = sum(niters)
        avgfitss = LCSVD.getdata(rst.traces,:avgfits); fxss = LCSVD.getdata(rst.traces,:fxs)
        avgfits = Float64[]; inner_fxs = Float64[]; rt2s = Float64[]
        for (i,(afs,fxs)) in enumerate(zip(avgfitss, fxss))
            isempty(afs) && continue
            append!(avgfits,afs); append!(inner_fxs,fxs)
            if i == 1
                rt2i = 0.
            else
                rt2i = collect(range(start=rst0.laps[i-1],stop=rst0.laps[i],length=length(afs)+1))[1:end-1].-rst0.laps[1]
            end
            append!(rt2s,rt2i)
        end
        dd = Dict()
        dd["niters"] = niters; dd["totalniters"] = totalniters; dd["rt1"] = rt1; dd["rt2s"] = rt2s
        dd["avgfits"] = avgfits; dd["f_xs"] = f_xs; dd["inner_fxs"] = inner_fxs
        if true#iter == num_experiments
            metadata = Dict()
            metadata["r"] = r; metadata["initmethod"] = initmethod
            metadata["maxiter"] = maxiter; metadata["useprecond"] = useprecond
            metadata["usedenoiseUVt"] = usedenoiseUVt; metadata["denoisefilter"] = alg.denoisefilter; 
            metadata["alpha"] = α; metadata["beta"] = β; metadata["wthr"] = wthr; metadata["hthr"] = hthr
        end
        save(joinpath(subworkpath,prefix,"$(fprex)$(tailstr)_results$(iter).jld2"),"metadata",metadata,"data",dd)
    end
end





# natural images
function init_dictionary(X, K)
    D = X[:, rand(1:end, K)]
    D ./= sqrt.(sum(D .^ 2, dims=1))  # Normalize columns
    return D
end

dataset = :natural
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
X = X_whitened

dd = load(joinpath(subworkpath,"allinit.jld2"))
U, Vt, D = dd["SVD"]; V = Vt'
noc = ncs; initmethod = :DICT
# prefix = "pcb_cnstrnt_$(dataset)"
# for hthr in [0.2]#:BPDN, :nndsvd, :SBC, :isvd]
#     α = 0. optim_method = :lbfgs_constraint; tol=-1; inner_tol = 1e-6; maxiter=500
#     inner_maxiter = 100 
#     @show hthr
prefix = "pcb_$(dataset)"
for α in [1e-4]
    optim_method = :lbfgs; tol=-1; inner_tol = 1e-7; maxiter=4000 # Int(ceil(log(eps(eltype(X)))/log(r)))
    inner_maxiter = 10; r=0.5 
    @show α
    if initmethod == :DICT
        Winit = init_dictionary(X, noc)
        M0 = U'Winit; N0t = rand(noc,noc)
    else
        Winit, Hinit, M0, N0t, _ = dd[String(initmethod)]
    end
    Hinit = N0t'*Vt
    fv = LCSVD.fitd(X,Winit*Hinit)
    LCSVD.normalizeW!(Winit,Hinit)
    Sw = norm(Winit,1); Sh = norm(Hinit,1) # Sh includes whole power
    imsave_data(dataset,joinpath(subworkpath,"$(initmethod)_f$(fv)_Sw$(Sw).png"),Winit,Hinit,imgsz,lengthT; saveH=false)

    β1 = β2 = β = 0; α1 =  0; α2 = α; wthr = 0. # only H thresholding
    useprecond=false; uselv=false
    alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r,
        useprecond=useprecond, usedenoiseUVt=false, optim_method=optim_method,
        denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
        store_inner_trace = false, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0, nr=wthr, ur=hthr);
    M1, N1t = copy(M0), copy(N0t)
    rst1 = LCSVD.solve!(alg, X, U, V, D, M1, N1t; gtW=gtW, gtH=gtH);
    # alg.show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
    # M1, N1t = copy(M0), copy(N0t)
    rt2 = 0#@elapsed LCSVD.solve!(alg, X, U, V, D, M1, N1t);

    W1, H1 = rst1.W, rst1.Ht'
    Esh = norm(H1,1) 
    LCSVD.normalizeW!(W1,H1); Sh = norm(W1,1)
    fv = LCSVD.fitd(X,W1*H1)
    fprex = "$(prefix)_$(initmethod)_$(optim_method)"
    #fprex = "$(prefix)_BPDN"
    regstr = optim_method == :lbfgs_constraint ? "_wthr$(wthr)_hthr$(hthr)_b$(β)" : "_a$(α)_b$(β)_r$(r)_inmaxiter$(inner_maxiter)"
    fname = joinpath(subworkpath,"$(fprex)$(regstr)_f$(fv)_Sh$(Sh)_it$(rst1.niters)_rt$(rt2)")
    imsave_data(dataset,fname,W1,H1,imgsz,lengthT; saveH=false)
    # Xest = W1*H1; mse = norm(X[:,1:72]-Xest[:,1:72])^2/length(X[:,1:72])
    # imsave_data(dataset,joinpath(subworkpath,"$(fprex)$(regstr)_mse$(mse)_Xest1to72.png"),Xest[:,1:72],Xest[1:72,:],imgsz,lengthT; saveH=false)
end
