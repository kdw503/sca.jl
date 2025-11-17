
using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"StochasticLM")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using StochasticLM
using LinearAlgebra

#========= stochastic gradient descent with fakecells image ==========#
dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.1
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))
bias = 0.1

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncs, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
(m,n,p) = (size(X)...,ncs)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

subtract_bg = false
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    bg = W*fill(mean(H),1,n)
    X .-= bg
end

noc = ncs; nac = 0; nc = noc+nac; initmethod = :isvd
rt1 = @elapsed U, Vt, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
V = copy(Vt'); N0t = copy(N0')



t = 1:nc        # independent parameters `ξ[i]` that affect each predicted value
y = a .* exp.(-t ./ τ)         # dependent values (we try to predict these)

# Initialize the cache for intermediate values. `RJCache` is the default type, and it stores
# the residuals and Jacobian.
rw = zeros(nc^2+m*noc)     # allocate space to store the residuals for W components
Jw = zeros(nc^2+m*noc, nc*noc)  # allocate space to store the Jacobian for W components
rjw = RJCache(rw, Jw)

# Create the function `f!` that we'll use in optimization.
# `f!` should take three arguments, `f!(rj, x, idx)`, where
#   - `x` is the current value of the parameter vector
#   - `idx` lists the indices of observations included in the minibatch
# `f!` should always return the value of the objective function evaluated with parameters `x`
# on minibatch `idx`.
# For the first argument, you must support two syntaxes:
#    f!(nothing, x, idx)        # just compute the objective value on the minibatch (no caching performed)
#    f!(rj, x, ix)              # populate the cache `rj` for later derivative computation
#
# The first syntax will be called when new settings of `x` are being tested for whether
# they improve the objective value; the second will be called when picking a descent direction
# for the next step.
#
# For performance reasons, enclose any global variables (here, `t` and `y`) by passing them as
# arguments to a function that creates `f!`.
# Here we write out the Jacobian by hand; below, we'll see how you can use Automatic Differentiation for these computations.

for i = 1:nc


function create_f!(U,N,D)
    return function(rj, x, idx)        # this is `f!`
        M = reshape(x,nc,noc)
        Rs = M*N-D
        Um = U[idx,:]; Wm = Um*M; Rmn = min.(Wm,0)
        if rj !== nothing              # Important! The `rj` input might be `nothing` (if the cache
            rj.r[1:nc^2] .= vec(Rs)             #  does not yet need updating)
            rj.r[1:nc^2] .= vec(Rs)
            rj.J[i:nc:end,i:nc:end] .= N'
            rj.J[idx,2] .= 
        end
        return dot(r, r) / 2
    end
end

f! = create_f!(t, y)    # f!(rj, x, idx)

# Here we use a minibatch size equal to the whole dataset, which makes the algorithm non-stochastic.
# (This is just to show that we can extract the exact values of `a` and `τ`.)
batchsize = length(t)
x0 = [0.9, 10.0]    # not the true minimum
x, objval = stochasticlm(f!, x0, batchsize, rj)

