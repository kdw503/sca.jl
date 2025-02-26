using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","X_Xt")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))
using Test

dataset = :fakecells; SNR=0; inhibitindices=[]; bias=0.1
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

lcsvd_maxiter = 150
compnmf_maxiter = 1000
hals_maxiter = 150

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))
bias = 0.1

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncells, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
ncells = 15 # ncs
(m,n,p) = (size(X)...,ncells)

α=0.005; β=5.0
β1 = β2= β; α1 = α2 = α
rt1 = @elapsed W0, H0, M, N, Wp, Hp, D = LCSVD.initlcsvd(X, ncells, 0; initmethod=:svd, svdmethod=:svd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0')
σ0=std(W0*M) # std(N0*H0t) #=10*std(W0)=#
r = 0.3

function symdiff(x1,x2,p,nc)
    m2 = reshape(x2[1:p*nc],p,nc)'; n2 = reshape(x2[p*nc+1:2*p*nc],p,nc)
    xt = vcat(vec(n2),vec(m2'))
    x1 == xt, norm(x1-xt)
end

T = eltype(W0); nc, p = size(M)

alg = LCSVD.LinearCombSVD(α1=α, α2=α, β1=β, β2=β, σ0=σ0, r=r, usedenoiseW0H0=false,
    uselv=false, imgsz=imgsz)
M1, N1t, N1 = copy(M), copy(N'), copy(N)
M2, N2t, N2 = copy(N'), copy(M), copy(M')
M3, N3t, N3 = copy(N)', copy(M), copy(M)'
state1 = LCSVD.prepare_state(W0, copy(H0'), M1, N1t, alg)
updater1 = LCSVD.LinearCombSVDUpd{T}(state1, D, alg)
state2 = LCSVD.prepare_state(H0', W0, M2, N2t, alg)
updater2 = LCSVD.LinearCombSVDUpd{T}(state2, D', alg)
state3 = LCSVD.prepare_state(copy(H0'), W0, M3, N3t, alg)
updater3 = LCSVD.LinearCombSVDUpd{T}(state3, D', alg)
W02=W0.^2; H0t2=Array(H0').^2

# Gradient, E and preconditioning
x1 = vcat(vec(state1.M'),vec(state1.Nt'))
fg1!, P1 = LCSVD.prepare_fg_invert_whole(W0, copy(H0'), W02, H0t2, D, state1, updater1, alg)
grad1 = zeros(T,2p*nc)
fg1!(nothing,grad1,x1); fval1 = fg1!(1,nothing,x1)
x2 = vcat(vec(state2.M'),vec(state2.Nt'))
fg2!, P2 = LCSVD.prepare_fg_invert_whole(copy(H0'), W0, H0t2, W02, D', state2, updater2, alg)
grad2 = zeros(T,2p*nc)
fg2!(nothing,grad2,x2); fval2 = fg2!(1,nothing,x2)
@test symdiff(grad1,grad2,p,nc)[1]
@test abs(fval1 - fval2) == 0

# invertivility
state1.grad=zeros(T,2*nc*p)
LCSVD.mmul!(state1.MNmD,M1,N1t';uselv=false)
state1.MNmD -= D; fval1 = LCSVD.fvalEi(state1.MNmD; uselv=false)
@show norm(state1.MNmD), sum(abs2,state1.MNmD)
LCSVD.gradEi!(state1.MNmD,state1.grad,M1,N1t; uselv=false)
state2.grad=zeros(T,2*nc*p)
LCSVD.mmul!(state2.MNmD,M2,N2t';uselv=false)
state2.MNmD -= D'; fval2 = LCSVD.fvalEi(state2.MNmD; uselv=false)
@show norm(state2.MNmD), sum(abs2,state2.MNmD)
LCSVD.gradEi!(state2.MNmD,state2.grad,M2,N2t; uselv=false)
@test symdiff(state1.grad,state2.grad,p,nc)[1]
@test abs(fval1 - fval2) == 0
# LCSVD.mmul!(state3.MNmD,M3,N3t';uselv=false)
# state3.MNmD -= D'; fval3 = LCSVD.fvalEi(state3.MNmD)
# LCSVD.gradEi!(state3.MNmD,state3.grad,M3,N3t;uselv=false)
# @test symdiff(state1.grad,state3.grad,p,nc)[1]
# @test abs(fval1 - fval3) == 0

# sparsity
aw1n1, ah1n1, aw3n1, ah3n1, regW11, regH11, regW31, regH31 = LCSVD.cal_params(W0, copy(H0'), M1, N1t, state1, updater1, alg)
aw1n2, ah1n2, aw3n2, ah3n2, regW12, regH12, regW32, regH32 = LCSVD.cal_params(copy(H0'), W0, M2, N2t, state2, updater2, alg)
# aw1n3, ah1n3, aw3n3, ah3n3, regW13, regH13, regW33, regH33 = LCSVD.cal_params(H0', W0, M3, N3t, state3, updater3, alg)
@test aw1n1 == ah1n2# == ah1n3
@test ah1n1 == aw1n2# == aw1n3
@test aw3n1 == ah3n2# == ah3n3
@test ah3n1 == aw3n2# == aw3n3
@test regW11 == regH12# == regH13
@test regH11 == regW12# == regW13
@test regW31 == regH32# == regH33
@test regH31 == regW32# == regW33

state1.grad=zeros(T,2*nc*p)
LCSVD.gradRelxEs1!(state1.W0TW,state1.TmpWs[1],W0,state1.W,updater1.σ2,uselv=false)
LCSVD.sprod!(state1.W0TW,state1.W0TW,aw1n1)
state1.grad[1:p*nc] .+= vec(state1.W0TW')
state2.grad=zeros(T,2*nc*p)
LCSVD.gradRelxEs1!(state2.H0HT,state2.TmpHts[1],W0, state2.Ht,updater2.σ2;uselv=false)
LCSVD.sprod!(state2.H0HT,state2.H0HT,ah1n2)
state2.grad[p*nc+1:end] .+= vec(state2.H0HT')
@test symdiff(state1.grad,state2.grad,p,nc)[1]
# state3.grad=zeros(T,2*nc*p)
# LCSVD.gradRelxEs1!(state3.H0HT,state3.TmpHts[1], (W0')', state3.Ht,updater3.σ2;uselv=false)
# LCSVD.sprod!(state3.H0HT,state3.H0HT,ah1n2)
# state3.grad[p*nc+1:end] .+= vec(state3.H0HT')
# @test symdiff(state1.grad,state3.grad,p,nc)[1]

state1.grad=zeros(T,2*nc*p)
LCSVD.gradRelxEs1!(state1.H0HT,state1.TmpHts[1],copy(H0'),state1.Ht,updater1.σ2;uselv=false)
LCSVD.sprod!(state1.H0HT,state1.H0HT,ah1n1)
state1.grad[p*nc+1:end] .+= vec(state1.H0HT')
state2.grad=zeros(T,2*nc*p)
LCSVD.gradRelxEs1!(state2.W0TW,state2.TmpWs[1],copy(H0'),state2.W,updater2.σ2,uselv=false)
LCSVD.sprod!(state2.W0TW,state2.W0TW,aw1n2)
state2.grad[1:p*nc] .+= vec(state2.W0TW')
@test symdiff(state1.grad,state2.grad,p,nc)[1]
# state3.grad=zeros(T,2*nc*p)
# LCSVD.gradRelxEs1!(state3.W0TW,state3.TmpWs[1],H0',state3.W,updater2.σ2,uselv=false)
# LCSVD.sprod!(state3.W0TW,state3.W0TW,aw1n3)
# state3.grad[1:p*nc] .+= vec(state3.W0TW')
# @test symdiff(state1.grad,state3.grad,p,nc)[1]

(aw11, ah11, aw31, ah31) = (updater1.αw, updater1.αh, updater1.βw, updater1.βh)
(aw12, ah12, aw32, ah32) = (updater2.αw, updater2.αh, updater2.βw, updater2.βh)
(aw41, ah41, aw51, ah51) = (ah11*regH11, aw11*regW11, ah31*regH31, aw31*regW31)
(aw42, ah42, aw52, ah52) = (ah12*regH12, aw12*regW12, ah32*regH32, aw32*regW32)
@test aw41 == ah42
@test ah41 == aw42
@test aw51 == ah52
@test ah51 == aw52

state1.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state1.TmpMs[1],M1,aw41/norm(M1))
state1.grad[1:p*nc] .+= vec(state1.TmpMs[1]')
state3.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state3.TmpMs[1],N3t,ah42/norm(N3t))
state3.grad[p*nc+1:end] .+= vec(state3.TmpMs[1]')
@test symdiff(state1.grad,state3.grad,p,nc)[1]

state1.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state1.TmpMs[1],N1t,ah41/norm(N1t))
state1.grad[p*nc+1:end] .+= vec(state1.TmpMs[1]')
state2.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state2.TmpMs[1],M2,aw42/norm(M2))
state2.grad[1:p*nc] .+= vec(state2.TmpMs[1]')
@test symdiff(state1.grad,state2.grad,p,nc)[1]

# non-negativity
state1.grad=zeros(T,2*nc*p)
state1.bnn .= 0.; LCSVD.gradEnn!(state1.bnn,W0,state1.W)
LCSVD.sprod!(state1.bnn,state1.bnn,2*aw3n1)
state1.grad[1:p*nc] .+= vec(reshape(state1.bnn,nc,p)')
state2.grad=zeros(T,2*nc*p)
state2.bnn .= 0.; LCSVD.gradEnn!(state2.bnn,(W0')',state2.Ht)
LCSVD.sprod!(state2.bnn,state2.bnn,2*ah3n2)
state2.grad[p*nc+1:end] .+= vec(reshape(state2.bnn,nc,p)')
@test symdiff(state1.grad,state2.grad,p,nc)[1]
# state3.grad=zeros(T,2*nc*p)
# state3.bnn .= 0.; LCSVD.gradEnn!(state3.bnn,Array(W0')',state3.Ht)
# LCSVD.sprod!(state3.bnn,state3.bnn,2*ah3n3)
# state3.grad[p*nc+1:end] .+= vec(reshape(state3.bnn,nc,p)')
# @test symdiff(state1.grad,state3.grad,p,nc)[1]

state1.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state1.TmpMs[1],M1,aw51*2.)
state1.grad[1:p*nc] .+= vec(state1.TmpMs[1]')
state2.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state2.TmpMs[1],N2t,ah52*2.)
state2.grad[p*nc+1:end] .+= vec(state2.TmpMs[1]')
@test symdiff(state1.grad,state2.grad,p,nc)[1]
state1.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state1.TmpMs[1],N1t,ah51*2.)
state1.grad[p*nc+1:end] .+= vec(state1.TmpMs[1]')
state2.grad=zeros(T,2*nc*p)
LCSVD.sprod!(state2.TmpMs[1],M2,aw52*2.)
state2.grad[1:p*nc] .+= vec(state2.TmpMs[1]')
@test symdiff(state1.grad,state2.grad,p,nc)[1]

# minMN_whole!
M1, N1t = copy(M), copy(N');
M2, N2t = copy(N'), copy(M)
state1 = LCSVD.prepare_state(W0, copy(H0'), M1, N1t, alg);
updater1 = LCSVD.LinearCombSVDUpd{T}(state1, D, alg);
state2 = LCSVD.prepare_state(H0', W0, M2, N2t, alg);
updater2 = LCSVD.LinearCombSVDUpd{T}(state2, D', alg);
gtW=Matrix{T}(undef,0,0); gtH=Matrix{T}(undef,0,0);
alg.inner_tol = -1e-6; alg.inner_maxiter = 100; trace = LCSVD.Trace(T);
LCSVD.minMN_whole!(X, W0, copy(H0'), W02, H0t2, D, gtW, gtH, state1, updater1, trace; alg=alg)
LCSVD.minMN_whole!(X', copy(H0'), W0, H0t2, W02, D, gtH, gtW, state2, updater2, trace; alg=alg)
@test state1.M == state2.Nt
@test state1.Nt == state2.M
@test norm(state1.M-state2.Nt) == 0
LCSVD.minMN_whole!(X, W0, copy(H0'), W02, H0t2, D, gtW, gtH, state1, updater1, trace; alg=alg)
LCSVD.minMN_whole!(X', copy(H0'), W0, H0t2, W02, D, gtH, gtW, state2, updater2, trace; alg=alg)
@test state1.M == state2.Nt
@test state1.Nt == state2.M
@test norm(state1.M-state2.Nt) == 0

# solve!
nac=10; r=0.3; tol=1e-6
W0, H0, M, N, Wp, Hp, D = LCSVD.initlcsvd(X, p, nac; initmethod=:isvd, svdmethod=:isvd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M'); N0t = copy(N')
M1, N1t = copy(M), copy(N')
M2, N2t = copy(N'), copy(M)
σ0=std(W0*M)
rst1 = LCSVD.solve!(LCSVD.LinearCombSVD(T,α1=0.005,α2=0.005,β1=5.0,β2=5.0,σ0=σ0,
                                        r=r,maxiter=10,inner_maxiter=1000,f_abstol=tol, f_reltol=tol,
                                        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol=1e-6),
                                        X, W0, copy(H0'), D, M1, N1t)
#σ0= std(M0t*W0t) # this is different from std(W0*M)
rst2 = LCSVD.solve!(LCSVD.LinearCombSVD(T,α1=0.005,α2=0.005,β1=5.0,β2=5.0,σ0=σ0,
                                        r=r,maxiter=10,inner_maxiter=1000,f_abstol=tol, f_reltol=tol,
                                        f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol=1e-6),
                                        X', copy(H0'), W0, D', M2, N2t)
@test norm(M1-N2t) == 0
@test norm(N1t'-M2') == 0
um = M1\N2t; un = M2'/N1t'
@test norm(um'*um-I) <= length(M1)*eps(T)
@test norm(un*un'-I) <= length(N1)*eps(T)
@test norm(rst1.W-rst2.Ht) == 0
@test norm(rst1.Ht-rst2.W) == 0
uw = rst1.W\rst2.Ht; uh = rst1.W'/rst2.Ht'
@test norm(uw'*uw-I) <= length(rst1.W)*eps(T)
@test norm(uh*uh'-I) <= length(rst1.Ht)*eps(T)
