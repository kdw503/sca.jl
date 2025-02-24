using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","col-wise")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using ForwardDiff

dataset = :fakecells; SNR=0; inhibitindices=[1,2,3]; bias=0.5
filter = dataset ∈ [] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))

@show bias; flush(stdout)
imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
X, imgsz, lengthT, ncs, gtncs, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
        lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
        isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
(m,n,p) = (size(X)...,ncs)
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))

subtract_bg = true
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    bg = W*fill(mean(H),1,n)
    X .-= bg
end

#============== component-wise 1 : σi|Umi|σ||N||₂ =================#
# initialization
initmethod = :isvd; 
noc = 15; nac=0; nc = noc+nac
σ2 = 0.1
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:svd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)

# sparsity
aw = rand(noc); ah = rand(noc)
Aw = Diagonal(aw); Ah = Diagonal(ah)
Esw(M) = (s = 0; for i in 1:noc s += LCSVD.relaxedL1(W0*M[:,i],σ2,uselv=false)*aw[i] end; s*norm(N0))
fdgradEw1, trueE1 = ForwardDiff.gradient(Esw,M0), Esw(M0)
gradEsw(M) = (W = W0*M; W_tilde=W./sqrt.(W.^2 .+σ2);W0'W_tilde*Aw*norm(N0))
gEsw = gradEsw(M0)
norm(fdgradEw1 - gEsw)
Esh(N) = (s = 0; for i in 1:noc s += LCSVD.relaxedL1(N[i,:]'*H0,σ2,uselv=false)*ah[i] end; s*norm(M0))
fdgradEh1, trueE1 = ForwardDiff.gradient(Esh,N0), Esh(N0)
gradEsh(N) = (H = N*H0; H_tilde=H./sqrt.(H.^2 .+σ2);Ah*H_tilde*H0'*norm(M0))
gEsh = gradEsh(N0)
norm(fdgradEh1 - gEsh)

# non-negativity
bw = rand(noc); bh = rand(noc)
Bw = Diagonal(bw); Bh = Diagonal(bh)
Enw(M) = (s = 0; for i in 1:noc s += LCSVD.sca2(W0*M[:,i])*bw[i] end; s*norm(N0)^2)
fdgradEw1, trueE1 = ForwardDiff.gradient(Enw,M0), Enw(M0)
gradEnw(M) = (W = W0*M; W[W.>0.].=0.; 2.0*W0'W*Bw*norm(N0)^2)
gEnw = gradEnw(M0)
norm(fdgradEw1 - gEnw)
Enh(N) = (s = 0; for i in 1:noc s += LCSVD.sca2(N[i,:]'*H0)*bh[i] end; s*norm(M0)^2)
fdgradEh1, trueE1 = ForwardDiff.gradient(Enh,N0), Enh(N0)
gradEnh(N) = (H = N*H0; H[H.>0.].=0.; 2.0*Bh*H*H0'*norm(M0)^2)
gEnh = gradEnh(N0)
norm(fdgradEh1 - gEnh)


# PCB
prefix = "pcb"
noc = ncs; nac = 0
(tailstr,initmethod,α,β) = ("_sp_nn",:isvd,0.005,5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,5.0)

rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)

β1 = β2= β; α1 = α2 = α
β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.
α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.
σ0=1.0; r=0.3 # std(N0*H0t) #=10*std(W0)=#
useprecond=false; uselv=false; tol=1e-6
maxiter = 100#Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
alg = LCSVD.LinearCombSVD(#α1=α1, α2=α2, β1=β1, β2=β2,
    α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec,
    σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rst1 = LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);
show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);

W1, H1 = rst1.W, rst1.Ht'
LCSVD.normalizeW!(W1,H1);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
fitval = LCSVD.fitd(X,W1*H1)
fv = dataset == :fakecells ? avgfit : fitval
nodr = LCSVD.matchedorder(ml,noc); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
LCSVD.flip2makepos!(Wlc1,Hlc1)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(noc)_$(initmethod)"
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]))_bvec($(β1vec[1]),$(β1vec[2]))" : "_a$(α)_b$(β)"
fname = joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)

#============== component-wise 2 : σi|Umi|σ||ni||₂ =================#
# initialization
initmethod = :isvd; 
noc = 15; nac=0; nc = noc+nac
σ2 = 0.1
rt1 = @elapsed U, Vt, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:svd)
Xt = copy(X'); Ut = copy(U'); V = copy(Vt'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)
M = rand(noc,noc); N = rand(noc,noc); Nt = copy(N')
W = U*M; H = N*Vt
# sparsity
aws = rand(noc); ahs = rand(noc)
bws = rand(noc); bhs = rand(noc)
norm_nis(N) = map(n -> norm(n), eachrow(N)); norm_nis2(N) = norm_nis(N).^2
norm_mis(M) = map(m -> norm(m), eachcol(M)); norm_mis2(M) = norm_mis(M).^2
sparwi(M) = map(m -> sum(sqrt.((U*m).^2 .+ σ2)), eachcol(M))
sparhi(N) = map(n -> sum(sqrt.((n'*Vt).^2 .+ σ2)), eachrow(N))
neg(x::T) where T = x < zero(T) ? x : zero(T)
nnwi(M) = map(m -> norm(neg.(U*m))^2, eachcol(M))
nnhi(N) = map(n -> norm(neg.(n'*Vt))^2, eachrow(N))
W_tilde(W) = map(w->w/sqrt(w^2+σ2),W); W_(W) = map(w->neg(w),W)
H_tilde(H) = map(h->h/sqrt(h^2+σ2),H); H_(H) = map(h->neg(h),H)

aws_nnis(N) = aws.*norm_nis(N); bws_nnis2 = bws.*norm_nis2(N);
ahs_nmis(N) = ahs.*norm_mis(M); bhs_nmis2 = bhs.*norm_mis2(M);
A_tilde_w(N) = Diagonal(aws_nnis(N)); A_tilde_h(M) = Diagonal(ahs_nmis(M))
B_tilde_w(N) = Diagonal(bws_nnis2(N)); B_tilde_h(M) = Diagonal(bhs_nmis2(M))

aws_spw_nnis(M,N) = aws.*sparwi(M)./norm_nis(N); bws_nnw_nnis2(M,N) = bws.*nnwi(M)./norm_nis2(N);
ahs_sph_nmis(M,N) = ahs.*sparhi(N)./norm_mis(M); bhs_nnh_nmis2(M,N) = bhs.*nnhi(N)./norm_mis2(M);
A_breve_w(M,N) = Diagonal(aws_spw_nnis(M,N)); A_breve_h(M,N) = Diagonal(ahs_sph_nmis(M,N))
B_breve_w(M,N) = Diagonal(bws_nnw_nnis2(M,N)); B_breve_h(M,N) = Diagonal(bhs_nnh_nmis2(M,N))

# sparsity
MtNt = hcat(M',Nt)
Esw0(M,N) = (s = 0; for i in 1:noc s += LCSVD.relaxedL1(U*M[:,i],σ2,uselv=false)*aws[i]*norm(N[i,:]) end; s)
Esh0(M,N) = (s = 0; for i in 1:noc s += LCSVD.relaxedL1(N[i,:]'*Vt,σ2,uselv=false)*ahs[i]*norm(M[:,i]) end; s)
Esw(MtNt) = (M=MtNt[:,1:noc]'; N=MtNt[:,noc+1:2noc]'; Esw0(M,N))
Esh(MtNt) = (M=MtNt[:,1:noc]'; N=MtNt[:,noc+1:2noc]'; Esh0(M,N))
Es(MtNt) = (M=MtNt[:,1:noc]'; N=MtNt[:,noc+1:2noc]'; Esw0(M,N)+Esh0(M,N))

fdgradEsw, trueEsw = ForwardDiff.gradient(Esw,MtNt), Esw(MtNt)
fdgradEsh, trueEsh = ForwardDiff.gradient(Esh,MtNt), Esh(MtNt)
fdgradEs, trueEs = ForwardDiff.gradient(Es,MtNt), Es(MtNt)

gradEswM(M,N) = (W = U*M; U'W_tilde(W)*A_tilde_w(N))
gradEswN(M,N) = A_breve_w(M,N)*N 
gradEsw0(M,N) = (gradEswM(M,N),gradEswN(M,N))

gradEshM(M,N) = M*A_breve_h(M,N)
gradEshN(M,N) = (H = N*Vt; A_tilde_h(M)*H_tilde(H)*V)
gradEsh0(M,N) = (gradEshM(M,N),gradEshN(M,N))

gradEsw(MtNt) = (M=MtNt[:,1:noc]'; N=MtNt[:,noc+1:2noc]'; (gM,gN) = gradEsw0(M,N); hcat(gM',gN'))
gradEsh(MtNt) = (M=MtNt[:,1:noc]'; N=MtNt[:,noc+1:2noc]'; (gM,gN )= gradEsh0(M,N); hcat(gM',gN'))
gradEs(MtNt) = (gradEsw(MtNt)+gradEsh(MtNt))
gEsw = gradEsw(MtNt)
gEsh = gradEsh(MtNt)
gEs = gradEs(MtNt)
norm(fdgradEsw[:,1:noc] - gEsw[:,1:noc])
norm(fdgradEsh[:,1:noc] - gEsh[:,1:noc])
norm(fdgradEsw[:,noc+1:2noc] - gEsw[:,noc+1:2noc])
norm(fdgradEsh[:,noc+1:2noc] - gEsh[:,noc+1:2noc])
norm(fdgradEsw - gEsw)
norm(fdgradEsh - gEsh)
norm(fdgradEs - gEs)


# non-negativity
bw = rand(noc); bh = rand(noc)
Bw = Diagonal(bw); Bh = Diagonal(bh)
Enw(M) = (s = 0; for i in 1:noc s += LCSVD.sca2(W0*M[:,i])*bw[i] end; s*norm(N0)^2)
fdgradEw1, trueE1 = ForwardDiff.gradient(Enw,M0), Enw(M0)
gradEnw(M) = (W = W0*M; W[W.>0.].=0.; 2.0*W0'W*Bw*norm(N0)^2)
gEnw = gradEnw(M0)
norm(fdgradEw1 - gEnw)
Enh(N) = (s = 0; for i in 1:noc s += LCSVD.sca2(N[i,:]'*H0)*bh[i] end; s*norm(M0)^2)
fdgradEh1, trueE1 = ForwardDiff.gradient(Enh,N0), Enh(N0)
gradEnh(N) = (H = N*H0; H[H.>0.].=0.; 2.0*Bh*H*H0'*norm(M0)^2)
gEnh = gradEnh(N0)
norm(fdgradEh1 - gEnh)


prepare_fg_colwise_params2


# PCB
prefix = "pcb"
noc = ncs; nac = 0
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,0.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initpcb(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)

β1 = β2= β; α1 = α2 = α
#β1 = β; β2 = 0; α1 = 0; α2 = α
β1vec = fill(β1,noc); β2vec = fill(β2,noc); β1vec[1] = 0.; β2vec[1] = 0.; #β1vec[2] = 0.; β2vec[2] = 0.
α1vec = fill(α1,noc); α2vec = fill(α2,noc); α1vec[1] = 0.; α2vec[1] = 0.; #α1vec[2] = 0.; α2vec[2] = 0.
σ0=1.0; r=0.3 # std(N0*H0t) #=10*std(W0)=#
useprecond=false; uselv=false; tol=1e-6
maxiter = 100#Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncs+350))# Int(ceil(0.75*ncs+100)) # 
alg = LCSVD.LinearCombSVD(#α1=α1, α2=α2, β1=β1, β2=β2,
    α1vec=α1vec, α2vec=α2vec, β1vec=β1vec, β2vec=β2vec, scaletype = 3,
    σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = true, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rst1 = LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);
show_trace = false; alg.store_trace = false; alg.store_inner_trace = false
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);

W1, H1 = rst1.W, rst1.Ht'
LCSVD.normalizeW!(W1,H1);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
fitval = LCSVD.fitd(X,W1*H1)
fv = dataset == :fakecells ? avgfit : fitval
nodr = LCSVD.matchedorder(ml,noc); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
LCSVD.flip2makepos!(Wlc1,Hlc1)
fprex = "$(prefix)$(SNR)db_bias$(bias)_inh$(inhibitindices)_ft$(factor)_nc$(noc)_$(sbgstr)_$(initmethod)"
regstr = alg.usecolparams ? "_avec($(α1vec[1]),$(α1vec[2]),$(alg.scaletype))_bvec($(β1vec[1]),$(β1vec[2]))" : "_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)"
fname = joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)
LCSVD.flip2makepos!(W1,H1)
fname = joinpath(subworkpath,"$(fprex)$(regstr)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)_")
imsave_data(dataset,fname,W1,H1,imgsz,100; saveH=false)
