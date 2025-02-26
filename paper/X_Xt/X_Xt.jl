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
gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
gtfname = "fakecells$(inhibitindices)_calcium_sz$(imgsz)_lengthT$(lengthT)_SNR$(SNR)_bias$(bias)"
imsave_data(dataset,joinpath(subworkpath,gtfname),gtW,gtH',imgsz,100; saveH=false)
plotH_data(joinpath(subworkpath,gtfname),gtH'; space=0.,ylabel="",ytickformat="{:.2f}")
X = LCSVD.noisefilter(filter,X,size)

subtract_bg = false
sbgstr = subtract_bg ? "sbg" : "nosbg"
if subtract_bg
    rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
    LCSVD.normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
    plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
    bg = W*fill(mean(H),1,n)
#        bg = W*H
    X .-= bg
end

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)

mfmethod = :LCSVD; useprecond=false; uselv=false; tol=1e-6
r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
maxiter = 100#Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
inner_tol = 1e-6; inner_maxiter = 1000#Int(ceil(2.5*ncells+350))# Int(ceil(0.75*ncells+100)) # 

usedenoiseW0H0 = false; makepositive = true

(tailstr,initmethod,α,β) = ("_sp",:svd,0.005,5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)

β1 = β2= β; α1 = α2 = α
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells, 0; initmethod=initmethod, svdmethod=:svd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)

# PCB(ISVD(X))
σ0=std(W0*M0); # std(N0*H0t) #=10*std(W0)=#
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
M1, N1t = copy(M0), copy(N0t)
rst1 = LCSVD.solve!(alg, X, W0, copy(H0'), D, M1, N1t);
alg.store_trace = false; alg.store_inner_trace = false
M1, N1t = copy(M0), copy(N0t)
rt2 = @elapsed LCSVD.solve!(alg, X, W0, copy(H0'), D, M1, N1t);

W1, H1 = rst1.W, rst1.Ht'
LCSVD.normalizeW!(W1,H1);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
fitval = LCSVD.fitd(X,W1*H1)
fv = dataset == :fakecells ? avgfit : fitval
nodr = LCSVD.matchedorder(ml,ncells); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
makepositive && LCSVD.flip2makepos!(Wlc1,Hlc1)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(ncells)_$(initmethod)"
fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_sigW_r$(r)_intol$(inner_tol)_inmiter$(inner_maxiter)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst1.niters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)
# plotH_data(fname*"_Hinhibit",Hlc[inhibitindices,:]; space=0.,ylabel="",ytickformat="{:.2f}")
#plotH_data(fname*"_H",Hlc[1:8,:]; space=0.,ylabel="",ytickformat="{:.2f}")
# save(joinpath(subworkpath,"xdiffs_ft$(factor)_nc$(ncells)_r$(r)_intol$(inner_tol)_inmiter$(inner_maxiter)_tol$(tol)_miter$(maxiter).png"),f)

# PCB(ISVD(X)')
σ0= std(M0t*W0t);# std(W0t*M0t) #=10*std(W0)=#
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0);
M2, Nt2 = copy(N0t), copy(M0)
rst2 = LCSVD.solve!(alg, X', H0t, W0, D', M2, Nt2);
alg.store_trace = false; alg.store_inner_trace = false
M2, Nt2 = copy(N0t), copy(M0)
rt2 = @elapsed LCSVD.solve!(alg, X', H0t, W0, D', M2, Nt2);

W2, H2 = copy(rst2.Ht), copy(rst2.W')
LCSVD.normalizeW!(W2,H2);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W2, H2; clamp=false)
fitval = LCSVD.fitd(X,W2*H2)
fv = dataset == :fakecells ? avgfit : fitval
nodr = LCSVD.matchedorder(ml,ncells); Wlc2, Hlc2 = W2[:,nodr], H2[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
makepositive && LCSVD.flip2makepos!(Wlc2,Hlc2)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(ncells)_$(initmethod)"
fname = joinpath(subworkpath,"$(fprex)t_a$(α)_b$(β)_sigH_r$(r)_intol$(inner_tol)_inmiter$(inner_maxiter)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst2.niters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc2,Hlc2,imgsz,100; saveH=false)


# PCB(ISVD(X'))
Xt = copy(X')
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(Xt, ncells, 0; initmethod=initmethod, svdmethod=:isvd)
σ0= std(N0*H0)# std(W0t*M0t) #=10*std(W0)=#
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M3, N3 = copy(M0), copy(N0)
rst3 = LCSVD.solve!(alg, Xt, W0, H0, D, M3, N3);
alg.store_trace = false; alg.store_inner_trace = false
M3, N3 = copy(M0), copy(N0)
rt2 = @elapsed LCSVD.solve!(alg, Xt, W0, H0, D, M3, N3);

W3, H3 = copy(rst3.Ht), copy(rst3.W')
LCSVD.normalizeW!(W3,H3);
# avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W3, H3; clamp=false)
fitval = LCSVD.fitd(X,W3*H3)
fv = dataset == :fakecells ? avgfit : fitval
nodr = LCSVD.matchedorder(ml,ncells); Wlc3, Hlc3 = W3[:,nodr], H3[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
makepositive && LCSVD.flip2makepos!(Wlc3,Hlc3)
fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(ncells)_$(initmethod)"
fname = joinpath(subworkpath,"$(fprex)Xt_a$(α)_b$(β)_sigH_r$(r)_intol$(inner_tol)_inmiter$(inner_maxiter)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst3.niters)_rt$(rt2)")
imsave_data(dataset,fname,Wlc3,Hlc3,imgsz,100; saveH=false)


r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
(tailstr,initmethod,α,β) = ("_sp",:svd,0.005,5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)
β1 = β2= β; α1 = α2 = α
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells, 0; initmethod=initmethod, svdmethod=:svd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)

maxiter = 10 #Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
tol = 1e-6
inner_tol = 1e-6; inner_maxiter = 1000 #Int(ceil(2.5*ncells+350))# Int(ceil(0.75*ncells+100)) # 

# PCB(ISVD(X))
σ0=std(W0*M0) # std(N0*H0t) #=10*std(W0)=#
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M1, N1t = copy(M0), copy(N0t)
rst1 = LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);
# alg.store_trace = false; alg.store_inner_trace = false
# M1, N1 = copy(M0), copy(N0)
# rt2 = @elapsed LCSVD.solve!(alg, X, W0, H0, D, M1, N1);

# W1, H1 = rst1.W, rst1.H
# LCSVD.normalizeW!(W,H);
# # avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
# avgfit, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
# fitval = LCSVD.fitd(X,W1*H1)
# fv = dataset == :fakecells ? avgfit : fitval
# nodr = LCSVD.matchedorder(ml,ncells); Wlc1, Hlc1 = W1[:,nodr], H1[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
# LCSVD.flip2makepos!(Wlc1,Hlc1)
# fprex = "$(prefix)$(SNR)db_ft$(factor)_nc$(ncells)_$(initmethod)"
# fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_sigW_r$(r)_intol$(inner_tol)_inmiter$(inner_maxiter)_tol$(tol)_miter$(maxiter)_f$(fv)_it$(rst0.niters)_rt$(rt2)")
# imsave_data(dataset,fname,Wlc1,Hlc1,imgsz,100; saveH=false)

# PCB(ISVD(X)')
σ0= std(M0t*W0t)# std(W0t*M0t) #=10*std(W0)=#
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M2, Nt2 = copy(N0t), copy(M0)
rst2 = LCSVD.solve!(alg, X', H0t, W0, D', M2, Nt2);

# norm(M1-N2')
# norm(N1-M2')
# for i in 2:length(rst1.traces)
#     print("$(rst1.traces[i].niters), ")
#     println()
# end
# for i in 2:length(rst2.traces)
#     print("$(rst2.traces[i].niters), ")
#     println()
# end

nc = ncells
fdiffs = Float64[]; ngdiffs = Float64[]; xdiffs = Float64[]
nrmWs = Float64[]; nrmHs = Float64[]; diffMs = Float64[]; diffNs = Float64[];
for (tr1,tr2) in zip(rst1.traces[2:9], rst2.traces[2:9])
    for (fx1, fx2) in zip(tr1.fxs,tr2.fxs)
        push!(fdiffs, abs(fx1-fx2))
    end
    for (gxs1, gxs2) in zip(tr1.gxs,tr2.gxs)
        gxs2t = [gxs2[nc*p+1:2*nc*p]...,gxs2[1:nc*p]...]
        push!(ngdiffs, norm(gxs1-gxs2t))
    end
    for (x1, x2) in zip(tr1.xs,tr2.xs)
        M1, N1t = reshape(x1[1:nc*p],p,nc)', reshape(x1[nc*p+1:2nc*p],p,nc)'
        M2, N2t = reshape(x2[1:nc*p],p,nc)', reshape(x2[nc*p+1:2nc*p],p,nc)'
        x2t = [x2[nc*p+1:2*nc*p]...,x2[1:nc*p]...]
        push!(nrmWs, norm(M1-N2t)); push!(nrmHs, norm(N1t'-M2'))
        um = M1\N2t; un = M2'/N1t'; umtumIdiff = norm(um'um-I); ununtIdiff = norm(un*un'-I)
        push!(diffMs, umtumIdiff); push!(diffNs, ununtIdiff); push!(xdiffs, norm(x1-x2t))
    end
end

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,nrmWs[3:end],label="norm(M1-N2')")
lines!(ax,nrmHs[3:end],label="norm(N1_M2')")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"MNnormdiff_im$(inner_maxiter).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,diffMs[3:end],label="M_diff")
lines!(ax,diffNs[3:end],label="N_diff")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"MNdiff_im$(inner_maxiter).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,xdiffs[3:end],linestyle=:dot,label="norm(x1-x2)")
lines!(ax,ngdiffs[3:end],linestyle=:dot,label="norm(g1-g2)")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"xgdiff_im$(inner_maxiter).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=identity)
lines!(ax,fdiffs[2:end],label="abs(fx1-fx2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"fxdiff_im$(inner_maxiter).png"),f)

f = Figure(size=(350,250))
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing), yscale=log10)
lines!(ax,ngdiffs[2:end],label="norm(g1-g2)")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"ngdiff_im$(inner_maxiter).png"),f)



W = zeros(eltype(W0),size(W0,1),size(M1,2))
LCSVD.mmul!(W,W0,M1)
TmpW = similar(W)
@inbounds @simd for i in eachindex(W)
    TmpW[i] = W[i]/sqrt(W[i]^2 + σ2)
end
W0TW = zeros(eltype(W0),size(W0,2),size(TmpW,2))
W0TW2 = zeros(eltype(W0),size(W0,2),size(TmpW,2))
LCSVD.mmul!(W0TW,W0',TmpW;uselv=false)
LCSVD.mmul!(W0TW2,copy(W0'),TmpW;uselv=false)
norm(W0TW-W0TW2) # 2.6044931271560082e-14

Ht = zeros(eltype(H0t),size(H0t,2),size(N2,1))
LCSVD.mmul!(Ht,H0t',N2')
TmpH = similar(Ht)
@inbounds @simd for i in eachindex(Ht)
    TmpH[i] = Ht[i]/sqrt(Ht[i]^2 + σ2)
end
H0HT = zeros(eltype(W0),size(H0t,1),size(TmpH,2))
LCSVD.mmul!(H0HT,H0t,TmpH;uselv=false)

norm(W0-H0t') # 0.0
norm(M1-N2') # 0.0
norm(W-Ht) # 0.0
norm(TmpW-TmpH) # 0.0
norm(W0TW-H0HT) # 2.6044931271560082e-14

function mymul!(C,A,B)
    nc, m = size(A)
    p = size(B,2)
    for i in 1:nc
        for j in 1:p
            for k in 1:m
                C[i,j] += A[i,k]*B[k,j] 
            end
        end
    end
end

A = rand(800,15); B = rand(800,15)

ATB = zeros(Float64,15,15);
ATB2 = zeros(Float64,15,15);
@time mul!(ATB, A',B);
@time mul!(ATB2, copy(A'),B);
norm(ATB-ATB2) # 2.3131849444980393e-12

ATB = zeros(Float64,15,15);
ATB2 = zeros(Float64,15,15);
@time mymul!(ATB, A',B);
@time mymul!(ATB2, copy(A'),B);
norm(ATB-ATB2) # 0.0


mfmethod = :LCSVD; useprecond=false; uselv=false
r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
usedenoiseW0H0 = false; makepositive = true

r=0.3 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,5.0)# ("_nn",:nndsvd,0.,5.0), ("_sp_nn",:isvd,0.005,0.005)
noc = 15; nac = 0; nc = noc+nac
β1 = β2= β; α1 = α2 = α
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, nc, 0; initmethod=initmethod, svdmethod=:svd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)

maxiter = 10 #Int(ceil(log(eps(eltype(X)))/log(r))) #lcsvd_maxiter # 
tol = 1e-6
inner_tol = 1e-6; inner_maxiter = 1000 #Int(ceil(2.5*ncells+350))# Int(ceil(0.75*ncells+100)) # 
σ0=std(W0*M0) # std(N0*H0t) #=10*std(W0)=#
r=0.3

M1, N1t = copy(M0), copy(N0t)
M02 = M0.+rand(size(M0)...)*eps(Float64) 
N0t2 = N0t.+rand(size(N0t)...)*eps(Float64) 

alg = LCSVD.LinearCombSVD(α1=α, α2=α, β1=β, β2=β, σ0=σ0, r=r, usedenoiseW0H0=false,
                            denoisefilter=:avg, uselv=false, imgsz=imgsz)

M1, N1t = copy(M0), copy(N0t)
state1 = LCSVD.prepare_state(W0, H0t, M1, N1t, alg)
updater1 = LCSVD.LinearCombSVDUpd{eltype(W0)}(state1, D, alg)
aw1n, ah1n, aw3n, ah3n, regW1, regH1, regW3, regH3 = LCSVD.cal_params(W0, H0t, M1, N1t, state1, updater1, alg)
(aw1, ah1) = (updater1.αw, updater1.αh)
(aw3, ah3) = (updater1.βw, updater1.βh)
(aw4, ah4) = (ah1*regH1, aw1*regW1)
(aw5, ah5) = (ah3*regH3, aw3*regW3)
state1.grad .= 0
LCSVD.gradRelxEs1!(state1.W0TW,state1.TmpWs[1],W0,state1.W,updater1.σ2,uselv=alg.uselv)
LCSVD.sprod!(state1.W0TW,state1.W0TW,aw1n)
state1.grad[1:p*nc] .+= vec(state1.W0TW')
LCSVD.gradRelxEs1!(state1.H0HT,state1.TmpHts[1],H0t,state1.Ht,updater1.σ2;uselv=alg.uselv)
LCSVD.sprod!(state1.H0HT,state1.H0HT,ah1n)
state1.grad[p*nc+1:end] .+= vec(state1.H0HT') # because x[1:p*nc] = vec(M')

M02 = M0.+(rand(size(M0)...).-0.5)*eps(Float64) 
N0t2 = N0t.+(rand(size(N0t)...).-0.5)*eps(Float64) 
M2, N2t = copy(M02), copy(N0t2)
state2 = LCSVD.prepare_state(W0, H0t, M2, N2t, alg)
updater2 = LCSVD.LinearCombSVDUpd{eltype(W0)}(state2, D, alg)
aw1n, ah1n, aw3n, ah3n, regW1, regH1, regW3, regH3 = LCSVD.cal_params(W0, H0t, M2, N2t, state2, updater2, alg)
(aw1, ah1) = (updater2.αw, updater2.αh)
(aw3, ah3) = (updater2.βw, updater2.βh)
(aw4, ah4) = (ah1*regH1, aw1*regW1)
(aw5, ah5) = (ah3*regH3, aw3*regW3)
state2.grad .= 0
LCSVD.gradRelxEs1!(state2.W0TW,state2.TmpWs[1],W0,state2.W,updater2.σ2,uselv=alg.uselv)
LCSVD.sprod!(state2.W0TW,state2.W0TW,aw1n)
state2.grad[1:p*nc] .+= vec(state2.W0TW')
LCSVD.gradRelxEs1!(state2.H0HT,state2.TmpHts[1],H0t,state2.Ht,updater2.σ2;uselv=alg.uselv)
LCSVD.sprod!(state2.H0HT,state2.H0HT,ah1n)
state2.grad[p*nc+1:end] .+= vec(state2.H0HT') # because x[1:p*nc] = vec(M')

norm(M1-M2)
norm(state1.grad[1:p*nc]-state2.grad[1:p*nc])
norm(N1t-N2t)
norm(state1.grad[p*nc+1:2p*nc]-state2.grad[p*nc+1:2p*nc])

f(x) = (M0 = reshape(x[1:p*nc],p,nc)'; N0t = reshape(x[p*nc+1:end],p,nc)'; Ws = W0*M0; Hst = H0t*N0t;
        norm(M0*N0t'-D)^2 + updater1.αw*LCSVD.relaxedL1(Ws,updater1.σ2,uselv=alg.uselv)*norm(N0t) +
        updater1.αh*LCSVD.relaxedL1(Hst,updater1.σ2,uselv=alg.uselv)*norm(M0) +
        updater1.βw*LCSVD.sca2(Ws)*norm(N0t)^2 + updater1.βh*LCSVD.sca2(Hst)*norm(M0)^2)
# Gradient, E and preconditioning
# rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, nc, 0; initmethod=:nndsvd, svdmethod=:svd)
x01 = vcat(vec(M1'),vec(N1t'))
x02 = vcat(vec(M2'),vec(N2t'))
using ForwardDiff
@time fdgradE1, trueE1 = ForwardDiff.gradient(f,x01), f(x01) # 4.6sec
@time fdhessE1 = ForwardDiff.hessian(f,x01) # 99sec
fdgradE2, trueE2 = ForwardDiff.gradient(f,x02), f(x02)
fdhessE2 = ForwardDiff.hessian(f,x02)
isposdef(fdhessE1)
eigen(fdhessE1).values[eigen(fdhessE1).values.<0]
norm(x01[1:p*nc]-x02[1:p*nc])
norm(fdgradE1[1:p*nc]-fdgradE2[1:p*nc])
norm(x01[p*nc+1:end]-x02[p*nc+1:end])
norm(fdgradE1[p*nc+1:end]-fdgradE2[p*nc+1:end])
norm(fdhessE1-fdhessE2)


#
(tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,5.0)
tol=1e-6; inner_tol=1e-5
noc = 15; nac = 0; nc = noc+nac
β1 = β2= β; α1 = α2 = α
rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, nc, 0; initmethod=initmethod, svdmethod=:svd)
Xt = copy(X'); W0t = copy(W0'); H0t = copy(H0'); M0t = copy(M0'); N0t = copy(N0'); gtWt = copy(gtH); gtHt = copy(gtW)
M02 = M0.+(rand(size(M0)...).-0.5)*eps(Float64) 
N0t2 = N0t.+(rand(size(N0t)...).-0.5)*eps(Float64) 

maxiter = 1; inner_maxiter=100
σ0=std(W0*M0); r=0.3
alg = LCSVD.LinearCombSVD(eltype(W0), α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false, usedenoiseW0H0=false,
    denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, inner_maxiter = inner_maxiter, store_trace = true,
    store_inner_trace = true, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, inner_tol = inner_tol, successive_f_converge=0)
M1, N1t = copy(M0), copy(N0t)
M2, N2t = copy(M02), copy(N0t2)
rst1 = LCSVD.solve!(alg, X, W0, H0t, D, M1, N1t);
rst2 = LCSVD.solve!(alg, X, W0, H0t, D, M2, N2t);
norm(M1-M2)
norm(N1t-N2t)
x1 = rst1.traces[end].xs[end];
x2 = rst2.traces[end].xs[end];
norm(x1-x2)
for i in 1:length(rst1.traces)
    @show length(rst1.traces[i].xs)
end
