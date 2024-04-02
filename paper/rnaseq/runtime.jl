using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","rnaseq")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :rnaseq
#X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
using FileIO, IncrementalSVD, TSVD
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)
file_name = feature_name*"-raw"
adata = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad"))
Xraw = adata.X # cell_label(adata.obs_names), gene_identifier(adata.var_names)
X = sqrt.(Xraw) # norm(X)=29511.377f0


mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)
f = Figure(size=(120,300))
ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
joint_limits = (-10., 10.)
Xs = Array(X[1:402:100562,1:323:32285]')
hm1 = heatmap!(ax, Xs, colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
hidedecorations!(ax)
Colorbar(f[:, end+1], hm1)
save(joinpath(subworkpath,"$(file_name)_X[250,100].png"),f)

noc = 500
rt = @elapsed U, s = isvd(X,noc) # noc=100, norm(X-U*(U'X)) = 16185.033f0
                                 # noc=500, norm(X-U*(U'X)) = 15374.808f0
                                 # noc=10000, norm(X-U*(U'X)) = 4098.025f0
save(joinpath(subworkpath,"svd","$(file_name)_isvd$noc.jld2"),"U",U,"s",s,"rt",rt)
rt = @elapsed U, s, V = tsvd(X,100) # noc=100, norm(X-U*Diagonal(s)*V') = 16172.704f0
save(joinpath(subworkpath,"$(file_name)_tsvd100.jld2"),"U",U,"s",s,"V",V,"rt",rt)
dd = load(joinpath(subworkpath,"$(file_name)_isvd$noc.jld2"))
U = dd["U"]; s = dd["s"]

rt = @elapsed Ut, st = isvd(X',noc)
Ht = Ut'X'
save(joinpath(subworkpath,"$(file_name)_Xt_isvd$noc.jld2"),"Ut",Ut,"st",st,"Ht",Ht,"rt",rt)
dd = load(joinpath(subworkpath,"$(file_name)_Xt_isvd$noc.jld2"))
Ut = dd["Ut"]; st = dd["st"]; Ht = dd["Ht"]; UtHt = Ut*Ht
aa = norm(X[1:50281,1:16142]-UtHt[1:16142,1:50281]')^2
ab = norm(X[50282:end,1:16142]-UtHt[1:16142,50282:end]')^2
ba = norm(X[1:50281,16143:end]-UtHt[16143:end,1:50281]')^2
bb = norm(X[50282:end,16143:end]-UtHt[16143:end,50282:end]')^2
nrm = sqrt(aa+ab+ba+bb)# noc=500, norm(X'-Ut*(Ut'X'))(this kill julia) = 15418.61f0
norm(X-U*Diagonal(s)*Ut') # 20770.834f0

f = Figure()
ax = AMakie.Axis(f[1,1])#,yscale=log10
lines!(ax,1:noc,s)
save(joinpath(subworkpath,"$(file_name)_isvd$(noc)_rt$(rt).png"),f)#log10_


noc = 500
initisvd(X,noc) = ((U,s)=isvd(X,noc); H = U'*X ; (U, H, copy(U), copy(H))) # H isn't normalized one
(m,n,p) = (size(X)...,noc)
# X = LCSVD.noisefilter(filter,X,imgsz)

lcsvd_maxiter = 100
compnmf_maxiter = 1000
hals_maxiter = 100

# LCSVD
prefix = "lcsvd"
@show prefix; flush(stdout)

mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter; tol=-1 
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased

(tailstr,initmethod,α,β) = ("_sp_nn",:custom,0.005,5.0)# ("_sp_nn",:custom,0.005,5.0) ,("_nn",:nndsvd,0.,5.0)

α1 = α; α2 = 0; β1 = 0; β2= β; 
fname = joinpath(subworkpath,"$(file_name)_isvd_init$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    W0, H0, M0, N0, Wp, Hp, D, rt1 = dd["W0"], dd["H0"], dd["M0"], dd["N0"], dd["Wp"], dd["Hp"], dd["D"], dd["rt1"]
    # noc = 500, norm(X-W0*M0*N0*H0) = 15375.242f0
    fitval = LCSVD.fitd(X,W0*M0*N0*H0) # noc = 500, 0.92099863f0
else
    println("calculating init.")
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, noc; initmethod=initmethod, initfn=initisvd)
    save(fname, "W0",W0,"H0",H0,"Wp",Wp,"Hp",Hp,"M0",M0,"N0",N0,"D",D,"rt1",rt1)
end

σ0=s*std(W0) #=10*std(W0)=#
r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
    # if this is too big iteration number would be increased
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false,
    denoisefilter=:avg, uselv=false, maxiter = maxiter, store_trace = false,
    store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
    f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
M, N = copy(M0), copy(N0)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
W, H = rst0.W, rst0.H
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X,W*H)
fname = joinpath(subworkpath,"$(file_name)_lcsvd_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(rst0.niters).jld2")
save(fname, "W",W,"H",H,"M",M,"N",N,"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)

dd = load(joinpath(subworkpath,"$(file_name)_lcsvd_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(maxiter).jld2"))
W, H, rt1, rt2, fitval = dd["W"], dd["H"], dd["rt1"], dd["rt2"], dd["fitval"]

#colormap
# Makie.available_gradients()
# Plasma, Inferno, Magma, Cividis, Jet, grays, heat, :Spectral

for i in 1:4
    f = Figure()
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    rowsizeq = size(W,1)÷4
    rows = (i==4 ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    joint_limits = (-0.0, 0.05)
    hm1 = heatmap!(ax, W[rows,:]', colormap = :viridis, colorrange = joint_limits) # , reverse_colormap = true
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(file_name)_lcsvd_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(maxiter)_heatmap_W$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,1000000)))
ht1 = hist!(ax,vec(W),bin=5)
save(joinpath(subworkpath,"$(file_name)_lcsvd_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(maxiter)_histo1000000_W.png"),f)
num_blocks = 16
for i in 1:num_blocks
    f = Figure(size=(2200,600))
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (0, 200) # 2470
    hm1 = heatmap!(ax, H[rows,cols]',  colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(file_name)_lcsvd_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(maxiter)_heatmap_H$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
ht1 = hist!(ax,vec(H),bin=5)
save(joinpath(subworkpath,"$(file_name)_lcsvd_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(maxiter)_histo100_H.png"),f)



# HALS
prefix="hals"; @show prefix
fname = joinpath(subworkpath,"$(file_name)_hals_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    Whals0, Hhals0, rt1 = dd["Whals0"], dd["Hhals0"], dd["rt1"]
else
    rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar);
    save(fname, "Whals0",Whals0,"Hhals0",Hhals0,"rt1",rt1)
end
mfmethod = :HALS; αhals=0.1; maxiter = hals_maxiter; tol=-1
W, H = copy(Whals0), copy(Hhals0);
rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, W, H)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X[1:10:end,1:10:end],W[1:10:end,:]*H[:,1:10:end])
fname = joinpath(subworkpath,"hals_WMB-10Xv2-TH-log2_a$(αhals).jld2")
save(fname, "W",W,"H",H,"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)

dd = load(joinpath(subworkpath,"$(file_name)_hals_noc$(noc)_a$(αhals)_iter$(maxiter).jld2"))
W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
for i in 1:4
    f = Figure()
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    rowsizeq = size(W,1)÷4
    rows = (i==4 ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    @show rows
    joint_limits = (-0.0, 0.08)
    hm1 = heatmap!(ax, W[rows,:]',  colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    i == 1 ? nothing : hidexdecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)                     # These three
    # scatter!(ax, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
    save(joinpath(subworkpath,"$(file_name)_hals_noc$(noc)_a$(αhals)_iter$(iter)_heatmap_W$i.png"),f)
end
num_blocks = 16
for i in 1:num_blocks
    f = Figure(size=(2200,600))
    ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    rows = noc:-1:1
    joint_limits = (0, 200) # 1591
    hm1 = heatmap!(ax, H[rows,cols]',  colorrange = joint_limits)
    hideydecorations!(ax, ticks = false)
    Colorbar(f[:, end+1], hm1)                     # These three
    # scatter!(ax, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
    save(joinpath(subworkpath,"$(file_name)_hals_noc$(noc)_a$(αhals)_iter$(iter)_heatmap_H$i.png"),f)
end
f = Figure()
ax = AMakie.Axis(f[1, 1],limits = (nothing,(0,100)))#
ht1 = hist!(ax,vec(H),bin=5)
save(joinpath(subworkpath,"$(file_name)_hals_noc$(noc)_a$(αhals)_histo100_H.png"),f)


# COMPNMF
prefix = "compnmf"
@show prefix; flush(stdout)
mfmethod = :COMPNMF; maxiter = compnmf_maxiter
(tailstr,initmethod) = ("_nn",:lowrank_nndsvd)
dd = Dict(); tol=-1
Wcn0, Hcn0, rt1 = dd["Whals0"], dd["Hhals0"], dd["rt1"]
W, H = copy(Wcn0), copy(Hcn0);
rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{eltype(X)}(maxiter=maxiter, tol=tol, verbose=false), X, W, H)
rt1 += rst0.inittime # add calculation time for compression matrices L and R
rt2 -= rst0.inittime
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X[1:10:end,1:10:end],W[1:10:end,:]*H[:,1:10:end])
fname = joinpath(subworkpath,"compnmf_WMB-10Xv2-TH-log2.jld2")
save(fname, "W",W,"H",H,"rt1",rt1,"rt2",rt2,"iter",maxiter,"fitval",fitval)

# SPCA
prefix = "spca"
@show prefix; flush(stdout)
makepositive = true
α = 0.5; ridge_alpha=0.01; max_iter=500; tol=1e-7
rtspca = @elapsed resultspca = fit_transform!(SparsePCA(n_components=noc,alpha=α,ridge_alpha=ridge_alpha,max_iter=max_iter,tol=tol,verbose=true),X) 
W = copy(resultspca); H = W\X
LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
nodr = LCSVD.matchedorder(ml,noc); Wspca, Hspca = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
makepositive && LCSVD.flip2makepos!(Wspca,Hspca,mask=:topNpix)
fprex = "$(prefix)$(SNR)db$(inhibitindices)_bias$(bias)_$(sbgstr)"
fname = joinpath(subworkpath,"sp_meanT","$(fprex)_f$(fitval)_it$(max_iter)_rt$(rtspca)")
imsave_data(dataset,fname,Wspca,Hspca,imgsz,100; saveH=false)
plotH_data(fname*"_Hinhibit",Hspca[inhibitindices,:]; space=0.,ylabel="",ytickformat="{:.2f}")


# result for 1 inhibit cell
colorindices=[1,2,7,5,4]; ftsize1 = 30; ftsize2=30; ftsize3=30; linewidth1 = 1.5
lstyles = [:dash,nothing,nothing,nothing,:dash]
imggt = TestData.mkimgW(gtW,imgsz); imglc = TestData.mkimgW(Wlc,imgsz);
imgcn = TestData.mkimgW(Wcn,imgsz,scalemtd=:maxcol); imghals = TestData.mkimgW(Whals,imgsz,scalemtd=:maxcol);
imgspca = TestData.mkimgW(Wspca,imgsz)
# scainhibitindices = (bias == 0.5) && (subtract_bg == false) ? 8 : inhibitindices
hdata = [gtH[:,inhibitindices[1]],Hlc[inhibitindices[1],:],Hcn[inhibitindices[1],:],Hhals[inhibitindices[1],:],Hspca[inhibitindices[1],:]] # Hlc inhibit index setting for plot
labels = ["Ground Truth","PCB-S","Compressed NMF","HALS NMF","SPCA"]
f = Figure(resolution = (1000,400))
ax11=AMakie.Axis(f[1,1],title=labels[2], aspect = DataAspect()); hidedecorations!(ax11)
ax21=AMakie.Axis(f[2,1],title=labels[3], aspect = DataAspect()); hidedecorations!(ax21)
ax31=AMakie.Axis(f[3,1],title=labels[4], aspect = DataAspect()); hidedecorations!(ax31)
ax41=AMakie.Axis(f[4,1],title=labels[5], aspect = DataAspect()); hidedecorations!(ax41)
axall2=AMakie.Axis(f[:,2],title="Inhibited H component",xlabel="time index")
image!(ax11, rotr90(imglc)); image!(ax21, rotr90(imgcn)); image!(ax31, rotr90(imghals)); image!(ax41, rotr90(imgspca))
lin = [lines!(axall2,hd,linewidth=linewidth1,color=mtdcolors[colorindices[i]],linestyle=lstyles[i]) for (i,hd) in enumerate(hdata)]
f[:,3] = Legend(f[:,2],lin,labels)
save(joinpath(subworkpath,"sp_meanT","idx$(inhibitindices[1])_bias$(bias)_$(sbgstr).png"),f)

A = zeros(100,10)
A[ 1:10,1] = rand(2:5,10)
A[11:20,2] = rand(2:5,10)
A[21:30,3] = rand(2:5,10)
A[31:40,4] = rand(2:5,10)
A[41:50,5] = rand(2:5,10)
A[51:60,6] = rand(2:5,10)
A[61:70,7] = rand(2:5,10)
A[71:80,8] = rand(2:5,10)
A[81:90,9] = rand(2:5,10)
A[91:100,10] = rand(2:5,10)
f = Figure()
ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
joint_limits = (0, 5)
hm1 = heatmap!(ax, A[100:-1:1,:]',  colorrange = joint_limits)
hideydecorations!(ax, ticks = false)
Colorbar(f[:, end+1], hm1)                     # These three
# scatter!(ax, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
#ax.yaxis.attributes.flipped = true
save(joinpath(subworkpath,"test_heatmap.png"),f)


diff(X,WH) = norm(X-WH)/norm(X)*100

noc = 1000
fname = joinpath(subworkpath,"svd","$(file_name)_isvd$(noc).jld2")
dd = load(fname)
U, s, rt1 = dd["U"], dd["s"], dd["rt"]
rt11 = @elapsed begin
    W0 = U; Wp = copy(W0)
    H0 = W0'X; Hp = copy(H0)
end
W0H0 = W0*H0
rt12 = @elapsed begin
    LCSVD.balanceWH!(Wp,Hp)
    d = LCSVD.normalizeWH!(W0,H0)
    D = Diagonal(d)
    M0, N0 = (W0\Wp, Hp/H0)
end
rt1 += rt11 + rt12
fname = joinpath(subworkpath,"$(file_name)_isvd_init$(noc).jld2")
save(fname,"W0", W0, "H0", H0, "M0", M0, "N0", N0, "Wp", Wp, "Hp", Hp, "D", D, "rt1", rt1)

dd = load(fname)
W0, H0, M0, N0, Wp, Hp, D, rt1 = dd["W0"], dd["H0"], dd["M0"], dd["N0"], dd["Wp"], dd["Hp"], dd["D"], dd["rt1"]
WMNH0 = W0*M0*N0*H0
f = LCSVD.fitd(X,WMNH0)
d = diff(X,W0H0)
@show noc, d, f
