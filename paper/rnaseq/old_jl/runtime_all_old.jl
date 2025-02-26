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
using FileIO

#colormap
# Makie.available_gradients()
# Plasma, Inferno, Magma, Cividis, Jet, grays, heat, :Spectral
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)

#======== Load Data ===========#
dataset = :rnaseq
#X, _ = load_data(dataset; feature_name="WMB-10Xv2-TH");
feature_name = "WMB-10Xv2-HY" # WMB-10Xv2-HY(100562×32285)
file_name = feature_name*"-raw"
adata = load(joinpath(datapath,"AllenBrain","expression_matrices","WMB-10Xv2","20230630","$(file_name).h5ad"))
Xraw = adata.X # cell_label(adata.obs_names), gene_identifier(adata.var_names)

#========= Matrix Factorization methods ===========#
noc = 500; nac = 0; nc = noc+nac

sizestep = 5
X = sqrt.(Xraw[1:sizestep:end,1:sizestep:end])'; Xtpstr = "t"

# LCSVD
prefix = "pcb"
@show prefix; flush(stdout)

(tailstr,initmethod,α,β) = ("_sp_nn",:isvd,0.005,5.0)# ("_sp_nn",:custom,0.005,5.0) ,("_nn",:nndsvd,0.,5.0)

initmethod = :isvd; initmtdstr="init$(initmethod)"
fname = joinpath(subworkpath,"$(file_name)$(Xtpstr)_ss$(sizestep)_$(initmtdstr)_nc$(nc).jld2")
if isfile(fname)
    println("reading init.")
    dd = load(fname)
    W0, H0t, M0, N0t, D, rt1 = dd["W0"], dd["H0t"], dd["M0"], dd["N0t"], dd["D"], dd["rt1"]
    # noc = 500, norm(X-W0*M0*N0*H0) = 15375.242f0
    # fitval = LCSVD.fitd(X,W0*M0*N0*H0) # noc = 500, 0.92099863f0
else
    println("calculating init.")
    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, nc; initmethod=initmethod)
    save(fname, "W0",W0,"H0t",H0',"Wp",Wp,"Hpt",Hp',"M0",M0,"N0t",N0',"D",D,"rt1",rt1)
end

α1 = α; α2 = α; β1 = β; β2= β; 
r = 0.3
maxiter = Int(ceil(log(eps(eltype(X)))/log(r)))
tol = 1e-6; inner_tol = 1e-6; inner_maxiter = 1000
alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, r=r, useprecond=false, uselv=false,
    maxiter = maxiter, inner_maxiter = inner_maxiter, inner_tol = inner_tol,
    store_trace = false, store_inner_trace = false, show_trace = true, allow_f_increases = true,
    f_abstol=tol, f_reltol=tol, f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
M, Nt = copy(M0), copy(N0t)
rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0t, D, M, Nt);
W, H = rst0.W, rst0.Ht'
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X,W*H)
jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_pcb_noc$(noc)_aw$(α1)_ah$(α2)_bw$(β1)_bh$(β2)_it$(rst0.niters)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "W",W,"H",H,"M",M,"N",Nt',"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)

dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
# W histogram
xlimit = round(quantile(vec(W),0.9999),sigdigits=3) # cf)  round(123456,digits=-2) = 123500.0
# for ylimits in [nothing, (0,*(size(W)...)÷1000)]
#     f = Figure()
#     ax = AMakie.Axis(f[1, 1],limits = (nothing,ylimits))#(0,ylimit)))
#     ht1 = hist!(ax,vec(W),bins=100)
#     ylimstr = ylimits === nothing ? "" : "$(ylimits[2])"
#     save(joinpath(subworkpath,"$(jldfprex)_histo$(ylimstr)_qt$(xlimit)_W.png"),f)#$(ylimit)_W.png"),f)
# end
# W heatmap
y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50
for i in 1:min(1,num_blocks)
    f = Figure(size=(xsize,ysize)) # 
    rowsizeq = size(W,1)÷num_blocks
    rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    ax = AMakie.Axis(f[1, 1],width=noc,height=length(rows))#,xaxisposition=:top
    joint_limits = (-xlimit, xlimit) # 0.08(), 0.1(small, 0.15)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax) # , ticks = false
    Colorbar(f[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprex)_heatmap_W$i.png"),f)
end
# H histogram
xlimit = round(quantile(vec(H),0.9999),sigdigits=3)
# for ylimits in [nothing, (0,*(size(H)...)÷1000)]
#     f = Figure()
#     ax = AMakie.Axis(f[1, 1],limits = (nothing,ylimits))
#     ht1 = hist!(ax,vec(H),bins=100)
#     ylimstr = ylimits === nothing ? "" : "$(ylimits[2])"
#     save(joinpath(subworkpath,"$(jldfprex)_histo$(ylimstr)_qt$(xlimit)_H.png"),f)
# end
# H heatmap
y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
for i in 1:min(1,num_blocks)
    f = Figure(size=(xsize,ysize)) # (2200,600), small(130,1100)
    rows = noc:-1:1
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    ax = AMakie.Axis(f[1, 1],width=length(cols),height=noc)
    joint_limits = (-xlimit, xlimit) # 200(1591), 30(small 171)
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprex)_heatmap_H$i.png"),f)
end


# HALS
prefix="hals"; @show prefix
hals_maxiter = 50

fname = joinpath(subworkpath,"$(file_name)$(Xtpstr)_ss$(sizestep)_initnndsvd_noc$(noc).jld2")
if isfile(fname)
    dd = load(fname)
    Whals0, Hhals0, rt1 = dd["Whals0"], dd["Hhals0"], dd["rt1"]
else
    @show "calculating nndsvd..."
    rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, noc, variant=:ar);
    save(fname, "Whals0",Whals0,"Hhals0",Hhals0,"rt1",rt1)
end
mfmethod = :HALS; αhals=0.1; maxiter = hals_maxiter; tol=-1
W, H = copy(Whals0), copy(Hhals0);
rt2 = @elapsed rst0 = NMF.solve!(NMF.CoordinateDescent{eltype(X)}(maxiter=maxiter, α=αhals, l₁ratio=1,
                tol=tol, verbose=false), X, W, H)
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
LCSVD.normalizeW!(W,H); fitval = LCSVD.fitd(X[1:10:end,1:10:end],W[1:10:end,:]*H[:,1:10:end])
jldfprex = "$(file_name)$(Xtpstr)_ss$(sizestep)_hals_noc$(noc)_a$(αhals)_mit$(maxiter)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "W",W,"H",H,"rt1",rt1,"rt2",rt2,"iter",rst0.niters,"fitval",fitval)

# jldfprex = "WMB-10Xv2-HY-raw_small25141_hals_noc200_a0.1_mit100"
dd = load(joinpath(subworkpath,"$(jldfprex).jld2"))
W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
# W histogram
xlimit = round(quantile(vec(W),0.99999),sigdigits=3)
# for ylimits in [nothing, (0,*(size(W)...)÷1000)]
#     f = Figure()
#     ax = AMakie.Axis(f[1, 1],limits = (nothing,ylimits))#(0,ylimit)))
#     ht1 = hist!(ax,vec(W),bins=100)
#     ylimstr = ylimits === nothing ? "" : "$(ylimits[2])"
#     save(joinpath(subworkpath,"$(jldfprex)_histo$(ylimstr)_qt$(xlimit)_W.png"),f)#$(ylimit)_W.png"),f)
# end
# W heatmap
y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50
for i in 1:min(1,num_blocks)
    f = Figure(size=(xsize,ysize)) # 
    rowsizeq = size(W,1)÷num_blocks
    rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    ax = AMakie.Axis(f[1, 1],width=noc,height=length(rows))#,xaxisposition=:top
    joint_limits = (-xlimit, xlimit) # 0.08(), 0.1(small, 0.15)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax) # , ticks = false
    Colorbar(f[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprex)_W$i.png"),f)
end
# H histogram
xlimit = round(quantile(vec(H),0.9999),sigdigits=3)
# for ylimits in [nothing, (0,*(size(H)...)÷1000)]
#     f = Figure()
#     ax = AMakie.Axis(f[1, 1],limits = (nothing,ylimits))
#     ht1 = hist!(ax,vec(H),bins=100)
#     ylimstr = ylimits === nothing ? "" : "$(ylimits[2])"
#     save(joinpath(subworkpath,"$(jldfprex)_histo$(ylimstr)_qt$(xlimit)_H.png"),f)
# end
# H heatmap
y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
for i in 1:min(1,num_blocks)
    f = Figure(size=(xsize,ysize)) # (2200,600), small(130,1100)
    rows = noc:-1:1
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    ax = AMakie.Axis(f[1, 1],width=length(cols),height=noc)
    joint_limits = (-xlimit, xlimit) # 200(1591), 30(small 171)
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)                     # These three
    save(joinpath(subworkpath,"$(jldfprex)_H$i.png"),f)
end


# COMPNMF
prefix = "compnmf"
compnmf_maxiter = 1000

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


jldfprex1 = "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it11"
jldfprex2 = "WMB-10Xv2-HY-rawt_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it12"
jldfprex3 = "WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it12"
fname1 = joinpath(subworkpath,"$(jldfprex1).jld2")
fname2 = joinpath(subworkpath,"$(jldfprex2).jld2")
fname3 = joinpath(subworkpath,"$(jldfprex3).jld2")
dd1 = load(fname1)
dd2 = load(fname2)
dd3 = load(fname3)

W2 = dd2["W"]; H2t = dd2["H"]'
W3 = dd3["W"]; H3t = dd3["H"]'
uw = W2\H3t; uh = W3'/H2t'; uwtuwIdiff = norm(uw'uw-I); uhuhtIdiff = norm(uh*uh'-I)
M2 = dd2["M"]; N2t = dd2["N"]'
M3 = dd3["M"]; N3t = dd3["N"]'
um = M2\N3t; un = M3'/N2t'; umtumIdiff = norm(um'um-I); ununtIdiff = norm(un*un'-I)



# initialization
noc = 500; nac=2500
file_name = "WMB-10Xv2-HY"*"t-raw"
initmethod = :isvd; initmtdstr="init$(initmethod)"
fname = joinpath(subworkpath,"$(file_name)_$(initmtdstr)_noc$(noc)_nac$(nac)_X.jld2")
dd = load(fname)
W0, H0t, M0, N0t, D, rt1 = copy(dd["H0"]'), copy(dd["W0"]), copy(dd["N0"]'), copy(dd["M0"]), copy(dd["D"]'), dd["rt1"]
M2 = M0; N2t = N0t
M3 = dd["M0"]; N3t = dd["N0"]'
um = M2\N3t; un = M3'/N2t'; umtumIdiff = norm(um'um-I); ununtIdiff = norm(un*un'-I)

# redraw figures
jldfprexs = [#"WMB-10Xv2-HY-raw_hals_noc500_a0.1_mit50",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it12",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.0_bw0.0_bh5.0_tol1.0e-6_it14",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.015_ah0.015_bw5.0_bh5.0_tol1.0e-6_it9",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.015_ah0.0_bw0.0_bh5.0_tol1.0e-6_it14",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.03_ah0.03_bw5.0_bh5.0_tol1.0e-6_it11",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.03_ah0.0_bw0.0_bh5.0_tol1.0e-6_it14",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw50.0_bh50.0_tol1.0e-6_it9",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.0_bw0.0_bh50.0_tol1.0e-6_it14",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw200.0_bh200.0_tol1.0e-6_it13",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.0_bw0.0_bh200.0_tol1.0e-6_it14",
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.0_bw0.0_bh500.0_tol1.0e-6_it14",
"WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw500.0_bh500.0_tol1.0e-6_it14"
# "WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it11",
# "WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it10",
# "WMB-10Xv2-HY-rawt_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it10"
]

jldfprexs = ["general/WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it11",
"general/WMB-10Xv2-HY-rawt_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it12",
"general/WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it12"
]

for jldfprex in jldfprexs
    @show jldfprex
fname = joinpath(subworkpath,"$(jldfprex).jld2")
dd = load(fname)
# W = load(fname,"W")
# H = load(fname,"H")
W, H, iter, rt1, rt2 = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"]
fitval = dd["fitval"]
#fitval = LCSVD.fitd(X,W*H)
LCSVD.normalizeWH!(W,H)
sw = norm(W,1)
sh = norm(H,1)

# W heatmap
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)
y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50; noc=x
xlimit = round(quantile(vec(W),0.9999),sigdigits=3) # cf)  round(123456,digits=-2) = 123500.0
for i in 1:min(2,num_blocks) # som many blocks, so now just plot the first twos.
    f = Figure(size=(xsize,ysize))
    rowsizeq = size(W,1)÷num_blocks
    rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    ax = CairoMakie.Axis(f[1, 1],width=noc,height=length(rows))
    joint_limits = (-xlimit, xlimit)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(jldfprex)_W$(i)_sw$(sw).png"),f)
end
# H heatmap
y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
xlimit = round(quantile(vec(H),0.9999),sigdigits=3)
for i in 1:min(2,num_blocks)
    f = Figure(size=(xsize,ysize))
    rows = noc:-1:1
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    ax = CairoMakie.Axis(f[1, 1],width=length(cols),height=noc)
    joint_limits = (-xlimit, xlimit) 
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(jldfprex)_H$(i)_sh$(sh)_fv$(fitval).png"),f)
end
end


file_name = "WMB-10Xv2-HYt-raw"; noc=500; nac=2500
initmethod = :isvd; initmtdstr="init$(initmethod)"
fname = joinpath(subworkpath,"$(file_name)_$(initmtdstr)_noc$(noc)_nac$(nac)_X.jld2")
dd = load(fname)
W01, H0t1 = copy(dd["W0"]), copy(dd["H0"]')
W02, H0t2 = copy(dd["H0"]'), copy(dd["W0"])

jldfprex1 = "WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it10"
jfname = joinpath(subworkpath,"$(jldfprex1).jld2")
dd1 = load(fname)
W1, H1, M1, N1, fitval1 = dd1["W"], dd1["H"], dd1["M"], dd1["N"], dd1["fitval"]
LCSVD.normalizeWH!(W1,H1)
ldfprex2 = "WMB-10Xv2-HY-rawt_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it10"
fname = joinpath(subworkpath,"$(jldfprex2).jld2")
dd2 = load(fname)
W2, H2, M2, N2, fitval2 = dd2["W"], dd2["H"], dd2["M"], dd2["N"], dd2["fitval"]
LCSVD.normalizeWH!(W2,H2)

W11 = W01*M1; H11 = N1*H0t1'
W22 = W02*M2; H22 = N2*H0t2'

W, H = copy(H11'), copy(W11')
LCSVD.normalizeWH!(W,H)
jldfprex = "WMB-10Xv2-HYt-raw_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it10t"


W, H = copy(W22), copy(H22)
LCSVD.normalizeWH!(W,H)
jldfprex = "WMB-10Xv2-HY-rawt_initisvd_pcb_noc500_nac2500_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it10test"

jldfprexs = ["WMB-10Xv2-HY-raw_n100562_initnndrsvd_hals_noc500_a0.1_tol0.003_mit100_nrrd",
"WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw0.015_ah0.0_bw0.0_bh5.0_tol1.0e-6_it2"]





#=
X = sqrt.(Xraw) # norm(X)=29511.377f0

X = Array(X[1:4:100562,1:3:32285]); sz = size(X,1); fnsmall = file_name*"_small$sz"

xlimit = round(quantile(vec(X),0.9999),sigdigits=3) # cf)  round(123456,digits=-2) = 123500.0
f = Figure(size=(120,300))
ax = AMakie.Axis(f[1, 1],xaxisposition=:top)
joint_limits = (-xlimit, xlimit)
Xs = Array(xqrt.(Xraw)[1:402:100562,1:323:32285])
hm1 = heatmap!(ax, Xs', colormap = mycmap, colorrange = joint_limits) # , reverse_colormap = true
hidedecorations!(ax)
save(joinpath(subworkpath,"$(fnsmall)_X[250,100].png"),f)

#======== SVD methods comparison ========#
noc = 500
rt = @elapsed Usvd, ssvd, Vsvd = svd(X) # noc=100, norm(X-U*(U'X)) = 16185.033f0
save(joinpath(subworkpath,"svd","$(fnsmall)_svd$noc.jld2"),"U",Usvd,"s",ssvd,"V",Vsvd,"rt",rt)
noc = 9000
rt = @elapsed Uisvd, sisvd = isvd(X,noc) # noc=100, norm(X-U*(U'X)) = 16185.033f0
                                 # noc=500, norm(X-U*(U'X)) = 15374.808f0
                                 # noc=10000, norm(X-U*(U'X)) = 4098.025f0
save(joinpath(subworkpath,"svd","$(fnsmall)_isvd$noc.jld2"),"U",Uisvd,"s",sisvd,"rt",rt)
rt = @elapsed Utsvd, stsvd, Vtsvd = tsvd(X,noc) # noc=100, norm(X-U*Diagonal(s)*V') = 16172.704f0, noc=500 fail
save(joinpath(subworkpath,"svd","$(fnsmall)_tsvd$noc.jld2"),"U",Utsvd,"s",stsvd,"V",Vtsvd,"rt",rt)
f = Figure()
ax = AMakie.Axis(f[1,1],yscale=log10)
lines!(ax,ssvd,label="SVD")
lines!(ax,sisvd,label="ISVD")
axislegend(ax, position = :rt)
save(joinpath(subworkpath,"svd","$(fnsmall)_s.png"),f)

dd = load(joinpath(subworkpath,"svd","$(fnsmall)_isvd$noc.jld2"))
U = dd["U"]; s = dd["s"]

rt = @elapsed Ut, st = isvd(X',noc)
Ht = Ut'X'
save(joinpath(subworkpath,"svd","$(fnsmall)_Xt_isvd$noc.jld2"),"Ut",Ut,"st",st,"Ht",Ht,"rt",rt)
dd = load(joinpath(subworkpath,"svd","$(fnsmall)_Xt_isvd$noc.jld2"))
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
save(joinpath(subworkpath,"$(fnsmall)_isvd$(noc)_rt$(rt).png"),f)#log10_
=#
