using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","inhibit")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

using ScikitLearn
@sk_import decomposition: SparsePCA

dataset = :fakecells; SNR=0; inhibitindices=[1,2,3,4]; bias=0.1
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

lcsvd_maxiter = 150
compnmf_maxiter = 1000
hals_maxiter = 150

imgsz0 = (40,20); factor = 1
sqfactor = Int(floor(sqrt(factor)))
for bias in [0.1,0.5]
    @show bias; flush(stdout)
    imgsz = (sqfactor*imgsz0[1],sqfactor*imgsz0[2]); lengthT = factor*1000; sigma = sqfactor*5.0
    X, imgsz, lengthT, ncs, gtncells, datadic = load_data(dataset; dpath=subworkpath, sigma=sigma, imgsz=imgsz,
            lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true, inhibitindices=inhibitindices, issave=true,
            isload=true, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
    ncells = ncs
    (m,n,p) = (size(X)...,ncells)
    gtW, gtH = dataset == :fakecells ? (datadic["gtW"], datadic["gtH"]) : (Matrix{eltype(X)}(undef,0,0),Matrix{eltype(X)}(undef,0,0))
    gtfname = "fakecells$(inhibitindices)_calcium_sz$(imgsz)_lengthT$(lengthT)_SNR$(SNR)_bias$(bias)"
    imsave_data(dataset,joinpath(subworkpath,gtfname),gtW,gtH',imgsz,100; saveH=false)
    plotH_data(joinpath(subworkpath,gtfname),gtH'; space=0.,ylabel="",ytickformat="{:.2f}")
    X = LCSVD.noisefilter(filter,X)

    for subtract_bg in [false, true]
        sbgstr = subtract_bg ? "sbg" : "nosbg"
        if subtract_bg
            rt1cd = @elapsed W, H = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
            NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, W, H)
            LCSVD.normalizeW!(W,H); imsave_data(dataset,joinpath(subworkpath,"Wr1"),W,H,imgsz,100; signedcolors=TestData.dgwm(), saveH=false)
            plotH_data(joinpath(subworkpath,"Hr1_gtH"),H)
            bg = W*fill(mean(H),1,n); X .-= bg
        end

        # LCSVD
        prefix = "lcsvd"
        @show prefix; flush(stdout)

        mfmethod = :LCSVD; useprecond=false; uselv=false; s=10; maxiter = lcsvd_maxiter; tol=-1 
        r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
            # if this is too big iteration number would be increased
        makepositive = true

        (tailstr,initmethod,α,β) = ("_sp",:isvd,0.005,0.0)# ("_sp",:isvd,0.005,.0) ,("_nn",:nndsvd,0.,5.0)

        β1 = β2= β; α1 = α2 = α
        rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, ncells; initmethod=initmethod, svdmethod=:isvd)
        σ0=s*std(W0) #=10*std(W0)=#
        r=(0.3)^1 #0.3 # decaying rate for relaxed L1, if this is too small result is very sensitive for setting α
            # if this is too big iteration number would be increased
        alg = LCSVD.LinearCombSVD(α1=α1, α2=α2, β1=β1, β2=β2, σ0=σ0, r=r, useprecond=false,
            denoisefilter=:avg, uselv=false, imgsz=imgsz, maxiter = maxiter, store_trace = false,
            store_inner_trace = false, show_trace = false, allow_f_increases = true, f_abstol=tol, f_reltol=tol,
            f_inctol=1e2, x_abstol=tol, x_reltol=tol, successive_f_converge=0)
        M, N = copy(M0), copy(N0)
        rt2 = @elapsed rst0 = LCSVD.solve!(alg, X, W0, H0, D, M, N);
        W, H = rst0.W, rst0.H
        # avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
        LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
        nodr = LCSVD.matchedorder(ml,ncells); Wlc, Hlc = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(W1,H1)
        makepositive && LCSVD.flip2makepos!(Wlc,Hlc,mask=:topNpix)
        fprex = "$(prefix)$(SNR)db$(inhibitindices)_bias$(bias)_$(sbgstr)"
        fname = joinpath(subworkpath,"$(fprex)_a$(α)_b$(β)_f$(fitval)_it$(rst0.niters)_rt$(rt2)")
        imsave_data(dataset,fname,Wlc,Hlc,imgsz,100; saveH=false)
        plotH_data(fname*"_Hinhibit",Hlc[inhibitindices,:]; space=0.,ylabel="",ytickformat="{:.2f}")
        #continue

        # HALS
        prefix="hals"; @show prefix
        # W0, H0, Mw0, Mh0, Wp, Hp, D = initsemisca(X, ncells, initmethod=:isvd,poweradjust=:wh_normalize) # for penmetric = :SCA
        rt1 = @elapsed Whals0, Hhals0 = NMF.nndsvd(X, ncells, variant=:ar);
        mfmethod = :HALS; αhals=0.1; maxiter = hals_maxiter; tol=-1
        W, H = copy(Whals0), copy(Hhals0);
        rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=αhals, l₁ratio=1,
                        tol=tol, verbose=false), X, W, H)
        # avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, Whals, Hhals; clamp=false)
        LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
        nodr = LCSVD.matchedorder(ml,ncells); Whals, Hhals = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
        makepositive && LCSVD.flip2makepos!(Whals,Hhals)
        fprex = "$(prefix)$(SNR)db$(inhibitindices)_bias$(bias)_$(sbgstr)"
        fname = joinpath(subworkpath,"$(fprex)_a$(αhals)_f$(fitval)_it$(maxiter)_rt$(rt2)")
        imsave_data(dataset,fname,Whals,Hhals,imgsz,100; saveH=false, scalemtd=:maxcol)
        plotH_data(fname*"_Hinhibit",Hhals[inhibitindices,:]; space=0.,ylabel="",ytickformat="{:.2f}")

        # COMPNMF
        prefix = "compnmf"
        @show prefix; flush(stdout)
        mfmethod = :COMPNMF; maxiter = compnmf_maxiter
        (tailstr,initmethod) = ("_nn",:lowrank_nndsvd)
        dd = Dict(); tol=-1
        rt1 = @elapsed Wcn0, Hcn0 = NMF.nndsvd(X, ncells, variant=:ar);
        Wcn, Hcn = copy(Wcn0), copy(Hcn0);
        result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=true), X, Wcn, Hcn)
        W, H = copy(Wcn0), copy(Hcn0);
        rt2 = @elapsed rst0 = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=maxiter, tol=tol, verbose=false), X, W, H)
        rt1 += rst0.inittime # add calculation time for compression matrices L and R
        rt2 -= rst0.inittime
        LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
        nodr = LCSVD.matchedorder(ml,ncells); Wcn, Hcn = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
        makepositive && LCSVD.flip2makepos!(Wcn,Hcn)
        fprex = "$(prefix)$(SNR)db$(inhibitindices)_bias$(bias)_$(sbgstr)"
        fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_it$(rst0.niters)_rt$(rt2)")
        imsave_data(dataset,fname,Wcn,Hcn,imgsz,100; saveH=false, scalemtd=:maxcol)
        plotH_data(fname*"_Hinhibit",Hcn[inhibitindices,:]; space=0.,ylabel="",ytickformat="{:.2f}")

        # SPCA
        prefix = "spca"
        @show prefix; flush(stdout)
        makepositive = true
        α = 0.5; ridge_alpha=0.01; max_iter=500; tol=1e-7
        rtspca = @elapsed resultspca = fit_transform!(SparsePCA(n_components=ncells,alpha=α,ridge_alpha=ridge_alpha,max_iter=max_iter,tol=tol,verbose=true),X) 
        W = copy(resultspca); H = W\X
        LCSVD.normalizeW!(W,H); avgnssda, ml, nssdas = LCSVD.matchedWnssda(gtW, W); fitval = LCSVD.fitd(X,W*H)
        nodr = LCSVD.matchedorder(ml,ncells); Wspca, Hspca = W[:,nodr], H[nodr,:]; # W3,H3 = sortWHslices(Whals,Hhals)
        makepositive && LCSVD.flip2makepos!(Wspca,Hspca,mask=:topNpix)
        fprex = "$(prefix)$(SNR)db$(inhibitindices)_bias$(bias)_$(sbgstr)"
        fname = joinpath(subworkpath,"$(fprex)_f$(fitval)_it$(max_iter)_rt$(rtspca)")
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
        labels = ["Ground Truth","LCSVD","Compressed NMF","HALS NMF","SPCA"]
        f = Figure(resolution = (1000,400))
        ax11=AMakie.Axis(f[1,1],title=labels[2], aspect = DataAspect()); hidedecorations!(ax11)
        ax21=AMakie.Axis(f[2,1],title=labels[3], aspect = DataAspect()); hidedecorations!(ax21)
        ax31=AMakie.Axis(f[3,1],title=labels[4], aspect = DataAspect()); hidedecorations!(ax31)
        ax41=AMakie.Axis(f[4,1],title=labels[5], aspect = DataAspect()); hidedecorations!(ax41)
        axall2=AMakie.Axis(f[:,2],title="Inhibited H component",xlabel="time index")
        image!(ax11, rotr90(imglc)); image!(ax21, rotr90(imgcn)); image!(ax31, rotr90(imghals)); image!(ax41, rotr90(imgspca))
        lin = [lines!(axall2,hd,linewidth=linewidth1,color=mtdcolors[colorindices[i]],linestyle=lstyles[i]) for (i,hd) in enumerate(hdata)]
        f[:,3] = Legend(f[:,2],lin,labels)
        save(joinpath(subworkpath,"idx$(inhibitindices[1])_bias$(bias)_$(sbgstr).png"),f)

        # result for 2 inhibit cells
        if length(inhibitindices) > 1
            hindices = [1,4]; ahindices = setdiff(collect(1:7),inhibitindices)
            # imggt = TestData.mkimgW(gtW,imgsz); imglc = TestData.mkimgW(Wlc,imgsz)
            # imgcn = TestData.mkimgW(Wcn,imgsz); imghals = TestData.mkimgW(Whals,imgsz)
            # scainhibitindices = (bias == 0.5) && (subtract_bg == false) ? 8 : inhibitindices
            hdata1 = #=subtract_bg=# false ? [gtH[:,inhibitindices[hindices[1]]],Hlc[inhibitindices[hindices[1]],:]] :
                    [gtH[:,inhibitindices[hindices[1]]],Hlc[inhibitindices[hindices[1]],:],Hcn[inhibitindices[hindices[1]],:],
                    Hhals[inhibitindices[hindices[1]],:],Hspca[inhibitindices[hindices[1]],:]] # Hlc inhibit index setting for plot
            hdata2 = #=subtract_bg=# false ? [gtH[:,inhibitindices[hindices[2]]],Hlc[inhibitindicse[hindices[2]],:]] :
                    [gtH[:,inhibitindices[hindices[2]]],Hlc[inhibitindices[hindices[2]],:],Hcn[inhibitindices[hindices[2]],:],
                    Hhals[inhibitindices[hindices[2]],:],Hspca[inhibitindices[hindices[2]],:]] # Hlc inhibit index setting for plot
            ahdata1 = #=subtract_bg=# false ? [gtH[:,ahindices[1]],Hlc[ahindices[1],:]] :
                    [gtH[:,ahindices[1]],Hlc[ahindices[1],:],Hcn[ahindices[1],:],Hhals[ahindices[1],:],Hspca[ahindices[1],:]] # Hlc inhibit index setting for plot
            labels = ["Ground Truth","LCSVD","Compressed NMF","HALS NMF","SPCA"]
            f = Figure(resolution = (2000,800))
            ax11=AMakie.Axis(f[1,1],title=labels[2], titlesize = ftsize1, aspect = DataAspect()); hidedecorations!(ax11)
            ax21=AMakie.Axis(f[2,1],title=labels[3], titlesize = ftsize1, aspect = DataAspect()); hidedecorations!(ax21)
            ax31=AMakie.Axis(f[3,1],title=labels[4], titlesize = ftsize1, aspect = DataAspect()); hidedecorations!(ax31)
            ax41=AMakie.Axis(f[4,1],title=labels[5], titlesize = ftsize1, aspect = DataAspect()); hidedecorations!(ax41)
            image!(ax11, rotr90(imglc)); image!(ax21, rotr90(imgcn)); image!(ax31, rotr90(imghals)); image!(ax41, rotr90(imgspca))
            ax122=AMakie.Axis(f[1:2,2],title="Inhibited H components (cell $(inhibitindices[hindices[1]]))", titlesize = ftsize2,
                    yticklabelsize=ftsize3)
            hidexdecorations!(ax122, grid = false, ticks=false)
            ax342=AMakie.Axis(f[3:4,2],title="Activated H components (cell $(ahindices[1]))",xlabel="time index", titlesize = ftsize2,
                    xlabelsize=ftsize3, xticklabelsize=ftsize3, yticklabelsize=ftsize3)
            linkxaxes!(ax122,ax342)
            lin1 = [lines!(ax122,hd,color=mtdcolors[colorindices[i]],linewidth=linewidth1,linestyle=lstyles[i]) for (i,hd) in enumerate(hdata1)]
            lin2 = [lines!(ax342,hd,color=mtdcolors[colorindices[i]],linewidth=linewidth1,linestyle=lstyles[i]) for (i,hd) in enumerate(ahdata1)]
            f[:,3] = Legend(f[:,2],lin1,labels,labelsize = ftsize2)
            save(joinpath(subworkpath,"idx$(inhibitindices)_bias$(bias)_$(sbgstr).png"),f)
        end
    end # subtract_bg
    # Input data
    imggt = TestData.mkimgW(gtW,imgsz)
    hdata = eachcol(gtH)
    labels = ["cell $i" for i in 1:length(hdata)]
    f = Figure(resolution = (900,300))
    ax11=AMakie.Axis(f[1,1],title="W component", aspect = DataAspect()); hidedecorations!(ax11)
    axall2=AMakie.Axis(f[:,2:3],title="H component",xlabel="time index")
    image!(ax11, rotr90(imggt))
    lin = [lines!(axall2,hd,color=dtcolors[i]) for (i,hd) in enumerate(hdata)]
    foreach(i->labels[i] *= " (inhibited)" ,inhibitindices)
    f[:,4] = Legend(f[:,2],lin,labels)
    save(joinpath(subworkpath,"idx$(inhibitindices)_bias$(bias)_gt.png"),f)
end # bias





fontsize = 30

f = Figure(resolution = (1450, 1010))
rowgap!(f.layout,0)

g1 = f[1, 1] = GridLayout()
g2 = f[2, 1] = GridLayout()
g3 = f[3, 1] = GridLayout()
g4 = f[4, 1] = GridLayout()
g5 = f[5, 1] = GridLayout()
g6 = f[6, 1] = GridLayout()

# Panel (a,d)
Label(g1[1,1],"(a) Inhibited neurons with 10% bias", font="Arial bold", fontsize=fontsize, width=700, padding=(10,2,2,0))
Label(g1[1,2],"(d) Inhibited neurons with 50% bias", font="Arial bold", fontsize=fontsize, width=700, padding=(10,2,2,0)) # g1p5[1,1,Bottom()]
rowsize!(g1,1,5)

fname = joinpath(subworkpath, "sp_meanT", "idx[1, 2, 3, 4]_bias0.1_gt.png")
axa=AMakie.Axis(g2[1,1], aspect = DataAspect())
hidedecorations!(axa, label=false); hidespines!(axa); image!(axa, rotr90(load(fname)))
fname = joinpath(subworkpath, "sp_meanT", "idx[1, 2, 3, 4]_bias0.5_gt.png")
axd=AMakie.Axis(g2[1,2], aspect = DataAspect())
hidedecorations!(axd, label=false); hidespines!(axd); image!(axd, rotr90(load(fname)));
rowsize!(g2,1,250); colgap!(g2,5)

# Panel (b,e)
Label(g3[1,1],"(b) Without background subtraction (10% bias)", font="Arial bold", fontsize=fontsize, width=700, padding=(10,2,0,0))
Label(g3[1,2],"(e) Without background subtraction (50% bias)", font="Arial bold", fontsize=fontsize, width=700, padding=(10,2,0,0)) # g1p5[1,1,Bottom()]
rowsize!(g3,1,5)

fname = joinpath(subworkpath, "sp_meanT", "idx[1, 2, 3, 4]_bias0.1_nosbg.png")
axb=AMakie.Axis(g4[1,1], aspect = DataAspect())
hidedecorations!(axb, label=false); hidespines!(axb); image!(axb, rotr90(load(fname)))
fname = joinpath(subworkpath, "sp_meanT", "idx[1, 2, 3, 4]_bias0.5_nosbg.png")
axe=AMakie.Axis(g4[1,2], aspect = DataAspect())
hidedecorations!(axe, label=false); hidespines!(axe); image!(axe, rotr90(load(fname)));
rowsize!(g4,1,300); colgap!(g4,5)

# Panel (c,f)
Label(g5[1,1],"(c) With background subtraction (10% bias)", font="Arial bold", fontsize=fontsize, width=700, padding=(10,2,0,0))
Label(g5[1,2],"(f) With background subtraction (50% bias)", font="Arial bold", fontsize=fontsize, width=700, padding=(10,2,0,0)) # g1p5[1,1,Bottom()]
rowsize!(g5,1,5)

fname = joinpath(subworkpath, "sp_meanT", "idx[1, 2, 3, 4]_bias0.1_sbg.png")
axc=AMakie.Axis(g6[1,1], aspect = DataAspect())
hidedecorations!(axc, label=false); hidespines!(axc); image!(axc, rotr90(load(fname)))
fname = joinpath(subworkpath, "sp_meanT", "idx[1, 2, 3, 4]_bias0.5_sbg.png")
axf=AMakie.Axis(g6[1,2], aspect = DataAspect())
hidedecorations!(axf, label=false); hidespines!(axf); image!(axf, rotr90(load(fname)));
rowsize!(g6,1,300); colgap!(g6,5)

save(joinpath(subworkpath,"inhibit_all_figures.png"),f,px_per_unit=2)



#= Broken Y axis
f = Figure()

lims = Node(((0.0, 1.3), (7.6, 10.0)))

g = f[1, 1] = GridLayout()

ax_top = Axis(f[1, 1][1, 1], title = "Broken Y-Axis")
ax_bottom = Axis(f[1, 1][2, 1])

on(lims) do (bottom, top)
    ylims!(ax_bottom, bottom)
    ylims!(ax_top, top)
    rowsize!(g, 1, Auto(top[2] - top[1]))
    rowsize!(g, 2, Auto(bottom[2] - bottom[1]))
end

hidexdecorations!(ax_top, grid = false)
ax_top.bottomspinevisible = false
ax_bottom.topspinevisible = false

linkxaxes!(ax_top, ax_bottom)
rowgap!(g, 10)

angle = pi/8
=#