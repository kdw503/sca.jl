using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

include("setup.jl")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include("dataset.jl")

plt.ioff()

#=
# ARGS =  ["[60]","50","[0]","[2]",":normalize",":balance2",":power1"]
# ARGS =  ["[60]","20","[0]","[2]",":balance",":balance2",":power1"]
# julia C:\Users\kdw76\WUSTL\Work\julia\sca\convexexpr.jl [10] 50 false 1 true :column [0] [0.5] :balance3 :none
SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]));
λs = eval(Meta.parse(ARGS[3])); βs = eval(Meta.parse(ARGS[4])) # be careful spaces in the argument
initpwradj = eval(Meta.parse(ARGS[5])); pwradj = eval(Meta.parse(ARGS[6]));
weighted = eval(Meta.parse(ARGS[7]))
Msparse = false; order = 1; Wonly = true; sd_group = :column; 
gtncells = 7; store_trace = false

@show SNRs, maxiter, order, Wonly, sd_group, λs, βs, initpwradj, pwradj, weighted
flush(stdout)

SNR = SNRs[1]; λ=λ1=λ2=λs[1]; β1=βs[1]; β2= Wonly ? 0 : βs[1]

for SNR in SNRs
    if SNR == :face
        filepath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
        nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
        X = zeros(nRow*nCol,nFace)
        for i in 1:nFace
            fname = "face"*@sprintf("%05d",i)*".pgm"
            img = load(joinpath(filepath,fname))
            X[:,i] = vec(img)
        end
        ncells = 49; borderwidth=1
        fprefix0 = "Wuc_face_nc$(ncells)_Convex_$(initpwradj)"
        rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
        # W1,H1 = copy(W0), copy(H0); normalizeWH!(W1,H1)
        # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
        # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W1,imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
        # normalizeWH!(Wp,Hp)
        # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wp,imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
    else
        lengthT=1000; jitter=0
        if gtncells == 2
            imgsz = (20,30); fname = "obj2_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
            ncls = 6
        else
            imgsz = (40,20); fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
            ncls = 15
        end
        X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell(fname, gt_ncells=gtncells, fovsz=imgsz, imgsz=imgsz,
                                                ncells=ncls, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
        gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
        fprefix0 = "Wuc_$(SNR)dB_nc$(ncells)_Convex_$(initpwradj)"
        rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
        # W0n, H0n = copy(W0), copy(H0); normalizeWH!(W0n,H0n)
        # Wpn, Hpn = copy(Wp), copy(Hp); normalizeWH!(Wpn,Hpn)
        # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W0n,imgsz, borderwidth=1)
        # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wpn,imgsz, borderwidth=1)
        # W0Mw0n, Mh0H0n = copy(W0*Mw), copy(Mh*H0); normalizeWH!(W0Mw0n,Mh0H0n)
        # imsaveW(fprefix0*"_W0Mw0_rt$(rt1).png",W0Mw0n,imgsz, borderwidth=1)
    end
    Mw0, Mh0 = copy(Mw), copy(Mh);
    for λ in λs, β in βs
        @show SNR, λ, β
        λ1 = λ; λ2 = λ; β1 = β; β2 = Wonly ? 0. : β
        flush(stdout)
        sparsestr = Msparse ? "M" : "WH"
        pwrstr = pwradj==:balance2 ? "M2" : pwradj==:balance3 ? "M$(order)" : ""
        paramstr="_$(sd_group)_$(weighted)_$(sparsestr)$(order)$(pwrstr)_lm$(λ)_bw$(β1)_bh$(β2)"
        fprefix = fprefix0*"_$(pwradj)"*paramstr
        sd_group ∉ [:column, :component, :pixel] && error("Unsupproted sd_group")
        Mw, Mh = copy(Mw0), copy(Mh0);
        rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,Msparse,order;
                    poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, SNR=SNR, store_trace=true, weighted=weighted, decifactor=4)
        W2,H2 = copy(W0*Mw), copy(Mh*H0)
        normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
        if SNR == :face
            clamp_level=0.5; W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
            signedcolors = (colorant"green1", colorant"white", colorant"magenta")
            imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W2, imgsz, gridcols=7, colors=signedcolors, borderval=W2_max, borderwidth=1)
        else
            imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W2, imgsz, borderwidth=1) #sortWHslices(W2,H2)[1]
        end
        ax = plotW([log10.(f_xs) log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_rt$(rt2)_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
            legendstrs = ["log(f_x)","log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
        if store_trace
            ax = plotW(log10.(f_xs), fprefix*"_rt$(rt2)_fxplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
                legendstrs = ["log(f_x)"], legendloc=1, separate_win=false)
            ax = plotW([log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_rt$(rt2)_xplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
                legendstrs = ["log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
            save(fprefix*"_iter$(iter)_rt$(rt2).jld", "f_xs", f_xs, "x_abss", x_abss, "xw_abss", xw_abss, "xh_abss", xh_abss, "SNR", SNR,
                    "order", order, "β1", β1, "β2", β2, "rt2", rt2, "trs", trs)
            dd = load(fprefix*"_iter$(iter)_rt$(rt2).jld"); plot_trs(dd["trs"], 8)
        end
    end
end
=#
# plt.show()


# initfn should return normalize Wp, Hp
# and normalize Wp and Hp
function initfn0(X, ncells)
    U, s = SCA.isvd(X, ncells)
    W0 = U; H0 = W0\X
    Wp, Hp = abs.(W0), abs.(H0)
    normalizeWH!(Wp,Hp)
    W0, H0, Wp, Hp
end
function initfn_SVD_PCA(X, ncells)
    U, s = SCA.isvd(X, ncells)
    W0 = U; H0 = W0\X
    Wp = fit(PCA, X; maxoutdim=ncells).proj
    Hp = Wp\X
    W0, H0, Wp, Hp
end
function initfn(X, ncells) #_PCA_PCA
    W0 = fit(PCA, X; maxoutdim=ncells).proj
    H0 = W0\X; Wp = copy(W0); Hp = copy(H0)
    W0, H0, Wp, Hp
end
initmethod=:custom

# julia C:\Users\kdw76\WUSTL\Work\julia\sca\convexexpr.jl [20] 200 [0] [0,0.01,0.1,1.0,5.0,10.0] :balance :balance3 [:nndsvd]
ARGS = ["[10]", "50","[0]", "[2.0]", ":balance", ":balance3", "[:nndsvd]"]
SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]));
λs = eval(Meta.parse(ARGS[3])); βs = eval(Meta.parse(ARGS[4])) # be careful spaces in the argument
initpwradj = eval(Meta.parse(ARGS[5])); pwradj = eval(Meta.parse(ARGS[6]));
initmethods = eval(Meta.parse(ARGS[7])); nclsrng = eval(Meta.parse(ARGS[8]))
weighted = :none
Msparse = false; order = 1; Wonly = true; sd_group = :column; 
store_trace = false
λ=λ1=λ2=λs[1]; β1=βs[1]; β2=0; SNR = SNRs[1]; initmethod=initmethods[1]
rt1s=[]; rt2s=[]; mssds=[]; mssdHs=[] 
for SNR in SNRs
    if true # 7cells
        gtncells = 7; imgsz = (40,20); ncls = 15
    else
        gtncells = 2; imgsz = (20,30); ncls = 6
    end
    lengthT=1000; jitter=0
    initpwradj=:balance; identityM=false
    fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"

    X, imgsz, lengthT, ncells, gtncells, datadic = load_data(:fakecell; SNR=SNR, save_gtimg=false)
    gtW = datadic["gtW"]; gtH = datadic["gtH"]
    W3,H3 = copy(gtW), copy(gtH')
    fprefixgt = "GT_$(SNR)B_n$(ncls)"
    imsaveW(fprefixgt*"_W.png", W3, imgsz, borderwidth=1)
    imsaveH(fprefixgt*"_H.png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))

    for ncells in nclsrng
    if identityM
        rt1 = @elapsed W0n, H0n, Mwn, Mhn, Wpn, Hpn = initsemisca(X, ncells, poweradjust=:normalize,
                            initmethod=initmethod, initfn=initfn) # initfn works only when initmethod==:custom
        Mwn0, Mhn0 = copy(Mwn), copy(Mhn);
    end
    rt1 = @elapsed W0b, H0b, Mwb, Mhb, Wpb, Hpb = initsemisca(X, ncells, poweradjust=:balance,
                            initmethod=initmethod, initfn=initfn)
    Mwb0, Mhb0 = copy(Mwb), copy(Mhb)
    push!(rt1s,rt1)
    for  β in  βs
        maxiter = 50; pwradj = :balance2; weighted = :none
        Msparse = false; order = 1; sd_group = :column; 
        β1=β; β2=0
        initpwradjstr = initpwradj==:balance ? "blnc" : "nmlz"
        pwradjstr = pwradj==:balance ? "blnc" :
                    pwradj==:normalize ? "nmlz" :
                    pwradj==:balance2 ? "blnc2" :
                    pwradj==:balance3 ? "blnc3" : "none"
        sparpwrstr = pwradj==:balance2 ? "M2" : pwradj==:balance3 ? "M$(order)" : ""
        sdgroupstr = sd_group==:column ? "col" : sd_group==:component ? "comp" : "pix"
        if identityM  
            fprefix0 = "SCA_$(SNR)B_n$(ncls)_Cnvx_$(sdgroupstr)_$(initmethod)_nmlzWH_idntyM_$(pwradjstr)"
        else
            fprefix0 = "SCA_$(SNR)B_n$(ncls)_Cnvx_$(sdgroupstr)_$(initmethod)_$(initpwradjstr)_$(pwradjstr)"
        end
        paramstr="_wgt$(weighted)_WH$(order)$(sparpwrstr)_lm$(λ)_bw$(β1)_bh$(β2)"
        fprefix = fprefix0*paramstr

        # Mwn, Mhn = copy(Mwn0), copy(Mhn0)
        Mwb, Mhb = copy(Mwb0), copy(Mhb0)
        if identityM
            Mwi = Matrix(1.0I,ncells,ncells); Mhi = Matrix(1.0I,ncells,ncells)
            rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mwi,Mhi,W0n,H0n,λ1,λ2,β1,β2,maxiter,Msparse,order;
                poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, SNR=SNR, store_trace=false, weighted=weighted, decifactor=4)
            @show norm(I-Mw*Mh)^2, norm(W0n*Mw,1), norm(W0n*(I-Mw*Mh)*H0n)^2
            W2,H2 = copy(W0n*Mw), copy(Mh*H0n)
        else
            rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mwb,Mhb,W0b,H0b,λ1,λ2,β1,β2,maxiter,Msparse,order;
            poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, SNR=SNR, store_trace=false, weighted=weighted, decifactor=4)
            @show norm(I-Mwb0*Mhb0)^2, norm(W0b*Mwb0,1)
            @show norm(I-Mw*Mh)^2, norm(W0b*Mw,1), norm(W0b*(I-Mw*Mh)*H0b)^2
            W2,H2 = copy(W0b*Mw), copy(Mh*H0b)
        end
        push!(rt2s,rt2)
        # match with ground truth
        normalizeWH!(W2,H2); W3,H3 = sortWHslices(W2,H2)
        imsaveW(fprefix*"_iter$(iter)_rt$(rt2)_W.png", W3, imgsz, borderwidth=1)
        imsaveH(fprefix*"_iter$(iter)_rt$(rt2)_H.png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
        ax = plotW([log10.(f_xs) log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_rt$(rt2)_plot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
            legendstrs = ["log(f_x)","log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, arrange=:combine)
        ax = plotW(log10.(f_xs), fprefix*"_rt$(rt2)_fxplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
            legendstrs = ["log(f_x)"], legendloc=1)
        ax = plotW([log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_rt$(rt2)_xplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
            legendstrs = ["log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, arrange=:combine)
        ax = plotW(H3[1:8,1:100]', fprefix*"_rt$(rt2)_Hplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
            legendstrs = ["log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, arrange=:combine)

        # make W components be positive mostly
        for i in 1:ncells
            (w,h) = view(W2,:,i), view(H2,i,:)
            psum = sum(w[w.>0]); nsum = -sum(w[w.<0])
            psum < nsum && (w .*= -1; h .*= -1) # just '*=' doesn't work
        end
        # match with ground truth
        mssd, ml, ssds = matchednssd(gtW,W2)
        mssdH = ssdH(ml,gtH,H2')
        push!(mssds,mssd); push!(mssdHs,mssdH)
        neworder = zeros(Int,length(ml))
        for (gti, i) in ml
            neworder[gti]=i
        end
        for i in 1:ncells
            i ∉ neworder && push!(neworder,i)
        end
        W3,H3 = copy(W2[:,neworder]), copy(H2[neworder,:])
        imsaveW(fprefix*"_iter$(iter)_rt$(rt2)_W_matched_ssd$(mssd).png", W3, imgsz, borderwidth=1)
        imsaveH(fprefix*"_iter$(iter)_rt$(rt2)_H_matched_ssd$(mssdH).png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
    end
    end # for ncells
end # for SNR
#save("SCA_SNR_vs_MSSD.jld","rt1s",rt1s,"rt2s",rt2s,"mssds",mssds,"mssdHs",mssdHs)
#save("SCA_NOC_vs_MSSD_$SNR.jld","rt1s",rt1s,"rt2s",rt2s,"mssds",mssds,"mssdHs",mssdHs)

#=
# save ground truth images
SNR = 20
imgsz = (40,20); lengthT=1000; jitter=0; gtncells = 7; ncls = 15; initpwradj=:balance
fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell(fname, gt_ncells=gtncells, fovsz=imgsz, imgsz=imgsz,
        ncells=ncls, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW = fakecells_dic["gtW"]; gtH = fakecells_dic["gtH"]
W3,H3 = sortWHslices(gtW,gtH')
fprefixgt = "GT_$(SNR)B_n$(ncls)"
imsaveW(fprefixgt*"_W.png", W3, imgsz, borderwidth=1)
imsaveH(fprefixgt*"_H.png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))


ax = plotW([log10.(f_xs) log10.(f_xs_spa)], "nndsvd_vs_spa_fx.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(f(x))",
    legendstrs = ["NNDSVD","SPA"], legendloc=1, separate_win=false)
ax = plotW([log10.(x_abss) log10.(x_abss_spa)], "nndsvd_vs_spa_xabs.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
    legendstrs = ["NNDSVD", "SPA"], legendloc=1, separate_win=false)

# PCA test
using MultivariateStats

SNR = 20
imgsz = (40,20); lengthT=1000; jitter=0; gtncells = 7; ncls = 15; initpwradj=:balance
fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell(fname, gt_ncells=gtncells, fovsz=imgsz, imgsz=imgsz,
        ncells=ncls, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW = fakecells_dic["gtW"]; gtH = fakecells_dic["gtH"]
W3,H3 = sortWHslices(gtW,gtH')
fprefixgt = "GT_$(SNR)B_n$(ncls)"
imsaveW(fprefixgt*"_W.png", W3, imgsz, borderwidth=1)
imsaveH(fprefixgt*"_H.png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))

M = fit(PCA, X; maxoutdim=15)
Y = predict(M, X)
Xr = reconstruct(M, Y)
imsaveW("PCA_$(SNR)B_n$(ncls)_W.png", M.proj, imgsz, borderwidth=1)
=#

#=
imgsz = (20,30); lengthT = 1000; SNR=10; distance = 10; overlap_rate = 0.3
X, imgsz, ncells0, fakecells_dic, img_nl, maxSNR_X = loadfakecell("obj2_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld";
        gt_ncells=2, fovsz=imgsz, imgsz=imgsz, ncells=15, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];
ncells = 5
fprefix0 = "Wuc_$(SNR)dB_nc$(ncells)"
rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj, use_nndsvd=true)
W0n, H0n = copy(W0), copy(H0); normalizeWH!(W0n,H0n)
imsaveW(fprefix0*"_SVD_rt$(rt1).png",W0n,imgsz, borderwidth=1)
img = reshape(imgrs,imgsize...,lengthT)
save("img.gif", RGB{N0f8}.(clamp01nan!(img./maximum(img))))



order = 1; weighted = :none
verbose=false; show_figure=false; xmin=-10; xmax=10; fn=""; cols=1;
poweradjust = :balance2

Mw,Mh=copy(Mw0),copy(Mh0)
nMh = norm(Mh,2)^order; Hi = Mh*H0; nMhH0 = norm(Hi,order)^order

λ1=λ2=0; β1=5.0; β2=0
m, p = size(W0); n = size(H0,2)
normw2 = norm(W0)^2; normh2 = norm(H0)^2
normwp = norm(W0,order)^order; normhp = norm(H0,order)^order
λw = λ1/normw2; λh = λ2/normh2
βw = β1/normwp; βh = β2/normhp

k=1

p = size(W0,2)
fw=gh=Matrix(1.0I,p,p)
mwk = Mw[:,k]; mhk = Mh[k,:]'
(Eprev,Ek) = weighted == :none ? (I-Mw*Mh, Mw[:,k]*mhk) : (fw*(I-Mw*Mh)*gh, fw*Mw[:,k]*mhk*gh)
E = Eprev+Ek
(poweradjust==:balance2 && order==1 && βh!=0) && (sqMwk = sqrt(norm(Mw)^2-norm(mwk)^2))
# Convex : set variable
x = Variable(p)
set_value!(x, Mw[:,k])
# Convex : set problem
invertibility = weighted == :none ? sumsquares(E-x*mhk) : sumsquares(E-fw*x*mhk*gh)
if poweradjust == :balance2
    sparw = βw==0 ? 0 : order == 1 ? βw*norm(W0*x, 1)*nMh : βw*sumsquares(W0*x)*nMh
    sparh = βh==0 ? 0 : order == 1 ? βh*nMhH0*norm(vcat(x,sqMwk),2) : βh*nMhH0*norm(x,2)^2
    sparsity = sparw+sparh
else
    sparsity = order == 1 ? βw*norm(W0*x, 1) : βw*sumsquares(W0*x)
end
nnegativity = λ*sumsquares(max(0,-W0*x))
expr = invertibility + nnegativity + sparsity
problem = minimize(expr)
Evalpre = Convex.evaluate(expr)
totalpen, _ = SCA.penaltyMw(Mw,Mh,W0,H0,fw,gh,λ,βw,βh; order=order, poweradjust=poweradjust, weighted=weighted)
totalpen -= βw*(norm(W0*Mw,1)-norm(W0*mwk,1))*nMh
Convex.evaluate(invertibility)
Convex.evaluate(sparsity)
Convex.evaluate(nnegativity)
# println("expression curvature = ", vexity(expr))
# println("expression sign = ", sign(expr))

# Convex : solve
solve!(problem, ECOS.Optimizer; warmstart = false, silent_solver = true, verbose=verbose) 
# other solver options : SCS, ECOS, (GLPK : run error), (Gurobi, Mosek : precompile error)
# verbose=false (turn off warning)
# warmstart doesn't work for SCS.GeometricConicForm and ECOS

# Convex : check the result
# @show round.(Convex.evaluate(x), digits = 2)
Eval = problem.optval # round(problem.optval, digits = 10)
Evalfromxsol = Convex.evaluate(expr) # round(Convex.evaluate(expr), digits = 10)

Mwnew = copy(Mw); Mwnew[:,k]=x.value
Evalfromxsol2 = SCA.penaltyMw(Mwnew,Mh,W0,H0,fw,gh,λ,βw,βh; order=order, poweradjust=poweradjust, weighted=weighted)
Evalfromxsol2 -= βw*(norm(W0*Mw,1)-norm(W0*mwk,1))*nMh


x = Variable(15)
set_value!(x, rand(15))
expr = norm(vcat(x,sqrt(2)),2)
# expr = sumsquares(x)
problem = minimize(expr)
Evalpre = Convex.evaluate(expr)
println("expression curvature = ", vexity(expr))
println("expression sign = ", sign(expr))
solve!(problem, ECOS.Optimizer; warmstart = false, silent_solver = true, verbose=true) 
problem.optval
x.value


using ECOS
dd=load("tmp.jld"); params = dd["all"]
SCA.mincol_convex(params...)

Mw,Mh,W0,H0,nMh,nhobj,fw,gh,k,λw,λh,βw,βh,Msparse,order = params; 
poweradjust=:none; weighted=:none; verbose=false; show_figure=false; xmin=-10; xmax=10; fn=""; cols=1
=#
