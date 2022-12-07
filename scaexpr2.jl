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

#ARGS =  ["[:fakecell]",[20]","200","[0]","[0,0.01,0.1,0.5,1.0,1.5]",":sca"]
#ARGS =  ["[:fakecell]","[20]","200","[0]","[0,0.01,0.1,0.5,1.0,1.5]",":cd"]
#ARGS =  ["[:fakecell]","[40]","1","[0]","[0.1]",":sca","10:2:14","false"] # Best : false means using Convex.jl
#ARGS =  ["[:fakecell]","-20:1:-18","1","[0]","[0.1]",":sca","[15]","false"] # Best : false means using Convex.jl
#ARGS =  ["[:fakecell]","[20]","50","[0]","[1.5]",":sca","[15]","false"]
#ARGS =  ["[:cbclface]","[20]","500","[100]","[4]",":sca"] # Best
#ARGS =  ["[:cbclface]","[20]","200","[0]","[0.1]",":cd"] # Best
#ARGS =  ["[:orlface]","[20]","200","[0]","[1.5]",":sca"] # Best
#ARGS =  ["[:orlface]","[20]","200","[0]","[0.1]",":cd"] # Best
#ARGS =  ["[:natural]","[20]","200","[100]","[7]",":sca"] # Best
#ARGS =  ["[:natural]","[20]","200","[0]","[0.1]",":cd"] # Best
#ARGS =  ["[:urban]","[20]","200","[0]","[1.5]",":sca"] # Best
#ARGS =  ["[:urban]","[20]","200","[0]","[0.1]",":cd"] # Best
#ARGS =  ["[:neurofinder]","[20]","200","[0]","[1.5]",":sca"] # Best
#ARGS =  ["[:neurofinder]","[20]","200","[0]","[0.1]",":cd"] # Best

datasets = eval(Meta.parse(ARGS[1])); SNRs = eval(Meta.parse(ARGS[2]));
maxiter = eval(Meta.parse(ARGS[3])); λs = eval(Meta.parse(ARGS[4]));
βs = eval(Meta.parse(ARGS[5])); # be careful spaces in the argument
method = eval(Meta.parse(ARGS[6])); αs = eval(Meta.parse(ARGS[5]))
nclsrng = eval(Meta.parse(ARGS[7])); usingFastSCA = eval(Meta.parse(ARGS[8]))
order = 1; Wonly = true; sd_group = :column; ls_method = :sca_full;
initmethod = :nndsvd; initfn = SCA.nndsvd2; initpwradj = :balance;
pwradj = :balance3; tol=-1 # -1 means don't use convergence criterion
makepositive = false; # flip W[:,i] and H[i,:] to make mostly positive
savefigure = true
@show datasets, SNRs, maxiter
@show λs, βs, method, αs, nclsrng
flush(stdout)
dataset = datasets[1]; SNR = SNRs[1]; λ = λs[1]; λ1 = λ; λ2 = λ;
β = βs[1]; β1 = β; β2 = Wonly ? 0. : β; α = αs[1]; ncls = nclsrng[1]
for dataset in datasets
    rt1s=[]; rt2s=[]; mssds=[]; mssdHs=[]
    for SNR in SNRs # this can be change ncellss, factors
        X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(true&&savefigure))
        SNRstr = dataset == :fakecell ? "_$(SNR)dB" : ""
        for ncls in nclsrng
        ncells = dataset != :fakecell ? ncells0 : ncls
        initdatastr = "_$(dataset)$(SNRstr)_nc$(ncells)"
        if method == :sca
            rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=initpwradj,
                                        initmethod=initmethod, initfn=initfn)
            fprefix0 = "SCAinit$(initdatastr)_$(initpwradj)_rt"*@sprintf("%1.2f",rt1)
            Mw0, Mh0 = copy(Mw), copy(Mh)
            W1,H1 = copy(W0), copy(H0); normalizeWH!(W1,H1)
            savefigure && imsave_data(dataset,fprefix0,W1,H1,imgsz,lengthT; saveH=false)
            fprefix1 = "SCA$(initdatastr)_$(initpwradj)"
            push!(rt1s,rt1)
            for λ in λs
                λ1 = λ; λ2 = λ
                for β in βs
                    @show SNR, ncells, λ, β
                    β1 = β; β2 = Wonly ? 0. : β
                    flush(stdout)
                    paramstr="_L$(order)_λw$(λ1)_λh$(λ2)_βw$(β1)_βh$(β2)"
                    fprefix2 = fprefix1*"_$(pwradj)"*paramstr
                    if usingFastSCA
                        stparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=:WkHk, order=order, hfirst=true, processorder=:none,
                                poweradjust=pwradj, method=:cbyc_uc, rectify=:pinv, objective=:normal, option=1)
                        lsparams = LineSearchParams(method=ls_method, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
                                iterations_to_show=[15])
                        cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                                x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
                        Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
                        rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
                        x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
                        W2,H2 = copy(W1), copy(H1); iter = length(trs)
                    else
                        Msparse = false; weighted = :none
                        rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,
                                Msparse,order; poweradjust=pwradj, fprefix=fprefix2, sd_group=sd_group, SNR=SNR, store_trace=false,
                                show_figure=false, weighted=weighted, decifactor=4)
                        W2,H2 = copy(W0*Mw), copy(Mh*H0)
                        fprefix2 = "CVX_"*fprefix2
                    end
                    normalizeWH!(W2,H2); W3,H3 = sortWHslices(W2,H2)
                    # make W components be positive mostly
                    makepositive && flip2makepos!(W3,H3)
                    # save W and H image data and plot
                    fprefix3 = fprefix2*"_iter$(iter)_rt"*@sprintf("%1.2f",rt2)
                    if savefigure
                        imsave_data(dataset,fprefix3,W3,H3,imgsz,100; saveH=true)
                        plot_convergence(fprefix3,x_abss,xw_abss,xh_abss,f_xs; title="convergence (SCA)")
                        plotH_data(dataset,fprefix3,H3)
                    end
                    push!(rt2s,rt2)

                    if dataset == :fakecell
                        gtW = datadic["gtW"]; gtH = datadic["gtH"]
                        # calculate MSD
                        mssd, ml, ssds = matchednssd(gtW,W3)
                        mssdH = ssdH(ml, gtH,H3')
                        # reorder according to GT image
                        neworder = matchedorder(ml,ncells)
                        savefigure && imsave_data(dataset,fprefix3,W3[:,neworder],H3[neworder,:],imgsz,100;
                            mssdwstr="_MSE"*@sprintf("%1.4f",mssd), mssdhstr="_MSE"*@sprintf("%1.4f",mssdH), saveH=true)
                        push!(mssds,mssd); push!(mssdHs, mssdH)
                    elseif dataset == :urban
                        savefigure && imsave_reconstruct_urban(fprefix3,X,W3,H3,imgsz; index=100)
                    end
                end
            end
        else # :cd
            rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
            fprefix0 = "CDinit$(initdatastr)_normalize_rt"*@sprintf("%1.2f",rt1)
            Wcd0 = copy(Wcd); Hcd0 = copy(Hcd)
            W1,H1 = copy(Wcd), copy(Hcd); normalizeWH!(W1,H1)
            savefigure && imsave_data(dataset,fprefix0,W1,H1,imgsz,lengthT; saveH=false)
            fprefix1 = "CD$(initdatastr)_normalize"
            push!(rt1s,rt1)
            Wcd0 = copy(Wcd); Hcd0 = copy(Hcd)
            for α in βs # best α = 0.1
                @show α
                Wcd, Hcd = copy(Wcd0), copy(Hcd0);
                usingNMF = true
                if usingNMF
                    paramstr="_α$(α)"
                    fprefix2 = fprefix1*"_usingNMF"*paramstr
                    rt2 = @elapsed result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=0.5), X, Wcd, Hcd)
                    iter = result.niters
                    fprefix3 = fprefix2*"_iter$(iter)_rt"*@sprintf("%1.2f",rt2)
                else
                    cdorder=1; cdpwradj=:none; cdβ1=α; cdβ2=0
                    stparams = StepParams(β1=cdβ1, β2=cdβ2, order=cdorder, hfirst=true, processorder=:none, poweradjust=cdpwradj,
                                        rectify=:truncate) 
                    cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                                x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
                    paramstr="_L$(cdorder)_βw$(cdβ1)_βh$(cdβ2)"
                    fprefix2 = fprefix1*"_$(cdpwradj)"*paramstr
                    rt2 = @elapsed objvals, trs = SCA.halssolve!(X, Wcd, Hcd; stparams=stparams, cparams=cparams);
                    iter = length(trs)
                    fprefix3 = fprefix2*"_iter$(iter)_rt"*@sprintf("%1.2f",rt2)
                    x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
                    savefigure && plot_convergence(fprefix3,x_abss,xw_abss,xh_abss,f_xs; title="convergence (CD)")
                end
                normalizeWH!(Wcd,Hcd); Wcd1,Hcd1 = sortWHslices(Wcd,Hcd)
                if savefigure
                    imsave_data(dataset,fprefix3,Wcd1,Hcd1,imgsz,100; saveH=true)
                    plotH_data(dataset,fprefix3,Hcd1)
                end
                push!(rt2s,rt2)

                if dataset == :fakecell
                    gtW = datadic["gtW"]; gtH = datadic["gtH"]
                    # calculate MSD
                    mssd, ml, ssds = matchednssd(gtW,Wcd1)
                    mssdH = ssdH(ml, gtH,Hcd1')
                    # reorder according to GT image
                    neworder = matchedorder(ml,ncells)
                    savefigure && imsave_data(dataset,fprefix3,Wcd1[:,neworder],Hcd1[neworder,:],imgsz,100;
                        mssdwstr="_MSE"*@sprintf("%1.4f",mssd), mssdhstr="_MSE"*@sprintf("%1.4f",mssdH), saveH=true)
                    push!(mssds,mssd); push!(mssdHs, mssdH)
                elseif dataset == :urban
                    savefigure && imsave_reconstruct_urban(fprefix3,X,Wcd1,Hcd1,imgsz; index=100)
                end
            end # for α
        end # if method
        end # for ncells
    end # for SNR
    if length(SNRs) > 1
        save("$(method)_SNR_vs_MSSD.jld","SNRs",SNRs,"rt1s",rt1s,"rt2s",rt2s,"mssds",mssds,"mssdHs",mssdHs)
    elseif length(nclsrng) > 1
        save("$(method)_NOC_vs_MSSD_$(SNR)dB.jld","nclsrng",nclsrng,"rt1s",rt1s,"rt2s",rt2s,"mssds",mssds,"mssdHs",mssdHs)
    end
#    rts = rt1s+rt2s
    # TODO : plot SNRs vs. rts
    # TODO : plot ncells vs. rts
    # TODO : plot factors vs. rts
end


# plt.show()

# Hessian and gradient check for sparsity of :balance3
#=
function Eswk(x,W0,Mw,Mh,k)
    mwk = Mw[:,k]
    Swk = norm(W0*Mw,1) - norm(W0*mwk,1)
    Swk+norm(W0*(mwk+x),1)*norm(Mh)
end

function Eshk(x,H0,Mw,Mh,k)
    mwk = Mw[:,k]
    nMwk = norm(Mw)^2-norm(mwk)^2
    norm(Mh*H0,1)*sqrt(nMwk+norm(mwk+x)^2)
end

W0 = rand(100,15); H0 = rand(15,200)
Mw = rand(15,15); Mh = rand(15,15)
Wim1 = W0*Mw; Him1 = Mh*H0
k = 1
Esw(x) = Eswk(x,W0,Mw,Mh,k)
Esh(x) = Eshk(x,H0,Mw,Mh,k)
using ForwardDiff
fdgradEsw = ForwardDiff.gradient(Esw,zeros(15))
fdHessEsw = ForwardDiff.hessian(Esw,zeros(15))
fdgradEsh = ForwardDiff.gradient(Esh,zeros(15))
fdHessEsh = ForwardDiff.hessian(Esh,zeros(15))
hesssw, gradsw = SCA.hessgradsparsityWbal3(W0, Wim1, Mh, k)
Hesssh, gradsh = SCA.HessgradsparsityHbal3(Mw, Him1, k)
norm(fdgradEsw-gradsw)
norm(fdHessEsw) # hesssw == 0
norm(fdgradEsh-gradsh)
norm(fdHessEsh-Hesssh)
=#


#=
#============ noc vs runtime (:symmetric_orthogonality) =============#
ncellsrng = 6:2:100; xlabelstr = "number of components"

imgsz=(40,20); ncells=15; lengthT=1000; SNR=40; # SNR=10(noisey), SNR=40(less noisey)
X, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells_sz$(imgsz)lengthT$(lengthT)_SNR$(SNR).jld",
                                            fovsz=imgsz, ncells=ncells, lengthT=lengthT, SNR=SNR, save=true);
#X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fakecells100_$(orthogstr).jld"; ncells=40, lengthT=100)
gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"];

lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-6, f_inctol=1e-5,
    maxiter=500, store_trace=false, show_trace=true)

rtcd1s = [];
rtcd2s = [];mssdcds=[];
rtcd2s0p1 = []; mssdcds0p1=[]
rtssca1s = []
rtssca2s = []; mssdsscas = []
rtssca2s0p1 = []; mssdsscas0p1 = []
for ncells in ncellsrng
    @show ncells
    @show "SSCA"
    stparams = StepParams(β1=0.0, β2=0.0, reg=:none)
    rt1 = @elapsed W0, H0, Mwinit, Mhinit = initsemisca(X, ncells)
    Mw, Mh = copy(Mwinit), copy(Mhinit)
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
    mssdssca, ml, ssds = matchedfiterr(gtW,W2);
    push!(rtssca1s, rt1); push!(rtssca2s, rt2); push!(mssdsscas,mssdssca)
    imsaveW("SSCA_SNR$(SNR)_n$(ncells)_mssd$(mssdssca)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)

    @show "SSCA with Mk reg"
    stparams = StepParams(β1=0.2, β2=0.2, reg=:WH)
    Mw, Mh = copy(Mwinit), copy(Mhinit)
    rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); imshowW(W2,imgsz, borderwidth=1)
    mssdssca, ml, ssds = matchedfiterr(gtW,W2);
    push!(rtssca2s0p1, rt2); push!(mssdsscas0p1,mssdssca)
    imsaveW("SSCA_SNR$(SNR)_n$(ncells)_β$(stparams.β1)_mssd$(mssdssca)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)
    # @show "SSCA+Rect"
    # W2,H2 = copy(W1), copy(H1)
    # W2[W2.<0].=0; H2[H2.<0].=0;
    # normalizeWH!(W2,H2); # imshowW(W2,imgsz, borderwidth=1)
    # mssdsscarect, ml, ssds = matchedfiterr(gtW,W2);
    # push!(mssdsscarects,mssdsscarect)
    # imsaveW("SSCA_rect_SNR$(SNR)_n$(ncells)_mssd$(mssdsscarect)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)

    # @show "SSCA+CD"
    # W2,H2 = copy(W1), copy(H1)
    # rtsscacd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=0.1, l₁ratio=0.5), X, W2, H2)
    # normalizeWH!(W2,H2); # imshowW(W2,imgsz, borderwidth=1)
    # mssdsscacd, ml, ssds = matchedfiterr(gtW,W2);
    # push!(rtsscacd2s, rtsscacd2); ; push!(mssdsscacds,mssdsscacd)
    # imsaveW("SSCA_CD_SNR$(SNR)_n$(ncells)_mssd$(mssdsscacd)_rt1$(rt1)_rt2$(rt2)_rt3$(rtsscacd2).png",W2,imgsz,borderwidth=1)

    @show "CD α = 0"
    α = 0.1
    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    Wcd0, Hcd0 = copy(Wcd), copy(Hcd)
    rtcd2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz, borderwidth=1)
    mssdcd, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd1s, rtcd1); push!(rtcd2s, rtcd2); push!(mssdcds,mssdcd)
    imsaveW("cd_SNR$(SNR)_n$(ncells)_a$(α)_mssd$(mssdcd)_rt1$(rtcd1)_rt2$(rtcd2).png",Wcd,imgsz)

    @show "CD α = 0.1"
    α = 3
    Wcd, Hcd = copy(Wcd0), copy(Hcd0)
    rtcd20p1 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd); imshowW(Wcd,imgsz, borderwidth=1)
    mssdcd0p1, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd2s0p1, rtcd20p1); push!(mssdcds0p1,mssdcd0p1)
    imsaveW("cd_SNR$(SNR)_n$(ncells)_a$(α)_mssd$(mssdcd)_rt1$(rtcd1)_rt2$(rtcd2).png",Wcd,imgsz)
end

fname = "noc_SNR$(SNR)_$(today())_$(hour(now()))_$(minute(now())).jld"
save(fname, "ncellsrng", ncellsrng, "imgsz", imgsz, "lengthT", lengthT, "SNR", SNR,
        "β1", stparams.β1, "β2", stparams.β2,
        # semi-sca
        "rtssca1s",rtssca1s,"rtssca2s",rtssca2s, "mssdsscas", mssdsscas,
        # semi-sca with reg
        "rtssca2s0p1",rtssca2s0p1, "mssdsscas0p1", mssdsscas0p1,
        # CD α = 0
        "rtcd1s", rtcd1s,"rtcd2s",rtcd2s, "mssdcds", mssdcds,
        # CD α = 0.1
        "rtcd2s0p1",rtcd2s0p1, "mssdcds0p1", mssdcds0p1
        )

# dd = load("noc_SNR-15_2022-04-15_14_58.jld")
# β1 = dd["β1"]; β2 = dd["β2"]; ncellsrng = dd["ncellsrng"]
# imgsz = dd["imgsz"]; lengthT = dd["lengthT"];
# rtssca1s = dd["rtssca1s"]; rtssca2s = dd["rtssca2s"]; mssdsscas = dd["mssdsscas"];
# mssdsscarects = dd["mssdsscarects"];
# rtsscacd2s = dd["rtsscacd2s"]; mssdsscacds = dd["mssdsscacds"];
# rtcd1s = dd["rtcd1s"]; rtcd2s = dd["rtcd2s"]; mssdcds = dd["mssdcds"];
# rtcd2s0p1 = dd["rtcd2s0p1"]; mssdcds0p1 = dd["mssdcds0p1"];

rtcds = rtcd1s + rtcd2s
rtcds0p1 = rtcd1s + rtcd2s0p1
rtsscas = rtssca1s + rtssca2s
rtsscas0p1 = rtssca1s + rtssca2s0p1

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtsscas rtsscas0p1 rtcds rtcds0p1])
ax1.legend(["Semi-SCA", "Semi-SCA w reg.", "CD", "CD w reg."],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtssca1s rtcd1s])
ax1.legend(["Semi-SCA", "CD"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_runtime1.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [rtssca2s rtssca2s0p1 rtcd2s rtcd2s0p1])
ax1.legend(["Semi-SCA", "Semi-SCA w reg.", "CD", "CD w reg."],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("noc_vs_runtime2.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [mssdsscas mssdsscas0p1 mssdcds mssdcds0p1])
ax1.legend(["Semi-SCA", "Semi-SCA w reg.", "CD", "CD w reg."],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("MSSD",fontsize = 12)
savefig("noc_vs_mssd.png")



#==== factor vs different svd runtime (rsvd test) =============#
factorrng = 1:20; cncells = 60; SNR = 10
fovsz=(20,20); lengthT0 = 100

stparams = StepParams(β1=0.0, β2=0.0)
lsparams = LineSearchParams(method=:none, c=1e-4, α0=0.1, ρ=0.5, maxiter=100, show_figure=false)
cparams = ConvergenceParams(allow_f_increases = true, f_abstol = 1e-6, f_reltol=1e-6, f_inctol=1e-5,
    maxiter=500, store_trace=false, show_trace=false)

# isvd
rtssca1s = []; rtssca2s = []; mssdsscas = []
rtcd1s = []; rtcd2s0 = []; mssdcds0 = []
rtcd2s0p1 = []; mssdcds0p1 = []
for factor in factorrng
    @show factor
    imgsz = (fovsz[1]*factor,fovsz[2])
    lengthT = lengthT0*factor
    println("imgsz=($(imgsz[1]),$(imgsz[2])), lengthT=$lengthT")
    X, W, H, imgsz, ncells, fakecells_dic, img_nl = loadfakecell("fc_SNR$(SNR)_sz$(imgsz)_fsz$(imgsz)_lT$(lengthT).jld";
            fovsz=imgsz, lengthT=lengthT, SNR=SNR, imgsz=imgsz)
    gtW, gtH, gt_ncells = fakecells_dic["gtW"], fakecells_dic["gtH"], fakecells_dic["gt_ncells"]

    ncells = cncells

    @show "SSCA"
    rt1 = @elapsed W0, H0, Mw, Mh = initsemisca(X, ncells)
    rt2 = @elapsed W1, H1, objval, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, cparams=cparams);
    W2,H2 = copy(W1), copy(H1)
    normalizeWH!(W2,H2); # imshowW(W2,imgsz, borderwidth=1)
    mssdssca, ml, ssds = matchedfiterr(gtW,W2);
    push!(rtssca1s, rt1); push!(rtssca2s, rt2); push!(mssdsscas,mssdssca)
    imsaveW("W2_SNR$(SNR)_f$(factor)_mssd$(mssdssca)_rt1$(rt1)_rt2$(rt2).png",W2,imgsz,borderwidth=1)

    rtcd1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells)
    push!(rtcd1s, rtcd1)

    @show "CD α = 0.0"
    Wcd0, Hcd0 = copy(Wcd),copy(Hcd)
    α = 0.0
    rtcd20 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd0, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd2s0, rtcd20); push!(mssdcds0,mssdcd0)
    imsaveW("cd_SNR$(SNR)_f$(factor)_a$(α)_mssd$(mssdcd0)_rt1$(rtcd1)_rt2$(rtcd20).png",Wcd,imgsz)

    @show "CD α = 0.1"
    Wcd, Hcd = copy(Wcd0),copy(Hcd0)
    α = 0.1
    rtcd20p1 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, Wcd, Hcd)
    normalizeWH!(Wcd,Hcd)
    mssdcd0p1, mlcd, ssdcd = matchedfiterr(gtW,Wcd)
    push!(rtcd2s0p1, rtcd20p1); push!(mssdcds0p1,mssdcd0p1)
    imsaveW("cd_SNR$(SNR)_f$(factor)_a$(α)_mssd$(mssdcd0p1)_rt1$(rtcd1)_rt2$(rtcd20p1).png",Wcd,imgsz)
end

currenttime = now()
fname = "factor_SNR$(SNR)_$(today())_$(hour(currenttime))-$(minute(currenttime)).jld"
save(fname, "β1", β1, "β2", β2, "factorrng", factorrng,
        "SNR", SNR, "fovsz", fovsz, "lengthT0", lengthT0,
        "rtcd1s", rtcd1s, "rtcd2s0", rtcd2s0, "mssdcds0", mssdcds0, # cd α = 0
        "rtcd2s0p1", rtcd2s0p1, "mssdcds0p1", mssdcds0p1, # cd α = 0.1
        "rtssca1s", rtssca1s,"rtssca2s",rtssca2s, "mssdsscas", mssdsscas) # ssca

# dd = load("factor_2022-04-14_21-50.jld")
# β1 = dd["β1"]; β2 = dd["β2"]; factorrng = dd["factorrng"]; fovsz = dd["fovsz"]; lengthT0 = dd["lengthT0"];
# rtcd1s = dd["rtcd1s"]; rtcd2s0 = dd["rtcd2s0"]; mssdcds0 = dd["mssdcds0"];
# rtcd2s0p1 = dd["rtcd2s0p1"]; mssdcds0p1 = dd["mssdcds0p1"];
# rtssca1s = dd["rtssca1s"]; rtssca2s = dd["rtssca2s"]; mssdsscas = dd["mssdsscas"];
rtsscas = rtssca1s + rtssca2s
rtcds0 = rtcd1s + rtcd2s0
rtcds0p1 = rtcd1s + rtcd2s0p1

xlabelstr = "factor"
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtsscas rtcds0 rtcds0p1])
ax1.legend(["Semi-SCA", "CD (α=0)", "CD (α=0.1)"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_totalruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtssca1s rtcd1s])
ax1.legend(["Semi-SCA", "CD"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_svdruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [rtssca2s rtcd2s0 rtcd2s0p1])
ax1.legend(["Semi-SCA", "CD (α=0)", "CD (α=0.1)"],fontsize = 12,loc=2)
xlabel(xlabelstr,fontsize = 12)
ylabel("runtime",fontsize = 12)
savefig("factor_vs_coreruntime.png")

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(factorrng, [mssdsscas mssdcds0 mssdcds0p1])
ax1.set_yscale(:log) # :linear
ax1.legend(["Semi-SCA", "CD (α=0)", "CD (α=0.1)"],fontsize = 12,loc=1)
xlabel(xlabelstr,fontsize = 12)
ylabel("MSSD",fontsize = 12)
savefig("factor_vs_mssd.png")



#============ noc vs. |X-WH|² =========================================#
ncellsrng = 2:2:200
W800, H800 = initWH(X, 800; svd_method=:svd)
err1=[]; err2=[]; err3=[]
for ncells in ncellsrng
    @show ncells
    W = W800[:,1:ncells]; H = W\X
    Wi, Hi = initWH(X, ncells; svd_method=:isvd)
    W0, H0, _ = initsemisca(X, ncells)
    push!(err1,norm(X-W*H)^2)
    push!(err2,norm(X-Wi*Hi)^2)
    push!(err3,norm(X-W0*H0)^2)
end
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(ncellsrng, [err1 err2 err3])
ax1.legend(["|X-Wsvd*Hsvd|^2", "|X-Wisvd*Hisvd|^2", "|X-Wmp*Hmp|^2"],fontsize = 12,loc=1)
xlabel("Number of cells",fontsize = 12)
ylabel("Error",fontsize = 12)
savefig("noc_vs_errors.png")

=#