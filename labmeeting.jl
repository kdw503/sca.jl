include("setup.jl")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))

plt.ioff()

#ARGS =  ["[\"face\"]","1","1","true",":column","[100]","[0]",":sca_full"]
#ARGS =  ["[10]","50","1","true",":column","[0]","[0.6,1.0,2.0,2.5,3.0,4.0,5.0]",":sca_full"]
ARGS =  ["[30]","200","1","true",":column","[100]","[1.5]",":sca_full"]

SNRs = eval(Meta.parse(ARGS[1])); maxiter = eval(Meta.parse(ARGS[2]))
order = eval(Meta.parse(ARGS[3])); Wonly = eval(Meta.parse(ARGS[4]));
sd_group = eval(Meta.parse(ARGS[5])) # subspace descent subspace group (:pixel subspace is same as CD)
λs = eval(Meta.parse(ARGS[6])); βs = eval(Meta.parse(ARGS[7])) # be careful spaces in the argument
ls_method = eval(Meta.parse(ARGS[8])) # line search method
initmethod = :nndsvd; initfn = SCA.nndsvd2
@show SNRs, maxiter
@show order, Wonly, sd_group
@show λs, βs
flush(stdout)
initpwradj = :balance; pwradj = :balance3; tol=-1
imgsize = (40,20); lengthT=1000; jitter=0
SNR = SNRs[1]; λ = λs[1]; λ1 = λ; λ2 = λ; β = βs[1]; β1 = β; β2 = Wonly ? 0. : β
for SNR in SNRs
    if SNR == "face"
        filepath = joinpath(datapath,"MIT-CBCL-face","face.train","train","face")
        nRow = 19; nCol = 19; nFace = 2429; imgsz = (nRow, nCol)
        X = zeros(nRow*nCol,nFace)
        for i in 1:nFace
            fname = "face"*@sprintf("%05d",i)*".pgm"
            img = load(joinpath(filepath,fname))
            X[:,i] = vec(img)
        end
        ncells = 49;  borderwidth=1
        fprefix0 = "Wuc_face"
        # W1,H1 = copy(W0), copy(H0); normalizeWH!(W1,H1)
        # signedcolors = (colorant"green1", colorant"white", colorant"magenta")
        # imsaveW(fprefix0*"_SVD_rt$(rt1).png",W1,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
        # normalizeWH!(Wp,Hp)
        # imsaveW(fprefix0*"_NNDSVD_rt$(rt1).png",Wp,imgsz, gridcols=7, colors=signedcolors, borderval=0, borderwidth=1)
    else
        if true # 7cells
            gtncells = 7; imgsz = (40,20); ncls = 15
        else
            gtncells = 2; imgsz = (20,30); ncls = 6
        end
        imgsize = (40,20); lengthT=1000; jitter=0
        fname = "fakecells_sz$(imgsz)_lengthT$(lengthT)_J$(jitter)_SNR$(SNR).jld"
        X, imgsz, ncells, fakecells_dic, img_nl, maxSNR_X = loadfakecell(Float64, fname, gt_ncells=gtncells, fovsz=imgsz,
                imgsz=imgsz, ncells=ncls, lengthT=lengthT, jitter=jitter, SNR=SNR, save=true);
        gtW = fakecells_dic["gtW"]; gtH = fakecells_dic["gtH"]
        W3,H3 = copy(gtW), copy(gtH')
        # fprefixgt = "GT_$(SNR)B_n$(ncls)"
        # imsaveW(fprefixgt*"_W.png", W3, imgsz, borderwidth=1)
        # imsaveH(fprefixgt*"_H.png", H3, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
        fprefix0 = "Wuc_$(SNR)dB"
    end
    rt1 = @elapsed W0, H0, Mw, Mh, Wp, Hp = initsemisca(X, ncells, poweradjust=:balance,
                            initmethod=initmethod, initfn=initfn)
    Mw0, Mh0 = copy(Mw), copy(Mh)
    # imsaveW(fprefix0*"_$(initmethod)_W_rt$(rt1).png", W0, imgsz, borderwidth=1)
    # imsaveH(fprefix0*"_$(initmethod)_H_rt$(rt1).png", H0, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
    for λ in λs
        λ1 = λ; λ2 = λ
        for β in βs
            @show SNR, λ, β
            β1 = β; β2 = Wonly ? 0. : β
            flush(stdout)
            paramstr="_L$(order)_lm$(λ)_aw$(β1)_ah$(β2)"
            fprefix =fprefix0*"_$(initpwradj)_$(pwradj)"*paramstr
            reg = :WkHk; method=:cbyc_uc
            stparams = StepParams(β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=reg, order=order, hfirst=true, processorder=:none,
                    poweradjust=pwradj, method=method, rectify=:pinv, objective=:normal, option=1)
            lsparams = LineSearchParams(method=ls_method, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_figure=false,
                    iterations_to_show=[15])
            cparams = ConvergenceParams(allow_f_increases = true, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
                    x_abstol=tol, successive_f_converge=2, maxiter=maxiter, store_trace=true, show_trace=true)
            fprefix = "Wuc_$(SNR)dB_$(initpwradj)_$(pwradj)"*paramstr
            Mw, Mh = copy(Mw0), copy(Mh0)
            rt2 = @elapsed W1, H1, objvals, trs = semiscasolve!(W0, H0, Mw, Mh; stparams=stparams, lsparams=lsparams, cparams=cparams);
            W2,H2 = copy(W1), copy(H1)
            normalizeWH!(W2,H2); iter = length(trs); W3,H3 = sortWHslices(W2,H2)
            if SNR == "face"
                clamp_level=0.5; W3_max = maximum(abs,W3)*clamp_level; W3_clamped = clamp.(W3,0.,W3_max)
                signedcolors = (colorant"green1", colorant"white", colorant"magenta")
                imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W3, imgsz, gridcols=7, colors=signedcolors, borderval=0.5, borderwidth=1)
            else
                imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W3,imgsz,borderwidth=1)
            end
            x_abss, xw_abss, xh_abss, f_xs, f_rel, semisympen, regW, regH = getdata(trs)
            ax = plotW([log10.(x_abss) log10.(f_xs) log10.(xw_abss) log10.(xh_abss)],fprefix*"_iter$(iter)_rt$(rt2)_log10plot.png"; title="convergence (SCA)", xlbl = "iteration",
                    ylbl = "log10(penalty)", legendstrs = ["log10(x_abs)", "log10(f_x)", "log10(xw_abs)", "log10(xh_abs)"], legendloc=1, separate_win=false)
                    #,axis=[480,1000,-0.32,-0.28])
            ax = plotW(log10.(f_xs), fprefix*"_iter$(iter)_rt$(rt2)_fxplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(penalty)",
                legendstrs = ["log(f_x)"], legendloc=1, separate_win=false)
            ax = plotW([log10.(x_abss) log10.(xw_abss) log10.(xh_abss) ], fprefix*"_iter$(iter)_rt$(rt2)_xplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
                legendstrs = ["log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
            ax = plotW(H3[1:8,1:100]', fprefix*"_iter$(iter)_rt$(rt2)_Hplot.png"; title="convergence (SCA)", xlbl = "iteration", ylbl = "log(x_abs)",
                legendstrs = ["log(x_abs)", "log(xw_abs)", "log(xh_abs)"], legendloc=1, separate_win=false)
        end
    end
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

# CD Test
using NMF
datart1scd = []; datart2scd = []; mssdscd = []
for ncells in [15]
    println("ncells=$ncells")
    runtime1 = @elapsed W88, H88 = NMF.nndsvd(X, ncells, variant=:ar)
    datart1cd=[]; datart2cd=[]; mssdcd=[]
    for α in [0] # best α = 0.1
        println("α=$(α)")
        runtime2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=1000, α=α, l₁ratio=0.5), X, W88, H88) # 2.5sec
        normalizeWH!(W88,H88)
        mssd88, ml88, ssds = matchednssd(gtW,W88)
        mssdH88 = ssdH(ml88,gtH,H88')
        push!(datart1cd, runtime1)
        push!(datart2cd, runtime2)
        push!(mssdcd, mssd88)
        #imshowW(W88,imgsz)
        imsaveW("CD_nc$(ncells)_a$(α)_W.png", W88, imgsz, borderwidth=1)
        imsaveH("CD_nc$(ncells)_a$(α)_H.png", H88, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
        neworder = zeros(Int,length(ml88))
        for (gti, i) in ml88
            neworder[gti]=i
        end
        for i in 1:ncells
            i ∉ neworder && push!(neworder,i)
        end
        W3cd,H3cd = copy(W88[:,neworder]), copy(H88[neworder,:])
        imsaveW("CD_nc$(ncells)_a$(α)_W_matched.png", W3cd, imgsz, borderwidth=1)
        imsaveH("CD_nc$(ncells)_a$(α)_H_matched.png", H3cd, 100, colors=(colorant"green1", colorant"white", colorant"magenta"))
    end
    push!(datart1scd, datart1cd)
    push!(datart2scd, datart2cd)
    push!(mssdscd, mssdcd)
end
datart1cd = getindex.(datart1scd,1)
datart2cd = getindex.(datart2scd,1)
datartcd = datart1cd+datart2cd


#================== Convex.jl ==================#
function plotP(penP1, penP2, penP3, penP4, penP5, extent, vmaxP)
    # ticks = [-π, -π/2, 0, π/2, π]
    # ticklabels = ["-π", "-π/2", "0", "π/2", "π"]

    fig, axs = plt.subplots(1, 6, figsize=(16, 4), gridspec_kw=Dict("width_ratios"=>[1,1,1,1,1,0.1]))

    ax = axs[1]
    himg1 = ax.imshow(reverse(penP1; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("total")

    ax = axs[2]
    himg2 = ax.imshow(reverse(penP2; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    #ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("∥A*x - b∥²")

    ax = axs[3]
    himg3 = ax.imshow(reverse(penP3; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    #ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("∥(B*x)₊∥²")

    ax = axs[4]
    himg4 = ax.imshow(reverse(penP4; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    #ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("|C*x|")

    ax = axs[5]
    himg5 = ax.imshow(reverse(penP5; dims=1); extent = extent, vmin=0, vmax=vmaxP)
    ax.set_xlabel("x1")
    #ax.set_ylabel("x2")
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels)
    ax.set_title("∥x∥")

    ax = axs[6]
    plt.colorbar(himg1, ax=ax, fraction=1.0, extend="max")
    ax.set_visible(false)

    axs
end

function penP1P2P3P4(penfn1, penfn2, penfn3, penfn4, x, x1s, x2s)
    [penfn1([x1+x[1], x2+x[2]])+penfn2([x1+x[1], x2+x[2]])+penfn3([x1+x[1], x2+x[2]])+penfn4([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s],
    [penfn1([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s],
    [penfn2([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s],
    [penfn3([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s],
    [penfn4([x1+x[1], x2+x[2]]) for x1 in x1s, x2 in x2s]
end

using Convex, ECOS
for i = 1:100
    @show i
    A = randn(2, 2).-0.5; b = randn(2) .- 0.5; B=randn(2,2) .- 0.5; C = randn(2, 2) .- 0.5;
    x = Variable(2)
    pen_data(x) = norm(A*x - b,2)^2
    pen_nn(x) = norm(min.(0,A*x))^2
    pen_spars1(x) = norm(C*x, 1)
    pen_spars2(x) = norm(x, 2)
    problem = minimize(sumsquares(A*x - b) + sumsquares(min(0,B*x)) + sumsquares(x))
    solve!(problem, ECOS.Optimizer; silent_solver = true)
    xsol = x.value
end

A = [  0.642854 0.406794;-0.728747 -1.3357]
b = [-0.7763834317899838, -0.4248008996625347]
B = [ -0.216811 -0.362751; -1.15347 -0.0527866]
C = [ -0.0681416 0.739463; -1.02285 -0.154347]
x = Variable(2)
pen_data(x) = norm(A*x - b,2)^2
pen_nn(x) = norm(min.(0,B*x))^2
pen_spars1(x) = norm(C*x, 1)
pen_spars2(x) = norm(x, 2)
problem = minimize(sumsquares(A*x - b) + sumsquares(min(0,B*x)) + sumsquares(x))
solve!(problem, ECOS.Optimizer; silent_solver = true)
xsol = x.value

n = 10; vmax = 6
pensum, penP1, penP2, penP3, penP4 = penP1P2P3P4(pen_data, pen_nn, pen_spars1, pen_spars2, xsol, -n:0.1:n , -n:0.1:n)
extent = (xsol[1]-n,xsol[1]+n,xsol[2]-n,xsol[2]+n)
axs = plotP(pensum, penP1, penP2, penP3, penP4, extent, vmax)
ax = axs[1]
ax.plot([xsol[1]], [xsol[2]], color="red", marker="x", linewidth = 3)

plt.savefig("convex2dplot.png")

slope = 4; αs = -1:0.01:1
fs = zeros(length(αs),5)
for (i,α) in enumerate(αs)
    x2 = α*10
    x1 = slope*α+3
    x = [x1,x2]
    fs[i,2] = pen_data(x) 
    fs[i,3] = pen_nn(x)
    fs[i,4] = pen_spars1(x)
    fs[i,5] = pen_spars2(x)
    fs[i,1] = fs[i,2]+100fs[i,3]+2fs[i,4]+2fs[i,5]
end
plotW(-10:0.1:10,fs,"convex_plot.png"; title="", xlbl = "α", ylbl = "penalty",
                legendstrs = ["1","1","1","1","1"], legendloc=0, arrange=:horizontal)
