function penalty(Mw,Mh,W0,H0,λw,λh,βw,βh; order=1)
    pen = norm(I-Mw*Mh)^2 + λw*sca2(W0*Mw)+λh*sca2(Mh*H0)
    spar = order==1 ? βw*norm(W0*Mw,1) + βh*norm(Mh*H0,1) :
                      βw*norm(W0*Mw)^2 + βh*norm(Mh*H0)^2
    pen+spar
end
penaltyL1(Mw,Mh,W0,H0,λw,λh,βw,βh) = penalty(Mw,Mh,W0,H0,λw,λh,βw,βh; order=1)
penaltyL2(Mw,Mh,W0,H0,λw,λh,βw,βh) = penalty(Mw,Mh,W0,H0,λw,λh,βw,βh; order=2)

penaltyW(Mw,Mh,W0,λ,βw; order=1) =
    norm(I-Mw*Mh)^2 + λ*sca2(W0*Mw) + βw*(order==1 ? norm(W0*Mw,1) : norm(W0*Mw)^2)
penaltyL1W(Mw,Mh,W0,λ,βw) = penaltyW(Mw,Mh,W0,λ,βw; order=1)
penaltyL2W(Mw,Mh,W0,λ,βw) = penaltyW(Mw,Mh,W0,λ,βw; order=2)

function penaltyWij(x,Mw,Mh,W0,λ,βw,l,k; order=1)
    mwk = Mw[:,k]; mwlk=mwk[l]; mhk = Mh[k,:]
    Edprev = I-Mw*Mh; Edprevml = copy(Edprev); Edprevml[l,:] .= 0
    Edprevl = Edprev[l,:]; El = Edprevl + mwlk*mhk
    w0lmk = W0*Mw; w0lmk[:,k].=0; w0mwk = W0*mwk
    w0l = W0[:,l]; w0mwlk = w0mwk-w0l*mwlk
    function Evalfn(x)
        w0mwlkx = w0mwlk+w0l*x
        sparsity = order == 1 ? norm(w0mwlkx, 1) : norm(w0mwlkx,2)^2
        nnegativity = norm(max.(0,-w0mwlkx))^2
        invertibility = norm(El-x*mhk,2)^2
        invertibility+λ*nnegativity+βw*sparsity
    end
    constE = norm(Edprevml)^2
    constE += order == 1 ? βw*norm(w0lmk, 1) : βw*norm(w0lmk,2)^2
    constE += λ*norm(max.(0,-w0lmk))^2
    Evalfn(x)+constE
end
penaltyL1Wij(x,Mw,Mh,W0,λ,βw,l,k) = penaltyWij(x,Mw,Mh,W0,λ,βw,l,k; order=1)
penaltyL2Wij(x,Mw,Mh,W0,λ,βw,l,k) = penaltyWij(x,Mw,Mh,W0,λ,βw,l,k; order=2)

function minpix_convex(Mw,Mh,W0,H0,l,k,λ,β,order; show_figure=false, row=1, col=1)
    p = size(W0,2)
    x = Variable(1)
    # invertibility
    mwk = Mw[:,k]; mwlk=mwk[l]; mhl = Mh[l,:]
    Edprev = I-Mw*Mh; Edprevl = Edprev[l,:]; El = Edprevl + mwlk*mhl
    invertibility = sumsquares(El-x*mhl)
    # sparsity and non-negativity
    w0l = W0[:,l]; w0mwk = W0*mwk; w0mwlk = w0mwk-w0l*mwlk
    w0mwlkx = w0mwlk + x*w0l
    sparsity = order == 1 ? norm(w0mwlkx, 1) : sumsquares(w0mwlkx)
    nnegativity = sumsquares(max(0,-w0mwlkx))
    # solve
    problem = minimize(invertibility + λ*nnegativity + β*sparsity)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    xsol = x.value

    function Evalfn(x)
        w0mwlkx = w0mwlk+w0l*x
        sparsity = order == 1 ? norm(w0mwlkx, 1) : norm(w0mwlkx,2)^2
        nnegativity = norm(max.(0,-w0mwlkx),2)^2
        norm(El-x*mhl,2)^2+λ*nnegativity+β*sparsity
    end

    if show_figure && l==row && k==col
        Eprev = Evalfn(mwlk)
        Eopt = Evalfn(xsol)
        xs = range(min(xsol-1,mwlk),max(xsol+1,mwlk),length=1000)
        ys = Evalfn.(xs); maxys = maximum(ys)
        fig, ax = plt.subplots(1,1, figsize=(5,4))
        ax.plot(xs, ys, color="orange")
        for x in xcross
            if x > xs[1] && x < xs[end]
                Eval = Evalfn(x)
                xs = [x,x]; ys = [Eopt-1,Eval]
                ax.plot(xs,ys,color="blue")
            end
        end
        # old value
        xs = [mwlk,mwlk]; ys = [Eopt-1,maxys]
        ax.plot(xs,ys,color="black")
        # new value
        xs = [xsol,xsol]; ys = [Eopt-1,maxys]
        @show xsol, mwlk, Eopt, Eprev
        ax.plot(xs,ys,color="red")
        xlabel("x",fontsize = 12)
        ylabel("E",fontsize = 12)
        savefig("x$(Eopt).png")
        close("all")
    end
    xsol
end

function minpix(Mw,Mh,W0,l,k,λ,β,order; show_figure=false, fpix="", row=1, col=1)
    m,p = size(W0)

    mwk = Mw[:,k]; mwlk=mwk[l]; mhk = Mh[k,:]
    Edprev = I-Mw*Mh; Edprevl = Edprev[l,:]; El = Edprevl + mwlk*mhk
    w0l = W0[:,l]; w0mwk = W0*mwk; w0mwlk = w0mwk-w0l*mwlk; xcross = -w0mwlk./w0l

    function Evalfn(x)
        w0mwlkx = w0mwlk+w0l*x
        sparsity = order == 1 ? norm(w0mwlkx, 1) : norm(w0mwlkx,2)^2
        nnegativity = norm(max.(0,-w0mwlkx),2)^2
        norm(El-x*mhk,2)^2+λ*nnegativity+β*sparsity
    end

    Eprev = Evalfn(mwlk)
    xsol = mwlk; Esol = Eprev
    # invertibility Ed=sum(avec*x^2+2*bvec*x+..)
    asum = mhk'*mhk; bsum = -mhk'*El
    if (order == 1 && β != 0) || λ != 0
        si = sortperm(xcross)
        signw0l = sign.(w0l[si]); signs = fill(-1,m)
        xmax = -Inf
        for i in 1:length(xcross)
            xmin=xmax; xmax=xcross[si[i]]
            xmin==xmax && (signs[i] = 1; continue)
            if λ != 0
                flags = (signs.-signw0l).*0.5
                avecn = flags.*w0l[si]; bvecn = flags.*w0mwlk[si]
                avecnn = avecn.*avecn; bvecnn = avecn.*bvecn
                # nnegativity Enn=sum(avecnn*x^2+2*bvecnn*x+..)
                asum += λ*sum(avecnn); bsum += λ*sum(bvecnn)
            end
            if order == 1 && β != 0
                # L1 sparsity Es=sum(bvecL1*x+..)
                bvecL1 = signs.*signw0l.*w0l[si]
                # cvecL1 = signs.*signw01.*w0mwlk[si]
                bsum += β*sum(bvecL1)
            end
            xsglr = -bsum/asum
            xcand = xsglr > xmin ? (xsglr <= xmax ? xcand = xsglr : xcand = xmax) : xmin
            Ecand = Evalfn(xcand)
            Ecand < Esol && (xsol = xcand; Esol = Ecand)
            signs[i] = 1
        end
    else
        # L2 sparsity Es=sum(avecL2*x^2+2*bvecL2*x+..)
        avecL2sum = w01'*w01; bvecL2sum = w01'*w0mwlk
        # total E=sum(avec)*x^2+2*sum(bvec)*x+..
        asum += β*avecL2sum; bvec += β*bvecL2sum
        xcand = -bsum/asum
        Ecand = Evalfn(xcand)
        Ecand < Esol && (xsol = xcand; Esol = Ecand)
    end

    if show_figure && l==row && k==col
        xs = range(min(xsol-1,mwlk),max(xsol+1,mwlk),length=1000)
        ys = Evalfn.(xs); maxys = maximum(ys)
        fig, ax = plt.subplots(1,1, figsize=(5,4))
        ax.plot(xs, ys, color="orange")
        for x in xcross
            if x > xs[1] && x < xs[end]
                Eval = Evalfn(x)
                xs = [x,x]; ys = [Esol-1,Eval]
                ax.plot(xs,ys,color="blue")
            end
        end
        # old value
        xs = [mwlk,mwlk]; ys = [Esol-1,maxys]
        ax.plot(xs,ys,color="black")
        # new value
        xs = [xsol,xsol]; ys = [Esol-1,maxys]
        ax.plot(xs,ys,color="red")
        xlabel("x",fontsize = 12)
        ylabel("E",fontsize = 12)
        savefig(fpix*"_r$(row)_c$(col).png")
        close("all")
#        pen = penaltyWij(xsol,Mw,Mh,W0,λ,β,l,k; order=order)
#        @show pen
    end
    xsol
end

using GLPK, ECOS
function mincol_convex(Mw,Mh,W0,H0,k,λ,β,order; verbose=false, show_figure=false, xmin=-10, xmax=10, fpix="", col=1)
    p = size(W0,2)
    Eprev = I-Mw*Mh; mhk = Mh[k,:]'; E = Eprev + Mw[:,k]*mhk

    # Convex : set variable
    x = Variable(p)
    set_value!(x, Mw[:,k])

    # Convex : set problem
    invertibiliity = sumsquares(E-x*mhk)
    sparsity = order == 1 ? norm(W0*x, 1) : sumsquares(W0*x)
    nnegativity = sumsquares(max(0,-W0*x))
    expr = invertibiliity + λ*nnegativity + β*sparsity
    problem = minimize(expr)
    Evalpre = Convex.evaluate(expr)
    # println("expression curvature = ", vexity(expr))
    # println("expression sign = ", sign(expr))

    # Convex : solve
    solve!(problem, ECOS.Optimizer; warmstart = false, silent_solver = true) 
    # other solver options : SCS, ECOS, (GLPK : run error), (Gurobi, Mosek : precompile error)
    # verbose=false (turn off warning)
    # warmstart doesn't work for SCS.GeometricConicForm and ECOS

    # Convex : check the result
    # @show round.(Convex.evaluate(x), digits = 2)
    Eval = problem.optval # round(problem.optval, digits = 10)
    Evalfromxsol = Convex.evaluate(expr) # round(Convex.evaluate(expr), digits = 10)
    fx_increased = false
    if Evalfromxsol > Evalpre + 1e-10 # sometime these two values are quite different in SCS
        @show Evalpre, Evalfromxsol, Eval
        @warn("Optimum variable is inaccurate")
        fx_increased = true
    end
    Evalpre < Eval && @show problem.status

    if show_figure && (fx_increased || k ∈ col)
        @show k, col
        mwk = Mw[:,k]; grad = x.value - mwk
        # pen(α) = (xvec=α*grad+mwk; norm(E-xvec*mhk)^2 + λ*norm(max.(0,-W0*xvec))^2
        #             + β*(order==1 ? norm(W0*xvec, 1) : norm(W0*xvec)^2))
        pen_convex(α) = (xvec=α*grad+mwk; set_value!(x,xvec); Convex.evaluate(expr))
        fig, ax = plt.subplots(1,1, figsize=(5,4))
        # function values
        αs = range(xmin,xmax,length=1000)
        ys = pen_convex.(αs); maxys = maximum(ys)
        ax.plot(αs, ys, color="orange")
        # previous point
        αs = [0,0]; ys = [0,maxys]
        ax.plot(αs,ys,color="black")
        # optimum point
        αs = [1,1]; ys = [0,maxys]
        ax.plot(αs,ys,color="red")
        xlabel("α",fontsize = 12); ylabel("E(α)",fontsize = 12)
        savefig(fpix*"$(k).png")
        close("all")

        # Check surrounding points of the optimum to see if this is optimum
        # (This check is not sufficient but necessary)
        # set_value!(x,1.1*grad+mwk)
        # Evalnear1p1 = Convex.evaluate(expr)
        # set_value!(x,0.9*grad+mwk)
        # Evalnear0p9 = Convex.evaluate(expr)
        # @show Evalnear0p9, Eval, Evalnear1p1
    end
    fx_increased ? Mw[:,k] : x.value
end

function minMw_pixel!(Mw,Mh,W0,H0,λ,β,order; show_figure=false, fpix="", row=1, col=1)
    p = size(Mw,2)
    for k in 1:p
        for l in 1:p
            #penb4 = penaltyWij(Mw[l,k],Mw,Mh,W0,λ,β,l,k; order=order)
            Mw[l,k] = minpix(Mw,Mh,W0,l,k,λ,β,order,show_figure=show_figure,fpix=fpix,row=row,col=col)
            # pen = penaltyWij(Mw[l,k],Mw,Mh,W0,λ,β,l,k; order=order)
            # if penb4 < pen - 1e-6
            #     @show k, l, penb4, pen
            # end
        end
    end
end

function minMw_cbyc!(Mw,Mh,W0,H0,λ,β,order; show_figure=false, fpix="", xmin=-10, xmax=10, col=1)
    p = size(Mw,2)
    for k in 1:p
        #@show k
        penb4 = penaltyW(Mw,Mh,W0,λ,β; order=order)
        Mw[:,k] = mincol_convex(Mw,Mh,W0,H0,k,λ,β,order,show_figure=show_figure,fpix=fpix,xmin=xmin,xmax=xmax, col=col)
        pen = penaltyW(Mw,Mh,W0,λ,β; order=order)
        if penb4 < pen - 1e-6
            @show k, penb4, pen
        end
    end
end

function minMw_ac!(Mw,Mh,W0,H0,λ,β,order)
    m, p = size(W0)
    x = Variable(p^2)
    Ivec = vec(Matrix(1.0I,p,p))
    A = zeros(p^2,p^2)
    SCA.directMw!(A, Mh) # vec(I-reshape(x,p,p)*Mh) == Ivec-A*x
    Aw = zeros(m*p,p^2); bw = zeros(m*p)
    SCA.direct!(Aw, bw, W0; allcomp = false) # vec(W*reshape(x,p,p)) == (Aw*x)
    spars = order == 1 ? norm(Aw*x, 1) : sumsquares(Aw*x)
    problem = minimize(sumsquares(Ivec-A*x)+ λ*sumsquares(max(0,-A*x)) + β*spars)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    Mw[:,:] .= reshape(x.value,p,p)
end

function minMwMh(Mw,Mh,W0,H0,λ1,λ2,β1,β2,maxiter,order; fprefix="", sd_group=:pixel, imgsz=(40,20), SNR=60)
    normw2 = norm(W0)^2; normh2 = norm(H0)^2
    λw = λ1/normw2; λh = λ2/normh2
    βw = order==1 ? β1/norm(W0,1) : β1/normw2; βh = order==1 ? β2/norm(H0,1) : β2/normh2

    f_xs=[]; x_abss=[]
    iter = 0
    while iter < maxiter
        iter += 1
        @show iter
        Mwprev, Mhprev = copy(Mw), copy(Mh)
        if iter==1
            show_figure = true
        else
            show_figure = true
        end
        if sd_group == :column
            minMw_cbyc!(Mw,Mh,W0,H0,λw,βw,order; show_figure=show_figure, fpix="W_cbyc_iter$(iter)_c", xmin=-5, xmax=5, col=[])
            minMw_cbyc!(Mh',Mw',H0',W0',λh,βh,order; show_figure=show_figure, fpix="H_cbyc_iter$(iter)_r", xmin=-5, xmax=5, col=[])
        elseif sd_group == :component
            minMw_ac!(Mw,Mh,W0,H0,λw,βw,order)
            minMw_ac!(Mh',Mw',H0',W0',λh,βh,order)
        elseif sd_group == :pixel
            minMw_pixel!(Mw,Mh,W0,H0,λw,βw,order; show_figure=show_figure, fpix="W_iter$(iter)", row=9, col=11)
            minMw_pixel!(Mh',Mw',H0',W0',λh,βh,order; show_figure=show_figure, fpix="H_iter$(iter)", row=9, col=11)
        else
            error("Unsupproted sd_group")
        end            
        pen = penalty(Mw,Mh,W0,H0,λw,λh,βw,βh; order=order)
        x_abs = norm(Mwprev-Mw)^2*norm(Mhprev-Mh)^2
        @show iter, x_abs, pen
        if isnan(x_abs)
            Mw, Mh = copy(Mwprev), copy(Mhprev)
            iter -= 1
            break
        end
        push!(f_xs, pen)
        push!(x_abss, x_abs)
        W2,H2 = copy(W0*Mw), copy(Mh*H0)
        normalizeWH!(W2,H2)
        if iter%10 == 0
            @show iter
            imsaveW(fprefix*"_iter$(iter).png",sortWHslices(W2,H2)[1],imgsz,borderwidth=1)
        end
    end
    Mw, Mh, f_xs, x_abss, iter
end
