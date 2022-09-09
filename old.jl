function minpix_old(Mw,Mh,W0,H0,l,k,λ,β,order)
    p = size(W0,2)
    x = Variable(1)
    # invertibility
    mwk = Mw[:,k]; mwlk=mwk[l]; mhl = Mh[l,:]
    Eprev = I-Mw*Mh; Eprevl = Eprev[l,:]; El = Eprevl + mwlk*mhl
    invertibility = sumsquares(El-x*mhl)
    # sparsity and non-negativity
    w0l = W0[:,l]; w0mwk = W0*mwk; w0mwlk = w0mwk-w0l*mwlk; w0mwlkx = w0mwlk+w0l*x
    sparsity = order == 1 ? norm(w0mwlkx, 1) : sumsquares(w0mwlkx)
    nnegativity = sumsquares(max(0,-w0mwlkx))
    # solve
    problem = minimize(invertibility + λ*nnegativity + β*sparsity)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    xsol = x.value
    # errprev = norm(Eprevl)^2; err = norm(El-xsol*mhk)^2
    # sparsprev = order == 1 ? β*norm(w0mwk,1) : β*norm(w0mwk)^2
    # spars = order == 1 ? β*norm(w0mwlk+w0l*xsol,1) : β*norm(w0mwlk+w0l*xsol)^2
    # nnegprev = λ*norm(min.(0,w0mwk))^2
    # nneg = λ*norm(min.(0,w0mwlk+w0l*xsol))^2
    # errprev+sparsprev+nnegprev < err+spars+nneg && @show problem.status
    xsol
end
