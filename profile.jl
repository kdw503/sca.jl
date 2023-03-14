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

using ProfileView

#=============== Orthogonality ==================================#
dataset = :fakecells; SNR = 0; ncells = 15
ls_method = :none; initmethod = :svd; initfn = SCA.nndsvd2; initpwradj = :balance
sd_group = :whole; optimmethod = :optim_lbfgs; objective = :normal; regularization = :W1
approx = true;
order = 1; useRelaxedL1=true; λ1 = λ2 = 0; β1 = β2 = 1; approxHo = false; maxiter = 100; inner_maxiter = 100; tol=1e-7
show_trace=false; store_trace=true; savefigure = true
makepositive = false

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure));
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, d = initsemisca(X, ncells, poweradjust=initpwradj,
            initmethod=initmethod, initfn=initfn);

σ0 = 100*round(std(W0)^2,digits=5); r=0.1
# ocasolve!
Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
stparams = StepParams(sd_group=:component, optimmethod=optimmethod, approx=approx,
        β1=β1, β2=0, λ1=λ1, λ2=0, reg=regularization, order=order,
        useRelaxedL1=true, σ0=100*std(W0)^2, r=0.1, objective=objective)
lsparams = LineSearchParams(method=ls_method, c=0.5, α0=1.0, ρ=0.5, maxiter=50, show_lsplot=true,
        iterations_to_show=[1])
cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=2, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:0)
GC.gc()
@profview W1, objvals, trs, nitersum, fxss, xdiffss, ngss = ocasolve!(W0, Mw, d; stparams=stparams,
                                                        lsparams=lsparams, cparams=cparams);
# scasolve!
Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
stparams = StepParams(sd_group=sd_group, optimmethod=optimmethod, approx=approx,
        β1=β1, β2=β2, λ1=λ1, λ2=λ2, reg=regularization, order=order, M2power=M2power,
        useRelaxedL1=false, σ0=100*std(W0)^2, r=0.1, hfirst=true, processorder=:none,
        poweradjust=pwradj, rectify=rectify, objective=objective)
lsparams = LineSearchParams(method=ls_method, c=0.5, α0=2.0, ρ=0.5, maxiter=maxiter, show_lsplot=false,
        iterations_to_show=[15])
cparams = ConvergenceParams(allow_f_increases = false, f_abstol = tol, f_reltol=tol, f_inctol=1e2,
        x_abstol=tol, successive_f_converge=1, maxiter=maxiter, inner_maxiter=inner_maxiter, store_trace=true,
        store_inner_trace=true, show_trace=false, show_inner_trace=false, plotiterrng=1:0, plotinneriterrng=1:0)
GC.gc()
@profview W1, H1, objvals, trs, nitersum, fxss, xdiffss, ngss = scasolve!(W0, H0, d, Mw, Mh; stparams=stparams,
                                                            lsparams=lsparams, cparams=cparams);

W2,H2 = copy(W0*Mw), copy(Mh*H0)

# save images
normalizeW!(W2,H2); W3,H3 = sortWHslices(W2,H2)
makepositive && flip2makepos!(W3,H3)
Hostr = optimmethod == :optim_newton ? (approxHo ? "_approxHo" : "_exactHo") :
                        (optimmethod == :optim_lbfgs ? "_lbfgs" : "")
imsave_data(dataset,"$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",W3,H3,imgsz,100; saveH=true)
plot_convergence("$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",x_abss,f_xs; title="convergence (SCA)")
plotH_data(dataset,"$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",H3)
