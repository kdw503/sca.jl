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
dataset = :fakecell; SNR = 60; ncells = 15
ls_method = :none; initmethod = :svd; initfn = SCA.nndsvd2; initpwradj = :balance
sd_group = :whole; optimmethod = :optim_lbfgs; objective = :normal; regularization = :W1
order = 1; usingRelaxedL1=true; λ1 = 0; β1 = 1; approxHo = false; maxiter = 100; innermaxiter = 100; tol=1e-7
show_trace=false; store_trace=true; savefigure = true
makepositive = false

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure));
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, d = initsemisca(X, ncells, poweradjust=initpwradj,
            initmethod=initmethod, initfn=initfn);

σ0 = 100*round(std(W0)^2,digits=5); r=0.1
Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
# minOrthogMw!
@profview Mw, Mh, d, f_xs, x_abss, iter, trs = minOrthogMw!(W0,Mw,d,λ1,β1,σ0,r,order,maxiter,innermaxiter,tol;
                    approxHo=approxHo, usingRelaxedL1=usingRelaxedL1, optimmethod=optimmethod, sd_group=sd_group,
                    show_trace=show_trace, store_trace=store_trace, show_plot=true, plotiterrng=1:0,
                    plotinneriterrng=1:1);

Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
rt2 = @elapsed Mw, Mh, d, f_xs, x_abss, iter, trs = minOrthogMw!(W0,Mw,d,λ1,β1,σ0,r,order,maxiter,innermaxiter,tol;
                    approxHo=approxHo, usingRelaxedL1=usingRelaxedL1, optimmethod=optimmethod, sd_group=sd_group,
                    show_trace=show_trace, store_trace=store_trace, show_plot=true, plotiterrng=1:0,
                    plotinneriterrng=1:1);

W2,H2 = copy(W0*Mw), copy(Mh*H0)

# save images
normalizeW!(W2,H2); W3,H3 = sortWHslices(W2,H2)
makepositive && flip2makepos!(W3,H3)
Hostr = optimmethod == :optim_newton ? (approxHo ? "_approxHo" : "_exactHo") :
                        (optimmethod == :optim_lbfgs ? "_lbfgs" : "")
imsave_data(dataset,"$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",W3,H3,imgsz,100; saveH=true)
plot_convergence("$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",x_abss,f_xs; title="convergence (SCA)")
plotH_data(dataset,"$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",H3)


#=============== Invertibility ==================================#
dataset = :fakecell; SNR = 20; ncells = 15
method=:sca; ls_method = :none; initmethod = :nndsvd; initfn = SCA.nndsvd2; initpwradj = :balance
sd_group = :whole; optimmethod = :optim_lbfgs; objective = :normal; regularization = :W1
order = 1; Msparse=false; usingRelaxedL1=true; λ1 = 0;  λ2 = 0; β1 = 1; β2 = 0; approxHo = false;
maxiter = 400; innermaxiter = 100; tol=1e-7
show_trace=true; store_trace=true; savefigure = true
makepositive = false

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure));
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, d = initsemisca(X, ncells, poweradjust=initpwradj,
            initmethod=initmethod, initfn=initfn);

σ0 = 100*round(std(W0)^2,digits=5); r=0.1
Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
# Initialize with orthogonality
rt2 = @elapsed Mw, Mh, d, f_xs, x_abss, iter, trs = minOrthogMw!(W0,Mw,d,λ1,β1,σ0,r,order,maxiter,innermaxiter,tol;
                    approxHo=approxHo, usingRelaxedL1=usingRelaxedL1, optimmethod=optimmethod, sd_group=:whole,
                    show_trace=show_trace, store_trace=store_trace, show_plot=true, plotiterrng=1:0,
                    plotinneriterrng=1:1);
Mh = Mw\Diagonal(d); Mw1, Mh1 = copy(Mw), copy(Mh)
# minMwMh!
Mw, Mh = copy(Mw1), copy(Mh1)
optimmethod = :optim_newton; sd_group = :component; useConvex = true; usingRelaxedL1 = false; regularization = :W1M2
rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, traces = minMwMh!(W0,H0,d,Mw,Mh,λ1,λ2,β1,β2,σ0,r,maxiter,Msparse,order;
                    tol=tol, innermaxiter=innermaxiter, optimmethod=optimmethod, usingRelaxedL1=usingRelaxedL1,
                    show_trace=show_trace, plotiterrng=1:0, reg=regularization, poweradjust=:none, sd_group=sd_group,
                    show_lsplot=false, store_trace=false, useConvex=useConvex, weighted=:none, decifactor=4)

W2,H2 = copy(W0*Mw), copy(Mh*H0)

# save images
normalizeW!(W2,H2); W3,H3 = sortWHslices(W2,H2)
makepositive && flip2makepos!(W3,H3)
Hostr = optimmethod == :optim_newton ? (approxHo ? "_approxHo" : "_exactHo") :
                        (optimmethod == :optim_lbfgs ? "_lbfgs" : "")
imsave_data(dataset,"$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",W3,H3,imgsz,100; saveH=true)
plot_convergence("$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",x_abss,f_xs; title="convergence (SCA)")
plotH_data(dataset,"$(method)$(Hostr)_b$(β1)_s$(σ0)_r$(r)_$(SNR)dB_it$(iter)_rt$(rt2)",H3)
