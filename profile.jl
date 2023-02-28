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
order = 1; useRelaxedL1=true; λ1 = 0; β1 = 1; approxHo = false; maxiter = 100; innermaxiter = 100; tol=1e-7
show_trace=false; store_trace=true; savefigure = true
makepositive = false

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure));
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, d = initsemisca(X, ncells, poweradjust=initpwradj,
            initmethod=initmethod, initfn=initfn);

σ0 = 100*round(std(W0)^2,digits=5); r=0.1
Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
# minOrthogMw!
@profview Mw, Mh, d, f_xs, x_abss, iter, trs = minOrthogMw!(W0,Mw,d,λ1,β1,σ0,r,order,maxiter,innermaxiter,tol;
                    approxHo=approxHo, useRelaxedL1=useRelaxedL1, optimmethod=optimmethod, sd_group=sd_group,
                    show_trace=show_trace, store_trace=store_trace, show_plot=true, plotiterrng=1:0,
                    plotinneriterrng=1:1);

Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
rt2 = @elapsed Mw, Mh, d, f_xs, x_abss, iter, trs = minOrthogMw!(W0,Mw,d,λ1,β1,σ0,r,order,maxiter,innermaxiter,tol;
                    approxHo=approxHo, useRelaxedL1=useRelaxedL1, optimmethod=optimmethod, sd_group=sd_group,
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
dataset = :fakecell; SNR = -25; ncells = 15
method=:sca; ls_method = :none; initmethod = :svd; initfn = SCA.nndsvd2; initpwradj = :balance
sd_group = :whole; optimmethod = :optim_lbfgs; objective = :normal; regularization = :W1
order = 1; Msparse=false; useRelaxedL1=true; λ1 = 0;  λ2 = 0; β1 = 1; β2 = 0; approxHo = false;
maxiter = 1000; innermaxiter = 100; tol=1e-5
show_trace=false; store_trace=true; savefigure = true
makepositive = false

X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure));
# Median filter
rsimg = reshape(X,imgsz...,lengthT)
rsimgm = mapwindow(median!, rsimg, (3,3,1));
X = reshape(rsimgm,*(imgsz...),lengthT)
# Initialize
rt1 = @elapsed W0, H0, Mw0, Mh0, Wp, Hp, d = initsemisca(X, ncells, poweradjust=initpwradj,
            initmethod=initmethod, initfn=initfn);

σ0 = 100*round(std(W0)^2,digits=5); r=0.1
Mw, Mh = copy(Mw0), copy(Mh0) # reload initialized Mw, Mh
# Initialize with orthogonality
method = :oca; optimmethod = :optim_cg; sd_group = :component; useRelaxedL1 = true
rt2 = @elapsed Mw, Mh, d, f_xs, x_abss, iter, trs = minOrthogMw!(W0,Mw,d,λ1,β1,σ0,r,order,maxiter,innermaxiter,tol;
                    approx=true, useRelaxedL1=useRelaxedL1, optimmethod=optimmethod, sd_group=:component,
                    show_trace=show_trace, store_trace=store_trace, plotiterrng=1:0, plotinneriterrng=1:10);
f_x_abss = []
Mh = Mw\Diagonal(d); Mw1, Mh1 = copy(Mw), copy(Mh)
# minMwMh! (Convex)
Mw, Mh = copy(Mw0), copy(Mh0)
β1 = 1.5; β2 = 0; λ1 = λ2 = 0
maxiter = 100
method = :sca; optimmethod = :convex; sd_group = :component; useRelaxedL1 = false; regularization = :W1M2
M2power=2
rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, traces = minMwMh!(W0,H0,d,Mw,Mh,λ1,λ2,β1,β2,σ0,r,maxiter,Msparse,order;
                    tol=tol, innermaxiter=innermaxiter, optimmethod=optimmethod, useRelaxedL1=useRelaxedL1,
                    reg=regularization, M2power=M2power, show_trace=show_trace, plotiterrng=1:0, poweradjust=:none, sd_group=sd_group,
                    store_trace=false, weighted=:none, decifactor=4)
# minMwMh! (Optim)
σ0 = 100*round(std(W0)^2,digits=5); r=0.1
Mw, Mh = copy(Mw0), copy(Mh0);
β1 = 1; β2 = 0; λ1 = λ2 = 0
maxiter = 500; show_trace = false; tol=1e-5
approx=true # use ForwardDiff
method = :sca; optimmethod = :optim_cg; sd_group = :component; useRelaxedL1 = true; regularization = :W1M2
M2power=1; allow_f_increase=true
rt2 = @elapsed Mw, Mh, f_xs, f_x_abss, x_abss, iter, traces = minMwMh!(W0,H0,d,Mw,Mh,λ1,λ2,β1,β2,σ0,r,maxiter,Msparse,order;
                    tol=tol, innermaxiter=innermaxiter, allow_f_increase=allow_f_increase, optimmethod=optimmethod, useRelaxedL1=useRelaxedL1,
                    approx=approx, reg=regularization, M2power=M2power, show_trace=show_trace, poweradjust=:none, sd_group=sd_group,
                    store_trace=store_trace, weighted=:none, decifactor=4, plotiterrng=1:0, plotinneriterrng=1:10)

W2,H2 = copy(W0*Mw), copy(Mh*H0)

# minMwMh! (Whole)
σ0 = 100*round(std(W0)^2,digits=5); r=0.1
Mw, Mh = copy(Mw0), copy(Mh0);
β1 = β2 = 1; λ1 = λ2 = 0
maxiter = 500; show_trace = false; tol=1e-5
approx=true # if false use ForwardDiff
method = :sca; optimmethod = :optim_lbfgs; sd_group = :whole; useRelaxedL1 = false; regularization = :W1
M2power = 1; allow_f_increase = true
rt2 = @elapsed Mw, Mh, f_xs, f_x_abss, x_abss, iter, traces = minMwMh!(W0,H0,d,Mw,Mh,λ1,λ2,β1,β2,σ0,r,maxiter,Msparse,order;
                    tol=tol, innermaxiter=innermaxiter, allow_f_increase=allow_f_increase, optimmethod=optimmethod, useRelaxedL1=useRelaxedL1,
                    approx=approx, reg=regularization, M2power=M2power, show_trace=show_trace, poweradjust=:none, sd_group=sd_group,
                    store_trace=store_trace, weighted=:none, decifactor=4, plotiterrng=1:0, plotinneriterrng=1:10)

W2,H2 = copy(W0*Mw), copy(Mh*H0)

# save images
normalizeW!(W2,H2); W3,H3 = sortWHslices(W2,H2)
optstr = optimmethod == :optim_newton ? (approxHo ? "_approxHo" : "_exactHo") :
        optimmethod == :optim_lbfgs ? "_lbfgs" :
        optimmethod == :optim_cg ? "_cg" :
        optimmethod == :convex ? "_cvx" : ""
σ0str = @sprintf("%1.3f",σ0)
relaxedL1str = useRelaxedL1 ? "_s$(σ0str)_r$(r)" : ""
imsave_data(dataset,"$(method)$(optstr)_b$(β1)$(relaxedL1str)_$(SNR)dB_filtered_it$(iter)_rt$(rt2)",W3,H3,imgsz,100; saveH=true)
plot_convergence("$(method)$(optstr)_b$(β1)$(relaxedL1str)_$(SNR)dB_filtered_it$(iter)_rt$(rt2)",x_abss,f_xs,f_x_abss; title="convergence (SCA)")
plotH_data(dataset,"$(method)$(optstr)_b$(β1)$(relaxedL1str)_$(SNR)dB_filtered_it$(iter)_rt$(rt2)",H3)
makepositive = true
makepositive && flip2makepos!(W3,H3)
imsave_data_gt(dataset,"$(method)$(optstr)_b$(β1)$(relaxedL1str)_$(SNR)dB_filtered_it$(iter)_rt$(rt2)_gt",
        W3,H3,datadic["gtW"],imgsz,100; saveH=true)

# CD
dataset = :fakecell
SNRstr = dataset == :fakecell ? "_$(SNR)dB" : ""
initdatastr = "_$(dataset)$(SNRstr)_nc$(ncells)"
rt1 = @elapsed Wcd, Hcd = NMF.nndsvd(X, ncells, variant=:ar)
fprefix0 = "CDinit$(initdatastr)_normalize_rt"*@sprintf("%1.2f",rt1)
fprefix1 = "CD$(initdatastr)_normalize"
α = 0.1
paramstr="_α$(α)"
fprefix2 = fprefix1*"_usingNMF"*paramstr
rt2 = @elapsed result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=0.5, tol=tol), X, Wcd, Hcd)
iter = result.niters
imsave_data_gt(dataset,fprefix2*"_filtered_rt"*@sprintf("%1.4f",rt2),Wcd,Hcd,datadic["gtW"],imgsz,100; saveH=true)
