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

#ARGS =  ["[:cbclface]","[0]","50","[2,5,10]",":balance"]
#ARGS =  ["[:onoffnatural]","[0]","1","[0.001]",":balance"]

datasets = eval(Meta.parse(ARGS[1])); SNRs = eval(Meta.parse(ARGS[2]));
maxiter = eval(Meta.parse(ARGS[3])); βs = eval(Meta.parse(ARGS[4]));
initpwradj = eval(Meta.parse(ARGS[5])); nclsrng=[15]
savefigure = true; show_trace = true
dataset=datasets[1]; SNR=SNRs[1]; ncls=nclsrng[1]; β=βs[1]
for dataset in datasets
    @show dataset
    for SNR in SNRs # this can be change ncellss, factors
        X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure))
        SNRstr = dataset == :fakecell ? "_$(SNR)dB" : ""
        for ncls in nclsrng, β in βs
        @show SNR, β
        ncells = dataset ∈ [:fakecell, :neurofinder] ? ncls : ncells0
        initdatastr = "_$(dataset)$(SNRstr)_nc$(ncells)"
        rt1 = @elapsed W0, H0, Mw, Mh, D, Epre, E, _ = initMwMhSparse(X, ncells, β, poweradjust=initpwradj,
                                                maxiter=maxiter,show_trace=show_trace)
        fprefix0 = "SCAinitsp$(β)$(initdatastr)_$(initpwradj)_iter$(maxiter)"
        Mw0, Mh0 = copy(Mw), copy(Mh)
        W1,H1 = W0*Mw, Mh*H0; normalizeWH!(W1,H1)
        savefigure && imsave_data(dataset,fprefix0,W1,H1,imgsz,lengthT; saveH=false)
        save(fprefix0*".jld","X",X,"W0",W0,"H0",H0,"Mw",Mw,"Mh",Mh,"D",D,"β",β,"rt1",rt1)
        end
    end
end

#=
ncells = 15; Imat = Matrix(1.0I,ncells,ncells); βw=1.0; Y=Imat
Mw = Variable(ncells,ncells); set_value!(Mw, Imat)
expr = expr = norm(vec(Mw'*Mw-Y),1)+βw*norm(vec(W0*Mw), 1)
vexity(expr)

invpen(Mw,D) = norm(Mw'*Mw-Diagonal(D))^2
sparsepen(Mw,W0) = norm(W0*Mw,1)

fprefixs = [
     "SCAinitsp1.0e-5_cbclface_nc49_balance",
     "SCAinitsp1.0e-5_cbclface_nc49_wh_normalize",
     "SCAinitsp0.001_cbclface_nc49_balance",
     "SCAinitsp0.001_cbclface_nc49_wh_normalize",
     "SCAinitsp0.1_cbclface_nc49_balance",
     "SCAinitsp0.1_cbclface_nc49_wh_normalize",
    ]
for fprefix in fprefixs
    fname = fprefix*".jld"
    dd = load(fname)
    W0 = dd["W0"]; Mw = dd["Mw"]; W = W0*Mw
    H0 = dd["H0"]; Mh = dd["Mh"]; H = Mh*H0
    clamp_level=1.0; index=100; imgsz=(19,19)
    reconimg = W*H[:,100]
    W_max = maximum(abs,reconimg)*clamp_level; W_clamped = clamp.(reconimg,0.,W_max)
    save(fprefix*"_recon$index.png", map(clamp01nan, reshape(W_clamped,imgsz...)))
end
=#

dataset = :fakecell; SNR = 0; savefigure = true
X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure))
SNRstr = dataset == :fakecell ? "_$(SNR)dB" : ""
ncls = 15; β = 0.001
ncells = dataset ∈ [:fakecell, :neurofinder] ? ncls : ncells0
initpwradj = :balance; maxiter = 500; show_trace = false
initdatastr = "_$(dataset)$(SNRstr)_nc$(ncells)"
usingSCAinit =false
if usingSCAinit
    fprefix0 = "SCAinitsp$(β)$(initdatastr)_$(initpwradj)_iter$(maxiter)"
    if isfile(fprefix0*".jld")
        dd = load(fprefix0*".jld")
        X = dd["X"]; W0 = dd["W0"]; H0 = dd["H0"]; Mw = dd["Mw"]; Mh = dd["Mh"]
    else
        rt1 = @elapsed W0, H0, Mw, Mh, D, Epre, E, _ = initMwMhSparse(X, ncells, β, poweradjust=initpwradj,
                                                maxiter=maxiter,show_trace=show_trace)
        save(fprefix0*".jld","X",X,"W0",W0,"H0",H0,"Mw",Mw,"Mh",Mh,"D",D,"β",β,"rt1",rt1)
    end
else
    fprefix0 = "SCAinitsp$(β)$(initdatastr)_$(initpwradj)_iter$(maxiter)"
    if isfile(fprefix0*".jld")
        dd = load(fprefix0*".jld")
        X = dd["X"]; W0 = dd["W0"]; H0 = dd["H0"]; Mw = dd["Mw"]; Mh = dd["Mh"]
    else
        rt1 = @elapsed W0, H0, Mw, Mh, D, Epre, E, _ = initMwMhSparse(X, ncells, β, poweradjust=initpwradj,
                                                maxiter=maxiter,show_trace=show_trace)
        save(fprefix0*".jld","X",X,"W0",W0,"H0",H0,"Mw",Mw,"Mh",Mh,"D",D,"β",β,"rt1",rt1)
    end
end
Mw0, Mh0 = copy(Mw), copy(Mh)
W1,H1 = W0*Mw, Mh*H0; normalizeWH!(W1,H1)
savefigure && imsave_data(dataset,fprefix0,W1,H1,imgsz,lengthT; saveH=false)

sd_group=:component; weighted=:none; Msparse=false; reg=:W1M2; order=1
pwradj=:none; λ=100; β=10; Wonly=true; maxiter=30
λ1 = λ; λ2 = λ; β1 = β; β2 = Wonly ? 0. : β
sparsestr = Msparse ? "M" : "WH"
pwrstr = pwradj==:balance2 ? "M2" : pwradj==:balance3 ? "M$(order)" : ""
paramstr="_$(sd_group)_$(weighted)_$(sparsestr)$(order)$(pwrstr)_lm$(λ)_bw$(β1)_bh$(β2)"
fprefix = fprefix0*"_$(pwradj)"*paramstr
sd_group ∉ [:column, :component, :pixel] && error("Unsupproted sd_group")
Mw, Mh = copy(Mw0), copy(Mh0);
rt2 = @elapsed Mw, Mh, f_xs, x_abss, xw_abss, xh_abss, iter, trs = minMwMh!(W0,H0,D,Mw,Mh,λ1,λ2,β1,β2,maxiter,Msparse,order;
            reg=:W1M2, poweradjust=pwradj, fprefix=fprefix, sd_group=sd_group, show_lsplot=false, store_trace=false, imgsz=imgsz,
            SNR=SNR, weighted=weighted, decifactor=4)
W2,H2 = copy(W0*Mw), copy(Mh*H0)
normalizeWH!(W2,H2); #imshowW(sortWHslices(W2,H2)[1],imgsz, borderwidth=1);# imshowW(W2,imgsz, borderwidth=1);
if SNR == :face
    clamp_level=0.5; W2_max = maximum(abs,W2)*clamp_level; W2_clamped = clamp.(W2,0.,W2_max)
    signedcolors = (colorant"green1", colorant"white", colorant"magenta")
    imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W2, imgsz, gridcols=7, colors=signedcolors, borderval=W2_max, borderwidth=1)
else
    imsaveW(fprefix*"_iter$(iter)_rt$(rt2).png", W2, imgsz, borderwidth=1) #sortWHslices(W2,H2)[1]
end

SNR = -10
X, imgsz, lengthT, ncells0, gtncells, datadic = load_data(dataset; SNR=SNR, save_gtimg=(false&&savefigure))

beta=0; lambda = 1; initmaxiter=400
SNRstr = dataset == :fakecell ? "_$(SNR)dB" : ""
initdatastr = "_$(dataset)$(SNRstr)_nc$(ncells)"
initstr = "spb$(beta)l$(lambda)i$(initmaxiter)"
fpfix = "SCAinit$(initstr)$(initdatastr)_$(initpwradj)"
rt1 = @elapsed W0, H0, Mw, Mh, D, Epre, E, _ = initMwMhSparse(X, ncells, beta, lambda,
    maxiter=initmaxiter, poweradjust=initpwradj)
fprefix0 = "SCAinit$(initstr)$(initdatastr)_$(initpwradj)_rt"*@sprintf("%1.2f",rt1)
Mw0, Mh0 = copy(Mw), copy(Mh)
W1,H1 = copy(W0*Mw), copy(Mh*H0); normalizeWH!(W1,H1)
gtW = datadic["gtW"]; gtH = datadic["gtH"]
# calculate MSD
mssd, ml, ssds = matchednssda(gtW,W1)
mssdH = ssdH(ml, gtH,H1')
# reorder according to GT image
neworder = matchedorder(ml,ncells)
savefigure && imsave_data(dataset,fprefix0,W1[:,neworder],H1[neworder,:],imgsz,100;
    mssdwstr="_MSE"*@sprintf("%1.4f",mssd), mssdhstr="_MSE"*@sprintf("%1.4f",mssdH), saveH=true)
