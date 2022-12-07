using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

generate_data = false
if generate_data
    # Convex SCA SNR vs MSSD
    ARGS =  ["[:fakecell]","-20:1:50","100","[0]","[1.0]",":sca","[15]","false"]
    include("scaexpr.jl") # save sca_SNR_vs_MSSD.jld
    # Convex SCA ncells vs MSSD (-10dB)
    ARGS =  ["[:fakecell]","[-10]","100","[0]","[1.0]",":sca","10:2:80","false"] 
    include("scaexpr.jl") # save sca_NOC_vs_MSSD_-10dB.jld
    # Convex SCA ncells vs MSSD (40dB)
    ARGS =  ["[:fakecell]","[40]","100","[0]","[1.0]",":sca","10:2:80","false"] 
    include("scaexpr.jl") # save sca_NOC_vs_MSSD_40dB.jld

    # HALS CD SNR vs MSSD
    ARGS =  ["[:fakecell]","-20:1:50","100","[0]","[0.1]",":cd","[15]","false"]
    include("scaexpr.jl") # save sca_SNR_vs_MSSD.jld
    # Convex SCA ncells vs MSSD (-10dB)
    ARGS =  ["[:fakecell]","[-10]","100","[0]","[0.1]",":cd","10:2:80","false"] 
    include("scaexpr.jl") # save sca_NOC_vs_MSSD_-10dB.jld
    # Convex SCA ncells vs MSSD (40dB)
    ARGS =  ["[:fakecell]","[40]","100","[0]","[0.1]",":cd","10:2:80","false"] 
    include("scaexpr.jl") # save sca_NOC_vs_MSSD_40dB.jld
end

dscasnr = load("sca_SNR_vs_MSSD.jld")
dscanocm10 = load("sca_NOC_vs_MSSD_-10dB.jld")
dscanoc40 = load("sca_NOC_vs_MSSD_40dB.jld")
dcdsnr = load("cd_SNR_vs_MSSD.jld")
dcdnocm10 = load("cd_NOC_vs_MSSD_-10dB.jld")
dcdnoc40 = load("cd_NOC_vs_MSSD_40dB.jld")

# MSSD
legendstrs = ["SCA","CD"]; ylbl="MSE"
# SNR vs MSSD
plotW(dscasnr["SNRs"], [dscasnr["mssds"] dcdsnr["mssds"]], "SNR_vs_MSSD.png";
    title="SNR vs MSE", xlbl="SNR", ylbl=ylbl, legendloc=1, arrange=:combine,
    legendstrs = legendstrs)
# NOC vs MSSD (-10dB)
plotW(dscanocm10["nclsrng"], [dscanocm10["mssds"] dcdnocm10["mssds"]], "NOC_vs_MSSD_-10dB.png";
    title="NOC vs MSE (-10dB)", xlbl="NOC", ylbl=ylbl, legendloc=1, arrange=:combine,
    legendstrs = legendstrs)
# NOC vs MSSD (40dB)
plotW(dscanoc40["nclsrng"], [dscanoc40["mssds"] dcdnoc40["mssds"]], "NOC_vs_MSSD_40dB.png";
    title="NOC vs MSE (40dB)", xlbl="NOC", ylbl=ylbl, legendloc=1, arrange=:combine,
    legendstrs = legendstrs)

plotW(dcdnocm10["nclsrng"], [dcdnocm10["mssds"] dcdnoc40["mssds"]], "NOC_vs_MSSD_-10dB_40dB.png";
    title="NOC vs MSE (40dB)", xlbl="NOC", ylbl=ylbl, legendloc=1, arrange=:combine,
    legendstrs = legendstrs)

# Runtime
ylbl="Time (sec)"
# SNR vs runtime
plowW(dscasnr["SNRs"], [dscasnr["rt1s"]+dscasnr["rt2s"] dcdsnr["rt1s"]+dcdsnr["rt2s"]], "SNR_vs_rt.png";
    title="SNR vs Runtime", xlbl="SNR", ylbl=ylbl, legendloc=2, arrange=:combine,
    legendstrs = legendstrs)
# NOC vs MSSD (-10dB)
plowW(dscanocm10["nclsrng"], [dscanocm10["rt1s"]+dscanocm10["rt2s"] dcdnocm10["rt1s"]+dcdnocm10["rt2s"]], "NOC_vs_rt_-10dB.png";
    title="NOC vs Runtime (-10dB)", xlbl="NOC", ylbl=ylbl, legendloc=2, arrange=:combine,
    legendstrs = legendstrs)
# NOC vs MSSD (40dB)
plowW(dscanoc40["nclsrng"], [dscanoc40["rt1s"]+dscanoc40["rt2s"] dcdnoc40["rt1s"]+dcdnoc40["rt2s"]], "NOC_vs_rt_40dB.png";
    title="NOC vs Runtime (40dB)", xlbl="NOC", ylbl=ylbl, legendloc=2, arrange=:combine,
    legendstrs = legendstrs)

# CBCL-Face
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(:cbclface; SNR=20, save_gtimg=false)
fprefix = "CVX_SCA_cbclface_nc49_balance_balance2_L1_λw0_λh0_βw5.0_βh0.0_iter100_rt1595.30"
dd = load(fprefix*".jld")
W0 = dd["W0"]; H0 = dd["H0"]; Mw = dd["Mw"]; Mh = dd["Mh"]
W = W0*Mw; H = Mh*H0
imsave_reconstruct(fprefix,X,W,H,imgsz; index=100, clamp_level=0.2)

clrs=(colorant"black", colorant"black", colorant"white")
reconimg = W0*H0[:,100]#; wmin=minimum(reconimg); reconimg .-=wmin
mxabs = maximum(abs, reconimg)
fsc = scalesigned(mxabs)
fcol = colorsigned(clrs...)
Wcolor = Array(mappedarray(fcol ∘ fsc, reshape(reconimg, Val(2))))
save(fprefix*"_recon100.png",reshape(Wcolor, imgsz...))

clrs=(colorant"green1", colorant"white", colorant"magenta")
p = ncells
reconimg = Mw*Mh
mxabs = maximum(abs, reconimg)
fsc = scalesigned(mxabs)
fcol = colorsigned(clrs...)
Wcolor = Array(mappedarray(fcol ∘ fsc, reshape(reconimg, Val(2))))
save(fprefix*"_recon100.png",reshape(Wcolor,p,p))
