using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)
subworkpath = joinpath(workpath,"paper")

include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))
using GLMakie

f = Figure(resolution = (500,400))
ax11=GLMakie.Axis(f[1,1], limits=(-0.1,1.1,-0.1,1.1), title="S of SVD(data)")
lns=[]; legends=[]
for (i,dataset) in enumerate([:fakecells,:audio,:cbclface,:urban,:inhibit_real#=,:neurofinder=#])
    SNR=0; inhibitindices=0; bias=0.0
    X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
            inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
    F = svd(X); S = F.S ./F.S[1]; rng=1:length(S); rng = rng./length(S)
    lin = lines!(ax11,rng,S,color=mtdcolors[i+1]); push!(lns,lin); push!(legends,string(dataset))
end
axislegend(ax11,lns,legends)

save(joinpath(subworkpath,"svd_powers.png"),f)
