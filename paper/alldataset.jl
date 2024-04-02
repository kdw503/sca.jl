using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

datasets = [:fakecells,:audio,:cbclface,:neurofinder]
Ss = []
for (i,dataset) in enumerate(datasets)
    SNR=0; inhibitindices=0; bias=0.0
    X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
            inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
    F = svd(X); S = F.S ./F.S[1]; push!(Ss,S)
    push!(legends,string(dataset))
end
f = Figure(resolution = (500,400))
ax11=GLMakie.Axis(f[1,1], limits=((1,250),(1e-6,1)), xscale=identity, yscale=log10)
lns=[]
for (i,S) in enumerate(Ss)
    rng=1:length(S)#; rng = rng./length(S)
    @show minimum(S), maximum(S)
    lin = lines!(ax11,rng,S,color=mtdcolors[i+1]); push!(lns,lin)
end
labels = ["fake cells", "audio", "CBCL face", "neurofinder"]
axislegend(ax11,lns,labels,position=:lb)

save(joinpath(subworkpath,"svd_powers2.png"),f)
