using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath)
subworkpath = joinpath(workpath,"paper","inhibit")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"dataset.jl"))
include(joinpath(workpath,"utils.jl"))
using GLMakie

dataset = :fakecells; SNR=0; inhibitindices=[1,2]; bias=0.1; initmethod=:isvd; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
imgsz = (40,20); lengthT=1000
datastr = dataset == :fakecells ? "_fc$(inhibitindices)_$(SNR)dB" : "_$(dataset)"

gt_ncells, imgrs, img_nl, gtW, gtH, gtWimgc, gtbg = gaussian2D(5, imgsz, lengthT, 10,
    bias=bias, useCalciumT=true,jitter=0, fovsz=imgsz, SNR=SNR, orthogonal=false,
    inhibitindices=inhibitindices,gtincludebg=false)

fname = joinpath(workpath,"jldtest.jld")
fakecells_dic = Dict()
fakecells_dic["gt_ncells"] = gt_ncells
fakecells_dic["imgrs"] = imgrs
fakecells_dic["img_nl"] = img_nl
fakecells_dic["gtW"] = gtW
fakecells_dic["gtH"] = gtH
fakecells_dic["gtWimgc"] = Array(gtWimgc)
fakecells_dic["gtbg"] = gtbg
fakecells_dic["imgsz"] = imgsz
fakecells_dic["SNR"] = SNR

jldopen(fname,"w") do file
    for key in keys(fakecells_dic)
        file[key] = fakecells_dic[key]
    end
end

jldopen(fname,"w") do file
    for key in keys(fakecells_dic)
        write(file,key,fakecells_dic[key])
    end
end

file = jldopen(fname,"w")
jldsave(file,fakecells_dic)

jldopen(fname, "r") do file
    for key in keys(file)
        fakecells_dic[key] = read(file[key]) # this cause error
    end
end

jldopen(fname, "r") do file
    fakecells_dic = read(file) # this cause error
end

fakecells_dic =  jldopen(fname, "r") do file
    read(file) # this cause error
end
