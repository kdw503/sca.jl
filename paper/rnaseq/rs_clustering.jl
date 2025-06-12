using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
allenbrainversion = "20241130"
subworkpath = joinpath(workpath,"paper","rnaseq",allenbrainversion)

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))
include(joinpath(workpath,"clustering.jl"))

# Base.ARGS = ["\"WMB-10Xv2-HY\"", "\"log2\"", ":pcb_sp", "28", "true",":average"]
feature_name = eval(Meta.parse(ARGS[1]))
pp = eval(Meta.parse(ARGS[2]))
mf_method = eval(Meta.parse(ARGS[3]))
clnoc = eval(Meta.parse(ARGS[4]))
normalization = eval(Meta.parse(ARGS[5]))
hc_linkage = eval(Meta.parse(ARGS[6]))

#======== load Allenbrain annotation data =======#
download_base = joinpath(datapath,"AllenBrain")
ddann = load(joinpath(download_base,allenbrainversion,"$(feature_name)_annotation.jld2"))
annindices = ddann["annindices"]; cell_lable = ddann["cell_lable"]
class = ddann["class"]; class_color = ddann["class_color"] # 28
subclass = ddann["subclass"]; subclass_color = ddann["subclass_color"] # 192
supertype = ddann["supertype"]; supertype_color = ddann["supertype_color"] # 515
cluster_alias = ddann["cluster_alias"]; cluster_color = ddann["cluster_color"] # 1646
x = ddann["x"]; y = ddann["y"]; keeping_indices = ddann["keeping_indices"]
cls_cpairs = ddann["cls_cpairs"]; scls_cpairs = ddann["scls_cpairs"]
styp_cpairs = ddann["styp_cpairs"]; clst_cpairs = ddann["clst_cpairs"]
uclass = unique(class); clscmap = countmap(class); class_counts = map(l->clscmap[l],uclass)

#======== load Matrix Factorization data ========#
feature_name_pp = feature_name*"-"*pp

# load SVD data
fname = joinpath(subworkpath,"$(feature_name_pp)_initisvd_noc500_X.jld2")
ddsvd = load(fname)
Genesvd = ddsvd["U"]'; Cellsvd = (ddsvd["V"]*ddsvd["D"])'; rt1svd = ddsvd["rt1"]

# load PCB data
#fname = joinpath(subworkpath,"$(feature_name_pp)_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw0.0_bh0.0_tol1.0e-6_it13.jld2")
fname = joinpath(subworkpath,"$(feature_name_pp)_initisvd_pcb_noc500_nac0_aw0.005_ah0.005_bw5.0_bh5.0_tol1.0e-6_it11.jld2")
ddpcb = load(fname)
Genepcb = ddpcb["W"]; Cellpcb = ddpcb["H"]
rt1pcb = ddpcb["rt1"]; rt2pcb = ddpcb["rt2"]; fitvalpcb = ddpcb["fitval"]

# load HALS data
fname = joinpath(subworkpath,"$(feature_name_pp)_hals_noc500_a0.1_iter100.jld2")
ddhls = load(fname)
Genehls = ddhls["W"]; Cellhls = ddhls["H"]
rt1hls = ddhls["rt1"]; rt2hls = ddhls["rt2"]; fitvalhls = ddhls["fitval"]

#=================== Hierarchical Clustering =================#
method = :hclust; nepmt = 1; hc_h = nothing
label = class
label_counts = class_counts
noc = clnoc
nepmt = nepmt
hc_h = hc_h

using NRRD, AxisArrays, FileIO

if mf_method ∈ [:pcb_sp, :pcb_sp_nn]
    @show mf_method; flush(stdout)
    fprex = "$(feature_name_pp)_$(mf_method)"
    hfname = fprex*"_cluster.nhdr"
    if isfile(joinpath(download_base,allenbrainversion,hfname))
        @show "Reading Dpcb"; flush(stdout)
        Dpcb = load(joinpath(download_base,allenbrainversion,fprex*"_cluster.nhdr")).data
    else
        @show "Calculating Dpcb"; flush(stdout)
        Worg = Genepcb'

        l = size(Worg,1); T = eltype(Worg)
        if normalization
            for r in eachrow(Worg)
                n = norm(r)
                r = n == 0 ? r : r ./= n
            end
        end
        Dpcb = pairwise(Clustering.Euclidean(), Worg, dims=1) # 3~4min

        # write data
        dfname = fprex*"_cluster.nrrd"
        open(joinpath(download_base,allenbrainversion,dfname),"w") do io
            write(io,Dpcb)
        end
        # write header
        sizey, sizex = size(Dpcb)
        axy = AxisArrays.Axis{:y}(1:sizey)
        axx = AxisArrays.Axis{:x}(1:sizex)
        header = NRRD.headerinfo(Float32, (axy, axx))
        header["datafile"] = dfname
        open(joinpath(download_base,allenbrainversion,hfname),"w") do io
            write(io,magic(format"NRRD"))
            NRRD.write_header(io,"0004",header)
        end
    end
    # Clustering PCB result
    # rtclustpcb = @elapsed pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, resultsnpcb, clustsnpcb =
    #         clustring_experi(method, Hpcb', class, class_counts; D=Dpcb, noc=clnoc, normalization=normalization,
    #                         nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
    @show "Clustering"; flush(stdout)
    rtclustpcb = @elapsed resultnpcb = hclust(Dpcb, linkage=hc_linkage)
    clustnpcb = cutree(resultnpcb; k=clnoc, h=hc_h)
    save(joinpath(subworkpath,fprex*"_cn$(clnoc)_$(normalization)_$(hc_linkage)_class.jld2"), "resultn", resultnpcb, "clustn",clustnpcb,"rtclust",rtclustpcb)
elseif mf_method == :svd
    @show mf_method; flush(stdout)
    fprex = "$(feature_name_pp)_svd"
    hfname = fprex*"_cluster.nhdr"
    if isfile(joinpath(download_base,allenbrainversion,hfname))
        @show "Reading Dsvd"; flush(stdout)
        Dsvd = load(joinpath(download_base,allenbrainversion,fprex*"_cluster.nhdr")).data
    else
        @show "Calculating Dsvd"; flush(stdout)
        Worg = Genesvd'

        l = size(Worg,1); T = eltype(Worg)
        if normalization
            for r in eachrow(Worg)
                n = norm(r)
                r = n == 0 ? r : r ./= n
            end
        end
        Dsvd = pairwise(Clustering.Euclidean(), Worg, dims=1) # 3~4min

        # write data
        dfname = fprex*"_cluster.nrrd"
        open(joinpath(download_base,allenbrainversion,dfname),"w") do io
            write(io,Dsvd)
        end
        # write header
        sizey, sizex = size(Dsvd)
        axy = AxisArrays.Axis{:y}(1:sizey)
        axx = AxisArrays.Axis{:x}(1:sizex)
        header = NRRD.headerinfo(Float32, (axy, axx))
        header["datafile"] = dfname
        open(joinpath(download_base,allenbrainversion,hfname),"w") do io
            write(io,magic(format"NRRD"))
            NRRD.write_header(io,"0004",header)
        end
    end
    # Clustering PCB result
    # rtclustpcb = @elapsed pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, resultsnpcb, clustsnpcb =
    #         clustring_experi(method, Hpcb', class, class_counts; D=Dpcb, noc=clnoc, normalization=normalization,
    #                         nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
    @show "Clustering"; flush(stdout)
    rtclustsvd = @elapsed resultnsvd = hclust(Dsvd, linkage=hc_linkage)
    clustnsvd = cutree(resultnsvd; k=clnoc, h=hc_h)
    save(joinpath(subworkpath,fprex*"_cn$(clnoc)_$(normalization)_$(hc_linkage)_class.jld2"), "resultn", resultnsvd, "clustn",clustnsvd,"rtclust",rtclustsvd)
elseif mf_method == :hals
    @show mf_method; flush(stdout)
    fprex = "$(feature_name_pp)_hals"
    hfname = fprex*"_cluster.nhdr"
    if isfile(joinpath(download_base,allenbrainversion,hfname))
        @show "Reading Dhls"; flush(stdout)
        Dhls = load(joinpath(download_base,allenbrainversion,fprex*"_cluster.nhdr")).data
    else
        @show "Calculating Dhls"; flush(stdout)
        Worg = Genehls'

        l = size(Worg,1); T = eltype(Worg)
        if normalization
            for r in eachrow(Worg)
                n = norm(r)
                r = n == 0 ? r : r ./= n
            end
        end
        Dhls = pairwise(Clustering.Euclidean(), Worg, dims=1) # 3~4min

        # write data
        dfname = fprex*"_cluster.nrrd"
        open(joinpath(download_base,allenbrainversion,dfname),"w") do io
            write(io,Dhls)
        end
        # write header
        sizey, sizex = size(Dhls)
        axy = AxisArrays.Axis{:y}(1:sizey)
        axx = AxisArrays.Axis{:x}(1:sizex)
        header = NRRD.headerinfo(Float32, (axy, axx))
        header["datafile"] = dfname
        open(joinpath(download_base,allenbrainversion,hfname),"w") do io
            write(io,magic(format"NRRD"))
            NRRD.write_header(io,"0004",header)
        end
    end
    # Clustering HALS result
    @show "Clustering"; flush(stdout)
    rtclusthls = @elapsed resultnhls = hclust(Dhls, linkage=hc_linkage)
    clustnhls = cutree(resultnhls; k=clnoc, h=hc_h)
    # rtclusthls = @elapsed pre_meansn, pre_stdsn, rec_meansn, rec_stdsn, wavg_pren, wavg_recn, precisionssn, recallssn, resultsnhls, clustsnhls =
    #         clustring_experi(method, Hhals', label, label_counts; noc=clnoc, normalization=normalization,
    #                         nepmt=nepmt, hc_linkage=hc_linkage, hc_h=hc_h)
    save(joinpath(subworkpath,fprex*"_cn$(clnoc)_$(normalization)_$(hc_linkage)_class.jld2"),"resultn", resultnhls, "clustn",clustnhls,"rtclust",rtclusthls)
else
    error("Unknown mf_method: $mf_method")
end
# save clustering result
