using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","rnaseq")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

# feature_name, ncells, noc, nac = "WMB-10Xv2-HY", 0, 500, 2500
feature_name = eval(Meta.parse(ARGS[1]))
ncells = eval(Meta.parse(ARGS[2]))
noc = eval(Meta.parse(ARGS[3]))
nac_init = eval(Meta.parse(ARGS[4]))
nac_step = eval(Meta.parse(ARGS[5]))
nac_end = eval(Meta.parse(ARGS[6]))

memsize = Int(Sys.total_memory())/1e9
@show memsize; flush(stdout)

# load data
using NRRD
feature_group=first(feature_name,9)
fgpath = joinpath(datapath,"AllenBrain","expression_matrices",feature_group)
file_name = feature_name*"-raw"
Xraw = load(joinpath(fgpath,file_name*".nhdr")).data
m,nraw = size(Xraw)
ncells = ncells == 0 ? nraw : (file_name*="_n$(ncells)"; ncells); @show ncells; flush(stdout)
ncells > nraw && error("ncells must be smaller than $(nraw)")
X = view(Xraw,:,1:ncells); n=ncells

# isvd

errs = Float64[]
for nac = nac_init:nac_step:nac_end
    nc = noc+nac
    fn = joinpath(subworkpath,"$(file_name)_isvd_nc$(nc).jld2")
    if isfile(fn)
        @show "Reading isvd..."
        dd = load(fn)
        U, s, Vt, rt0, memsize0 = dd["U"], dd["s"], dd["Vt"], dd["rt0"], dd["memsize"]
    else
        @show "Calculating isvd..."
        rt0 = @elapsed ((U, s) = isvd(X, nc); Vt = Array(Diagonal(s.^-1)*(U'*X))) # Vt = Array((pinv(U*Diagonal(s))*X))
        save(fn, "U",U,"s",s,"Vt",Vt,"rt0",rt0, "memsize", memsize)
    end
    push!(errs,norm(X-U*Diagonal(s)*Vt))
end
fn = joinpath(subworkpath,"$(file_name)_isvd_$(noc)_$(nac_init)_$(nac_step)_$(nac_end).jld2")
save(fn, "errs",errs)

f = Figure(size=(350,250))
nc100 = collect(0:500:2500) .+ 500
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing))
lines!(ax,nc100,dd500["errs"],label="X'")
lines!(ax,nc100,dd500t["errs"],label="X")
axislegend(ax; position = :lb)
save(joinpath(subworkpath,"nc_vs_errer_HY_500.png"),f)

noc = 500; errs = Float32[]; errst = Float32[]
nrmX = norm(X)
for nac in 0:500:2500
    @show nac
    nc = noc+nac
    fn = joinpath(subworkpath,"WMB-10Xv2-HY-raw_isvd_nc$(nc).jld2")
    dd = load(fn)
    fnt = joinpath(subworkpath,"WMB-10Xv2-HYt-raw_isvd_nc$(nc).jld2")
    ddt = load(fnt)
    nrmdiff = norm(X-dd["U"]*Diagonal(dd["s"])*dd["Vt"])
    nrmdifft = norm(X'-ddt["U"]*Diagonal(ddt["s"])*ddt["Vt"])
    push!(errs,nrmdiff/nrmX)
    push!(errst,nrmdifft/nrmX)
end
save(joinpath(subworkpath,"WMB-10Xv2-HY-raw_Xerrs500.jld2"),"errs", errs,"errst", errst)

f = Figure(size=(350,250))
nc100 = collect(0:500:2500) .+ 500
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing))
lines!(ax,nc100,errs,label="X'")
lines!(ax,nc100,errst,label="X")
axislegend(ax; position = :lb)
save(joinpath(subworkpath,"nc_vs_Xerrs_HY_500.png"),f)

noc = 500; uerrs = Float32[]; verrs = Float32[]; serrs = Float32[]
for nac in 0:100:1000
    @show nac
    nc = noc+nac
    fn = joinpath(subworkpath,"WMB-10Xv2-HY-raw_isvd_nc$(nc).jld2")
    dd = load(fn)
    fnt = joinpath(subworkpath,"WMB-10Xv2-HYt-raw_isvd_nc$(nc).jld2")
    ddt = load(fnt)
    nrmU = zero(Float32); nrmV = zero(Float32)
    for (c,ct) in zip(eachcol(dd["U"]),eachcol(ddt["Vt"]'))
        nm1 = norm(c-ct); nm2 = norm(c+ct)
        nrmU += nm1 > nm2 ? nm2^2 : nm1^2
    end
    for (r,rt) in zip(eachrow(dd["Vt"]),eachrow(ddt["U"]'))
        nm1 = norm(r-rt); nm2 = norm(r+rt)
        @show nm1, nm2
        nrmV += nm1 > nm2 ? nm2^2 : nm1^2
    end
    push!(uerrs,sqrt(nrmU)/norm(dd["U"]))
    push!(verrs,sqrt(nrmV)/norm(ddt["U"]))
    push!(serrs,norm(dd["s"]-ddt["s"])/norm(dd["s"]))
end
save(joinpath(subworkpath,"WMB-10Xv2-HY-raw_errs100.jld2"),"uerrs", uerrs,"verrs", verrs,"serrs", serrs)

f = Figure(size=(350,250))
nc100 = collect(0:100:1000) .+ 500
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing))
lines!(ax,nc100,uerrs,label="RNAs")
lines!(ax,nc100,verrs,label="Cells")
lines!(ax,nc100,serrs,label="powers")
axislegend(ax; position = :lb)
save(joinpath(subworkpath,"nc_vs_uvserrs_HY_100.png"),f)


noc = 500; uerrs = Float32[]; verrs = Float32[]; serrs = Float32[]
for nac in 0:500:2500
    nc = noc+nac
    fn = joinpath(subworkpath,"WMB-10Xv2-HY-raw_isvd_nc$(nc).jld2")
    dd = load(fn)
    fnt = joinpath(subworkpath,"WMB-10Xv2-HYt-raw_isvd_nc$(nc).jld2")
    ddt = load(fnt)
    nrmU = zero(Float32); nrmV = zero(Float32)
    for (c,ct) in zip(eachcol(dd["U"]),eachcol(ddt["Vt"]'))
        nm1 = norm(c-ct); nm2 = norm(c+ct)
        nrmU += nm1 > nm2 ? nm2^2 : nm1^2
    end
    for (r,rt) in zip(eachrow(dd["Vt"]),eachrow(ddt["U"]'))
        nm1 = norm(r-rt); nm2 = norm(r+rt)
        nrmV += nm1 > nm2 ? nm2^2 : nm1^2
    end
    push!(uerrs,sqrt(nrmU)/norm(dd["U"]))
    push!(verrs,sqrt(nrmV)/norm(ddt["U"]))
    push!(serrs,norm(dd["s"]-ddt["s"])/norm(dd["s"]))
end
save(joinpath(subworkpath,"WMB-10Xv2-HY-raw_errs500.jld2"),"uerrs", uerrs,"verrs", verrs,"serrs", serrs)

f = Figure(size=(350,250))
nc100 = collect(0:500:2500) .+ 500
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing))
lines!(ax,nc100,uerrs,label="RNAs")
lines!(ax,nc100,verrs,label="Cells")
axislegend(ax; position = :lt)
save(joinpath(subworkpath,"nc_vs_uverrs_HY_500.png"),f)

f = Figure(size=(350,250))
nc100 = collect(0:500:2500) .+ 500
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing))
lines!(ax,nc100,serrs,label="powers")
axislegend(ax; position = :rb)
save(joinpath(subworkpath,"nc_vs_serrs_HY_500.png"),f)

f = Figure(size=(350,250))
ss = 1:3000
ax = AMakie.Axis(f[1,1],limits=(nothing,nothing),yscale=log10)
lines!(ax,ss,dd["s"],label="powers of X'")
lines!(ax,ss,ddt["s"],label="powers of X")
axislegend(ax; position = :rt)
save(joinpath(subworkpath,"powers_nc_3000.png"),f)

function tii(a1,a2,b1,b2,iter)
    fn = joinpath(subworkpath,"5_16","WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw$(a1)_ah$(a2)_bw$(b1)_bh$(b2)_tol1.0e-6_it$(iter).jld2")
    dd = load(fn)
    rst = dd["rst"]
    s = 0
    for i in 1:length(rst.traces) s += rst.traces[i].niter end
    s
end

function spnn(a1,a2,b1,b2,iter)
    fn = joinpath(subworkpath,"5_16","WMB-10Xv2-HY-raw_initisvd_pcb_noc500_nac0_aw$(a1)_ah$(a2)_bw$(b1)_bh$(b2)_tol1.0e-6_it$(iter).jld2")
    dd = load(fn)
    rst = dd["rst"]
    @show norm(rst.W,1), LCSVD.sca2(rst.W), norm(rst.H,1), LCSVD.sca2(rst.H)
end

