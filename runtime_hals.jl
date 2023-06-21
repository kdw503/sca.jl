using Pkg

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

include("setup_light.jl")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))
include("dataset.jl")
include("utils.jl")

using ProfileView, BenchmarkTools

dataset = :fakecells; SNR=0; inhibitidx=0; bias=0.1; initmethod=:lowrank; initpwradj=:wh_normalize
filter = dataset ∈ [:neurofinder,:fakecells] ? :meanT : :none; filterstr = "_$(filter)"
datastr = dataset == :fakecells ? "_fc$(inhibitidx)_$(SNR)dB" : "_$(dataset)"
subtract_bg=false

data_hals_nn = Dict[]; data_hals_sp_nn = Dict[]
num_experiments = 30

for iter in 1:num_experiments
    @show iter
X, imgsz, lengthT, ncells, gtncells, datadic = load_data(dataset; SNR=SNR, bias=bias, useCalciumT=true,
        inhibitidx=inhibitidx, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);

(m,n,p) = (size(X)...,ncells); dataset == :fakecells && (gtW = datadic["gtW"]; gtH = datadic["gtH"])
X = noisefilter(filter,X)

if subtract_bg
    rt1cd = @elapsed Wcd, Hcd = NMF.nndsvd(X, 1, variant=:ar) # rank 1 NMF
    NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=60, α=0), X, Wcd, Hcd)
    normalizeW!(Wcd,Hcd); imsave_data(dataset,"Wr1",Wcd,Hcd,imgsz,100; signedcolors=dgwm(), saveH=false)
    close("all"); plot(Hcd'); savefig("Hr1.png"); plot(gtH[:,inhibitidx]); savefig("Hr1_gtH.png")
    bg = Wcd*fill(mean(Hcd),1,n); X .-= bg
end

# HALS
rt1cd = @elapsed Wcd0, Hcd0 = NMF.nndsvd(X, ncells, variant=:ar);
mfmethod = :HALS; maxiter=100; 
for (tailstr,α) in [("_nn",0.),("_sp_nn",0.1)]
    dd = Dict()
    avgfits=Float64[]
    Wcd, Hcd = copy(Wcd0), copy(Hcd0); normalizeW!(Wcd,Hcd)
    avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, Wcd, Hcd; clamp=false)
    push!(avgfits,avgfit)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    result = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=true), X, Wcd, Hcd)
    Wcd, Hcd = copy(Wcd0), copy(Hcd0);
    rt2 = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=1,
                    tol=tol, verbose=false), X, Wcd, Hcd)
    for (iter,(W3,H3)) in enumerate(zip(result.Ws,result.Hs))
        @show iter
        normalizeW!(W3,H3)
        avgfit, ml, merrval, rerrs = matchedfitval(gtW, gtH, W3, H3; clamp=false)
        push!(avgfits,avgfit)
    end
    rt2s = range(start=0,stop=rt2,length=length(avgfits))
    ddhals["hals_alpha$(tailstr)"] = α
    ddhals["hals_maxiter$(tailstr)"] = maxiter; ddhals["hals_niters$(tailstr)"] = result.niters;
    ddhals["hals_rt2s$(tailstr)"] = rt2s; ddhals["hals_avgfits$(tailstr)"]=avgfits
    lines!(axi, avgfits); lines!(axt, rt2s, avgfits)
end
end
save("hals_results.jld","data_hals_nn",data_hals_nn,"data_hals_sp_nn",data_hals_sp_nn)
