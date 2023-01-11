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

using Random

genW(p,sqrtr) = hcat(sqrtr*normalize.(eachcol(rand(MersenneTwister(123),50,p).-0.5))...)
genH(p,sqrtr) = hcat(sqrtr*normalize.(eachrow(rand(MersenneTwister(123),p,100).-0.5))...)'
genX(p,sqrtr) = genW(p,sqrtr)*genH(p,sqrtr)

p = 5; sqrtr=2; r=sqrtr^2;
W = genW(p,sqrtr); H = genH(p,sqrtr); X=W*H
# P vs. ∥X∥², ∥W∥², ∥H∥², |W|, |H|
prng = 2:30
close("all")
plot(prng,map(p->norm(genX(p,sqrtr))^2,prng))
plot(prng,map(p->norm(genW(p,sqrtr))^2,prng))
plot(prng,map(p->norm(genH(p,sqrtr))^2,prng))
plot(prng,map(p->norm(genW(p,sqrtr),1),prng))
plot(prng,map(p->norm(genH(p,sqrtr),1),prng))
legend(["∥X∥²","∥W∥²","∥H∥²","|W|","|H|"])
savefig("D_P_vs_normXWH.png")
# P vs. ∥W0∥², ∥H0∥², ∥D∥²
close("all")
W0H0D(p,sqrtr) = svd(genX(p,sqrtr))
plot(prng,map(p->norm(W0H0D(p,sqrtr).U[:,1:p])^2,prng))
plot(prng,map(p->norm(W0H0D(p,sqrtr).Vt[1:p,:])^2,prng))
plot(prng,map(p->norm(W0H0D(p,sqrtr).S[1:p])^2,prng))
legend(["∥W0∥²","∥H0∥²","∥D∥²"])
savefig("D_P_vs_normW0H0D.png")


# R vs. ∥X∥², ∥W∥², ∥H∥², |W|, |H|
rrng = 1:10
close("all")
plot(rrng,map(r->norm(genX(p,sqrt(r)))^2,rrng))
plot(rrng,map(r->norm(genW(p,sqrt(r)))^2,rrng))
plot(rrng,map(r->norm(genH(p,sqrt(r)))^2,rrng))
plot(rrng,map(r->norm(genW(p,sqrt(r)),1),rrng))
plot(rrng,map(r->norm(genH(p,sqrt(r)),1),rrng))
legend(["∥X∥²","∥W∥²","∥H∥²","|W|","|H|"])
savefig("D_R_vs_normXWH.png")
# R vs. ∥W0∥², ∥H0∥², ∥D∥²
close("all")
W0H0D(p,sqrtr) = svd(genX(p,sqrtr))
plot(rrng,map(r->norm(W0H0D(p,sqrt(r)).U[:,1:p])^2,rrng))
plot(rrng,map(r->norm(W0H0D(p,sqrt(r)).Vt[1:p,:])^2,rrng))
plot(rrng,map(r->norm(W0H0D(p,sqrt(r)).S[1:p])^2,rrng))
legend(["∥W0∥²","∥H0∥²","∥D∥²"])
savefig("D_R_vs_normW0H0D.png")

