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

#ARGS =  ["[:fakecell]",[20]","200","[0]","[0,0.01,0.1,0.5,1.0,1.5]",":sca"]

arng = -2:0.001:2
f(u) = 3(u+1)^2
J(u) = 2*norm(u,1)+norm(u-1,1)
E(u) = f(u) + J(u)
fs = f.(arng); Js = J.(arng); Es = E.(arng)

fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(arng, [fs Js Es])
ax1.legend(["f(u)", "J(u)", "J(u)+f(u)"],fontsize = 12,loc=2)
xlabel("u",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("Bregman_Problem.png")

v = 0.5; gradJ(u) = (u<0 ? -3 : u > 1 ? 3 : -1 )
D(u,v) = J(u) - (J(v)+gradJ(u)*(u-v))
Datv(u) = D(u,v)
Datvs = Datv.(arng)
fig, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.plot(arng, [fs Js Datvs])
ax1.legend(["f(u)", "J(u)", "D(u,v)"],fontsize = 12,loc=2)
xlabel("u",fontsize = 12)
ylabel("penalty",fontsize = 12)
savefig("Bregman_Distance.png")

gradJ(u) = (u<0 ? -3 : u > 1 ? 3 : -1 )
D(u,v) = J(u) - (J(v)+gradJ(u)*(u-v))
for v in [-0.5, 0.5, 1.5]
    Datv(u) = D(u,v)
    Datvs = Datv.(arng)
    αs = [0.1,1,10]
    for α in αs
        Eb(u) = α*Datv(u) + f(u)
        Ebs = Eb.(arng)
        fig, ax1 = plt.subplots(1,1, figsize=(5,4))
        ax1.plot(arng, [fs Js Ebs])
        ax1.legend(["f(u)", "J(u)", "$(α)*D(u)+f(u)"],fontsize = 12,loc=2)
        xlabel("u",fontsize = 12)
        ylabel("penalty",fontsize = 12)
        savefig("Bregman_obj_v$(v)α$(α).png")
    end
end
