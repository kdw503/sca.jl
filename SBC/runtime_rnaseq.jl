using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")

include(joinpath(workpath,"setup_light.jl"))
include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))
subworkpath = joinpath(workpath,"SBC")

# feature_name, ncells, noc, nac, sbc_maxiter = "WMB-10Xv2-HY", 0, 500, 2500, 100
feature_name = eval(Meta.parse(ARGS[1]))
ncells = eval(Meta.parse(ARGS[2]))
noc = eval(Meta.parse(ARGS[3]))
nac = eval(Meta.parse(ARGS[4]))
sbc_maxiter = eval(Meta.parse(ARGS[5]));

@show feature_name, ncells, noc, nac, sbc_maxiter

nc = noc+nac
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

# sbc
method = "sbc"
@show method; flush(stdout)

initmethod = :isvd; initmtdstr="init$(initmethod)"
fname = joinpath(subworkpath,"$(file_name)_$(initmtdstr)_noc$(noc)_nac$(nac)_X.jld2")
if isfile(fname)
    @show "Reading initfile..."
    dd = load(fname)
    W0, H0t, M0, N0t, D, rt1 = dd["W0"], copy(dd["H0"]'), dd["M0"], copy(dd["N0"]'), dd["D"], dd["rt1"]
    # H0, W0, N0, M0, Hp, Wp, D, rt1 = dd["W0"]', dd["H0"]', dd["M0"]', dd["N0"]', dd["Wp"]', dd["Hp"]', dd["D"]', dd["rt1"]
    # noc = 500, norm(X-W0*M0*N0*H0) = 15375.242f0
else
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
    rt1 = @elapsed begin 
                        W0 = U; H0 = Vt
                        D = Diagonal(s); D2 = Diagonal(sqrt.(s[1:noc])); T = eltype(W0)
                        M0 = Matrix{T}(undef,nc,noc); M0[1:noc,:].=D2; M0[noc+1:end,:].=zero(T)
                        N0 = Matrix{T}(undef,noc,nc); N0[:,1:noc].=D2; M0[:,noc+1:end].=zero(T)
                    end
    rt1 += rt0
#    rt1 = @elapsed W0, H0, M0, N0, Wp, Hp, D = LCSVD.initlcsvd(X, noc, nac; initmethod=initmethod, svdmethod=:isvd)
    save(fname, "W0",W0,"H0",H0,"M0",M0,"N0",N0,"D",D,"rt1",rt1, "memsize", memsize)
    H0t, N0t = copy(dd["H0"]'), copy(dd["N0"]')
end

# sbc
rt2 = @elapsed M0 = sbc(W0; maxiter=sbc_maxiter)
W = W0*M0
H = W\X
# avgfit, ml, merrval, rerrs = SCA.matchedfitval(gtW, gtH, W1, H1; clamp=false)
# LCSVD.normalizeW!(W,H); 
fitval = LCSVD.fitd(X,W*H)
jldfprex = "$(file_name)_$(initmtdstr)_$(method)_noc$(noc)_nac$(nac)_it$(sbc_maxiter)"
fname = joinpath(subworkpath,"$(jldfprex).jld2")
save(fname, "W",W,"H",H,"M",M0,"rt1",rt1,"rt2",rt2,"iter",sbc_maxiter,"fitval",fitval)
# W, H, iter, rt1, rt2, fitval = dd["W"], dd["H"], dd["iter"], dd["rt1"], dd["rt2"], dd["fitval"]
# jldfprex = "WMB-10Xv2-HY-raw_initisvd_sbc_noc500_nac0_it100"
# jldfprex = "WMB-10Xv2-HY-raw_initisvd_sbc_noc500_nac2500_it100"
# fname = joinpath(subworkpath,"$(jldfprex).jld2")
# dd = load(fname)

# W heatmap
LCSVD.normalizeWH!(W,H)
LCSVD.flip2makepos!(W,H)
sw = norm(W,1)
sh = norm(H,1)
mycolors =[colorant"green", colorant"white", colorant"magenta"]
mycmap = cgrad(mycolors, categorical=false, rev=false)
y,x = size(W); num_blocks = y÷800+1; xsize = x + 100; ysize = y÷num_blocks + 50
xlimit = round(quantile(vec(W),0.9999),sigdigits=3) # cf)  round(123456,digits=-2) = 123500.0
for i in 1:min(2,num_blocks) # som many blocks, so now just plot the first twos.
    f = Figure(size=(xsize,ysize))
    rowsizeq = size(W,1)÷num_blocks
    rows = (i==num_blocks ? size(W,1) : rowsizeq*i):-1:rowsizeq*(i-1)+1
    ax = CairoMakie.Axis(f[1, 1],width=noc,height=length(rows))
    joint_limits = (-xlimit, xlimit)
    hm1 = heatmap!(ax, W[rows,:]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(jldfprex)mp_W$(i)_sw$(sw).png"),f)
end
# H heatmap
y,x = size(H); num_blocks = x÷1000+1; ysize = y + 20; xsize = x÷num_blocks + 100
xlimit = round(quantile(vec(H),0.9999),sigdigits=3)
for i in 1:min(2,num_blocks)
    f = Figure(size=(xsize,ysize))
    rows = noc:-1:1
    colsizeq = size(H,2)÷num_blocks
    cols = colsizeq*(i-1)+1:(i==num_blocks ? size(H,2) : colsizeq*i)
    ax = CairoMakie.Axis(f[1, 1],width=length(cols),height=noc)
    joint_limits = (-xlimit, xlimit) 
    hm1 = heatmap!(ax, H[rows,cols]', colormap = mycmap, colorrange = joint_limits)
    hidedecorations!(ax)
    Colorbar(f[:, end+1], hm1)
    save(joinpath(subworkpath,"$(jldfprex)mp_H$(i)_sh$(sh)_fv$(fitval).png"),f)
end
