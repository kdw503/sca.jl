using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
subworkpath = joinpath(workpath,"paper","avgfit_measure")

include(joinpath(workpath,"setup_plot.jl"))
include(joinpath(workpath,"utils.jl"))

dataset = :fakecells; inhibitindices=0; bias=0.1
imgsz = (40,20); lengthT = 1000; sigma = 5.0; SNR = 0

X, imsz, lhT, ncs, gtncells, datadic = load_data(dataset; sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias, useCalciumT=true,
        inhibitindices=inhibitindices, issave=false, isload=false, gtincludebg=false, save_gtimg=true, save_maxSNR_X=false, save_X=false);
gtW, gtH = (datadic["gtW"], datadic["gtH"])

#======= Ground truth and average fit measure =======#
cls = distinguishable_colors(10)
imggt = load(joinpath(subworkpath,"GT_W.png"))
imgaf1 = load(joinpath(subworkpath,"SCA_a100_af0.840_pen774.31_it2_rt14.667.png"))
imgaf2 = load(joinpath(subworkpath,"SCA_a100_af0.941_pen742.83_it3_rt0.0486.png"))
imgaf3 = load(joinpath(subworkpath,"SCA_a100_af0.964_pen719.25_it10_rt0.0799.png"))
# scainhibitindices = (bias == 0.5) && (subtract_bg == false) ? 8 : inhibitindices
labels = ["cell$(i)" for i in collect(1:7)]
f = Figure(resolution = (1050,250))
ax12=AMakie.Axis(f[1,2],title="(a) Cells", width=200, aspect = DataAspect()); hidedecorations!(ax12)
ax13=AMakie.Axis(f[1,3:4],title="(b) Activities of cells",xlabel="time index")
ax21=AMakie.Axis(f[2,1],title="(c)", width=10, aspect = DataAspect()); hidedecorations!(ax21); hidespines!(ax21)
#ax22=AMakie.Axis(f[2,2],title="AF=0.840, penalty=774.31", width=300, aspect = DataAspect()); hidedecorations!(ax22)
ax22=AMakie.Axis(f[2,2],title="Average fit = 0.840", width=300, aspect = DataAspect()); hidedecorations!(ax22)
#ax23=AMakie.Axis(f[2,3],title="AF=0.941, penalty=742.83", width=300, aspect = DataAspect()); hidedecorations!(ax23)
ax23=AMakie.Axis(f[2,3],title="Average fit = 0.941", width=300, aspect = DataAspect()); hidedecorations!(ax23)
#ax24=AMakie.Axis(f[2,4],title="AF=0.964, penalty=719.25", width=300, aspect = DataAspect()); hidedecorations!(ax24)
ax24=AMakie.Axis(f[2,4],title="Average fit = 0.964", width=300, aspect = DataAspect()); hidedecorations!(ax24)
image!(ax12, rotr90(imggt)); image!(ax22, rotr90(imgaf1)); image!(ax23, rotr90(imgaf2)); image!(ax24, rotr90(imgaf3))
lin = [lines!(ax13,hd,color=cls[i],label=labels[i]) for (i,hd) in enumerate(eachcol(gtH))]
# axislegend(ax12,position=:rt)
save(joinpath(subworkpath,"GT_cells_activities.png"),f)

#======= compare SMF, Compressed NMF, HALS and Sparse PCA =======#
img11 = load(joinpath(subworkpath,"SCA_fc0_0dB_meanT_isvd_optim_lbfgs_a100.0_b0.0_af0.970334652187394_r0.3_it100_rt0.2957052_gt_W_MSE0.0241.png"))
img12 = load(joinpath(subworkpath,"ADMM_fc0_0dB_meanT_lowrank_nndsvd_a0.0_af0.9402539647801446_it1500_rt0.9892015_gt_W_MSE0.0545.png"))
img21 = load(joinpath(subworkpath,"HALS_fc0_0dB_meanT_svd_a0.0_af0.9639339590110941_it100_rt0.2408629_gt_W_MSE0.0341.png"))
img22 = load(joinpath(subworkpath,"SPCA_fc0_0dB_meanT_a0.5_af0.8448469430010129_rt8.2085074_gt_W_MSE0.2669.png"))
f = Figure(resolution = (700,200))
ax11=AMakie.Axis(f[1,1],title="SMF (AF:0.970)", width=300, aspect = DataAspect()); hidedecorations!(ax11)
ax12=AMakie.Axis(f[1,2],title="Compressed NMF (AF:0.940)", width=300, aspect = DataAspect()); hidedecorations!(ax12)
ax21=AMakie.Axis(f[2,1],title="HALS (AF:0.964)", width=300, aspect = DataAspect()); hidedecorations!(ax21)
ax22=AMakie.Axis(f[2,2],title="Sparse PCA (AF:0.845)", width=300, aspect = DataAspect()); hidedecorations!(ax22)
image!(ax11, rotr90(img11)); image!(ax12, rotr90(img12)); image!(ax21, rotr90(img21)); image!(ax22, rotr90(img22))
# axislegend(ax12,position=:rt)
save(joinpath(subworkpath,"result_all.png"),f)

