#using MultivariateStats # for ICA
using Images, LinearAlgebra, Printf, Colors, Interpolations, JLD2
using FakeCells, AxisArrays, ImageCore, MappedArrays, NMF, Statistics, TiledFactorizations
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using ScikitLearn, Cthulhu, TestData
# using Convex, SCS, VideoIO

#import SymmetricComponentAnalysis as SCA
using SymmetricComponentAnalysis
SCA = SymmetricComponentAnalysis

is_X11_available = true
try
    Sys.islinux() && run(`ls /usr/bin/x11vnc`) # check if this is noVNC graphical platform
    using GLMakie
    GLMakie.activate()
    AMakie = GLMakie
catch # not a graphical platform
    @warn("Not a RIS noVNC graphical platform")
    using CairoMakie
    global is_X11_available = false
    CairoMakie.activate()
    AMakie = CairoMakie
end

scapath = joinpath(dirname(pathof(SCA)),"..")

include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"utils.jl"))
