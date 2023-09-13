using Pkg

Pkg.activate(".")

#using MultivariateStats # for ICA
using Images, LinearAlgebra, Printf, Colors, Interpolations, JLD2, GLMakie
using FakeCells, AxisArrays, ImageCore, MappedArrays, NMF, Statistics, TiledFactorizations
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using ScikitLearn, Cthulhu, TestData
# using Convex, SCS, VideoIO

#import SymmetricComponentAnalysis as SCA
using SymmetricComponentAnalysis
SCA = SymmetricComponentAnalysis

scapath = joinpath(dirname(pathof(SCA)),"..")

include(joinpath(scapath,"test","testutils.jl"))
include(joinpath(workpath,"utils.jl"))
