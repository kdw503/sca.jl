Pkg.activate(".")

#using MultivariateStats # for ICA
using Images, LinearAlgebra, Printf, Colors
using FakeCells, AxisArrays, ImageCore, MappedArrays, NMF, Statistics
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using ScikitLearn, Cthulhu
# using Convex, SCS, VideoIO

#import SymmetricComponentAnalysis as SCA
using SymmetricComponentAnalysis
SCA = SymmetricComponentAnalysis

Images.save("dummy.png",rand(2,2)) # If we didn't call this line before `using PyPlot`,
                            # ImageMagick is used when saving image. That doesn't
                            # properly close the file after saving.
using PyPlot
scapath = joinpath(dirname(pathof(SCA)),"..")
