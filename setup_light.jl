#using MultivariateStats # for ICA
using LinearAlgebra, JLD2
using FakeCells, AxisArrays, ImageCore, MappedArrays, NMF, Statistics
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using TestData
using LCSVD, CompNMF, IncrementalSVD, TSVD
