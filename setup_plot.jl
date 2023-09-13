using Pkg
Pkg.activate(".")

using GLMakie, JLD2, Colors

# mtdcolors = [RGB{N0f8}(0.00,0.00,0.00), # black
#             RGB{N0f8}(0.00,0.45,0.70),  # greenish blue
#             RGB{N0f8}(0.90,0.62,0.00),  # orange
#             RGB{N0f8}(0.1,0.4,0.0),     # dark green
#             RGB{N0f8}(0.70,0.20,0.70),  # bright magenta
#             RGB{N0f8}(0.84,0.37,0.00),  # redish orange
#             RGB{N0f8}(0.94,0.89,0.26),  # bright yellow
#             RGB{N0f8}(0.00,0.62,0.45),  # greenish cyan
#             RGB{N0f8}(0.34,0.71,0.91)]  # sky blue

# Wong colors : Blue, Orange, Bluish green, Reddish purple, Sky blue, Vermillion, Yellow
mtdcolors = convert.(RGB,Makie.wong_colors()); pushfirst!(mtdcolors,RGB{Float32}(0.0f0,0.0f0,0.0f0))
alpha = 0.2; mtdcoloras = convert.(RGBA,mtdcolors,alpha)
