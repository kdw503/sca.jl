using JLD2, Colors
is_X11_available = true
try
    Sys.islinux() && run(`ls /usr/bin/x11vnc`) # check if this is noVNC graphical platform
    using ImageView, GLMakie
    GLMakie.activate!()
    global AMakie = GLMakie
catch # not a graphical platform
    @warn("Not a RIS noVNC graphical platform")
    using CairoMakie
    global is_X11_available = false
    CairoMakie.activate!()
    global AMakie = CairoMakie
end

# mtdcolors = [RGB{N0f8}(0.00,0.00,0.00), # 1 black
#             RGB{N0f8}(0.00,0.45,0.70),  # 2 greenish blue
#             RGB{N0f8}(0.90,0.62,0.00),  # 3 orange
#             RGB{N0f8}(0.1,0.4,0.0),     # 4 dark green
#             RGB{N0f8}(0.70,0.20,0.70),  # 5 bright magenta
#             RGB{N0f8}(0.84,0.37,0.00),  # 6 redish orange
#             RGB{N0f8}(0.94,0.89,0.26),  # 7 bright yellow
#             RGB{N0f8}(0.00,0.62,0.45),  # 8 greenish cyan
#             RGB{N0f8}(0.34,0.71,0.91)]  # 9 sky blue

# Wong colors : Blue, Orange, Bluish green, Reddish purple, Sky blue, Vermillion, Yellow
mtdcolors = convert.(RGB,Makie.wong_colors()); pushfirst!(mtdcolors,RGB{Float32}(0.0f0,0.0f0,0.0f0))
alpha = 0.2; mtdcoloras = convert.(RGBA,mtdcolors,alpha)
dtcolors = distinguishable_colors(15; lchoices=range(0, stop=50, length=15))
