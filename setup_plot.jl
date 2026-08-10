using JLD2, Colors
is_X11_available = true
# try
#     Sys.islinux() && run(`ls /usr/bin/x11vnc`) # check if this is noVNC graphical platform
#     using ImageView, GLMakie
#     GLMakie.activate!()
#     global AMakie = GLMakie
# catch # not a graphical platform
    @warn("Not a RIS noVNC graphical platform")
    using CairoMakie
    global is_X11_available = false
    CairoMakie.activate!()
    global AMakie = CairoMakie
# end

function rgbtext(r, g, b, text)
    r = round(Int, 255r)
    g = round(Int, 255g)
    b = round(Int, 255b)
    # \e : escape which indicate the start of the ANSI control command
    # [ : CSI (Control Sequence Introducer)
    # 38 : foreground color, cf) 48 : background color
    # ; separates each parameters
    # 2 : 24bit rgb color, cf) 5 : 256 color palette, ex) "\e[48;5;196m" -> set background color as the 196th color in the palette
    # m : end of the ANSI control command
    # "\e[38;2;$(r);$(g);$(b)m" set foreground(38) color with 24bit rgb color(2)
    # "\e[0m" reset the ANSI style
    return "\e[38;2;$(r);$(g);$(b)m$(text)\e[0m" 
end

"""
    $(rgbtext(0.0, 0.0, 0.0, " 1. RGB(0.0, 0.0, 0.0) : black"))
    $(rgbtext(0.0, 0.44705883, 0.69803923, " 2. RGB(0.0, 0.44705883, 0.69803923) : greenish blue"))
    $(rgbtext(0.9019608, 0.62352943, 0.0, " 3. RGB(0.9019608, 0.62352943, 0.0) : orange"))
    $(rgbtext(0.0, 0.61960787, 0.4509804, " 4. RGB(0.0, 0.61960787, 0.4509804) : bluish green"))
    $(rgbtext(0.8, 0.4745098, 0.654902, " 5. RGB(0.8, 0.4745098, 0.654902) : bright purple"))
    $(rgbtext(0.3372549, 0.7058824, 0.9137255, " 6. RGB(0.3372549, 0.7058824, 0.9137255) : sky blue"))
    $(rgbtext(0.8352941, 0.36862746, 0.0, " 7. RGB(0.8352941, 0.36862746, 0.0) : redish orange"))
    $(rgbtext(0.9411765, 0.89411765, 0.25882354, " 8. RGB(0.9411765, 0.89411765, 0.25882354) : bright yellow"))
    $(rgbtext(0.1,0.4,0.0, " 9. RGB(0.1,0.4,0.0) : dark green"))
    $(rgbtext(0.70,0.20,0.70, "10. RGB(0.70,0.20,0.70) : bright magenta"))
    (2~8) Wong colors : Blue, Orange, Bluish green, Reddish purple, Sky blue, Vermillion, Yellow
"""
mtdcolors = convert.(RGB,Makie.wong_colors()); pushfirst!(mtdcolors,RGB{Float32}(0.0f0,0.0f0,0.0f0))
push!(mtdcolors,RGB{N0f8}(0.1,0.4,0.0)); push!(mtdcolors,RGB{N0f8}(0.70,0.20,0.70))

alpha = 0.2; mtdcoloras = convert.(RGBA,mtdcolors,alpha)
dtcolors = distinguishable_colors(15; lchoices=range(0, stop=50, length=15))
