using GLMakie

img = load(assetpath("cow.png"))
fig = Figure()#resolution=(400,300))

if true
	# for images
	image(fig[1, 1], img, axis = (title = "Default",))
	image(fig[1, 2], img, axis = (title = "Default",))
	image(fig[2, 1], img, axis = (title = "Default",))
	image(fig[2, 2], img, axis = (title = "Default",))
else
	# for plots
	AMakie.Axis(fig[1, 1], title = "Default")
	AMakie.Axis(fig[1, 2], title = "Default")
	AMakie.Axis(fig[2, 1], title = "Default")
	AMakie.Axis(fig[2, 2], title = "Default")
end

embed_text = false
if embed_text
    # embed labels
	labels = [
          "a" "b" 
          "c" "d"
        ]
else
    # embed images
	image_names = [
		assetpath("icon_transparent.png") assetpath("icon_transparent.png")
		assetpath("icon_transparent.png") assetpath("icon_transparent.png")
	]
end

nb_rows,nb_cols = size(fig.layout)
square_size = 25
for i ∈ 1:nb_rows, j ∈ 1:nb_cols
    px_area = content(fig.layout[i,j]).scene.px_area[]

    x,y = px_area.origin
    _,h = px_area.widths

    # white background
    b = poly!(fig.scene, 
            Point2f[
                    (x, y+h), 
                    (x+square_size, y+h), 
                    (x+square_size, y+h-square_size), 
                    (x, y+h-square_size)
                   ], 
            color = :white, 
            strokecolor = :black, 
            strokewidth = 1
           )
    if embed_text
        # text
        t = text!(
                fig.scene, 
                labels[i,j], position = Point2f(x+12, y+h-25+12),
                align = (:center, :center), fontsize=20,
            )
    else
        # image
        t = image!(
                fig.scene, 
                rotr90(load(image_names[i,j])), position = Point2f(x+12, y+h-25+12),
                #  align = (:center, :center)
            )
    end
    translate!(b, 0, 0, 100) # z >= 0 bring to front
    translate!(t, 0, 0, 100)
end
