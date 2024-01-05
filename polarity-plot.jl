alpha = .85

function userLegend(f, userColors, userColorIndex)
	uniqueColors = ( unique(userColors) )
	labels = map(c -> userColorIndex[c], uniqueColors)
	group_colors = [MarkerElement(color = color, strokecolor = :transparent, marker = :utriangle, markersize=10) for color in uniqueColors]

    # using MakieLayout
	leg = Legend(
		f, 
		[group_colors], 
		[labels],
		[userColorIndex[:title]]
	)
	# leg.tellheight = true

	return leg
end

function itemLegend(f, itemColors, itemColorIndex)

		uniqueColors = ( unique(itemColors) )
		labels = map(c -> itemColorIndex[c], uniqueColors)
		group_colors = [MarkerElement(color = color, strokecolor = :transparent, marker = :circle, markersize=8) for color in uniqueColors]

	    # using MakieLayout
		leg = Legend(
			f, 
			[group_colors], 
			[labels],
			[itemColorIndex[:title]]
		)
		# leg.tellheight = true

		# leg.tellwidth = false
		return leg

end


function plotUsers(pos, W, userColors; title="Users")
	scatter(pos, W, color=map(c -> (c,alpha), userColors), markersize = 10, marker = :utriangle,
		 axis = (; title=title, xlabel = "polarity factor", ylabel = "common ground factor")
	)

end

function plotItems(pos, X, itemColors; itemColorIndex=nothing, title="Items")
	scatter(pos, X', color=map(c -> (c,alpha), itemColors), markersize = 8, marker = :circle,
		 axis = (; title=title, xlabel = "polarity factor", ylabel = "common ground factor")
	)
end


function polarityPlot(W, X, userColors, itemColors; userColorIndex=nothing, itemColorIndex=nothing, title=nothing)

  f = Figure(size = (600, 400))

	if title !== nothing
		Label(f[0, 1:2], title, padding = (0, 0, 0, 0), font = "Noto Sans Bold", fontsize=18)
	end

	# map(c -> (c,0.5), userColors)

	plotUsers(f[1,1], W, userColors; title="Users")
	plotItems(f[1,2], X, itemColors; title="Items")

	sublayout = GridLayout()
	if userColorIndex !== nothing
		sublayout[1,1] = userLegend(f, userColors, userColorIndex)
	end

	if itemColorIndex !== nothing
		sublayout[1,2] = itemLegend(f, itemColors, itemColorIndex)
	end
	f[2, 1:2] = sublayout

	
	# x = for color in unique(colors)
	groupAverages = map(unique(itemColors)) do color
		x = [ mean( (itemColors .== color) .* X', dims=1) color]
		x[1,:]
	end	
	groupAverages = permutedims(hcat(groupAverages...))

	return f

end


# W = model.W
# X =  model.X
# b
# ww = model.W * b
# xx = b' * model.X
function polarityPlotWithBasis(W, X, ww, xx, userColors, itemColors; userColorIndex=nothing, itemColorIndex=nothing, title=nothing)

  f = Figure(size = (600, 800))

	if title !== nothing
		Label(f[0, 1:2], title, padding = (0, 0, 0, 0), font = "Noto Sans Bold", fontsize=18)
	end


	# ax = Axis(f[1, 1])

	# a1 = arrows!([0], [0], [b[1,1]], [b[2,1]], arrowsize = 10, lengthscale = 0.3,
	#     arrowcolor = [:red], linecolor = [:red], linestyle=:dash, label="arrow 1")

	# a2 = arrows!([0], [0], [b[1,2]], [b[2,2]], arrowsize = 10, lengthscale = 0.3,
	#     arrowcolor = [:blue], linecolor = [:blue], linestyle=:dot, label="arrow 2")



	plotUsers(f[1,1], W, userColors; title="Users Before Change of Basis")

	plotItems(f[1,2], X, itemColors; title="Items Before Change of Basis")

	plotUsers(f[2,1], ww, userColors; title="Users After Change of Basis")
	plotItems(f[2,2], xx, itemColors; title="Items After Change of Basis")

	sublayout = GridLayout()
	if userColorIndex !== nothing
		sublayout[1,1] = userLegend(f, userColors, userColorIndex)
	end

	if itemColorIndex !== nothing
		sublayout[1,2] = itemLegend(f, itemColors, itemColorIndex)
	end
	f[3, 1:2] = sublayout

	
	return f

end




function polarityPlotUsers(W, userColors; userColorIndex=nothing, title=nothing)


  f = Figure()

	if title !== nothing
		Label(f[0, 1], title, padding = (0, 0, 0, 0), font = "Noto Sans Bold", fontsize=18)
	end

	plotUsers(f[1,1], W, userColors; title="Users")


	if userColorIndex !== nothing
		f[1,2] = userLegend(f, userColors, userColorIndex)
	end

	return f

end



function polarityPlotItems(X, itemColors; itemColorIndex=nothing, title=nothing)


  f = Figure()

	if title !== nothing
		Label(f[0, 1:2], title, padding = (0, 0, 0, 0), font = "Noto Sans Bold", fontsize=18)
	end


	plotItems(f[1,1], X, itemColors; title="Items")


	if itemColorIndex !== nothing
		f[1,2] = itemLegend(f, itemColors, itemColorIndex)
	end

	return f

end




polarityPlot3d = function(W, colors, title="Dimension 1 vs. Dimension 2")

	f = Figure()
	pos = f[1,1]
	s1 = scatter(pos, W[:,1], W[:,2], W[:,3], markersize = 10, marker = :utriangle ,color=colors, transparency=true)


	s1
end


	