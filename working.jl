
# (Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix();
itemColorIndex = Dict(
	:red => "Unhelpful",
	:green => "Helpful",
	:gray => "Needs More Ratings",
	:title => "Commmunity Notes Status"
)

# (Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix(table="sampleDataSet3", coreOnly=true);

(Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix(table="sampleDataSet2");
include("matrix-factorization.jl")


lambda = .1

# First run 1-d model
begin
	Random.seed!(6)
	model = factorizeMatrixIntercepts(Matrix(Y), 1, lambda, false)
	# modelCNlarge1dCoreOnly = model
	# model = modelCNlarge1dCoreOnly

	b = swapLeftRight(model.X, itemIds)
	polarityPlot([model.W * b .+ model.μ model.B .+ model.μ ], [model.X * b .+ model.μ; model.C .+ model.μ ], map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="Community Notes Data (1D)")

	# ratedItems = findall(i -> i !== :gray, itemColors)
	# Xr = model.X[ratedItems]'
	# Cr = model.C[ratedItems]'
	# itemColorsR = itemColors[ratedItems]
	# polarityPlot([model.W .+ model.μ model.B .+ model.μ ], [Xr .+ model.μ; Cr .+ model.μ ], map(c -> :gray, model.W[:,1]), itemColorsR, itemColorIndex=itemColorIndex, title="Community Notes Data (1D)")
	# scene = polarityPlotItems([Xr .+ model.μ; Cr .+ model.μ ], itemColorsR, itemColorIndex=itemColorIndex, title="Community Notes Polarity Plot (Notes)")

	scene = polarityPlotItems([model.X*b .+ model.μ; model.C .+ model.μ ], itemColors, itemColorIndex=itemColorIndex, title="Community Notes Polarity Plot (Notes)")
	save("plots/community-notes-large-items-1d.png", scene)

	score1d = model.C[1,:]


	# model = factorizeMatrixPrimed(Matrix(Y), modelCNlarge1d, lambda)
	# scene = polarityPlot(model.W,  model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="Primed")
end


# Run 2-D model
begin
	include("matrix-factorization.jl")

	model = factorizeMatrixNoIntercepts(Matrix(Y), 2, .1);
	# modelCNlarge2d = model

	(bestBasis, loss) = findLowEntropyDimension(model.W)
	# (worstBasis, loss) = findHighEntropyDimension(model.W)
	# orthogonal = [0 -1; 1 0] * bestBasis
	# bestBasis = normalize([1, -.2])

	orthogonal = [0 -1; 1 0] * bestBasis
	b = [orthogonal bestBasis]

	dimensionEntropy(model.W * bestBasis)
	# save("plots/failed-aligrhment-2d-community-notes-large.png", scene)
	lossMasked(( model.W * b) * (b' * model.X), Y)


	# Wf = model.W[findall(c -> c == :green, userColors), :]
	# (bestBasis, loss) = findLowEntropyDimension(Wf)
	# dimensionEntropy(Wf * bestBasis)
	# orthogonal = [0 -1; 1 0] * bestBasis
	# b = [orthogonal bestBasis]


	f = polarityPlotWithBasis(model.W, model.X, model.W * b, b' * model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex = itemColorIndex, title="2d Bridge-Based Ranking: Synthetic Data")
	save("plots/community-notes-large-polarity-plot-with-basis-2d.png", scene)
	# scatter(f[3,2], entropyChartData)

	f = polarityPlot(model.W * b, b' * model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="2D Bridge-Based Ranking: Community Notes (Sample)")
	save("plots/community-notes-large-polarity-plot-2d.png", f)

	score2d = model.X' * bestBasis

end

# New 3d model
begin
	model = factorizeMatrixNoIntercepts(Matrix(Y), 3, lambda);
	# modelCNlarge3dCoreOnly = model

	# model = modelCNlarge3d

	lossMasked(model(Y), Y)
	penalty(lambda, model)


	(bestBasis, loss) = findLowEntropyDimension(model.W)
	dimensionEntropy(model.W * bestBasis)
	(worstBasis, loss) = findHighEntropyDimension(model.W)

	(highInfoBasis, loss) = findHighInformationDimension(model.W)


	b2 = normalize(cross(bestBasis, worstBasis))
	b3 = cross(b2, bestBasis)
	b = [b2 b3 bestBasis]


	# Okay now plot users in 3d, and show arrows with the direction of max entropy, min entropy, and max information.

	f = Figure()

	s1 = scatter(model.W[:,1], model.W[:,2], model.W[:,3], markersize = 10, marker = :utriangle ,color=userColors, transparency=true)


	b = highInfoBasis
	a1 = arrows!([0], [0], [0], b[1,:], b[2,:], b[3,:], arrowsize = .1, lengthscale = 1,
		    arrowcolor = [:green], linecolor = [:green], linestyle=:dash, label="arrow 1")

	b = worstBasis
	a1 = arrows!([0], [0], [0], b[1,:], b[2,:], b[3,:], arrowsize = .1, lengthscale = 1,
		    arrowcolor = [:red], linecolor = [:red], linestyle=:dash, label="arrow 1")


	b = bestBasis
	a1 = arrows!([0], [0], [0], b[1,:], b[2,:], b[3,:], arrowsize = .1, lengthscale = 1,
		    arrowcolor = [:blue], linecolor = [:blue], linestyle=:dash, label="arrow 1")



	# save("plots/3d.png", s1)

	score3d = model.X' * bestBasis



	W2 = model.W[:,1:1:2]
	X2 = model.X[1:1:2,:]
	
	(bestBasis2, loss) = findLowEntropyDimension(W2)
	orthogonal = [0 -1; 1 0] * bestBasis2
	b2 = [orthogonal bestBasis2]

	f = polarityPlotWithBasis(W2, X2, W2 * b2, b2' * X2, map(c -> :gray, W2[:,1]), itemColors, itemColorIndex = itemColorIndex, title="2d Bridge-Based Ranking: Synthetic Data")



	userWeights = model.W * bestBasis

	Y

	f = Figure()

	# Weighted average votes
	priorWeight = 200
	newScore = ( (userWeights' * Y) ./ ( userWeights' * (Y .!= 0) .+ priorWeight ) )[:]
	d = newScore .- score1d
	sum((d) .^ 2)
	norm(d)

	# scatter(f[1,1], score1d, score2d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	# scatter(f[1,2], score1d, score3d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	scatter(f[2,2], score1d, newScore[:], color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))


	priorWeight = 2000
	newScore = ( sum(Y, dims=1) ./ ( sum(Y .!= 0, dims=1) .+ priorWeight) )[:]
	d = newScore .- score1d
	sum((d) .^ 2)

	scatter(f[3,2], score1d, newScore[:], color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))




	# polarityPlotUsers(twoDProjection * b2 , userColors)
end



# lossMasked(model.W * b * (b' * model.X), Y)



begin
	model = factorizeMatrixNoIntercepts(Matrix(Y), 4, lambda);
	# modelCNlarge4d = model
	(bestBasis, loss) = findLowEntropyDimension(model.W)

	score4d = model.X' * bestBasis


	# orthogonal = [0 -1; 1 0] * bestBasis
	# b = [orthogonal bestBasis]
	# dimensionEntropy(model.W * bestBasis)
	# polarityPlot(model.W * b, b' * model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="2D Bridge-Based Ranking: Community Notes (Sample)")
	# save("plots/basis10d.png", scene)

end



begin	
	f = Figure()
	scatter(f[1,1], score1d, score2d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	scatter(f[1,2], score1d, score3d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	scatter(f[1,3], score1d, score4d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	# scatter(f[2,3], score3d, score4d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	save("plots/1d vs 2d and 3d.png", f)
end



begin
	(Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix(table="sampleDataSet3", coreOnly = true);
	model = factorizeMatrixNoIntercepts(Matrix(Y), 3, lambda);
	# modelCNlarge3dCoreOnly = model

end


	# m = mean(W * d.Vt' * bestBasis)
	# if m < 0 
	# 	bestBasis = bestBasis * -1
	# end

	# # Now flip the axis based on known left-wing items
	# bb = alignLeftRight(bb' * X, bb, itemIds)


