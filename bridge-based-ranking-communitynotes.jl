include("matrix-factorization.jl")
include("change-basis.jl")
include("polarity-plot.jl")
include("create-training-set.jl")
include("create-training-set-buterin.jl")
include("community-notes-data.jl")
include("entropy-based-dimension-reduction.jl")


# (Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix();
itemColorIndex = Dict(
	:red => "Unhelpful",
	:green => "Helpful",
	:gray => "Needs More Ratings",
	:title => "Community Notes Status"
)

(Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix(table="sampleDataSet3", coreOnly=false);

# (Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix(table="sampleDataSet3");
include("matrix-factorization.jl")


lambda1d = .03
lambda = .01

# First run 1-d model
begin
	# Random.seed!(7)
	model = factorizeMatrixIntercepts(Matrix(Y), 1, lambda1d, false)
	# modelCNlarge1dCoreOnly = model
	# model = modelCNlarge1dCoreOnly

	b = swapLeftRight(model.X, itemIds)
	polarityPlot([model.W * b .+ model.μ model.B .+ model.μ ], [model.X * b .+ model.μ; model.C .+ model.μ ], map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="Community Notes Data (1D)")

	f = polarityPlotItems([model.X*b .+ model.μ; model.C .+ model.μ ], itemColors, itemColorIndex=itemColorIndex, title="Community Notes Polarity Plot (Notes)")
	save("plots/community-notes-large-items-1d.png", f)

	score1d = model.C[1,:]
end

# Run 2-D model
begin

	model = factorizeMatrixNoIntercepts(Matrix(Y), 2, lambda, true);
	# modelCNlarge2dCoreOnly = model
	# modelCNlarge2d = model
	model = modelCNlarge2d

	(bestBasis, loss) = findLowEntropyDimension(model.W)

	orthogonal = [0 -1; 1 0] * bestBasis
	b = [orthogonal bestBasis]

	dimensionEntropy(model.W * bestBasis)

	f = polarityPlotWithBasis(model.W, model.X, model.W * b, b' * model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex = itemColorIndex, title="2d Bridge-Based Ranking: Synthetic Data")
	save("plots/community-notes-large-with-basis-2d.png", f)
	# scatter(f[3,2], entropyChartData)

	f = polarityPlot(model.W * b, b' * model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="2D Bridge-Based Ranking: Community Notes (Sample)")
	save("plots/community-notes-large-2d.png", f)

	score2d = model.X' * bestBasis

	f = polarityPlotUsers(model.W * b, map(c -> :gray, model.W[:,1]), title="Community Notes Polarity Plot (Users)")
	save("plots/community-notes-large-users-2d.png", f)


	f = polarityPlotItems(b' * model.X, itemColors,  itemColorIndex=itemColorIndex, title="Community Notes Polarity Plot (Items)")
	save("plots/community-notes-large-items-2d.png", f)

end

# New 3d model
# lambda = .005
begin
	# include("matrix-factorization.jl")
	model = factorizeMatrixNoIntercepts(Matrix(Y), 3, lambda, true);
	modelCNlarge3d = model
	# model = modelCNlarge3d

	(bestBasis, loss) = findLowEntropyDimension(model.W)
	(worstBasis, loss) = findHighEntropyDimension(model.W)

	orthogonal = normalize(cross(bestBasis, worstBasis))
	b2 = normalize(cross(orthogonal, bestBasis))
	b3 = cross(b2, bestBasis)
	b = [b2 b3 bestBasis]


	# Translate our data to a new basis where the low-entropy basis is up.
	W = model.W * b
	(worstBasis, loss) = findHighEntropyDimension(W)

	f = Figure()

	# s1 = scatter(model.W[:,1], model.W[:,2], model.W[:,3], markersize = 10, marker = :utriangle ,color=map(c -> (:gray, .2), model.W[:,1]), transparency=true)
	s1, a, p = scatter(W[:,1], W[:,2], W[:,3], markersize = 10, marker = :utriangle ,color=map(c -> (:gray, .2), W[:,1]), transparency=true)


	a1 = arrows!([0], [0], [0], [0], [0], [1], arrowsize = .15, lengthscale = 1,
		    arrowcolor = [:blue], linecolor = [:blue], linestyle=:dash, label="arrow 1")


	b = worstBasis
	a1 = arrows!([0], [0], [0], b[1,:], b[2,:], b[3,:], arrowsize = .15, lengthscale = 1,
		    arrowcolor = [:red], linecolor = [:red], linestyle=:dash, label="arrow 1")

	save("plots/community-notes-3d.png", s1)

	score3d = model.X' * bestBasis

end



begin	
	f = Figure()
	scatter(f[1,1], score1d, score2d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors),
				 axis = (; title="1d vs 2d model", xlabel = "1d model intercept", ylabel = "2d model common ground factor")
	)
	scatter(f[1,2], score1d, score3d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors),
				 axis = (; title="1d vs 3d model", xlabel = "1d model intercept", ylabel = "3d model common ground factor")

		)

	f[1,3] = itemLegend(f, itemColors, itemColorIndex)

	# scatter(f[1,3], score1d, score4d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	# scatter(f[2,3], score3d, score4d, color=map(c -> (c,c == :gray ? grayAlpha : alpha), itemColors))
	save("plots/1d vs 2d and 3d.png", f)
end



# Here are some incomplete experiments with estimating the common-ground factor for an item as a weighted average of the user votes.
begin

	W2 = model.W[:,1:1:2]
	X2 = model.X[1:1:2,:]
	
	(bestBasis2, loss) = findLowEntropyDimension(W2)
	orthogonal = [0 -1; 1 0] * bestBasis2
	b2 = [orthogonal bestBasis2]

	f = polarityPlotWithBasis(W2, X2, W2 * b2, b2' * X2, map(c -> :gray, W2[:,1]), itemColors, itemColorIndex = itemColorIndex, title="2d Bridge-Based Ranking: Synthetic Data")


	userWeights = model.W * bestBasis

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

end
