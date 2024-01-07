"""

An experiment with bridge-based ranking using matrix factorization.

"""

include("matrix-factorization.jl")
include("change-basis.jl")
include("polarity-plot.jl")
include("create-training-set.jl")
include("create-training-set-buterin.jl")
include("community-notes-data.jl")
include("entropy-based-dimension-reduction.jl")

# First, use some synthesized data. This data is very unbalanced -- there are many more right-wing users than left-wing users, and most users are "thugs" -- they
# vote primarily based on politics and not helpfulness.


n = 400
m = 200
lambda = .08

Random.seed!(6)
s = createTrainingSet(trunc(Int,n/8), trunc(Int,m/8),.1); 
Y = s.upvotes .+ s.votes .* (s.upvotes .- 1)
M = s.votes

userColorIndex = Dict(
	:cyan => "Good-Faith Liberal",
	:blue => "Liberal Thug",
	:magenta => "Good-Faith Conservative",
	:red => "Conservative Thug",
	:title => "User Type"
)
itemColorIndex = Dict(
	:cyan => "Helpful Right",
	:blue => "Unhelpful Right",
	:magenta => "Helpful Left",
	:red => "Unhelpful Left",
	:title => "Item Type"
)

lambda = .15
# First, do Matrix factorization with one dimension -- the same algorithm used by community notes.
model = factorizeMatrixIntercepts(Y, 1, lambda)

# Plot the polarity factor against the common ground factor (the intercepts)
# The intercept for users doesn't mean much, but helpful items should have a positive intercept, and unhelpful items should have a negative intercept
# However, with synthetic data we sometimes get weird results where there is a linear relationship between the slope and intercept 
polarityPlot([model.W .+ model.μ model.B .+ model.μ], [model.X .+ model.μ;	model.C .+ model.μ], s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="Bridge-Based Ranking: Synthetic Data (1-D)")

# Now use an alternative algorithm: matrix factorization with two dimensions but without intercepts. The idea is that one dimension will correspond roughly to the polarity factor, and one dimension will correspond to the common ground factor.
model = factorizeMatrixNoIntercepts(Y, 2, lambda)


# include("matrix-factorization.jl")
# model = factorizeMatrixNoIntercepts(Y, 3, .08)
# mean(Y)
# model.μ

# However, the results will be arbitrary rotated in 2d space. We use the changeBasis function to find the min/max entropy axes and make these correspond to the horizontal/vertical.
(b, entropyPlotData) = changeBasis(model.W, model.X, []);

polarityPlot(model.W * b, b' * model.X, s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="2D Bridge-Based Ranking: Synthetic Data")

scene = polarityPlotWithBasis(model.W, model.X, model.W * b, b' * model.X, s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="@d Bridge-Based Ranking: Synthetic Data")
save("plots/synthetic-data-polarity-plot-with-basis-change.png", scene)


# Do the same analysis using a data set created by Vitalik Buterin for his own simplified implementation described here:
# https://vitalik.eth.limo/general/2023/08/16/communitynotes.html
# https://github.com/ethereum/research/blob/master/community_notes_analysis/basic_algo.py

(Y, itemColors) = createTrainingSetButerin()

itemColorIndex = Dict(
	:cyan => "Good",
	:green => "Good but Extra Polarizing",
	:brown => "Neutral",
	:orange => "Bad",
	:title => "Note Type"
)


# First, using the single-dimension with intercept model
model = factorizeMatrixIntercepts(Y, 1, lambda)
userColors = map(c -> :gray, model.W[:,1])
polarityPlot([model.W .+ model.μ model.B .+ model.μ], [model.X .+ model.μ;	model.C .+ model.μ], userColors, itemColors, itemColorIndex=itemColorIndex)

# Next, using the two-dimensional model. 
model = factorizeMatrixNoIntercepts(Y, 2, 0)

# Again changing basis to make the horizontal align with the polarity factor and the vertical with the "common ground factor"
(b, entropyPlotData) = changeBasis(model.W, model.X, itemIds);

polarityPlot(model.W*b, b'model.X, userColors, itemColors, itemColorIndex=itemColorIndex)




# Now, use community notes data

(Y, userColors, itemColors, userIds, itemIds) = loadCommunityNotesRatingsMatrix();

itemColorIndex = Dict(
	:red => "Unhelpful",
	:green => "Helpful",
	:gray => "Needs More Ratings",
	:title => "Commmunity Notes Status"
)

# First using the single-dimension with intercept model
lambda = .15
model = factorizeMatrixIntercepts(Matrix(Y), 1, lambda)
polarityPlot([model.W .+ model.μ model.B .+ model.μ ], [model.X .+ model.μ; model.C .+ model.μ ], map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="Community Notes Data (1D)")


scene = polarityPlotItems([model.X .+ model.μ; model.C .+ model.μ ], itemColors, itemColorIndex=itemColorIndex, title="Community Notes Polarity Plot (Notes)")
save("plots/community-notes-items-polarity-plot-1d.png", scene)


# Then using the two-dimensional model without intercepts
model = factorizeMatrixNoIntercepts(Matrix(Y), 2, lambda);
# Again changing basis to make the horizontal align with the polarity factor and the vertical with the "common ground factor"
(b, entropyPlotData) = changeBasis(model.W, model.X, itemIds);


# X = b' * model.X
# findall(item -> item > 1.0, X[1,:])
# rightWingItems = itemIds[findall(item -> item > 1.0, X[1,:])]


scene = polarityPlot(model.W * b, b' * model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="2D Bridge-Based Ranking: Community Notes (Sample)")

scene = polarityPlotWithBasis(model.W, model.X, model.W * b, b' * model.X, map(c -> :gray, model.W[:,1]), itemColors, itemColorIndex=itemColorIndex, title="2D Bridge-Based Ranking: Community Notes (Sample)")

save("plots/community-notes-polarity-plot-with-basis-change.png", scene)

scene = polarityPlotItems(b' * model.X, itemColors, itemColorIndex=itemColorIndex, title="Community Notes Polarity Plot (Notes)")
save("plots/community-notes-items-polarity-plot.png", scene)


# model = factorizeMatrixNoIntercepts(Matrix(Y1), 2, lambda);
# scene = polarityPlot(model.W, model.X, map(c -> :gray, model.W[:,1]), itemColors1, itemColorIndex=itemColorIndex, title="Community Notes Sample Chart")

scene = polarityPlotUsers(model.W * b, map(c -> :gray, model.W[:,1]), title="Community Notes Polarity Plot (Users)")
save("plots/community-notes-users-polarity-plot.png", scene)







