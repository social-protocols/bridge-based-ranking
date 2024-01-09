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

Random.seed!(6)
s = createTrainingSet(n, m,.1); 
Y = s.upvotes .+ s.votes .* (s.upvotes .- 1);
# M = s.votes

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
model = factorizeMatrixNoIntercepts(Y, 2, lambda, true)
polarityPlot(model.W, model.X, s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="2D Bridge-Based Ranking: Synthetic Data")



include("matrix-factorization.jl")

# First, do Matrix factorization with one dimension -- the same algorithm used by community notes.
lambda = .01
lambda1d = .02
model = factorizeMatrixIntercepts(Y, 1, lambda1d, true)

# f = polarityPlot([model.W .+ model.μ model.B .+ model.μ], [model.X .+ model.μ;	model.C .+ model.μ], s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="Bridge-Based Ranking: Synthetic Data (1-D)")

# Plot the polarity factor against the common ground factor (the intercepts)
# The intercept for users doesn't mean much, but helpful items should have a positive intercept, and unhelpful items should have a negative intercept
# However, with synthetic data we sometimes get weird results where there is a linear relationship between the slope and intercept 
f = polarityPlot([model.W .+ model.μ model.B .+ model.μ], [model.X .+ model.μ;	model.C .+ model.μ], s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="Bridge-Based Ranking: Synthetic Data (1-D)")
save("plots/synthetic-data-1d.png", f)

# Now use an alternative algorithm: matrix factorization with two dimensions but without intercepts. The idea is that one dimension will correspond roughly to the polarity factor, and one dimension will correspond to the common ground factor.
model = factorizeMatrixNoIntercepts(Y, 2, lambda, true)
polarityPlot(model.W, model.X, s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="2D Bridge-Based Ranking: Synthetic Data")

(b, entropyPlotData) = changeBasis(model.W, model.X, []);
f = polarityPlotWithBasis(model.W, model.X, model.W * b, b' * model.X, s.userColors, s.itemColors, userColorIndex = userColorIndex, itemColorIndex = itemColorIndex, title="@d Bridge-Based Ranking: Synthetic Data")
save("plots/synthetic-data-with-basis-change.png", f)



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


include("matrix-factorization.jl")
# First, using the single-dimension with intercept model
model = factorizeMatrixIntercepts(Y, 1, lambda, false)
userColors = map(c -> :gray, model.W[:,1])
f = polarityPlot([model.W .+ model.μ model.B .+ model.μ], [model.X .+ model.μ;	model.C .+ model.μ], userColors, itemColors, itemColorIndex=itemColorIndex)
save("plots/buterin-1d.png", f)


# Next, using the two-dimensional model. 
model = factorizeMatrixNoIntercepts(Y, 2, lambda, true)
# Again changing basis to make the horizontal align with the polarity factor and the vertical with the "common ground factor"
(b, entropyPlotData) = changeBasis(model.W, model.X, []);
f = polarityPlot(model.W*b, b'model.X, userColors, itemColors, itemColorIndex=itemColorIndex)
save("plots/buterin-2d.png", f)







