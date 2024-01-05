# This is an exercise in creating a model to estimate the priorWeight used to
# calculate the vote probability given a count of upvotes and votes. THe idea is that
# the correct priorWeight is whatever makes the bayesian average estimates of the 
# upvote rate closest to our final best estimate of the upvote rate.
# 
# In this simulation, we create nItems items with upvote rates drawn from a
# beta distribution with arbitrary parameters. We then have users vote on
# those items nSteps times (with the probability of an upvote equal to that
# items upvote rate). This gives us a matrix X representing vote data we might
# actually get (except every post has the exact same number of votes).
#
# We then have a model with a single parameter to optimize -- the priorWeight.
# We estimate the probability of an upvote for each item at each step using the
# Bayesian Average formula using this prior weight. We compare this estimate at
# each step to the final, best estimate (after the final step). The value of priorWeight
# that makes the estimates at each step closest to the final estimate is the best
# value for priorWeight.


using PoissonRandom
using Random, Distributions
using DataFrames
using Flux: params
using Flux
using ProgressMeter
using Optimisers


# Create a custom Layer. See "Building Layers" under https://fluxml.ai/Flux.jl/stable/models/basics/

struct PriorWeightModel
  PriorWeight
  PriorAverage
end

PriorWeightModel(mean) =
  PriorWeightModel(rand(1)*10, mean)


# This model takes as input the cumulative number of upvotes for each item at each step
# and returns out bayesian average probability estimate for each item at each step.
(priorWeightModel::PriorWeightModel)(C) = begin

	nItems, nSteps = size(C)

  steps = collect(1:nSteps)

	w = priorWeightModel.PriorWeight[1]
	priorAverage = priorWeightModel.PriorAverage[1]

	# This is thie Bayesian average formula
	Y_hat = (C .+ w*priorAverage) ./ (steps' .+ w)


	# Forget the last estimate, because we don't have a subsequent data point to compare it to
	Y_hat[:,1:nSteps-1]

	# Y_hat[:,1:5]
end

Flux.@functor PriorWeightModel

# If called with a single scalar, the standard library's entropy function calculates entropy as -p * log(p)
#  -- it doesn't add the -(1-p) * log(1-p). The reason is it's argument is supposed to represent a discrete probability distribution,
# which should have two values for a Bernoulli distribution. So basically calling entropy with a single scalar
# makes no sense at all. Anyway, here I define entropy of a bernoulli distribution with the given parameter.
binaryentropy(p) = entropy([p, 1-p])


# function binaryentropy(p)
#   if p == 1
#     return 0
#   end
#   p * log(p,2)

# end



# This loss function is the total relative entropy in bits between the
# estimated probabilities at each step (Y_hat) and the best final estimate.
# This loss function es effectively the standard cross-entropy (it differs
# from cross entropy by a constant linear transformation*), except the
# values are more human-interpretable (at least to some people). Zero entropy means
# all estimates were perfect (Y_hat always equals Y). Then for
# reference, always estimating 50% results in average relative entropy of 1
# bit every time, so if your loss is greater than 1 your model sucks. 

function loss(Y_hat, Y)
	Hp = binaryentropy.(Y)
	# Hpq = Flux.binarycrossentropy.(Y_hat, Y)
	# return mean(Hpq .- Hp ) / log(2)
	# ( mean(Flux.binarycrossentropy.(Y_hat, Y) .- Hp ) ) / log(2)

	sum(Flux.binarycrossentropy.(Y_hat, Y) .- Hp ) / log(2)
	# sum(Flux.binarycrossentropy.(Y_hat, Y) ) / log(2)
end


createTrainingSet = function(alpha, beta, nItems, nSteps) 

	d = Beta(alpha,beta)
	mean(d)

	# sample from beta distribution nItems times to get the "true"
	# vote probabilities for each item
	itemProbs = rand(d, nItems)


	# Get votes on each item at each step (1 or 0) by drawing
	# from Bernoulli distribution for each item nSteps times
	X_vec = rand.(Bernoulli.(itemProbs),nSteps)

	# Convert to a matrix
	X = reduce(vcat,transpose.(X_vec))

	# Now we want the cumulative votes at each step.
	C = cumsum(X, dims=2)


	# Now get our best estimate -- the actual vote ratio after the last step.
	Y = C[:,nSteps-1] ./ nSteps

	return itemProbs, mean(X), C, Y

end


itemProbs, u, C, Y = createTrainingSet(100, 200, 50, 100)


# opt = Flux.Nesterov(0.001, 2.0)
# opt = Flux.Adam(0.1)
# opt = Flux.AdaGrad(0.1, 1.0e-6)
# opt = Flux.AdaDelta(0.8, 1.0e-7)

# Best so far
opt = Flux.AMSGrad(0.1)
# opt = Flux.NAdam(0.002, (0.89, 0.995))

# opt = AdamW(0.001, (0.89, 0.995), 0.1)
# opt = AdamW(0.001, (0.89, 0.995), 0.1)
# opt = AdaBelief()

# p = [10.0]
# priorWeightModel = PriorWeightModel(p,u)

priorWeightModel = PriorWeightModel(u)

optim = Optimisers.setup(opt, priorWeightModel)  # will store optimiser momentum, etc.

Optimisers.freeze!(optim.PriorAverage)

@showprogress for epoch in 1:100

	Flux.train!(priorWeightModel, [[C,Y]], optim) do m, item
		x, y = item
	  loss(m(x), y)
	end
end

print("Model: ", priorWeightModel)

print("Average Cost: ", loss(priorWeightModel(C), Y) / ( size(C)[1] * size(C)[2] ))


