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


n = 200
m = 100

Random.seed!(6)
s = createTrainingSet(n, m,.1); 
Y = s.upvotes .+ s.votes .* (s.upvotes .- 1);

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



struct PredictiveModel 
  W
  X
  priorWeight
end


hyperPriorWeight = 10
NewPredictiveModel(n, m, k) =
  PredictiveModel(
    # rand(n,k)*.1 .- .05, rand(k,m)*.1 .- .05, [rand()*hyperPriorWeight]
    rand(n,k)*.1 .- .05, rand(k,m)*.1 .- .05, [rand()*hyperPriorWeight]
  )

(model::PredictiveModel)(Y) = begin

	priorWeight = model.priorWeight

	# Average vote (given a vote) across all items
	# priorAverage = sum((Y .!== 0) .* Y) / sum((Y .!== 0))

	M = (Y .!= 0)
	Yadj = M .* (Y .- priorAverage)

	# weightedSum = model.W'*Y
	# weight = model.W'*M

	# bayesianAverages = (weightedSum .+ priorAverage*model.priorWeight) ./ ( weight .+ model.priorWeight)
	inv = (1 .- Matrix(I, n, n))

	# Here we calculate the weighted Bayesian average of all votes on this product not include this user
	weightedSumSkip = inv .* model.W * Y
	weightSkip = abs.(inv .* model.W * M)


	# total weight for each item based on all users not including current user.
	bayesianAveragesSkip = (weightedSumSkip) ./ ( weightSkip .+ priorWeight)

	# prediction for each vote based on all other users weights
	Y_hat = (bayesianAveragesSkip .* model.X) .* M 


	return Y_hat
end

Flux.@functor PredictiveModel

function predictiveModelLoss(Y_hat, Y)
  err = ( (Y .!= 0) .* (Y .- Y_hat) )
  return norm( err )
end



function trainPredictiveModel(Y, k)
  # (Y, itemMeans) = meanNormalize(Y)
  (n, m) = size(Y) 
  model = NewPredictiveModel(n,m,k)

  Y
  Y_hat = model(Y)
  predictiveModelLoss(model(Y), Y)

  # model = MatrixFacorizationModel(n,m,k)

  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.


  trainGeneric(model, optim, Y, predictiveModelLoss)
end


model = factorizeMatrixNoIntercepts(Y, 1, .03, true)

model = trainPredictiveModel(Y, 1)
f = Figure()
scatter(f[1,1], collect(1:n), model.W[:], color=s.userColors)
model.priorWeight







"""
principal of the peer truth serum is that the reward is inversely proportional to the prior. If the prior is already high, and you vote 1, you don't get a big reward if the next guy
votes 1. So a simple information cascade where everyone votes 1 produces little value. 

So what we need is a cost function where the derivative of the cost function, wrt the user's weight, is positive as long as the user's vote is "helpful".


If we produce an estimate basded on a weighted bayesian average of all users who voted this product before and including me, and the cost function is based on how closely this estimate 
predicts votes of users after me, then the derivative of the cost will be in the right direction.

SO next step is to create a matrix that has the weighted sum and weight of all users up to including me, for each product.
"""


# randomly assign an order to each vote
Random.seed!(6)
voteOrder = reshape(shuffle(1:(n*m)), n, m) .* M

itemNumber = 2
function priorUserMatrix(voteOrder, itemNumber)
	v = voteOrder[:, itemNumber]
	(v .!= 0) .* (v .< v')
end


# pum = Matrix(undef, n, n)
# for i in 1:n
# 	for j in 1:n
# 		pum[i,j] = voteOrder[i, itemNumber] != 0 && voteOrder[j, itemNumber] != 0 && i != j ? voteOrder[i, itemNumber] < voteOrder[j, itemNumber] : missing
# 	end
# end

pum = priorUserMatrix(voteOrder, itemNumber)

pums = [priorUserMatrix(voteOrder, itemNumber) for itemNumber in 1:m]


weightedSumPrev = [pum .* model.W * Y[:, itemNumber] for itemNumber in 1:m]
weightPrev = [abs.(pum .* model.W * M[:, itemNumber]) for itemNumber in 1:m]


# total weight for each item based on all users not including current user.
bayesianAveragesPrev = [ (weightedSumPrev[itemNumber]) ./ ( weightPrev[itemNumber] .+ priorWeight) for itemNumber in 1:m ]
# bayesianAveragesPrev[1]
bayesianAveragesPrev = permutedims(vcat(bayesianAveragesPrev'...))





function trainGeneric(model, optim, Y, lossFunction)
  losses = []

  nEpochs = 100
  p = Progress(nEpochs)

  for epoch in 1:nEpochs
    Flux.train!(model, [Y], optim) do m, item
      lossFunction(m(item), item)
    end

    # loss = lossMasked(model(Y), Y, lambda, model)
    loss = lossFunction(model(Y), Y)

    # if length(losses) > 0
    #   lastLoss = losses[end]
    #   if abs((loss - lastLoss)/lastLoss) < 0.00001
    #     print("Stopping at loss", loss)
    #     break
    #   end
    # end

    push!(losses, loss)

    # @show loss
    next!(p; showvalues = [(:loss,loss), (:loss, loss)])
  end

  # print("LOsses", losses)
  finalLoss = losses[end]
  @show finalLoss

  finalloss = lossFunction(model(Y), Y)
  @show finalloss

  return model

end







