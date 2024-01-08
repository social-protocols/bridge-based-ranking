using Flux
using LinearAlgebra
using Optimisers
using ProgressMeter
using LinearAlgebra


function meanNormalize(Y)

  itemMeans = sum(Y, dims=1) ./ sum(Y .!= 0, dims=1)
  userMeans = sum(Y, dims=2) ./ sum(Y .!= 0, dims=2)

  return (Y .- userMeans .* (Y .!= 0), userMeans)
end


"""
A matrix factorization model, where the matrix factorizes to W*X .+ B .+ C + u

B and C are a column and row vectors representing the intercepts (e.g. user intercepts and item intercepts)

And μ is a single value representing the overall intercept.

"""
struct MatrixFacorizationModel 
  W
  B
  X
  C
  μ
end



MatrixFacorizationModel(n, m, k) =
  MatrixFacorizationModel(
    rand(n,k)*.1 .- .05, rand(n,1)*.1 .- .05, rand(k,m)*.1 .- .05, rand(1,m)*.1 .- .05, [rand()*.05]
  )

"""
MatrixFacorizationModelNointercept instantiates a model where all intercepts are set to zero.
"""

MatrixFacorizationModelNointercept(n, m, k) =
  MatrixFacorizationModel(
    # rand(n,k), rand(n,1), rand(k,m), rand(m,1)
    rand(n,k)*.1 .- .05, repeat([0],n), rand(k,m)*.1 .- .05, repeat([0],m)', [0]
  )


(model::MatrixFacorizationModel)(Y) = begin
  # return sigmoid(model.W * model.X .+ model.B .+ model.C')
  return model.W * model.X .+ model.B .+ model.C .+ model.μ
end

Flux.@functor MatrixFacorizationModel

function lossMaskedSqrt(Y_hat, Y)
  err = ( (Y .!= 0) .* (Y .- Y_hat) )
  return norm( err )
end


"""
lossMasked calculates mean square error loss, but only including loss for items in the original matrix where Y != 0
For example, if Y indicates votes (1 for upvote and -1 for downvote and 0 for now vote), we only calculate the loss
when users voted on an item.

"""
function lossMaskedSSE(Y_hat, Y)
  # return norm( (Y .!= 0) .* (Y .- Y_hat))
  
  # Y_hat = model(Y)

  # Y_hat = model(Y)*2


  # Y = [1 2; 0 4]
  # Y_hat = [10 10; 10 10]
  # Y_hat = [5 5; 5 5]
  # # Y_hat = [2 3 ; 4 5]

  # err = ( (Y .!= 0) .* (Y .- Y_hat) )
  # n = norm( err )
  # sum(err .^ 2 ) .^ (1/2)

  # rmse = mean( err .^ 2 )/m
  # c = n / rmse




  err = ( (Y .!= 0) .* (Y .- Y_hat) )
  m = sum((Y .!= 0))
  return sum( err .^ 2)  

# (Y .!= 0)[101,:]

# Y_hat[1,:]
# err[1,:]
# Y[1,:]

# err[101,:] .^ 2

# sum(err[101,:] .^ 2)

# penalty(0.15, model)

end

# Y_hat = model(Y)
# penalty(1, model)
# lossMasked(Y_hat, Y)

function penalty(lambda, model)
  (n, k) = size(model.W)
  (k, m) = size(model.X)

  return lambda * m * ( sum(model.W .^ 2) + sum(model.B .^ 2) + sum(model.μ .^ 2 ) ) + lambda * n * ( sum(model.X .^ 2) + sum(model.C .^ 2) + sum(model.μ .^ 2 ) )
end



function penaltyCNOld(lambda, model)
  # The loss function from the birdwatch paper
  # Where weights for intercepts are 5x other weights
  return (lambda) * ( sum(model.W .^ 2) + sum(model.X .^ 2) ) + lambda*5 * ( sum(model.B .^ 2) + sum(model.C .^ 2) + sum(model.μ .^ 2 ) )
end


function penaltyCN(lambda, model)
  (n, k) = size(model.W)
  (k, m) = size(model.X)

  return lambda * m * ( sum(model.W .^ 2) + 5 * sum(model.B .^ 2) + 5 * sum(model.μ .^ 2 ) ) + lambda * n * ( sum(model.X .^ 2) + 5 * sum(model.C .^ 2) + 5 * sum(model.μ .^ 2 ) )
end



"""

So I think that a penalty function that just takes the sum of squares of all parameters may be wrong. If the lambda * w_i^2 is be applied to the cost function, then the total penalty
should be calucate
  total cost for w_i is
         ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 + lambda * w_i^2 )
         = lambda * m * w_i^2 + ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 )

  total cost is
         ∑_i lambda * m * w_i^2 + ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 )
         ( lambda * m * ∑_i w_i^2 ) + ∑_i ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 )

  so the total penalty is:
    ( lambda * m * ∑_i w_i^2 )

  and not
    ( lambda * ∑_i w_i^2 )

  so we have to multiply 

This is the derivative of the loss function for w_i (using mean cost /2 as loss function instead of total cost) 
  derivatives are
    d/dw_i 1/2m * ∑_j (w_i*x_j + b_i + c_j - y_i,j)^2 + lambda * w_i^2 
      = d/dw_i 1/2m * ∑_j(w_i^2*x_j^2 + 2*w_i*x_j(b_i + c_j - y_i) + (b_i + c_j - y_i,j)^2) + lambda * w_i^2 
      = 1/2m * ∑_j(2w_i*x_j^2 + 2*x_j(b_i + c_j - y_i,j) + 0) + 2 * lambda * w_i
      = 1/m * ∑_j(w_i*x_j + b_i + c_j - y_i,j)*x_j + lambda * w_i
      = 1/m * ∑_j(f(x_j) - y_i,j)*x_j + lambda * w_i
      = 1/m * ( ∑_j(f(x_j) - y_i,j)*x_j ) + lambda * w_i/m

      we want d/dw_i to have a term lambda * w_i/m, so that if lambda is one, cost is directly proportional to w_i, so the gradient descent wants to resets w_i to zero


Adding all the regularization terms
  cost = (w_i*x_j + b_i + c_j - y_i,j)^2 + lambda * w_i^2 + lambda * x_j^2 + lambda * b_i^2 + lambda * c_j^2

  total cost for w_i is
         ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 + lambda * ( w_i^2 + x_j^2 + b_i^2 + c_j^2 )
         = lambda * m * ( w_i^2 + b_i^2 ) + ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 + x_j^2 + c_j^2 )

  total cost is
         ∑_i lambda * m * ( w_i^2 + b_i^2 ) + ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 + x_j^2 + c_j^2 )
        = ( lambda * m * ∑_i ( w_i^2 + b_i^2) ) + ∑_i ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 + x_j^2 + c_j^2 )
        = ( lambda * m * ∑_i (w_i^2 + b_i^2) ) 
        + ( lambda * n * ∑_j (x_j^2 + c_j^2) ) 
          + ∑_i ∑_j ( (w_i*x_j + b_i + c_j - y_i,j)^2 )
"""




"""
Factorize the matrix but without intercepts (all intercepts = 0)

"""
function factorizeMatrixNoIntercepts(Y, k, lambda, altModel)

  # (Y, itemMeans) = meanNormalize(Y)

  (n, m) = size(Y) 
  model = MatrixFacorizationModelNointercept(n,m,k)

  # opt = Optimisers.Adam(1.0)
  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

  Optimisers.freeze!(optim.B)
  Optimisers.freeze!(optim.C)
  Optimisers.freeze!(optim.μ)

  train(model, optim, Y, lambda, altModel)
end



"""
Factorize the matrix including intercepts for users and items and a global intercept

"""

function factorizeMatrixIntercepts(Y, k, lambda, altModel)

  # (Y, itemMeans) = meanNormalize(Y)
  (n, m) = size(Y) 
  model = MatrixFacorizationModel(n,m,k)

  # model = MatrixFacorizationModel(n,m,k)

  opt = Optimisers.Adam(1.0)
  # opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

  # Optimisers.freeze!(optim.B)
  # Optimisers.freeze!(optim.C)
  # lossMasked(model(Y), Y, lambda)

  train(model, optim, Y, lambda, altModel)
end



function train(model, optim, Y, lambda, altModel)
  losses = []

  nEpochs = 100
  p = Progress(nEpochs)


  if altModel
    lossFunction = lossMaskedSqrt
    penaltyFunction = penaltyCNOld
  else
    lossFunction = lossMaskedSSE
    penaltyFunction = penalty
    # lossFunction = lossMaskedSqrt
    # penaltyFunction = penaltySqrt
  end



  for epoch in 1:nEpochs
    Flux.train!(model, [Y], optim) do m, item
      lossFunction(m(item), item) + penaltyFunction(lambda, m)
    end

    # loss = lossMasked(model(Y), Y, lambda, model)
    lossWithoutPenalty = lossFunction(model(Y), Y)
    loss = lossWithoutPenalty + penaltyFunction(lambda, model)

    if length(losses) > 0
      lastLoss = losses[end]
      if abs((loss - lastLoss)/lastLoss) < 0.00001
        print("Stopping at loss", loss)
        break
      end
    end

    push!(losses, loss)

    # @show loss
    next!(p; showvalues = [(:loss,loss), (:lossWithoutPenalty, lossWithoutPenalty)])
  end

  # print("LOsses", losses)
  finalLoss = losses[end]
  @show finalLoss

  finalLossWithoutPenalty = lossFunction(model(Y), Y)
  @show finalLossWithoutPenalty

  return model

end







# MatrixFacorizationModelNointerceptPrimed(initModel1d) = begin
#   k = 2
#   initW = initModel1d.W
#   initX = initModel1d.X
#   initC = initModel1d.C
#   initU = initModel1d.μ  

#   n = size(initW)[1]
#   m = size(initX)[2]

#   newW = [initW[:,1] repeat([1], n)]
#   newX = [initX[1,:] initC[1,:]]


#   return MatrixFacorizationModel(
#     # rand(n,k), rand(n,1), rand(k,m), rand(m,1)
#     newW, initModel1d.B, newX', repeat([0],m)', initU

#     # newW, repeat([0],n), newX', repeat([0],m)', initU
#   )
# end


# function factorizeMatrixPrimed(Y, initModel1d, lambda)
#   (n, m) = size(Y) 
#   model = MatrixFacorizationModelNointerceptPrimed(initModel1d)
#   # lossMasked(model(Y), Y)

#   lossMasked(model(Y), Y)
#   penalty(lambda, model)

#   # lossMasked(initModel1d(Y), Y)


#   # opt = Optimisers.Adam(0.1)
#   opt = Flux.AMSGrad(0.1)
#   optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

#   Optimisers.freeze!(optim.B)
#   Optimisers.freeze!(optim.C)

#   train(model, optim, Y, lambda)

# end


