using Flux
using LinearAlgebra
using Optimisers
using ProgressMeter
using LinearAlgebra

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
    rand(n,k), rand(n,1), rand(k,m), rand(1,m), [rand()]
  )

"""
MatrixFacorizationModelNointercept instantiates a model where all intercepts are set to zero.
"""

MatrixFacorizationModelNointercept(n, m, k) =
  MatrixFacorizationModel(
    # rand(n,k), rand(n,1), rand(k,m), rand(m,1)
    rand(n,k), repeat([0],n), rand(k,m), repeat([0],m)', [0]
  )



(model::MatrixFacorizationModel)(Y) = begin
  # return sigmoid(model.W * model.X .+ model.B .+ model.C')
  return model.W * model.X .+ model.B .+ model.C .+ model.μ
end

Flux.@functor MatrixFacorizationModel


# sqnorm(x) = sum(abs2, x)


"""
lossMasked calculates mean square error loss, but only including loss for items in the original matrix where Y != 0
For example, if Y indicates votes (1 for upvote and -1 for downvote and 0 for now vote), we only calculate the loss
when users voted on an item.

"""
function lossMasked(Y_hat, Y)
  return norm( (Y .!= 0) .* (Y .- Y_hat))
  
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
  return sum( err .^ 2)  / m

# (Y .!= 0)[101,:]

# Y_hat[1,:]
# err[1,:]
# Y[1,:]

# err[101,:] .^ 2

# sum(err[101,:] .^ 2)

# penalty(0.15, model)

end

function penalty(lambda, model)
  # Lambda for intercepts is 5-times greater than lambda for slopes.
  # Mentioned in the birdwatch papaer: https://github.com/twitter/communitynotes/blob/main/birdwatch_paper_2022_10_27.pdf 
  # m = sum((Y .!= 0))

  # return  lambda/(5) * ( sum(model.W .^ 2) + sum(model.X .^ 2) )  + (lambda) * ( sum(model.B .^ 2) + sum(model.C .^ 2) + sum(model.μ .^ 2 ) )
  return  ( 
    lambda/(5) * ( sum(model.W .^ 2) + sum(model.X .^ 2) )  + (lambda) * ( sum(model.B .^ 2) + sum(model.C .^ 2) + sum(model.μ .^ 2 ) )
  ) ^ (1/2)
end

"""
Factorize the matrix but without intercepts (all intercepts = 0)

"""
function factorizeMatrixNoIntercepts(Y, k, lambda)
  (n, m) = size(Y) 
  model = MatrixFacorizationModelNointercept(n,m,k)

  # opt = Optimisers.Adam(1.0)
  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

  Optimisers.freeze!(optim.B)
  Optimisers.freeze!(optim.C)
  Optimisers.freeze!(optim.μ)

  train(model, optim, Y)
end

function train(model, optim, Y)
  losses = []

  nEpochs = 100
  p = Progress(nEpochs)

  for epoch in 1:nEpochs
    Flux.train!(model, [Y], optim) do m, item
      lossMasked(m(item), item) + penalty(lambda, m)
    end

    # loss = lossMasked(model(Y), Y, lambda, model)
    lossWithoutPenalty = lossMasked(model(Y), Y)
    loss = lossWithoutPenalty + penalty(lambda, model)

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

  finalLossWithoutPenalty = lossMasked(model(Y), Y)
  @show finalLossWithoutPenalty

  return model

end


"""
Factorize the matrix including intercepts for users and items and a global intercept

"""

function factorizeMatrixIntercepts(Y, k, lambda)
  (n, m) = size(Y) 
  model = MatrixFacorizationModel(n,m,k)
  # model = MatrixFacorizationModel(n,m,k)

  # opt = Optimisers.Adam(1.0)
  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

  # Optimisers.freeze!(optim.B)
  # Optimisers.freeze!(optim.C)
  # lossMasked(model(Y), Y, lambda)

  train(model, optim, Y)
end





MatrixFacorizationModelNointerceptPrimed(initModel1d) = begin
  k = 2
  initW = initModel1d.W
  initX = initModel1d.X
  initC = initModel1d.C
  initU = initModel1d.μ  

  n = size(initW)[1]
  m = size(initX)[2]

  newW = [initW[:,1] repeat([1], n)]
  newX = [initX[1,:] initC[1,:]]


  return MatrixFacorizationModel(
    # rand(n,k), rand(n,1), rand(k,m), rand(m,1)
    newW, initModel1d.B, newX', repeat([0],m)', initU

    # newW, repeat([0],n), newX', repeat([0],m)', initU
  )
end


function factorizeMatrixPrimed(Y, initModel1d, lambda)
  (n, m) = size(Y) 
  model = MatrixFacorizationModelNointerceptPrimed(initModel1d)
  # lossMasked(model(Y), Y)

  lossMasked(model(Y), Y)
  penalty(lambda, model)

  # lossMasked(initModel1d(Y), Y)


  # opt = Optimisers.Adam(0.1)
  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

  Optimisers.freeze!(optim.B)
  Optimisers.freeze!(optim.C)

  train(model, optim, Y)

end


