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
    rand(n,k), rand(n,1), rand(k,m), rand(1,m), rand()
  )

"""
MatrixFacorizationModelNointercept instantiates a model where all intercepts are set to zero.
"""

MatrixFacorizationModelNointercept(n, m, k) =
  MatrixFacorizationModel(
    # rand(n,k), rand(n,1), rand(k,m), rand(m,1)
    rand(n,k), repeat([0],n), rand(k,m), repeat([0],m)', 0
  )


(model::MatrixFacorizationModel)(Y) = begin
  # return sigmoid(model.W * model.X .+ model.B .+ model.C')
  return model.W * model.X .+ model.B .+ model.C .+ model.μ
end

Flux.@functor MatrixFacorizationModel


sqnorm(x) = sum(abs2, x)


"""
lossMasked calculates mean square error loss, but only including loss for items in the original matrix where Y != 0
For example, if Y indicates votes (1 for upvote and -1 for downvote and 0 for now vote), we only calculate the loss
when users voted on an item.

"""
function lossMasked(Y_hat, Y, lambda, model)
  # norm(M .* (Y .- Y_hat))
  totalCost = norm( (Y .!= 0) .* (Y .- Y_hat))
  # regularization = lambda * sum(sqnorm, Flux.params(m))

  regularization = lambda * ( sum(model.W .^ 2) + sum(model.X .^ 2) + sum(model.B .^ 2) + sum(model.C .^ 2) + (model.μ .^ 2 ) )

  return totalCost + regularization
end

"""
Factorize the matrix but without intercepts (all intercepts = 0)

"""
function factorizeMatrixNoIntercepts(Y, k, lambda)
  (n, m) = size(Y) 

  model = MatrixFacorizationModelNointercept(n,m,k)

  # opt = Optimisers.Adam(0.1)
  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

  Optimisers.freeze!(optim.B)
  Optimisers.freeze!(optim.C)
  Optimisers.freeze!(optim.μ)

  losses = []


  nEpochs = 100
  p = Progress(nEpochs)

  for epoch in 1:nEpochs

    Flux.train!(model, [Y], optim) do m, item
      Y = item
      lossMasked(m(Y), Y, lambda, m)
    end

    loss = lossMasked(model(Y), Y, lambda, model)

    # if length(losses) > 0
    #   lastLoss = losses[end]
    #   if abs((loss - lastLoss)/lastLoss) < 0.001
    #     print("Stopping at loss", loss)
    #     break
    #   end
    # end

    push!(losses, loss)

    # @show loss
    next!(p; showvalues = [(:loss,loss)])
  end

  # print("LOsses", losses)

  return model

end


"""
Factorize the matrix including intercepts for users and items and a global intercept

"""

function factorizeMatrixIntercepts(Y, k, lambda)
  (n, m) = size(Y) 
  model = MatrixFacorizationModel(n,m,k)
  # model = MatrixFacorizationModel(n,m,k)

  # opt = Optimisers.Adam(0.1)
  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, model)  # will store optimiser momentum, etc.

  # Optimisers.freeze!(optim.B)
  # Optimisers.freeze!(optim.C)
  # lossMasked(model(Y), Y, lambda)


  losses = []

  nEpochs = 100
  p = Progress(nEpochs)

  for epoch in 1:nEpochs

    Flux.train!(model, [Y], optim) do m, item
      Y = item
      lossMasked(m(Y), Y, lambda, m)
    end

    loss = lossMasked(model(Y), Y, lambda, model)

    # if length(losses) > 0
    #   lastLoss = losses[end]
    #   if abs((loss - lastLoss)/lastLoss) < 0.001
    #     print("Stopping at loss", loss)
    #     break
    #   end
    # end

    push!(losses, loss)

    # @show loss
    next!(p; showvalues = [(:loss,loss)])
  end

  return model

end

