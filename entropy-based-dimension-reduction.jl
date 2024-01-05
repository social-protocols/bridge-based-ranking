using Flux
using LinearAlgebra
using Optimisers
using ProgressMeter
using LinearAlgebra


struct DimensionEntropyModel
  B
end



NewDimensionEntropyModel(j::Int) = begin
  v = rand(j,1).*2 .- 1
  v = v / norm(v)
  return DimensionEntropyModel(
    v
  )
end


(model::DimensionEntropyModel)(W) = begin
  return W * model.B ./ norm(model.B)
end


Flux.@functor DimensionEntropyModel

function findLowEntropyDimension(W)
  (n, m) = size(W) 

  dimensionEntropyModel = NewDimensionEntropyModel(m)
  # model = DimensionEntropyModel(n,m,k)
  # dimensionEntropyModel.B

  # opt = Optimisers.Adam(.5)
  opt = Flux.AMSGrad(0.1)
  optim = Optimisers.setup(opt, dimensionEntropyModel)  # will store optimiser momentum, etc.


  @showprogress for epoch in 1:100

    gs = Flux.gradient(dimensionEntropyModel) do m
      result = m(W)
      # println("Result", result)
      loss = dimensionEntropy(result)
      # println("Loss", loss)
      # println("B", dimensionEntropyModel.B)
      loss
    end


    u = Flux.update!(optim, dimensionEntropyModel, gs[1])

  end

    # loss = dimensionEntropy(dimensionEntropyModel(W))
    # dimensionEntropyModel.B

  return (dimensionEntropyModel.B ./ norm(dimensionEntropyModel.B), loss)
end






