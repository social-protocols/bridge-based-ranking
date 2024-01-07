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
  v = v[:,1]
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

    loss = dimensionEntropy(dimensionEntropyModel(W))
    # dimensionEntropyModel.B

  b = dimensionEntropyModel.B ./ norm(dimensionEntropyModel.B)

  m = mean(W * b)
  if m < 0 
    b = b * -1
  end

  return (b, loss)
end


function findHighEntropyDimension(W)
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
      loss = -dimensionEntropy(result)
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



function findHighInformationDimension(W)
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
      loss = -dimensionInformation(result)
      # println("Loss", loss)
      # println("B", dimensionEntropyModel.B)
      loss
    end


    u = Flux.update!(optim, dimensionEntropyModel, gs[1])

  end

    loss = dimensionInformation(dimensionEntropyModel(W))
    # dimensionEntropyModel.B

  b = dimensionEntropyModel.B ./ norm(dimensionEntropyModel.B)

  m = mean(W * b)
  if m < 0 
    b = b * -1
  end

  return (b, loss)
end




function binaryentropy(p)
  if p == 1
    return 0
  end

  if p == 0
    return 0
  end

  -(p * log2(p)) -(1-p)*log2(1-p)

end


function dimensionEntropy(xs)
  positives = [x for x in xs if x > 0]
  negatives = [x for x in xs if x < 0]


  up = sum(positives)
  down = abs(sum(negatives))

  p = up / (up + down)
  return  binaryentropy(p)
end



function dimensionInformation(xs)
  positives = [x for x in xs if x > 0]
  negatives = [x for x in xs if x < 0]

  up = sum(positives)
  down = abs(sum(negatives))

  p = up / (up + down)

  return up * (1 + log2(p)/log2(2))
end




