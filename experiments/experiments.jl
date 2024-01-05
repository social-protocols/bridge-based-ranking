



# @showprogress for epoch in 1:100

# 	input, label = C, Y 

#   # Calculate the gradient of the objective
#   # with respect to the parameters within the model:
#   grads = Flux.gradient(priorWeightModel) do m
#       result = m(input)
#       loss(result, label)
#   end

#   # Update the parameters so as to reduce the objective,
#   # according the chosen optimisation rule:
#   Flux.update!(optim, priorWeightModel, grads[1])
# end




priorWeight = [.1]

# gr = gradient(l, X, Y, Flux.Params([priorWeight])
gr = gradient(() -> l(C,Y), Flux.params([priorWeight]))
gr[priorWeight]
l(C,Y)


# https://fluxml.ai/Flux.jl/stable/tutorials/2020-09-15-deep-learning-flux/

using Flux: params






x = rand(Float32, 2)

W = Float32[-0.24037038 -1.1160736]

m = function(x) 
	sigmoid(W*x)
end


# m = Chain(Dense(2, 1), sigmoid)
# params(m)
# m = Chain(Dense(10, 5, relu), Dense(5, 2), softmax)
l(x) = sum(Flux.binarycrossentropy(m(x), [0.5]))
l(x)


# Flux.gradient(m, W)

# params(m)
ps = [W]
grads = gradient(ps) do
    l(x)
end
for p in params(m)
    println(grads[p])
end


using Flux: gradient

mymethod = function(x, W)
	sigmoid(W*x)[1]
end

mymethod(x, W)


gradient(mymethod, x, W)

# Okay important thing I learned: 
# - gradient takes a function, and then a list of **values**, one for each
# - arg in the function, and calculates the gradient **at that value**
# The function should return a scalar, not a matrix

gradient(x, W) do x, W
	z = W*x
	sigmoid(z)[1]
end

grads = gradient(()->mymethod(x, W), Flux.params([x, W]))
grads[x]
grads[W]
# gradient(mymethod, model)

# model = params([x, W])
model = function(x) 
		mymethod(x, W)
end


# opt_state = Flux.setup(rule, model)
optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.


f(x) = 3x^2 + 2x + 1
f(5)
f(x)

df(x) = gradient(f, x)[1]

df(x)








struct CustomModel
  W
end

(customModel::CustomModel)(x) = m.W*x 


CustomModel(in::Integer, out::Integer) =
  CustomModel(randn(out, in))

customModel = CustomModel(2, 1)
customModel.W

customModel.W*x


# Overload call, so the object can be used as a function
(customModel::CustomModel)(x) = begin
	z = customModel.W * x
	sigmoid(z[1])
end

customModel(x)

Flux.@functor CustomModel



train_set = [[[1, 1], 1]]

loss(x, y) = sum(Flux.binarycrossentropy(customModel(x), y))
loss([1,1], 1)

input, label = train_set[1]

grads = Flux.gradient(customModel) do m
  result = m(input)[1]
  loss(result, label)
end

optim = Flux.setup(Flux.Adam(0.01), customModel)  # will store optimiser momentum, etc.


for data in train_set
  # Unpack this element (for supervised training):
  input, label = data

  # Calculate the gradient of the objective
  # with respect to the parameters within the model:
  grads = Flux.gradient(customModel) do m
      result = m(input)
      loss(result, label)
  end

  # Update the parameters so as to reduce the objective,
  # according the chosen optimisation rule:
  Flux.update!(optim, customModel, grads[1])
end

customModel.W