# # using PCA
# using MultivariateStats
using MultivariateStats 

"""

Changes the basis so that the right side of the chart actually corresponds to right-wing items. Uses a list of known left-wing items -- which all should have a very negative polarity
factor -- and flips the axis if they have an average positive polarity

"""

function alignLeftRight(X, b, itemIds)

	knownLeftWingItems = [
		  "1694922274426835377"
		 ,"1695026217353617600"
		 ,"1694960172832002373"
		 ,"1694994009003786467"
		 ,"1694890907470803080"
		 ,"1694905747132325923"
		 ,"1694889447634997665"
		 ,"1694894299647750158"
		 ,"1694888520643494275"
		 ,"1733799198170976425"
	]

	# Find the mean polarity factor of these items (if theya re in the sample)
	indices = findall(item -> in(item, knownLeftWingItems), itemIds)
	leftWing = mean(X[1,indices])


	println("Left wing mean", leftWing, b)
	if leftWing > 0 
		b[:,1] = -1 * b[:,1]
		println("Flipping axis", b)

	end

	return b
end

"""

Changes the basis so that the maximum-entry dimension (the polarity factor) is on the horizontal axis and the minimum-entropy dimension (the common ground factor) is the vertical axis.

Also tries to make sure positive values of the polarity factor align with real-world right-wing items.

"""

function changeBasis(W, X, itemIds)

	d = svd(W, full=true, alg=LinearAlgebra.QRIteration())
	ww = W * d.Vt'


	bestBasis = nothing
	worstBasis = nothing
	lowestEntropy = Inf
	highestEntropy = 0


	entropyChartData = zeros(360,2)
	entropyChartData2 = zeros(360,2)
	for a in 0:1:359
		basis=[cos(a * π/180), sin(a * π / 180)]


		converted = W * d.Vt' * basis
		e = dimensionEntropy(converted)

		if e < lowestEntropy
			lowestEntropy = e
			bestBasis = basis
		end

		if e > highestEntropy
			highestEntropy = e
			worstBasis = basis
		end

		# push!(entropyChartData, e*basis)

		# println("$(a) degrees: $(basis): $(var(converted)), $(e)")
		entropyChartData[a+1,:] = e*basis
		entropyChartData2[a+1,:] = basis

	end

	m = mean(W * d.Vt' * bestBasis)
	if m < 0 
		bestBasis = bestBasis * -1
	end


	orthogonal = [0 -1; 1 0] * bestBasis
	b = reshape([orthogonal; bestBasis], 2,2)

	bb = d.Vt' * b


	# Using the model in entropy-based-dimension-reduction.jl will generalize better beyond two dimensions
	# But the simple algorithm above works and is actually faster.

	# include("entropy-based-dimension-reduction.jl")
	# (bestBasis, loss) = findLowEntropyDimension(W)
	# orthogonal = [0 -1; 1 0] * bestBasis
	# bb2 = reshape([orthogonal; bestBasis], 2,2)


	# Now flip the axis based on known left-wing items
	bb = alignLeftRight(bb' * X, bb, itemIds)


	return (bb, entropyChartData)
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
	length(positives)
	length(negatives)

	up = sum(positives)
	down = abs(sum(negatives))

	p = up / (up + down)
	p
	return  binaryentropy(p)
end

