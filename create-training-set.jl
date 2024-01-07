# L = targets left-wing post
# R = targets right-wing post
# H = helpful
# U = unhelpful


# V() = vote rate
# U() = upvote probability 

# profile 1: honest but biased right-winger

# # Mostly looks for helpful notes targeting left-wing posts
# V(L,H) = 3.0
# U(L,H) = .99

# # Pays little attention to helpful notes targeting right-wing posts.
# But if he does vote, he votes fairly accurately 
# V(R,H) = .25
# U(R,H) = .85


# # Pays little attention to unhelpful notes targeting left-wing posts
# # But if he does vote, he votes fairly accurately
# V(L,U) = .25
# U(L,U) = .15

# # Pays a fair amount of attention to unhelpful notes targeting right-wing posts
# V(R,U) = 2.0
# U(R,U) = .01

#row is help, unhelpful
# col is left, right

profile1 = (
	rates = [3.0 .25; .25 2.0],
	probabilities = [.99 .85; .15 .01]
)


# profile 2: right-wing culture war thug

# # Mostly looks for helpful notes targeting left-wing posts
# V(L,H) = 3.0
# U(L,H) = .99

# # Also downvotes helpful notes targeting right-wing posts
# V(R,H) = 2.5
# U(R,H) = .1

# # Doesn't really care if they are helpful or not. Also very willing to upvote
# # unhelpful notes targeting left wing posts
# V(L,U) = 2.5
# U(L,U) = .9

# # Also pays a fair amount of attention to unhelpful notes targeting right-wing posts
# V(R,U) = 3.0
# U(R,U) = .01


profile2 = (
	rates = [3.0 2.5; 2.5 3.0],
	probabilities = [.99 .1; .9 .01]
)


# profile 3: honest but biased left-winger

# ...mirror image of profile 1

profile3 = (
	rates = [.25 3.0; 2.0 .25],
	probabilities = [.85 .99; .01 .15 ]
)


# profile 4: left-wing culture-war thug

# ... mirror image of profile 2


profile4 = (
	rates = [2.5 3.0; 3.0 2.5],
	probabilities = [.1 .99; .01 .9]
)



type1 = [1,0] # helpful left
type2 = [1,1] # unhelpful left
type3 = [0,0] # helpful right
type4 = [0,1] # unhelpful right

# voteRate = 
# 	b0 + b1 * H + b2 * R = n
# 	b0 + b1 * 1 + b2 * 1 = 3.0
# 	b0 + b1 * 1 + b2 * 0 = 2.0
# 	b0 + b1 * 0 + b2 * 1 = .25
# 	b0 + b1 * 0 + b2 * 0 = .25
# b = [3.0, 2.0, .25, .25]


using LinearSolve

# Convert the user profile into a matrix that we can multiply by an item type
# matrix to get the vote rate and probability for a user and item type. Do
# this by representing the user profile as linear equation that gives us the
# rate and probability as a function of the features of the item. 

profileToMatrix = function(p) 
	# p = profile1
	# so each y (upvote probability or vote rate) is equal to
	# where the x vector is [1, left, helpful, left*helpful]
	# y = w0 + w1 * x1 + w2 * x2 + w3 * x3  = n
	#   = w0 + w1 * RIGHT + w2 * HELPFUL + w3 RIGHTANDHELPFUL = n

	# X = [1.0 1.0 1.0 1.0; 1.0 1.0 0.0 0.0; 1.0 0.0 1.0 0.0; 1.0 0.0 0.0 0.0]
	X = [1.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 1.0 1.0 1.0 1.0; 1.0 0.0 1.0 0.0]

	w = [p.rates[1,:]; p.rates[2,:]]
	prob = LinearProblem(X, w)
	sol = solve(prob)
	v1 = sol.u

	w = [p.probabilities[1,:]; p.probabilities[2,:]]
	prob = LinearProblem(X, w)
	sol = solve(prob)
	v2 = sol.u

	return [v1 v2]
end


typeToMatrix = function(t) 
	v = [1 t[1] t[2] t[1]*t[2]]
end

# t1 = typeToMatrix(type3)

# p1 = profileToMatrix(profile1)
# typeToMatrix(type1)*p1
# typeToMatrix(type2)*p1
# typeToMatrix(type3)*p1
# typeToMatrix(type4)*p1


	# using PoissonRandom
using Random, Distributions


createTrainingSet = function(n, m, baseProb)


# profile1 = (
# 	rates = [3.0 .25; .25 2.0],
# 	probabilities = [.99 .85; .15 .01]
# )

	p1 = profileToMatrix(profile1)
	p2 = profileToMatrix(profile2)
	p3 = profileToMatrix(profile3)
	p4 = profileToMatrix(profile4)

	users = repeat([p1 p2 p3 p4 p3 p4 p4 p4], outer=[1,trunc(Int, n)])

	t1 = typeToMatrix(type1)
	t2 = typeToMatrix(type2)
	t3 = typeToMatrix(type3)
	t4 = typeToMatrix(type4)


	# t1*p1
	# t2*p1
	# t3*p1
	# t4*p1



	items = repeat([t1; t2; t3; t4],m)

	probs = (items*users)
	rates = probs[:, 1:2:end]
	probabilities = probs[:, 2:2:end]

	# Average user votes on 1 out of 10 items

	baseProb = .1
	votes = rand.(Bernoulli.(baseProb .* rates))
	upvotes = rand.(Bernoulli.(votes .* probabilities))

	userColors = repeat([:magenta; :red; :cyan; :blue; :cyan; :blue; :blue; :blue; ],n)
	itemColors = repeat([:magenta; :red; :cyan; :blue],m)

	return (users=users, items=items, votes=votes', upvotes=upvotes', userColors=userColors, itemColors=itemColors)

end

