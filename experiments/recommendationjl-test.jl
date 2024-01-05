s = createTrainingSet(10,20,.1)

# import Pkg; Pkg.add("Recommendation")
using Recommendation

# Load a sample dataset
# data = Recommendation.load_movielens_100k()

using SparseArrays
using Recommendation
 # .* (s.upvotes .- 1)
# events = s.votes
 # .+ s.votes .* (s.upvotes .- 1)

s = createTrainingSet(10,20,.1)

events = s.votes


data = DataAccessor(sparse(events))

# data = load_movielens_100k()

recommender = MF(data, 2)
# recommender = MostPopular(data)
fit!(recommender, learning_rate=15e-4, max_iter=1000)
build!(recommender)
recommend(recommender, 8, 1, collect(1:10))
recommender.P
recommender.Q
recommender.R
s.users[:,2]
recommender.Q[2,:]

