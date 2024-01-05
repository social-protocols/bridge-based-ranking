
using NMF

s = createTrainingSet(10,20,.1)

events = s.upvotes

r = nnmf(events .* 1.0, 2)
r.W
r.H

dot(r.W[1,:], r.H[:,5])
dot(r.W[1,:], r.H[:,6])
dot(r.W[1,:], r.H[:,7])
dot(r.W[1,:], r.H[:,8])

r.H[:,1:4:end]
r.H[:,2:4:end]
r.H[:,3:4:end]
r.H[:,4:4:end]


# type1 items
mean(r.W[1:4:end,:], dims=1)
# type2 items
mean(r.W[2:4:end,:], dims=1)
# type3 items
mean(r.W[3:4:end,:], dims=1)
# type4 items
mean(r.W[4:4:end,:], dims=1)