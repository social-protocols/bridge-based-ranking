## Notes


### Regularization

So I seem to have found a regularization that accidentally works, but I don't understand why. It users a cost function and a penalty function that don't match: the cost function is the norm of errors -- the sqrt of the sum of errors, and the regularization is the sum of parameters, with a lambda for intercepts 5x that for the lambda for other
parameters per the original birdwatch paper. The results seem to align better with the original community notes item categorizations.

In the functions in matrix-factorization.jl, setting the altModel parameter to true causes the model to use this regularzation function (penaltyCNOld) with a norm cost function.

The regularization function that I figure *should* work is called "penalty". My reasoning for this function is in the code. 

### Non-zero Polarity Origin

One result I get, which may be a problem of overfitting/bad regularization, is the average polarization factor is much greater or less than zero. This shift in the polarity factors results in a very large shift in the intercepts. E.g. if you start with users polarity factors centered at zero, and you add delta to the value of each items's polarity factors, then the regression lines for the users will become
	w*(x + delta) + b = w*x + w*delta + c


### 3d Model

- bridge-based-ranking-communitynotes.jl contains some incomplete experimentation using three-dimensional model

I have found that doing 3d matrix factorization, and then taking the vector with the lowest entropy and projecting onto this, gives me very similar results to the 1d model. The advantage of this model is it also gives us common-ground factors for users. We can use the common-ground factor of the user as the weight and take a weighted average of users votes as an alternative way to estimate common ground factor for items.







