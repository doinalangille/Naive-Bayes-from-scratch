# Naive Bayes Classifier From Scratch

The Naive Bayes algorithm is a classification technique based on Bayes Theorem. It assumes that the presence of a feature in a class is unrelated to the presence on any other feature. The algorithm rely on the posterior probability of the class given a predictor, as we can see in the following formula:

![proba](https://bit.ly/2CKbbtQ)

where:

P(c\|x) - the posterior probability of class given a predictor
P(x\|c) - the probability of the predictor $$x$$ given the class. Also known as Likelihood
P(c) - the prior probability of the class
P(x) - the prior probability of predictor.

Find [here](https://medium.com/@doina.jitoreanu/naive-bayes-classifier-from-scratch-fcfad8f145a2) the blog post for a better understanding.