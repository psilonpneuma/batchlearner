# batchlearner
Bayesian batch learner simulation. A single agent observes a starting proportion of 1s and 0s (e.g. 8:2), infers a distribution over possible grammars (i.e. ratios) and selects a language using one of 3 possible strategies - maximum a posteriori, sampling or averaging. The agent then produces another 10 datapoints using the same language, over n runs - default = 10000 using either maximisation, soft-maximisation or sampling. 
The code returns a graph of production frequencies of variant x across runs (shown on the x-axis) from the input observation ratios x:y. 
A matrix is extracted from the iterations.


ADD
- how to run (python/miniconda + filename + parameters)
- GUI/process visualization
- settingsFile? 
