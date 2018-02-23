# batchlearner
Bayesian batch learner simulation. A single agent observes a starting ratio of 1s and 0s, given as one of the parameters, for a total of 10 observations (e.g. 8:2). It then calculates a distribution over possible ratios (grammars) and selects one of them (a language) using one of 3 possible strategies - maximum a posteriori, sampling or averaging from the distribution.
The agent then produces another 10 datapoints using the same language. This is done over n runs - default = 10000 using either maximisation, soft-maximisation or sampling. 
The code returns a graph of production frequencies of variant x across runs (shown on the x-axis) from the input observation ratios x:y. 
A matrix is extracted from the iterations.


ADD
- how to run (python/miniconda + filename + parameters)
  - n runs: int :
        Number of simulation runs (iterations)
  - starting_observations: int :
        Number of observed 1s at first iteration
  - STRATEGY
- GUI/process visualization
- settingsFile? 
