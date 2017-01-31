##
# Bayesian batch learner simulation. A single agent observes a starting proportion of 1s and
# 0s (e.g. 8:2), infers a distribution over possible grammars (i.e. ratios) and selects a language
# using one of 3 possible strategies - maximum a posteriori, sampling or averaging. The agent then
# produces another 10 datapoints using the same language, over n runs - default = 10000 using either
# maximisation, soft-maximisation or sampling. The code returns a graph of production frequencies of
# variant x across runs (shown on the x-axis) from the input observation ratios x:y. A matrix is extracted
# from the iterations and decomposed in its eigenvectors to calculate the matrix steady states.
###




import random as rnd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import beta
import scipy.linalg
import argparse
import numpy as np
from prettytable import PrettyTable


parser = argparse.ArgumentParser(description='Batch frequency learning simulation using a single Bayesian agent')
parser.add_argument('--observations', '-o', default=5, type=int,
                    help="An integer representing the starting count of 1s")
parser.add_argument('--alpha', '-a', default=1, type=float,
                    help="A float representing the prior bias (alpha)")
parser.add_argument('--runs', '-r', default=1000, type=int,
                    help="An integer representing the number of runs wanted")
parser.add_argument('--learning', '-l', default="sample", type=str,
                    help="Learning strategy, can be max, avg or sample")
parser.add_argument('--production', '-p', default="sample", type=str,
                    help="Production strategy, can be max softmax or sample")
parser.add_argument('--exponent', '-e', default="2", type=float,
                    help="Exponent used in softmax")

args = parser.parse_args()

#starting_count_w1=args.observations
production=args.production
alpha = args.alpha
#expt=args.exponent

def generate(starting_count_w1,n_productions):
    data=[1]*starting_count_w1 + [0]*(n_productions-starting_count_w1)
    return data

#output
def produce(p):
	p0=1-p
	if production == "sample":
	    if rnd.random()<p:
	       return 1
	    else:
	       return 0
	#maximization
	elif production == "max":
	    if p >= 0.5:
	       return 1
	    else:
	       return 0
	#soft maximization
	elif production == "softmax":
	    	p1=p**expt/(p**expt+p0**expt)
	#	    	print p, p**expt, p1
	#	    	print p,p**expt,p0**expt
    		if rnd.random()<p1:
	    	   return 1
	    	else:
	       	   return 0

#----
# Hypothesis choice
# every run counts the occurrencies of x
def iterate(number_of_ones):
	ones=[] #count of x in every run
	runs=args.runs
	alpha=args.alpha
	learning=args.learning
	#expt=args.exponent
	for r in range(runs):
		if learning == "sample":
			language=beta.rvs(alpha+number_of_ones, alpha+(10-number_of_ones)) # sampling
		elif learning == "max":
			language=(alpha+number_of_ones-1)/(alpha*2+10-2) # maximising
		elif learning == "avg":
			language=(alpha+number_of_ones)/(alpha*2+10) # averaging
		data=[produce(language) for _ in range(10)] # one list of 01s
		#print data
		count_of_ones=float(data.count(1))
		ones.append(count_of_ones)
		#if r < 10:
			#print number_of_ones

	# dictionary with x_possible_values:freqs(x), ordered by n_of_x
	dictionary = {}
	for c in ones:
		count=ones.count(c)
		dictionary[c] = count
		tot = 0
		for key in dictionary:
			tot+=1
		if tot < 11:
			for suspect_x in range(0,11):
				if suspect_x not in dictionary.iterkeys():
					dictionary[suspect_x]= 0
	print "DICTIONARY: ",dictionary
	#print "dictionary: ",d.items()[1:10]

	# get probabilities of proportion_of_ones as list of tuples (n,prob(n))
	prob=[(n,float(freq)/len(ones)) for n, freq in dictionary.items()]
	print "PROBABILITIES: ",prob
	return prob


#----
# starting input ratio and number of productions
fig, axes = plt.subplots(nrows=3, ncols=11, figsize=(16, 8)) #grid dimension
exponents= [1.5,5,15] # rows
obs = range(0,11) # columns
row = -1
bigprob = []
for exponent in exponents:
	row += 1
	expt=exponent
	print "- outer loop - exponent={0},row={1},".format(exponent,row)
	for o in obs: # populate one row
		print "--inner loop - observations={0},".format(o)
		prob=iterate(o)
		print "-- inner loop - prob={0},".format(prob)
		bigprob.append(prob)
		print "-- inner loop - BigProb={0},".format(bigprob)
		#print "-- inner loop - xarray={0},".format([x[0] for x in bigprob[o]])
		

		# graphs

		# possible values of x(=observations), correspondent probability
		axes[row, o].bar([x[0] for x in bigprob[o]],[x[1] for x in bigprob[o]],align='center',width=0.8,color='g')
		axes[row, o].axvline(x=o, color='0', linestyle='dashed', linewidth=1.5)
		axes[row, o].set_xlim([0,10])
		axes[row, o].set_ylim([0,0.6])
		#axes[row, i].yaxis.grid(True, linestyle="-", color="0.75")
		if o == 0:
			axes[row, o].set_ylabel("exp "+str(exponent),size=18)
			axes[row, o].yaxis.set_ticks(np.arange(0, 0.6, 0.20))
		else:
			axes[row, o].yaxis.set_ticks([])
		if row == 0:
			axes[row, o].set_title(str(o)+":"+str(10-o))
		if row == 2:
			axes[row, o].xaxis.set_ticks(np.arange(0, 10, 1))
		else:
			axes[row, o].xaxis.set_ticks([])

	## list to matrix 

	A_list = []
	for pb in bigprob:
		A_list.append([x[1] for x in pb])
	A=np.array([pp for pp in A_list])
	bigprob = []
	p = PrettyTable() #prettyprint A
	for roar in A:
		p.add_row(roar)

	print p.get_string(header=False, border=True)

	theEigenvalues, leftEigenvectors = scipy.linalg.eig(A, right=False, left=True)  
	theEigenvalues = theEigenvalues.real
	leftEigenvectors = leftEigenvectors.real
	tolerance = 1e-10
	mask = abs(theEigenvalues - 1) < tolerance 
	theEigenvalues = theEigenvalues[mask]    
	leftEigenvectors = leftEigenvectors[:, mask] 
	leftEigenvectors[leftEigenvectors < tolerance] = 0  
	attractorDistributions = leftEigenvectors / leftEigenvectors.sum(axis=0, keepdims=True)   
	attractorDistributions = attractorDistributions.T
	theSteadyStates = np.sum(attractorDistributions, axis=0)
	print "The Steady States: ",theSteadyStates




#plots

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 8)
plt.subplots_adjust(wspace = .0,hspace=.0,right=.9,top=.9)
plt.show()
