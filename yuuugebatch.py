import random as rnd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import beta
import argparse
import numpy as np


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
def iterate(number_of_ones,alpha):
	ones=[] #count of x in every run
	runs=args.runs
	learning=args.learning
	#expt=args.exponent
	for r in range(runs):
		if learning == "sample":
			language=beta.rvs(alpha+number_of_ones, alpha+(10-number_of_ones)) # sampling
		elif learning == "max":
			language=(alpha+number_of_ones-1)/(alpha*2+10-2) # maximising
		elif learning == "avg":
			language=(alpha+number_of_ones)/(alpha*2+10) # averaging
		data=[produce(language) for _ in range(10)] #one list of 01s
		#print data
		count_of_ones=float(data.count(1))
		ones.append(count_of_ones)
		#if r < 10:
			#print number_of_ones
	#print "list of ones: ",ones[1:10]

	#dictionary with x_possible_values:freqs(x), ordered by n_of_x
	d = {}
	for c in ones:
	    count=ones.count(c)
	    d[c] = count
	#print "dictionary: ",d.items()[1:10]

	#get probabilities of proportion_of_ones as list of tuples (n,prob(n))
	prob=[(n,float(freq)/len(ones)) for n, freq in d.items()]
	return prob
	#print "probabilities: ",prob[1:10]

#----
#starting input ratio and number of productions
fig, axes = plt.subplots(nrows=9, ncols=6, figsize=(14, 8)) #
fs = 10
alphas = [0.01,1,100]
exponents= [1.5,5,15]
obs = range(0,6)
row = -1
bigprob = []
for a in alphas:
	for expt in exponents:
		row += 1
		print "- outer loop - exponent={0},row={1},".format(expt,row)
		for i in obs: ##populate one row
			print "--inner loop - i={0},".format(i)
			prob=iterate(i+5,a)
			print "-- inner loop - prob={0},".format(prob)
			bigprob.append(prob)
			print "-- inner loop - bigprob={0},".format(bigprob)
			print "-- inner loop - xarray={0},".format([x[0] for x in bigprob[i]])
			axes[row, i].bar([x[0] for x in bigprob[i]],[x[1] for x in bigprob[i]],align='center',width=0.3,color='r')
			axes[row, i].set_title("exponent: "+str(expt), fontsize=fs)
			axes[row, i].axvline(x=i+5, color='r', linestyle='dashed', linewidth=2)
			axes[row, i].xaxis.set_ticks(np.arange(0, 10, 1))
			axes[row, i].yaxis.set_ticks(np.arange(0, 1, 0.5))
			axes[row, i].yaxis.grid(True, linestyle="-", color="0.75")
			if row == 0:
				axes[row, i].set_title(str(i+5)+":"+str(10-(i+5)))

		bigprob = []
	#plots

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 8)
plt.show()