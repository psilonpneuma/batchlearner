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
		data=[produce(language) for _ in range(10)] #one list of 01s
		#print data
		count_of_ones=float(data.count(1))
		ones.append(count_of_ones)
		#if r < 10:
			#print number_of_ones

	#dictionary with x_possible_values:freqs(x), ordered by n_of_x
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

	#get probabilities of proportion_of_ones as list of tuples (n,prob(n))
	prob=[(n,float(freq)/len(ones)) for n, freq in dictionary.items()]
	print "PROBABILITIES: ",prob
	return prob


#----
#starting input ratio and number of productions
fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(16, 11)) #
fs = 10
exponents= [1.5,1.9,2.1]
obs = range(0,6)
row = -1
trace_bp = []
bigprob = []
for exponent in exponents:
	row += 1
	expt=exponent
	print "- outer loop - exponent={0},row={1},".format(expt,row)
	for i in obs: ##populate one row
		print "--inner loop - i={0},".format(i)
		prob=iterate(i+5)
		print "-- inner loop - prob={0},".format(prob)
		bigprob.append(prob)
		print "-- inner loop - bigprob={0},".format(bigprob)
		print "-- inner loop - xarray={0},".format([x[0] for x in bigprob[i]])
		axes[row, i].bar([x[0] for x in bigprob[i]],[x[1] for x in bigprob[i]],align='center',width=0.5,color='0.85')
		axes[row, i].axvline(x=i+5, color='0', linestyle='dashed', linewidth=1.5)
		axes[row, i].set_xlim([0,10])
		axes[row, i].set_ylim([0,1])
		#axes[row, i].yaxis.grid(True, linestyle="-", color="0.75")
		if i == 0:
			axes[row, i].set_ylabel("exp "+str(exponent),size=18)
			axes[row, i].yaxis.set_ticks(np.arange(0, 1, 0.25))
		else:
			axes[row, i].yaxis.set_ticks([])
		if row == 0:
			axes[row, i].set_title(str(i+5)+":"+str(10-(i+5)))
		if row == 2:
			axes[row, i].xaxis.set_ticks(np.arange(0, 10, 1))
		else:
			axes[row, i].xaxis.set_ticks([])

	trace_bp.append(bigprob)
	bigprob = []
print "TRACKING THE PROBS:  ",trace_bp[0]
print len(trace_bp[0])

#plots

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 8)
plt.subplots_adjust(wspace = .0,hspace=.0)
plt.show()




