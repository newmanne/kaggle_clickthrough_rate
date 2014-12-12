#!/usr/bin/python
import sys, math, subprocess

def run_command(command, cwd=None):
	print "Running command " + command
	p = subprocess.call(command, shell=True, cwd=cwd)
	return p

# For black box function optimization, we can ignore the first 5 arguments. 
# The remaining arguments specify parameters using this format: -name value 

alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized
epoch = 1       # learn training data for N passes

for i in range(len(sys.argv)-1):  
	if (sys.argv[i] == '-alpha'):
    	alpha = float(sys.argv[i+1])
  	elif(sys.argv[i] == '-beta'):
    	beta = float(sys.argv[i+1])   
  	elif(sys.argv[i] == '-L1'):
  		L1 = float(sys.argv[i+1])   
  	elif(sys.argv[i] == '-L2'):
		L2 = float(sys.argv[i+1])  	
	elif(sys.argv[i] == '-epoch'):
		epoch = float(sys.argv[i+1])

# Compute the logloss here
	run_command('thing')

# SMAC has a few different output fields; here, we only need the 4th output:
print "Result for SMAC: SUCCESS, 0, 0, %f, 0" % yValue