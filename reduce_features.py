import pandas as pd
import numpy as np

print "reading file"
train = pd.read_csv('../train.csv')
feature = train.columns.values[2:] 
for i in feature:
	print "Stating new feature"
	for j in np.unique(train[i]):
	   	if train[train[i]==j].shape[0] < 10: # let's say 10 is the threshold
	   		train.loc[train[i]==j, i]='dummy'   # replace it with any value you want
print "writing to file"
train.to_csv('../train_reduced.csv')
