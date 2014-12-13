# Convert from your indices to a submission
import pandas as pd

REAL_TEST = '../test.csv'
YOUR_TEST = '../foo.csv'
OUTPUT = '../submission4444.csv'

p = pd.read_csv(YOUR_TEST, header=None)[0]
indices = pd.read_csv(REAL_TEST, usecols=['id'])['id']

with open(OUTPUT, 'w') as outfile:
    outfile.write('id,click\n')
    for i in range(len(p)):
        outfile.write('%s,%s\n' % (indices[i], p[i]))