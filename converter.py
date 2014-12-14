# Convert from your indices to a submission
# TODO: this script takes insanely long for what it does - find otu what you are doing wrong...
import pandas as pd

REAL_TEST = '../test.csv'
YOUR_TEST = '../foo.csv'
OUTPUT = '../submission4444.csv'

print "Reading p values"
p = pd.read_csv(YOUR_TEST, header=None)[0]
print "Reading id values"
indices = pd.read_csv(REAL_TEST, usecols=['id'])['id']

print "Starting output"
with open(OUTPUT, 'w') as outfile:
    outfile.write('id,click\n')
    for i in range(len(p)):
        outfile.write('%s,%s\n' % (indices[i], p[i]))