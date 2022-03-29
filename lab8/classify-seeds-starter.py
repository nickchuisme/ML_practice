# classify-seeds-starter.py
# parsons/25-feb-2017
#
# The input data is the seeds data Set from the UCI Machine Learning repository.
#
# https://archive.ics.uci.edu/ml/datasets/seeds
#
# The code to load a csv file is based on code written by Elizabeth
# Sklar for Lab 1, and the code to create multiple plots is based on
# code Elizabeth Sklar wrte for Lab 0.

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Parameters
plot_step = 0.02

#
# Define constants
#

datafilename = 'seeds_dataset.csv' # input filename
area          = 0                  # column indexes in input file
perimeter     = 1
compactness   = 2
kernel_length = 3
kernel_width  = 4
asymmetry     = 5
groove        = 6
num           = 7 # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'area', 'perimeter', 'compactness', 'kernel_length', 'kernel_width', 'asymmetry', 'groove', 'num' ]

num_samples = 210 # size of the data file. 
num_features = 7

#
# Open and read data file in csv format
#
# After processing:
# 
# data   is the variable holding the features;
# target is the variable holding the class labels.

try:
    with open( datafilename ) as infile:
        # Note that while the file is a .csv, it is actually tab
        # separated, so we tell that to file reader
        indata = csv.reader( infile, dialect='excel-tab')
        data = np.empty(( num_samples, num_features ))
        target = np.empty(( num_samples,), dtype=int )
        i = 0
        for j, d in enumerate( indata ):
            ok = True
            for k in range(0,num_features): # If a feature has a missing value
                if ( d[k] == "?" ):         # we don't use that record.
                    ok = False
            if ( ok ):
                data[i] = np.asarray( d[:-1], dtype=float )
                target[i] = np.asarray( d[-1], dtype=int )
                i = i + 1
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print('there was an error: ' + str( x ))
    sys.exit()

# Here is are the sets of feastures:
data
# Here is the diagnosis for each set of features:
target

# How many records do we have?
num_samples = i
print("Number of samples:", num_samples)

# Now build your classifier
