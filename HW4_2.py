"""
STAT 672 HW4
Author: Bryce Dunn
"""
import pandas as pd
import numpy as np
#from sklearn.model_selection import KFold
from itertools import combinations

Xtrain = pd.read_csv("Xtrain.csv",header=None)
Ytrain = pd.read_csv("Ytrain.csv",header=None)
Xtest = pd.read_csv("Xtest.csv",header=None)
Ytest = pd.read_csv("Ytest.csv",header=None)

n = len(Xtrain) # 40 samples

# 20 random permutations of Ytrain
permutations = np.concatenate([np.random.permutation(Ytrain) for _ in range(20)],axis=1)
# Feature Subset Combination Indices
sub = np.arange(15)
subsets = [np.array(list(combinations(sub,i))) for i in range(1,16)]
subsets = [item for sublist in subsets for item in sublist]
# num_subsets = len(subsets) # 32767
# 5-Fold CV Indices
cross_validation = [np.arange(i, i+n//5).astype(int) for i in range(0, n, n//5)]
test_sets = [np.array(Xtrain)[cv,:] for cv in cross_validation]
training_sets = [np.delete(np.array(Xtrain),(cv),axis=0) for cv in cross_validation]
training_labels = [np.delete(np.array(Ytrain),(cv),axis=0) for cv in cross_validation]
test_labels = [np.array(Ytrain)[cv,:] for cv in cross_validation]
#---------------------------------------------------------------
best_feature_subset = 0
error = 0#sum(sum(L1))/L1.size
success = 0
best_feature_subset = 0
L_grid = np.zeros([len(test_sets[0]),5])

for subset in subsets:
    training_folds = [training_set[:,subset] for training_set in training_sets]   
    #p_inverse = [np.linalg.pinv(training_fold) for training_fold in training_folds]
    w_list = [np.linalg.lstsq(training_folds[i],training_labels[i],rcond=None) for i in range(len(training_folds))]
    # w_list = [np.matmul(p_inverse[i],training_labels[i]) for i in range(len(training_labels))]    
    # w = np.linalg.lstsq(Xtrain, Ytrain, rcond=None)[0]
    
    for i in range(len(test_sets)):
        test_set = test_sets[i]
        w = w_list[i]
        test_label = test_labels[i]
        
        for j in range(len(test_set)):
            fX = np.sign(np.dot(test_set[j,:],w))
            L_grid[j,i] = np.absolute(test_label[j]-fX)/2
         
    error_tmp = sum(sum(L_grid))/L_grid.size
    success_tmp = 1-error_tmp
    if success_tmp > success:
        success = success_tmp
        error = error_tmp
        best_feature_subset = subset
        
    

  