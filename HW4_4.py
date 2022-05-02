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

n = len(Xtrain) # 40 samples

# 20 random permutations of Ytrain
permutations = np.concatenate([np.random.permutation(Ytrain) for _ in range(20)],axis=1)
# Feature Subset Combination Indices
sub = np.arange(15)
subsets = [np.array(list(combinations(sub,i))) for i in range(1,16)]
subsets = [item for sublist in subsets for item in sublist]
# num_subsets = len(subsets) # 32767
# 5-Fold CV Indices, Training Sets, Test Sets, and Labels
cross_validation = [np.arange(i, i+n//5).astype(int) for i in range(0, n, n//5)]
test_sets = [np.array(Xtrain)[cv,:] for cv in cross_validation]
training_sets = [np.delete(np.array(Xtrain),(cv),axis=0) for cv in cross_validation]
training_labels = [np.delete(np.array(Ytrain),(cv),axis=0) for cv in cross_validation]
test_labels = [np.array(Ytrain)[cv,:] for cv in cross_validation]
"""
#---------------------------------------------------------------
best_feature_subset = []
error = 1
#L_grid = np.zeros([len(test_sets[0]),5])

for subset in subsets:
    L_grid = np.zeros([len(test_sets[0]),5])
    training_folds = [training_set[:,subset] for training_set in training_sets]   
    p_inverse = [np.linalg.pinv(training_fold) for training_fold in training_folds]
    w_list = [np.matmul(p_inverse[i],training_labels[i]) for i in range(len(training_labels))]
    
    for i in range(len(test_sets)):
        test_obs = test_sets[i][i]
        w = w_list[i].transpose()
        test_label = test_labels[i]
        
        for j in range(len(test_sets[0])):
            fX = np.sign(np.inner(test_obs[subset],w))
            L_grid[j,i] = np.absolute(test_label[j][0]-fX)/2
         
    error_tmp = sum(sum(L_grid))/L_grid.size
    if error > error_tmp:
        error = error_tmp
        best_feature_subset = subset

print("Part A")
print("The best feature subset is: ",best_feature_subset,". Its validation error rate is: ",error)
print("The training validation error using all 15 features is: ",error_tmp)
"""
#__________________________________________________________________________________________________
# Part B
Xtest = pd.read_csv("Xtest.csv",header=None)
Ytest = pd.read_csv("Ytest.csv",header=None)

best_training_subset = Xtrain[subsets[3]]
p_inverse = np.linalg.pinv(best_training_subset)
w = np.matmul(p_inverse,Ytrain)
fX = np.sign(np.inner(Xtest[subsets[3]],w))
L = np.absolute(Ytest-fX)/2
best_feature_test_error = sum(L[0])/len(Ytest)

all_features = Xtrain
p_inverse = np.linalg.pinv(all_features)
w = np.matmul(p_inverse,Ytrain)
fX = np.sign(np.inner(w.transpose(),Xtest))
L = np.absolute(Ytest-fX.transpose())/2
all_features_test_error = sum(L[0])/len(Ytest)

print("Part B:")
print("The test error of the best feature subset is: ",best_feature_test_error)
print("The test error using all 15 features is: ",all_features_test_error)
#___________________________________________________________________________________________________
# Part C
print("Part C:")
print("Comparing model accuracy to chance can be misleading.")
print("Randomly permutating the training labels generates a null distribution and remove dependency between features and labels.")
print("Using a p-value based method, a null hypothesis can be tested by comparing the average cross-validation accuracies.")
print("This p-value would show if there is any statistical depdency between features and labels.")
#___________________________________________________________________________________________________
# Part D
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LinearRegression

score=[]
perm_scores = []
pvalue= []

for idx,subset in zip(range(len(subsets)),subsets):
    #training = np.array(Xtrain[subset][0]).reshape(40,1)
    training = Xtrain[subset]
    clf = LinearRegression()
    score, perm_scores, pvalue = permutation_test_score(clf, training, Ytrain, cv=None, n_permutations=20)


