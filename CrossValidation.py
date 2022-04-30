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

n = len(Xtrain)

# 20 Random Permutations of Ytrain
j1 = np.random.permutation(Ytrain)
j2 = np.random.permutation(Ytrain)
j3 = np.random.permutation(Ytrain)
j4 = np.random.permutation(Ytrain)
j5 = np.random.permutation(Ytrain)
j6 = np.random.permutation(Ytrain)
j7 = np.random.permutation(Ytrain)
j8 = np.random.permutation(Ytrain)
j9 = np.random.permutation(Ytrain)
j10 = np.random.permutation(Ytrain)
j11 = np.random.permutation(Ytrain)
j12 = np.random.permutation(Ytrain)
j13 = np.random.permutation(Ytrain)
j14 = np.random.permutation(Ytrain)
j15 = np.random.permutation(Ytrain)
j16 = np.random.permutation(Ytrain)
j17 = np.random.permutation(Ytrain)
j18 = np.random.permutation(Ytrain)
j19 = np.random.permutation(Ytrain)
j20 = np.random.permutation(Ytrain)
permutations = np.concatenate((j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18,j19,j20),axis=1)

# Feature Subset Combination Indices
sub = np.arange(15)
sub1 = np.array(list(combinations(sub,1))).transpose() # Single Features
sub2 = np.array(list(combinations(sub,2))) # Two-Feature Combos
sub3 = np.array(list(combinations(sub,3))) # Three-Feature Combos
sub4 = np.array(list(combinations(sub,4))) # Four-Feature Combos
sub5 = np.array(list(combinations(sub,5))) # 5-Feature Combos
sub6 = np.array(list(combinations(sub,6))) # 6-Feature Combos
sub7 = np.array(list(combinations(sub,7))) # 7-Feature Combos
sub8 = np.array(list(combinations(sub,8))) # 8-Feature Combos
sub9 = np.array(list(combinations(sub,9))) # 9-Feature Combos
sub10 = np.array(list(combinations(sub,10))) # 10-Feature Combos
sub11 = np.array(list(combinations(sub,11))) # 11-Feature Combos
sub12 = np.array(list(combinations(sub,12))) # 12-Feature Combos
sub13 = np.array(list(combinations(sub,13))) # 13-Feature Combos
sub14 = np.array(list(combinations(sub,14))) # 14-Feature Combos
sub15 = np.array(list(combinations(sub,15))) # All 15 Features

# 5-Fold CV Indices
cv1 = np.arange(n/5).astype(int)
cv2 = np.arange(n/5,2*n/5).astype(int)
cv3 = np.arange(2*n/5,3*n/5).astype(int)
cv4 = np.arange(3*n/5,4*n/5).astype(int)
cv5 = np.arange(4*n/5,n).astype(int)

# 5-Fold CV Training and Test Sets
test1 = np.array(Xtrain)[cv1,:]
test2 = np.array(Xtrain)[cv2,:]
test3 = np.array(Xtrain)[cv3,:]
test4 = np.array(Xtrain)[cv4,:]
test5 = np.array(Xtrain)[cv5,:]

train1 = np.delete(np.array(Xtrain),(cv1),axis=0)
train2 = np.delete(np.array(Xtrain),(cv2),axis=0)
train3 = np.delete(np.array(Xtrain),(cv3),axis=0)
train4 = np.delete(np.array(Xtrain),(cv4),axis=0)
train5 = np.delete(np.array(Xtrain),(cv5),axis=0)

Ytrain1 = np.delete(np.array(Ytrain),(cv1),axis=0)
Ytrain2 = np.delete(np.array(Ytrain),(cv2),axis=0)
Ytrain3 = np.delete(np.array(Ytrain),(cv3),axis=0)
Ytrain4 = np.delete(np.array(Ytrain),(cv4),axis=0)
Ytrain5 = np.delete(np.array(Ytrain),(cv5),axis=0)

Ytest1 = np.array(Ytrain)[cv1]
Ytest2 = np.array(Ytrain)[cv2]
Ytest3 = np.array(Ytrain)[cv3]
Ytest4 = np.array(Ytrain)[cv4]
Ytest5 = np.array(Ytrain)[cv5]
#---------------------------------------------------------------
#___________________Single Features_____________________
L1 = np.zeros([len(test1),5])
error1 = sum(sum(L1))/L1.size
success1 = 0
best_feature_subset1 = 0
num = -1
for i1 in sub1:
    num = num+1
    #print(i1)
    training_fold1 = train1[:,i1]
    training_fold2 = train2[:,i1]
    training_fold3 = train3[:,i1]
    training_fold4 = train4[:,i1]
    training_fold5 = train5[:,i1]
    
    pinv1 = np.linalg.pinv(training_fold1)
    pinv2 = np.linalg.pinv(training_fold2)
    pinv3 = np.linalg.pinv(training_fold3)
    pinv4 = np.linalg.pinv(training_fold4)
    pinv5 = np.linalg.pinv(training_fold5)
    
    w1 = pinv1.dot(Ytrain1)
    w2 = pinv2.dot(Ytrain2)
    w3 = pinv3.dot(Ytrain3)
    w4 = pinv4.dot(Ytrain4)
    w5 = pinv5.dot(Ytrain5)
    
    ww = np.concatenate((w1,w2,w3,w4,w5),axis=1)
    
    for i11 in np.arange(len(test1)):
        fX = np.sign(np.dot(test1[i11,:],w1))
        L = np.absolute(Ytest1[i11][0]-fX)/2
        L1[i11,0] = L
        
    for i11 in np.arange(len(test2)):
        fX = np.sign(np.dot(test2[i11,:],w2))
        L = np.absolute(Ytest2[i11][0]-fX)/2
        L1[i11,1] = L
        
    for i11 in np.arange(len(test3)):
        fX = np.sign(np.dot(test3[i11,:],w3))
        L = np.absolute(Ytest3[i11][0]-fX)/2
        L1[i11,2] = L
        
    for i11 in np.arange(len(test4)):
        fX = np.sign(np.dot(test4[i11,:],w4))
        L = np.absolute(Ytest4[i11][0]-fX)/2
        L1[i11,3] = L
        
    for i11 in np.arange(len(test5)):
        fX = np.sign(np.dot(test5[i11,:],w5))
        L = np.absolute(Ytest5[i11][0]-fX)/2
        L1[i11,4] = L
    
    error_tmp = sum(sum(L1))/L1.size
    success_tmp = 1-error_tmp
    if success_tmp > success1:
        success1 = success_tmp
        error1 = error_tmp
        best_feature_subset1 = num+1
        
print("The best single feature is column: ",best_feature_subset1,". Its error rate is: ",error1)
#____________2-Feature Combos__________________________________

#---------------------------------------------------------------
#_____________ All 15 Features __________________
pinv15_1 = np.linalg.pinv(train1) # fold 1 p-inverse
pinv15_2 = np.linalg.pinv(train2) # fold 2
pinv15_3 = np.linalg.pinv(train3) # fold 3
pinv15_4 = np.linalg.pinv(train4) # fold 4
pinv15_5 = np.linalg.pinv(train5) # fold 5

w15_1 = pinv15_1.dot(Ytrain1) # fold 1 weights
w15_2 = pinv15_2.dot(Ytrain2) # fold 2
w15_3 = pinv15_3.dot(Ytrain3) # fold 3
w15_4 = pinv15_4.dot(Ytrain4) # fold 4
w15_5 = pinv15_5.dot(Ytrain5) # fold 5

L15 = np.zeros([len(test1),5])

# First Fold
for i15 in np.arange(len(test1)):
    fX15_1 = np.sign(np.dot(test1[i15,:],w15_1))
    L = np.absolute(Ytest1[i15][0]-fX15_1)/2
    L15[i15,0] = L

# Second Fold
for i15 in np.arange(len(test2)):
    fX15_2 = np.sign(np.dot(test2[i15,:],w15_2))
    L = np.absolute(Ytest1[i15][0]-fX15_2)/2
    L15[i15,1] = L
    
# Third Fold
for i15 in np.arange(len(test3)):
    fX15_3 = np.sign(np.dot(test3[i15,:],w15_3))
    L = np.absolute(Ytest3[i15][0]-fX15_3)/2
    L15[i15,2] = L
    
# Fourth Fold
for i15 in np.arange(len(test4)):
    fX15_4 = np.sign(np.dot(test4[i15,:],w15_4))
    L = np.absolute(Ytest4[i15][0]-fX15_4)/2
    L15[i15,3] = L
    
# Fifth Fold
for i15 in np.arange(len(test5)):
    fX15_5 = np.sign(np.dot(test5[i15,:],w15_5))
    L = np.absolute(Ytest5[i15][0]-fX15_5)/2
    L15[i15,4] = L

error15 = sum(sum(L15))/L15.size # Error rate: 45%

