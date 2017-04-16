#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 
c = 10000.
my_kernel = 'rbf'
#clf = svm.SVC(kernel='linear')
def my_predic(c,my_kernel):
	time_start = time()
	clf = svm.SVC(C=c,kernel=my_kernel)
	clf.fit(features_train,labels_train)
	print clf
	predict_test = clf.predict(features_test)
	time_end = time()
	print "predict_test: ",np.sum(predict_test)
#	print "c = {0} accuracy_score: {1} ".format(c, accuracy_score(labels_test,predict_test))
#	print "time: ", time_end - time_start
my_predic(c,my_kernel)