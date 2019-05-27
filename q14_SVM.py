#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:39:40 2018

@author: abhijay
"""

# Comment
import os
if not os.path.exists('Q14_plots'):
    os.mkdir('Q14_plots')

#os.getcwd()
#os.chdir("/home/abhijay/Documents/ML/hw_1/11632196/")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
#import pickle

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler

# Separate Y label and X from data
def separate_target_variable(data):
    data_x = data.iloc[:,0:-1]
    data_y = data.iloc[:,-1]
    data_y = data_y.apply(lambda y: -1 if y==" <=50K" else 1).tolist()
    return data_x, data_y

# Make train, dev and test set have same number of features
def make_features_correspond(train_, data_):
    missing_features = set(train_.columns) - set(data_.columns)
    missing_features = pd.DataFrame(0, index=np.arange(len(data_)), columns=missing_features)
    data_ = pd.concat( [data_, missing_features], axis = 1)
    return data_ 

# Make dummy variables
def one_hot_encode(data, categorical_features):    
    for categorical_feature in categorical_features:
        dummies = pd.get_dummies(data[categorical_feature],prefix=categorical_feature)
        dummies = dummies.iloc[:,0:-1]
        data = pd.concat( [data, dummies], axis = 1).drop(categorical_feature, axis=1) # Also add logic to remove redundant info
    return data


if __name__ == "__main__":

    print ("\n\n =============== Support Vector Machine ===============\n")
    
    # Get data
    col_names = ["age","workclass","education","marital_status","occupation","race","gender","hours-per-week","native-country","salary-bracket"]
    
    train_data = pd.read_csv("income-data/income.train.txt", names = col_names)
    dev_data = pd.read_csv("income-data/income.dev.txt", names = col_names)
    test_data = pd.read_csv("income-data/income.test.txt", names = col_names)
    
    # Converting numerical data into buckets
    train_data['age'] = train_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
    dev_data['age'] = dev_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
    test_data['age'] = test_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
    
    train_data['hours-per-week'] = train_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
    dev_data['hours-per-week'] = dev_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
    test_data['hours-per-week'] = test_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
    
    # Separating out features from target variable
    train_x, train_y = separate_target_variable(train_data)
    dev_x, dev_y = separate_target_variable(dev_data)
    test_x, test_y = separate_target_variable(test_data)
    
    # One hot encode all categorical variables
    categorical_features = [train_x.columns[col] for col, col_type in enumerate(train_x.dtypes) if col_type == np.dtype('O') ]
    #numerical_features = [col for col, col_type in enumerate(train_x.dtypes) if col_type != np.dtype('O') ]
    
    train_x = one_hot_encode(train_x, categorical_features)
    dev_x = one_hot_encode(dev_x, categorical_features)
    test_x = one_hot_encode(test_x, categorical_features)
    
    
    # Make features in dev categories consistent with train
    dev_x = make_features_correspond( train_x, dev_x)
    test_x = make_features_correspond( train_x, test_x)
    
    train_features = set(train_x.columns[(train_x.var(axis=0)>0.05)].tolist())
    #dev_features = set(dev_x.columns[(dev_x.var(axis=0)>0.1)].tolist())
    test_features = set(test_x.columns[(test_x.var(axis=0)>0.05)].tolist())
    
    print ("Number of features selected for training: ", len(train_features))
    
    final_list_features = list(train_features.intersection(test_features))
    final_list_features.sort()
    
    train_x = train_x[final_list_features]
    dev_x = dev_x[final_list_features]
    test_x = test_x[final_list_features]
    
    # Now that the features are consistent I can convert my datasets into numpy arrays
    train_x = np.array(train_x.values)
    dev_x = np.array(dev_x.values)
    test_x = np.array(test_x.values)
    
    #scaler = StandardScaler().fit(train_x)
    #train_x = scaler.transform(train_x)
    #test_x = scaler.transform(test_x)
    #dev_x = scaler.transform(dev_x)
    
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_x.astype(float))
    train_x = scaling.transform(train_x)
    dev_x = scaling.transform(dev_x)
    test_x = scaling.transform(test_x)
    
    print ("\n##### 14(a) Train and find best C #####")
           
    trainAccuracy = []
    devAccuracy = []
    testAccuracy = []
    numSupportVectors = []
    bestValidationAccuracy = 0
    bestC = 0

    for power in range(-4,5):
        start_time = time.time()
        c = 10**power # change the way it's written
        svm = SVC(kernel='linear', C=c)
        svm_fit = svm.fit(train_x, train_y)
        train_yHat = svm_fit.predict(train_x)
        
        print("\n\n-----\nC = 10^"+str(power),"\n-----\n",classification_report(train_y, train_yHat))
        trainAccuracy_ = round(accuracy_score(train_y, train_yHat),2)
        trainAccuracy.append(trainAccuracy_)
        print ("Train Accuracy: ", trainAccuracy_)
        print ("Number of support vectors:",svm_fit.n_support_)
        
        dev_yHat = svm_fit.predict(dev_x)
        devAccuracy_ = round(accuracy_score(dev_y, dev_yHat),3)
        if bestValidationAccuracy < devAccuracy_:
            bestValidationAccuracy = devAccuracy_
            bestC = c
        devAccuracy.append(devAccuracy_)
        print("\n",classification_report(dev_y, dev_yHat))
        print ("Dev Accuracy: ", devAccuracy_)
        
        test_yHat = svm_fit.predict(test_x)
        testAccuracy_ = round(accuracy_score(test_y, test_yHat),2)
        testAccuracy.append(testAccuracy_)
        print("\n",classification_report(test_y, test_yHat))
        print ("Test Accuracy: ", testAccuracy_)
        
        numSupportVectors.append(svm_fit.n_support_)
        print ("Number of support vectors:",numSupportVectors[-1])
        print("--- %s seconds ---" % (time.time() - start_time))
    
    # with open('svm_results.pkl', 'wb') as f:
    #     strObj = [trainAccuracy, devAccuracy, testAccuracy, numSupportVectors, bestValidationAccuracy, bestC]
    #     pickle.dump( strObj, f)
    
    ## Getting back the objects:
    #with open('svm_results.pkl','rb') as f:
    #    trainAccuracy, devAccuracy, testAccuracy, numSupportVectors, bestValidationAccuracy, bestC = pickle.load(f)
    
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot( range(-4,5), trainAccuracy, label='Training Accuracy')
    ax.plot( range(-4,5), devAccuracy, label='Dev Accuracy')
    ax.plot( range(-4,5), testAccuracy, label='Test Accuracy')
    #ax.set_ylim(5800, 6250)
    plt.xticks( range(-4,5), ["10^"+str(c) for c in range(-4,5)])
    ax.set_title('Accuracies with changing C', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_xlabel('C', fontsize=15)
    ax.legend(loc='lower right')
    fig.savefig('Q14_plots/Q14a_AccuracyPlot.png')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot( range(-4,5), np.array(numSupportVectors)[:,0], label='Num of Support Vectors for -1')
    ax.plot( range(-4,5), np.array(numSupportVectors)[:,1], label='Num of Support Vectors for +1')
    plt.xticks( range(-4,5), ["10^"+str(c) for c in range(-4,5)])
    ax.set_title('Number of Support Vectors with changing C', fontsize=18)
    ax.set_ylabel('Support Vectors', fontsize=15)
    ax.set_xlabel('C', fontsize=15)
    ax.legend(loc='lower right')
    fig.savefig('Q14_plots/Q14b_numSupportVectors.png')

    #########################
    
    print ("##### 14(b) Train on train+dev data #####")
    
    trainDev_x = np.concatenate((train_x, dev_x), axis=0)
    trainDev_y = train_y+dev_y
    
    svm = SVC(kernel='linear', C=bestC)
    svm_fit = svm.fit(trainDev_x, trainDev_y)
    test_yHat = svm_fit.predict(test_x)
    testAccuracy_ = round(accuracy_score(test_y, test_yHat),2)
    print ("Test Accuracy: ", testAccuracy_)
    print("\n",classification_report(test_y, test_yHat))
    
    
    #########################
    
    print ("##### 14(c) Best C and changing polynomial kernel degree #####")
    trainAccuracy = [trainAccuracy[np.argmax(devAccuracy)]]
    devAccuracy = [devAccuracy[np.argmax(devAccuracy)]]
    testAccuracy = [testAccuracy[np.argmax(devAccuracy)]]
    numSupportVectors = [numSupportVectors[np.argmax(devAccuracy)]]
    
    for degree in range(2,5):
        svm = SVC(kernel='poly', C=bestC, degree=degree)
        svm_fit = svm.fit(train_x, train_y)
        train_yHat = svm_fit.predict(train_x)
        print("\n-----\nC = 10^"+str(power)+", Degree = "+str(degree),"\n-----\n")
        trainAccuracy_ = round(accuracy_score(train_y, train_yHat),2)
        trainAccuracy.append(trainAccuracy_)
        print ("Train Accuracy: ", trainAccuracy_)
        print ("Number of support vectors:",svm_fit.n_support_)
        
        dev_yHat = svm_fit.predict(dev_x)
        devAccuracy_ = round(accuracy_score(dev_y, dev_yHat),2)
        devAccuracy.append(devAccuracy_)
        print("\n",classification_report(dev_y, dev_yHat))
        print ("Dev Accuracy: ", devAccuracy_)
        
        test_yHat = svm_fit.predict(test_x)
        testAccuracy_ = round(accuracy_score(test_y, test_yHat),2)
        testAccuracy.append(testAccuracy_)
        print("\n",classification_report(test_y, test_yHat))
        print ("Test Accuracy: ", testAccuracy_)
        
        numSupportVectors.append(svm_fit.n_support_)
        print ("Number of support vectors:",numSupportVectors[-1])
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot( range(1,5), trainAccuracy, label='Training Accuracy')
    ax.plot( range(1,5), devAccuracy, label='Dev Accuracy')
    ax.plot( range(1,5), testAccuracy, label='Test Accuracy')
    plt.xticks( range(1,5), ["linear 10^"+str(bestC)]+["Poly x^"+str(degree) for degree in range(2,5)])
    ax.set_title('Accuracies with changing Polynomial Degree', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_xlabel('Kernel linear and polynomial degree', fontsize=15)
    ax.legend(loc='upper right')
    fig.savefig('Q14_plots/Q14c_BestC_AccuracyPlot.png')
    
    print ("Best polynomial degree based on testAccuracy is: ",str(range(2,5)[np.argmax(testAccuracy)]))
    # # Saving the objects:
    # with open('svm_results.pkl', 'wb') as f:
    #     strObj = str(range(2,5)[np.argmax(testAccuracy)])
    #     pickle.dump( strObj, f)