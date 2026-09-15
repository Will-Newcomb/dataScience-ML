import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score




def OvO(yTest, decisionScore):
    fprData = []
    tprData = []
    ThresholdsData = []
    for i in range(len(decisionScore[0])):
        for j in range(i + 1,len(decisionScore[0])):
            #removing the data that we arent looking to compare

            filteredyTest = yTest[(yTest == i) | (yTest == j)]
            #taking the confidance we are in one as the dicsion factor 
            filteredDecisionScore = decisionScore[(yTest == i) | (yTest == j),i]
            fpr, tpr, thresholds = roc_curve(filteredyTest, filteredDecisionScore, pos_label=i)

            #makes lists for one vs one sets
            fprData.append(fpr)
            tprData.append(tpr)
            ThresholdsData.append(thresholds)
    return fprData, tprData, ThresholdsData   


def OvR(yTest, decisionScore, type= None, AUC=False):
    #have to binarize the data so its either the class or not, this is a maths trick to compress multiple calsses into 1 ie it is this or not
    #ie the coloums are is it this or not
    binainarized = label_binarize(yTest, classes = [0,1,2])
    fprData = []
    tprData = []
    ThresholdsData = []
    #have to run though coloum by coloum to compare each class vs all calsses as roc curve only take 1d arrays   
    
    #ie it works with a 1d array of 2 classes but for more classes we have to binarize into is that class or not ie 1 is 1 then all others are 0
    if type == "micro":
        #take the coloums and put them ened on end to make a 1d array this means when calcuting auc score all tp ect are added to from the micro metric
        binainarized = binainarized.ravel()
        decisionScore = decisionScore.ravel()
        fprData, tprData, ThresholdsData = roc_curve(binainarized, decisionScore)     

    elif type == "All" or type == "Macro" or type == None:
        for i in range(0,len(binainarized[0])):
            fpr, tpr, thresholds = roc_curve(binainarized[:,i], decisionScore[:,i]) #returns fpr ect values for a specific class vs all others in a list for different threshold values 
            fprData.append(fpr)
            tprData.append(tpr)
            ThresholdsData.append(thresholds)


        #need to come back to this as thershold values are different as fpr data is different lengths so need interpolation
        if type == "Macro":
            fprData = np.sum(fprData, axis = 1)
            tprData = np.sum(tprData, axis = 1)


    return fprData, tprData, ThresholdsData   