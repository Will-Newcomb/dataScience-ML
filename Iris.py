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


def OvR(yTest, decisionScore):
    #have to binarize the data so its either the class or not, this is a maths trick to compress multiple calsses into 1 ie it is this or not
    #ie the coloums are is it this or not
    binainarized = label_binarize(yTest, classes = [0,1,2])
    fprData = []
    tprData = []
    ThresholdsData = []
    #have to run though coloum by coloum to compare each class vs all calsses as roc curve only take 1d arrays   
    #ie it works with a 1d array of 2 classes but for more classes we have to binarize into is that class or not ie 1 is 1 then all others are 0
    for i in range(0,len(binainarized[0])):
        fpr, tpr, thresholds = roc_curve(binainarized[:,i], decisionScore[:,i]) #returns fpr ect values for a specific class vs all others in a list for different threshold values 
        fprData.append(fpr)
        tprData.append(tpr)
        ThresholdsData.append(thresholds)
    return fprData, tprData, ThresholdsData   

iris=load_iris()
irisFrame = pd.DataFrame(iris.data,columns=iris.feature_names)

#pd.plotting.scatter_matrix(irisFrame, c = iris.target, hist_kwds={'alpha':0.5,'edgecolor':'black'},figsize=(9,9), cmap = cm.cividis)
#plt.show()

XTrain, XTest, yTrain, yTest = train_test_split(irisFrame, iris.target,test_size = 0.2,random_state = 3)


Weight = ["uniform","distance"]
results = []


# Binarize yTest once before the loop

#for aoc curves for these as theres more than 2 classes you have to do some dunky stuff 
for i in range(1,20):
    for j in Weight:
        Obj = KNeighborsClassifier(i,weights=j)
        crossVal = np.mean(cross_val_score(Obj,XTrain,yTrain))
        Obj.fit(XTrain,yTrain)

        prediction = Obj.predict(XTest)
        score = accuracy_score(yTest,prediction)
        score1 = accuracy_score(yTest[yTest==0],prediction[yTest==0])
        score2 = accuracy_score(yTest[yTest==1],prediction[yTest==1])
        score3 =accuracy_score(yTest[yTest==2],prediction[yTest==2])


        temp = [i,j,score,score1,score2,score3, crossVal]
        results.append(temp)


resultsSVC = []
kernals= ["linear","poly", "rbf", "sigmoid" ]

for i in range(1,20):
    for j in kernals:
        Obj = SVC(C=i,kernel=j)
        crossVal = np.mean(cross_val_score(Obj,XTrain,yTrain))
        Obj.fit(XTrain,yTrain)

        decisionScore = Obj.decision_function(XTest) # Calculate scores for test points

        fpr, tpr, thresholds= OvO(yTest, decisionScore)
        for k in range(0, len(fpr)):
            plt.plot(fpr[k],tpr[k])     

        rocScore = roc_auc_score(yTest, decisionScore, multi_class= "ovo")
        print(rocScore)

        prediction = Obj.predict(XTest)
        score = accuracy_score(yTest,prediction)
        score1 = accuracy_score(yTest[yTest==0],prediction[yTest==0])
        score2 = accuracy_score(yTest[yTest==1],prediction[yTest==1])
        score3 =accuracy_score(yTest[yTest==2],prediction[yTest==2])


        temp = [i,j,score,score1,score2,score3, crossVal]
        resultsSVC.append(temp)




headers = ["Neighbours", "Weight", "Score","Setosa","Versicolour","virginica", "CrossVal"]
results = pd.DataFrame(results,columns=headers)
print(results)

headers = ["C", "Kernal", "Score","Setosa","Versicolour","virginica", "CrossVal"]
resultsSVC = pd.DataFrame(resultsSVC, columns=headers)
print(resultsSVC)