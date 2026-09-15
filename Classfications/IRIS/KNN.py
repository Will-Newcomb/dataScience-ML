import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris




#Loading in the iris dataset
iris=load_iris()
irisFrame = pd.DataFrame(iris.data,columns=iris.feature_names)

#Creating the test train split
XTrain, XTest, yTrain, yTest = train_test_split(irisFrame, iris.target,test_size = 0.2,random_state = 3)

#Prepareing Hyperparamaetrs to test and a results list to save too
Weight = ["uniform","distance"]
results = []


#varying neighbours from 1 to 20 and weights between uniform and distance
#calculating crossvalue scores and accuracy score for all plant types
#creating a final 2d list containg all results and hyper paramteres
#Calculating score inside loop ot investigate hoe hyper parameters chnage accuracy

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


#converting the 2d list to a data frame
headers = ["Neighbours", "Weight", "Score","Setosa","Versicolour","virginica", "CrossVal"]
results = pd.DataFrame(results,columns=headers)


