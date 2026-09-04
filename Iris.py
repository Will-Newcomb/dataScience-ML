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

iris=load_iris()
irisFrame = pd.DataFrame(iris.data,columns=iris.feature_names)

#pd.plotting.scatter_matrix(irisFrame, c = iris.target, hist_kwds={'alpha':0.5,'edgecolor':'black'},figsize=(9,9), cmap = cm.cividis)
#plt.show()

XTrain, XTest, yTrain, yTest = train_test_split(irisFrame, iris.target,test_size = 0.2,random_state = 3)


Weight = ["uniform","distance"]
results = []

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