import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#making a data set with 2 features that is centered around 2 areas (number of classes)
X, y = make_blobs(300,2, center_box=(-5,5), centers = 2, random_state = 2) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 3) 


def KNN(neighbours,Weight):
#initlaizing the object and traning it
    Obj = KNeighborsClassifier(neighbours,weights=Weight)
    Cv =np.mean(cross_val_score(Obj, X_train, y_train))#This does a k-fold test to measure accuracy for out of trainnig data and evlauated overfitting 

    Obj.fit(X_train,y_train)
    #predicditing using the test data
    pred = Obj.predict(X_test)
    #calsucing correct over total could be done manually but this is easier
    return accuracy_score(y_test,pred), Cv


def Svc(C,Kernal):
    Obj = SVC(C=C,kernel=Kernal)
    Cv =np.mean(cross_val_score(Obj, X_train, y_train))

    Obj.fit(X_train,y_train)
    pred = Obj.predict(X_test)

    return accuracy_score(y_test,pred), Cv




data=[]
data2=[]
Weights = ["uniform", "distance"]


for i in range(1,50):
    for j in range(0,2):
        accuracy, Cv = KNN(i,Weights[j])
        test= [i,Weights[j],accuracy, Cv]
        data.append(test)

kernals= ["linear","poly", "rbf", "sigmoid" ]
for i in range(1,100):
    for j in kernals:
        SvcTest, Cv = Svc(i, j)
        test = [i, j , SvcTest, Cv]
        data2.append(test)


coloums= ["C","Kernal","Accuracy","Cross Val"]
frame2 = pd.DataFrame(data2, columns= coloums)

coloums=["Number Neigh","Weights","Accuracy","Cross Val"]
frame = pd.DataFrame(data,columns=coloums)
print(frame)
print(frame2)