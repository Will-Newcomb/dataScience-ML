import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples = 300, noise = 0.25, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 2) 


def Svc(C,Kernal):
    Obj = SVC(C=C,kernel=Kernal)
    Cv =np.mean(cross_val_score(Obj, X_train, y_train))

    Obj.fit(X_train,y_train)

    pred = Obj.predict(X_test)

    return accuracy_score(y_test,pred), Cv


data2=[]
Weights = ["uniform", "distance"]


kernals= ["linear","poly", "rbf", "sigmoid" ]
for i in range(1,100):
    for j in kernals:
        SvcTest, Cv = Svc(i, j)
        test = [i, j , SvcTest, Cv]
        data2.append(test)


coloums= ["C","Kernal","Accuracy","Cross Val"]
frame2 = pd.DataFrame(data2, columns= coloums)
print(frame2)