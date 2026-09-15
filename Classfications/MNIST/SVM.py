from sklearn.datasets import fetch_openml
import sklearn.preprocessing as skp
import sklearn.model_selection as skms
import sklearn.svm as svm
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm  as cm
import pandas as pd
from tqdm import tqdm




mnist = fetch_openml('mnist_784')

imageData = pd.DataFrame(mnist.data) # Get the first point in the dataset
imageTarget = mnist.target # Reshape into its image form

#could have used a coloum of binarized data but did this for speed reasons
ReducedTarget = np.zeros(len(imageTarget))
ReducedTarget[imageTarget == "5"] = "1" # string as data set stores them as this

Xtrain, Xtest, ytrain, ytest = skms.train_test_split(imageData, ReducedTarget, test_size= 0.99, random_state=2)


#setting up the SCV to vary hyperparamters though

kernals = ["linear", "poly", "rbf", "sigmoid"]
bestAuc = 0

for i in tqdm(range(1,50)):
    for j in kernals:
        
        Classaifier = svm.SVC(C=i, kernel=j)
        Classaifier.fit(Xtrain, ytrain)

        predScore = Classaifier.decision_function(Xtest)
        Auc = skm.roc_auc_score(ytest,predScore)

        if Auc>bestAuc:
            bestAuc = Auc
            bestHyperParams = [i, j]



Classaifier = svm.SVC(C=bestHyperParams[0], kernel=bestHyperParams[1])
Classaifier.fit(Xtrain, ytrain)


pred = Classaifier.predict(Xtest)
accuracy = skm.accuracy_score(ytest, pred)
print(accuracy)

predScore = Classaifier.decision_function(Xtest)
fpr, tpr, thresholds = skm.roc_curve(ytest,predScore)
Auc = skm.roc_auc_score(ytest,predScore)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label = Auc)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.legend()
plt.show()
