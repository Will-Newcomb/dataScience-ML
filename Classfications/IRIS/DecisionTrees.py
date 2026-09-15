import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import sklearn.metrics as skm
from tqdm import tqdm



iris = load_iris()
Xtrain, Xtest, ytrain, ytest = train_test_split(iris['data'],iris['target'], test_size = 0.2, random_state = 20) 

best=0
hyperparams = []
for i in tqdm(range(1,20)):
    for j in range(2,20):
        Dtc = tree.DecisionTreeClassifier(max_depth = i,min_samples_split=j, random_state = 1)
        Dtc.fit(Xtrain, ytrain)

        pred = Dtc.predict(Xtest)
        acc = skm.accuracy_score(ytest,pred)


        settings = [i,j, acc]
        hyperparams.append(settings)

        if best<acc:
            bestHyperParams = settings


titles = ["Depth", "MinSplitSamples", "Accuracy"]
results = pd.DataFrame(hyperparams,columns=titles)
print(results)


Dtc = tree.DecisionTreeClassifier(max_depth = bestHyperParams[0],min_samples_split= bestHyperParams[1], random_state = 1)
Dtc.fit(Xtrain, ytrain)
print(bestHyperParams)

tree.plot_tree(Dtc, filled = True)
plt.show()