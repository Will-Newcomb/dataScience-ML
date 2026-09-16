from sklearn.datasets import make_moons
import sklearn.model_selection as skms
import sklearn.tree as skt
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import sklearn.ensemble as ske
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm
import pandas as pd


custom_cmap = ListedColormap(['#D55E00','#56B4E9'])

X, y = make_moons(n_samples = 300, noise = 0.25, random_state = 42)


Xtrain, Xtest,ytrain ,ytest = skms.train_test_split(X,y,test_size=0.2, random_state=2)


bestscore = 0
for i in tqdm(range(1,100,10)):
    for j in range(100,500,100):
        for k in range(1,10,9):
            for l in range(2,10,2):
                Tree = ske.RandomForestClassifier(n_estimators=j,max_depth=i,min_samples_split=l,min_samples_leaf = k, oob_score = True, n_jobs = 8)
                Tree.fit(Xtrain,ytrain)


                pred = Tree.predict(Xtest)
                acc = skm.accuracy_score(ytest,pred)

                oob = Tree.oob_score_

                if bestscore<acc:
                    bestscore = acc
                    bestoob = oob
                    besthype = [i,l,k,acc,oob,j]
                elif bestscore == acc and oob>bestoob:
                    bestscore = acc
                    bestoob = oob
                    besthype = [i,l,k,acc,oob,j]           
titles = ["depth", "minSamplesSplit","min_samples_leaf","acc","oob","nEstimators"]

ForestFrame = pd.DataFrame([besthype], columns = titles )

Forest = ske.RandomForestClassifier(n_estimators=besthype[5],max_depth=besthype[0], min_samples_leaf = besthype[2],min_samples_split= besthype[1])
Forest.fit(Xtrain, ytrain)


Axis = np.linspace(-3,3,100)
grid1, grid2 = np.meshgrid(Axis, Axis)


xAxis = grid1.ravel()
yAxis = grid2.ravel()
coords = np.column_stack((xAxis, yAxis))


pred = Forest.predict(coords)
plt.contourf(grid1, grid2, pred.reshape(grid1.shape), alpha=0.2)
plt.scatter(X[:,0],X[:,1], c=y)
plt.show()

print(ForestFrame)



