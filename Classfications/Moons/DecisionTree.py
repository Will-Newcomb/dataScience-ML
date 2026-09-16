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
for i in tqdm(range(1,100)):
    for j in range(2,10):
        for k in range(1,10):
            Tree = skt.DecisionTreeClassifier(max_depth=i,min_samples_leaf = k, min_samples_split=j)
            Tree.fit(Xtrain,ytrain)


            pred = Tree.predict(Xtest)
            acc = skm.accuracy_score(ytest,pred)

            if bestscore<acc:
                bestscore = acc
                besthype = [i,j,k,acc]
titles = ["depth", "minSamplesSplit","min_samples_leaf","acc"]
bestFrame = pd.DataFrame([besthype],columns=titles)



Tree = skt.DecisionTreeClassifier(max_depth=besthype[0], min_samples_split=besthype[1], min_samples_leaf = besthype[2])
Tree.fit(Xtrain, ytrain)


Axis = np.linspace(-3,3,100)
grid1, grid2 = np.meshgrid(Axis, Axis)


xAxis = grid1.ravel()
yAxis = grid2.ravel()
coords = np.column_stack((xAxis, yAxis))


pred = Tree.predict(coords)
plt.contourf(grid1, grid2, pred.reshape(grid1.shape), alpha=0.2)
plt.scatter(X[:,0],X[:,1], c=y)
plt.show()

print(bestFrame)