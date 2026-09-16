from sklearn.datasets import fetch_california_housing
from scipy.stats import kstest
import sklearn.svm as svm
import sklearn.preprocessing as skp
import sklearn.pipeline as skpipe
import sklearn.model_selection as skms
import sklearn.metrics as skm
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm


california_housing = fetch_california_housing()

#pd.plotting.scatter_matrix(DataFeatres,c=DataTargets,cmap = cm.cividis)
#plt.show()

DataFeatres = pd.DataFrame(california_housing.data,columns=california_housing.feature_names)
DataTargets = california_housing.target


Xtrain, Xtest, ytrain, ytest = skms.train_test_split(DataFeatres, DataTargets, test_size=0.2, random_state= 2)


results = []

for i in tqdm(range(1,50)):
    for j in range(0.1,1):
        for k in range(1,10):

            Regressor = skpipe.make_pipeline(skp.RobustScaler(), svm.SVR(C=i,kernel="rbf", epsilon=j, gamma=k) )
            Regressor.fit(Xtrain, ytrain)
            pred = Regressor.predict(Xtest)

            mse = skm.mean_squared_error(ytest, pred)
            mae = skm.mean_absolute_error(ytest, pred)
            rmse = skm.root_mean_squared_error(ytest, pred)

            r2 = skm.r2_score(ytest, pred)
            k = kstest(ytest, pred)
            results.append([mae,mse,rmse,r2,k])




titles = ["MAE", "MSE", "RMSE", "R2", "K"]
results = pd.DataFrame(results,columns=titles)
print(results)




