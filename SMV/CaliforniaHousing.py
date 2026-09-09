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

california_housing = fetch_california_housing()

#pd.plotting.scatter_matrix(DataFeatres,c=DataTargets,cmap = cm.cividis)
#plt.show()

DataFeatres = pd.DataFrame(california_housing.data,columns=california_housing.feature_names)
DataTargets = california_housing.target





Xtrain, Xtest, ytrain, ytest = skms.train_test_split(DataFeatres, DataTargets, test_size=0.2, random_state= 2)



Regressor = skpipe.make_pipeline(skp.RobustScaler(), svm.SVR(kernel="rbf") )
Regressor.fit(Xtrain, ytrain)
pred = Regressor.predict(Xtest)

mse = skm.mean_squared_error(ytest, pred)
mae = skm.mean_absolute_error(ytest, pred)
rmse = skm.root_mean_squared_error(ytest, pred)

r2 = skm.r2_score(ytest, pred)
k = kstest(ytest, pred)
print(k)




