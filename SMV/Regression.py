from scipy.stats import kstest
import pandas as pd
import sklearn.model_selection as sk
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm


poly_df = pd.DataFrame(pd.read_csv('poly_data.csv'))


#cant call single coloum as pandas convertes it to a pandas series (so data points arent disticnt from features) so have to do this
#Basically in a pandas series is the next element the next feature or the next data point so we make it a data frame again ie a 2d list
Xtrain, Xtest, ytrain, ytest =  sk.train_test_split(poly_df[["X"]], poly_df.y, test_size=0.2, random_state= 2)

poly_model = SVR(kernel='poly',degree=4,epsilon=10,coef0=2,C=1)
rbf_model = SVR(kernel='rbf',gamma = 20, coef0=2, epsilon = 10, C = 100)

poly_model.fit(Xtrain, ytrain)
rbf_model.fit(Xtrain, ytrain)




fig, ax = plt.subplots(1,1,figsize = (6,4),dpi = 150)
X_plot =pd.DataFrame(np.linspace(-4,4,1000),columns=["X"]) # making it a data frame so the coloum names match up

ax.plot(X_plot,poly_model.predict(X_plot), color='#D55E00', label = "Kernel = 'poly'")
ax.plot(X_plot, rbf_model.predict(X_plot), color='#56B4E9',label = "Kernel = 'rbf'")
ax.scatter(Xtrain, ytrain, color='black',label='Training data', s = 10)
ax.scatter(Xtest, ytest, color='#009E73',label='Test data', s = 10 )


ax.set_xlabel('X',fontsize = 20)
ax.set_ylabel('y',fontsize = 20)
ax.tick_params(which='both',labelsize = 12,direction='in',top=True,right=True)
ax.legend(loc='upper center',fontsize = 12, edgecolor='black')
plt.show()


#performing a Kolmogorov-Smirnov to evalue the goodness of fit
print(kstest(poly_model.predict(X_plot),ytest))
print(kstest(rbf_model.predict(X_plot),ytest))
