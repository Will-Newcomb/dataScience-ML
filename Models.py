from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


models = {

        "SVC": {"model":SVC(probability=True), "params":{"C":np.linspace(1,100,99),"kernel":["linear","poly","rbf","sigmoid"]}},
          
        "DecisionTreeClassifier": DecisionTreeClassifier(),

        "KNN":KNeighborsClassifier(),

        "Forest":RandomForestClassifier()


}
#checking branch