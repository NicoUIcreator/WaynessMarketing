#importar las librerias necesarias

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
import pickle


dfPuntos = pd.read_csv("../dataLimpio/dfPuntos.csv",index_col=0)

X = dfPuntos[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate','female', 'male', 'wep', 'activity_category']]

yLog = np.log1p(dfPuntos["Calories"])


Xscal = StandardScaler().fit_transform(X)
gbr_model = GradientBoostingRegressor(n_estimators=400 , 
                                      learning_rate= 0.1,
                                      max_depth= 5,
                                      min_samples_leaf= 1,
                                      min_samples_split= 6,
                                      verbose=1)
gbr_model.fit(Xscal, yLog)


with open('../models/finished_modelgrb.pkl', 'wb') as gbr:
    pickle.dump(gbr_model, gbr)


