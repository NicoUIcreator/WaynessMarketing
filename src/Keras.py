import tensorflow as tf
from tensorflow import keras
import keras
import pandas as pd
import numpy as np


dfPuntos = pd.read_csv("../dataLimpio/dfPuntos.csv",index_col=0)

X = dfPuntos[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate','female', 'male', 'wep', 'activity_category']]
yLog = np.log1p(dfPuntos[["Calories"]])

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, yLog, test_size = 0.20, random_state=42) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)

model = tf.keras.Sequential([
  keras.layers.Dense(9,activation='linear', input_shape=(X.shape[1],)),  # Input layer
  keras.layers.Dense(32, activation='relu'),  # Hidden layer with ReLU activation
 
  keras.layers.Dense(units=64, activation='relu'),  # One-hot encoding for activity category
  keras.layers.Dense(units=1, activation='linear')  # Output layer with linear activation
])

model.compile(optimizer='adam', loss='mse',metrics=['mae', 'mse'])


model.fit(X_train, y_train, epochs=50, validation_split=0.2,callbacks=[early_stopping])


model.save('../models/kerasModel.h5')  # Replace with your desired filename and path
