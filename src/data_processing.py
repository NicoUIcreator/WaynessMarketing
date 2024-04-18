import numpy as np
import pandas as pd

#----------------------------------------------------------------

df = pd.read_csv("data/exercise_Calories.csv",index_col=0)
#----------------------------------------------------------------

df[["female","male"]] = pd.get_dummies(df["Gender"])
#----------------------------------------------------------------

def calculate_wep(heart_rate, duration):
  if 70 <= heart_rate <= 89:
    wep = duration * 4  
  elif 90 <= heart_rate <= 109:
    wep = duration * 5
  elif 110 <= heart_rate <= 129:
    wep = duration * 6
  elif heart_rate >= 130:
    wep = duration * 7
  else:
    wep = 0  
  return wep


df['wep'] = df.apply(lambda row: calculate_wep(row["Heart_Rate"], row["Duration"]), axis=1)


#----------------------------------------------------------------
def categorize_activity(heart_rate):

    if heart_rate <= 79:
        return 0  
    elif 80 <= heart_rate <= 99:
        return 5
    elif 100 <= heart_rate <= 120:
        return 7 
    else:
        return 10 

df['activity_category'] = df['Heart_Rate'].apply(categorize_activity)


df.to_csv("dataLimpio/dfPuntos.csv")