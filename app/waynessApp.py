#importar librerias
import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np
import funciones as f
import tensorflow 
import keras


st.set_page_config(page_title="Wayness Points",
                   page_icon="../img/favicon.png")

seleccion = st.sidebar.selectbox("Selecciona menu", ["Home", "Data", "Predicciones"])

if seleccion == "Home":
    st.title("Wayness App")

    with st.expander("¿Qué es esta aplicación?"):
        st.write("Somos una nueva empresa Start-up cuyo objetivo es crear la motivación para lograr el equilibrio adecuado entre el deber y el placer, la responsabilidad y los hobbies, mejorando así la calidad de vida de las personas. Muy convencidos de que el correcto equilibrio entre la salud física, mental y emocional, es fundamental para que las personas sean más felices y productivas, pensamos en crear una App que, gracias a un sistema de puntos y a los incentivos adecuados, nos diera la oportunidad de animar a los usuarios a realizar cualquier tipo de actividad fisica")

    img = Image.open("../img/favicon.png")
    st.image(img)

elif seleccion == "Data":
    st.write("Datos para entrenar Modelo")
    dfPuntos = pd.read_csv("../dataLimpio/dfPuntos.csv",index_col=0)
    st.write(dfPuntos.head(15))

    img1 = Image.open("../img/heatmap.png")
    st.image(img1)

elif seleccion == "Predicciones":
    def main():

        #Extrar los archivos pickle
        with open('../models/best_model.pkl', 'rb') as gbr:
            gbr_reg = pickle.load(gbr)

        with open('../models/xgb_model.pkl', 'rb') as xgb:
            xgb_m = pickle.load(xgb)

        
        #titulo
        st.title('Prediccion Gasto Calorico y Ganancia de puntos Wayness')
        #titulo de sidebar
        st.sidebar.header('Datos Usuario')
        #escoger el modelo preferido
        option = ['GradientBoostingRegressor', 'XGB Model']
        model = st.sidebar.selectbox('Que modelo te gustaria usar?', option)


        #funcion para poner los parametrso en el sidebar
        def user_input_parameters():

                Age = st.sidebar.slider('Edad', 10, 60, 1)
                Height = st.sidebar.slider('Altura', 140, 210, 1)
                Weight = st.sidebar.slider('Peso', 45, 130, 1)
                Duration = st.sidebar.slider('Duracion de Ejercicio en Minutos', 5, 180, 1)
                Heart_Rate= st.sidebar.slider('Intensidad de Ejercicio en HR', 60, 180, 1)
                female= st.sidebar.slider('Mujer', 0,1)
                male= st.sidebar.slider('Hombre', 0,1)
                wep= f.calculate_wep(Heart_Rate, Duration)
                activity_category= f.categorize_activity(Heart_Rate)

                data = {'Age': Age,
                        'Height': Height,
                        'Weight': Weight,
                        'Duration': Duration,
                        "Heart_Rate":Heart_Rate,
                        "female":female,
                        "male":male,
                        "wep":wep,
                        "activity_category":activity_category,
                        }
                
                features = pd.DataFrame(data, index=[0])
                return features

        df = user_input_parameters()

        st.subheader('User Input Parameters')
        st.subheader(model)
        st.write(df)

        dfPuntos = pd.read_csv("../dataLimpio/dfPuntos.csv",index_col=0)

        X=dfPuntos[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate','female', 'male', 'wep', 'activity_category']]
        scaler = StandardScaler()
        XScal = scaler.fit_transform(X)

        if st.button('RUN'):
            if model == 'GradientBoostingRegressor':
                    resultado = (np.expm1(gbr_reg.predict(scaler.transform(df)))).round(2)
                    st.success(f"Vas a quemar {resultado} Calorias!")
                    img3 = Image.open("../img/gym.jpg")
                    st.image(img3)
            elif model == 'XGB Model':
                    resultado = (np.expm1(xgb_m.predict(scaler.transform(df)))).round(2)
                    st.success(f"Vas a quemar {resultado} Calorias!")
                    img4 = Image.open("../img/corriendo.jpg")
                    st.image(img4)
                 
            else:
                None

    if __name__ == '__main__':
        main()