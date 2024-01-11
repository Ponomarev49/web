import streamlit as st
import pandas as pd
import pickle
from keras.models import load_model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

import author


# Страница с инференсом моделей
def page_predictions():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        predict_input = pd.DataFrame(
            {'time_left': [st.number_input("time_left", min_value=1.0, max_value=180.0, step=5.0, value=90.0)],
             'ct_score': [author.data['ct_score'].mode()[0]],
             't_score': [author.data['t_score'].mode()[0]],
             'map': [author.data['map'].mode()[0]],
             'ct_health': [st.number_input("ct_health", min_value=1.0, max_value=500.0, step=20.0, value=100.0)],
             't_health': [st.number_input("t_health", min_value=1.0, max_value=500.0, step=20.0, value=100.0)],
             'ct_armor': [author.data['ct_armor'].mode()[0]],
             't_armor': [author.data['t_armor'].mode()[0]],
             'ct_money': [author.data['ct_money'].mode()[0]],
             't_money': [author.data['t_money'].mode()[0]],
             'ct_helmets': [author.data['ct_helmets'].mode()[0]],
             't_helmets': [author.data['t_helmets'].mode()[0]],
             'ct_defuse_kits': [author.data['ct_defuse_kits'].mode()[0]],
             'ct_players_alive': [st.number_input("ct_players_alive", min_value=1.0, max_value=5.0, step=1.0,
                                                  value=1.0)],
             't_players_alive': [st.number_input("t_players_alive", min_value=1.0, max_value=5.0, step=1.0,
                                                 value=1.0)]})
        if 'Unnamed: 0' in predict_input.columns:
            predict_input = predict_input.drop(['Unnamed: 0'], axis=1)


        st.write(predict_input)

        if st.button('Сделать предсказание'):
            with open('Models/knn_model.pkl', 'rb') as file:
                knn = pickle.load(file)
            with open('Models/kmeans_model.pkl', 'rb') as file:
                kmeans_model = pickle.load(file)
            nn_model = load_model('Models/bin_class.h5')
            with open('Models/bagging_model.pkl', 'rb') as file:
                bagging_model = pickle.load(file)
            with open('Models/gradient_model.pkl', 'rb') as file:
                gradient_model = pickle.load(file)
            with open('Models/stacking_model.pkl', 'rb') as file:
                stacking_model = pickle.load(file)

            st.header("Stacking:")
            stacking_pred = stacking_model.predict(predict_input)[0]
            st.write(f"{stacking_pred}")

            st.header("gradient:")
            gradient_pred = gradient_model.predict(predict_input)[0]
            st.write(f"{gradient_pred}")

            st.header("bagging:")
            bagging_pred = bagging_model.predict(predict_input)[0]
            st.write(f"{bagging_pred}")

            st.header("knn:")
            knn_pred = knn.predict(predict_input)[0]
            st.write(f"{knn_pred}")

            st.header("Binary Classification:")
            nn_pred = int(nn_model.predict(predict_input, verbose=None))
            st.write(f"{nn_pred}")

            st.header("kmeans:")
            kmeans_pred = kmeans_model.predict(predict_input)[0]
            st.write(f"{kmeans_pred}")
