import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf


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
            {'age': [st.number_input("Возраст", min_value=16, max_value=42, step=1, value=20)],
             'pace': [st.number_input("Скорость", min_value=24, max_value=100, step=1, value=70)],
             'shooting': [st.number_input("Удары", min_value=15, max_value=100, step=1, value=70)],
             'passing': [st.number_input("Пасы", min_value=24, max_value=100, step=1, value=70)],
             'dribbling': [st.number_input("Дриблинг", min_value=23, max_value=100, step=1, value=70)],
             'defending': [st.number_input("Защита", min_value=23, max_value=100, step=1, value=70)],
             'physic': [st.number_input("Физика", min_value=27, max_value=100, step=1, value=70)]})
        if 'Unnamed: 0' in predict_input.columns:
            predict_input = predict_input.drop(['Unnamed: 0'], axis=1)

        st.write(predict_input)

        if st.button('Сделать предсказание'):
            with open('Models/bagging.pkl', 'rb') as file:
                bagging_model = pickle.load(file)
            with open('Models/ridge.pkl', 'rb') as file:
                ridge_model = pickle.load(file)
            with open('Models/gradient.pkl', 'rb') as file:
                gradient_model = pickle.load(file)
            with open('Models/stacking.pkl', 'rb') as file:
                stacking_model = pickle.load(file)
            loaded_model = tf.keras.models.load_model('Models/Regression')

            pred = []

            stacking_pred = int(stacking_model.predict(predict_input)[0])
            pred.append(stacking_pred)
            st.header(f"stacking: {stacking_pred}")

            gradient_pred = int(gradient_model.predict(predict_input)[0])
            pred.append(gradient_pred)
            st.header(f"gradient boosting: {gradient_pred}")

            bagging_pred = int(bagging_model.predict(predict_input)[0])
            pred.append(bagging_pred)
            st.header(f"bagging: {bagging_pred}")

            ridge_pred = int(ridge_model.predict(predict_input)[0])
            pred.append(ridge_pred)
            st.header(f"ridge: {ridge_pred}")

            nn_pred = int(loaded_model.predict(predict_input)[0])
            pred.append(nn_pred)
            st.header(f"neural network: {nn_pred}")

            st.header(f"Final predict: {int(sum(pred) / len(pred))}")
