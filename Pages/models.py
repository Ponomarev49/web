import pickle


import pandas as pd
import streamlit as st
from tensorflow.keras.layers import TFSMLayer


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

            # from keras.models import load_model
            # model_regression = load_model('Models/regression.h5')

            # nn_model = load_model('Models/regression.h5')
            # with open('Models/model_config.json', 'r') as f:
            #     saved_model_config = json.load(f)
            # model = model_from_config(saved_model_config)

            # model_regression = tf.keras.Sequential(
            #     [
            #         # Dense - полносвязный слой (каждый нейрон следующего слоя связан со всеми нейронами предыдущего)
            #         tf.keras.layers.Dense(64, activation="relu", input_shape=(7,)),
            #         # на втором скрытом слое будет 32 нейрона
            #         tf.keras.layers.Dense(32, activation="linear"),
            #         # Dropout позволяет внести фактор случайности - при обучении часть нейронов будет отключаться
            #         # каждый нейрон, в данном случае, будет отключаться с вероятностью 0.1
            #         tf.keras.layers.Dropout(0.1),
            #         tf.keras.layers.Dense(16, activation="relu"),
            #         tf.keras.layers.Dropout(0.1),
            #         # на выходе один нейрон, функция активации не применяется
            #         tf.keras.layers.Dense(1, activation="linear"),
            #     ]
            # )
            # model_regression.summary()
            # # компилируем
            # model_regression.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            #                          loss=tf.keras.losses.MeanAbsoluteError())
            # model_regression.fit(author.X_train, author.y_train, epochs=50, verbose=None)
            #
            # tf.saved_model.save(model_regression, 'Models\Regression')
            # model_regression = tf.saved_model.load('Models\Regression')
            # model_regression.save(filepath='Models/RegressionModel')
            # model_regression = tf.keras.models.load_model('Models/Regression')

            smlayer = TFSMLayer('Models/Regression', call_endpoint='serving_default')
            # output = smlayer(predict_input)
            # model = tf.keras.Model(inputs=predict_input, outputs=output)
            # predictions = model.predict(predict_input)
            # nn_pred=predictions[0]

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

            nn_pred = int((smlayer(predict_input))['output_0'][0][0])
            pred.append(nn_pred)
            st.header(f"neural network: {nn_pred}")

            st.header(f"Final predict: {pred}")
