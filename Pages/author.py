import streamlit as st
import pandas as pd

import info
import models
import visualisation

st.set_option('deprecation.showPyplotGlobalUse', False)

# Навигация
st.sidebar.title('Навигация:')
page = st.sidebar.radio(
    "Выберите страницу",
    ("Разработчик", "Датасет", "Визуализация", "Инференс модели"),
    key="navi"
)

# Загрузка датасета
data = pd.read_csv('Data/players.csv')
if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)
y_train = data['overall']
X_train = data.drop(columns=['overall'])

st.title('Расчётно графичесикая работа ML')


# Информация о разработчике
def page_developer():
    st.title("Информация о разработчике")
    st.header("Тема РГР:")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Фотография")
        st.image("Images/my_photo.png")  # Укажите путь к вашей фотографии
    with col2:
        st.header("Контактная информация")
        st.write("ФИО: Пономарев Михаил Евгеньевич")
        st.write("Номер учебной группы: ФИТ-221")


if page == "Разработчик":
    page_developer()
elif page == "Датасет":
    info.page_dataset()
elif page == "Инференс модели":
    models.page_predictions()
elif page == "Визуализация":
    visualisation.page_data_visualization()
