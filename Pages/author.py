import streamlit as st

import classification
import regression

st.set_option('deprecation.showPyplotGlobalUse', False)

# Навигация
st.sidebar.title('Навигация:')
page = st.sidebar.radio(
    "Выберите страницу",
    ("Разработчик", "Классификация", "Регрессия"),
    key="navi"
)

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
elif page == "Классификация":
    classification.classification()
elif page == "Регрессия":
    regression.regression()
