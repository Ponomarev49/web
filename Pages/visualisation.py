import streamlit as st
import author


# Страница с визуализацией

def page_data_visualization():
    st.title("Визуализации данных")

    st.image(f"Images/heatmap.png")

    st.subheader("Гистограммы")

    def plot_histogram(column_name):
        st.image(f"Images/Hist/hist_{column_name}.png")

    # Отображение выбора колонки и гистограммы
    selected_column_hist = st.selectbox('Выберите колонку', author.data.columns, key="select_hist")
    plot_histogram(selected_column_hist)

    st.subheader("Боксплоты")

    def plot_boxplot(column_name):
        st.image(f"Images/Boxplot/box_{column_name}.png")

    # Отображение выбора колонки и гистограммы
    selected_column_box = st.selectbox('Выберите колонку', author.data.columns, key="select_box")
    plot_boxplot(selected_column_box)

    st.subheader("Зависимость рейтинга от возраста")
    st.image(f"Images/pie.png")
