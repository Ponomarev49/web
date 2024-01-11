import streamlit as st

import author


# Страница с визуализацией

def page_data_visualization():
    st.title("Визуализации данных")

    # plt.figure(figsize=(12, 8))
    # selected_cols = ['bomb_planted', 'time_left', 'ct_health', 't_health', 'ct_players_alive', 't_players_alive']
    # selected_df = data[selected_cols]
    # sns.heatmap(selected_df.corr(), annot=True, cmap='coolwarm')
    # plt.title('Тепловая карта с корреляцией')
    # plt.savefig(f"Images/heatmap.png")
    st.image(f"Images/heatmap.png")

    st.subheader("Гистограммы")

    def plot_histogram(column_name):
        # plt.figure(figsize=(8, 6))
        # plt.hist(data[column_name], bins=20, color='skyblue', edgecolor='black')
        # plt.title(f'Гистограмма {column_name}')
        # plt.xlabel(column_name)
        # plt.ylabel('Частота')
        # plt.savefig(f"Images/Hist/hist_{column_name}.png")
        st.image(f"Images/Hist/hist_{column_name}.png")

    # Отображение выбора колонки и гистограммы
    selected_column_hist = st.selectbox('Выберите колонку', author.data.columns, key="select_hist")
    plot_histogram(selected_column_hist)

    st.subheader("Боксплоты")

    def plot_boxplot(column_name):
        # plt.figure(figsize=(8, 6))
        # plt.boxplot(data[column_name])
        # plt.title(f'Ящик с усами для {column_name}')
        # plt.xlabel(column_name)
        # plt.ylabel('Значения')
        # plt.savefig(f"Images/Boxplot/box_{column_name}.png")
        st.image(f"Images/Boxplot/box_{column_name}.png")

    # Отображение выбора колонки и гистограммы
    selected_column_box = st.selectbox('Выберите колонку', author.data.columns, key="select_box")
    plot_boxplot(selected_column_box)

    st.subheader("Круговая диаграмма целевого признака")
    # plt.figure(figsize=(8, 8))
    # data['bomb_planted'].value_counts().plot.pie(autopct='%1.1f%%')
    # plt.title('Bomb planted')
    # plt.ylabel('')
    # plt.savefig(f"Images/Pie.png")
    st.image(f"Images/Pie.png")
