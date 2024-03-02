import streamlit as st
import pandas as pd
import pickle


def classification() :
    # # Загрузка датасета
    # data = pd.read_csv('Data/new_laptops.csv')
    # if 'Unnamed: 0' in data.columns:
    #     data = data.drop(['Unnamed: 0'], axis=1)
    # y_train = data['overall']
    # X_train = data.drop(columns=['overall'])

    brands = {'HP': 0, 'Apple': 1, 'Lenovo': 2, 'ASUS': 3, 'DELL': 4, 'Acer': 5, 'SAMSUNG': 6, 'MSI': 7, 'Infinix': 8,
              'Ultimus': 9, 'CHUWI': 10, 'WINGS': 11, 'ZEBRONICS': 12, 'Primebook': 13, 'GIGABYTE': 14, 'realme': 15,
              'MICROSOFT': 16, 'LG': 17}
    st.header("Choose Brand")
    option = st.selectbox(
        'Which Club do you like best?',
        brands.keys())
    'You selected: ', option

    processor = {'Core i3': 0, 'M1': 1, 'Core i7': 2, 'Core i5': 3, 'Ryzen 5 Hexa Core': 4, 'Celeron Dual Core': 5,
                 'Ryzen 7 Octa Core': 6, 'Ryzen 5 Quad Core': 7, 'Ryzen 3 Dual Core': 8, 'Ryzen 3 Quad Core': 9,
                 'M2': 10, 'Celeron Quad Core': 11, 'Athlon Dual Core': 12, 'MediaTek Kompanio 1200': 13,
                 'Ryzen 9 Octa Core': 14, 'MediaTek MT8788': 15, 'Ryzen Z1 HexaCore': 16, 'MediaTek Kompanio 500': 17,
                 'Core i9': 18, 'MediaTek Kompanio 520': 19, 'Ryzen Z1 Octa Core': 20, 'Pentium Silver': 21,
                 'Ryzen 5': 22, 'M1 Max': 23, 'M2 Max': 24, 'M3 Pro': 25, 'M1 Pro': 26, 'Ryzen 7 Quad Core': 27,
                 'Ryzen 5 Dual Core': 28, 'Ryzen 9 16 Core': 29}
    st.header("Choose Processor")
    option1 = st.selectbox(
        'Which Club do you like best?',
        processor.keys())
    'You selected: ', option1

    op_system = {'Windows 11 Home': 0, 'Mac OS Big Sur': 1, 'DOS': 2, 'Mac OS Monterey': 3, 'Chrome': 4, 'Windows 10': 5,
                 'Windows 10 Home': 6, 'Prime OS': 7, 'Windows 11 Pro': 8, 'Ubuntu': 9, 'Windows 10 Pro': 10,
                 'macOS Ventura': 11, 'macOS Sonoma': 12, 'Mac OS Mojave': 13}
    st.header("Choose Operating System")
    option2 = st.selectbox(
        'Which Club do you like best?',
        op_system.keys())
    'You selected: ', option2

    storage = {0.5: 0, 0.25: 1, 1.0: 2, 2.0: 3, 4.0: 4, 0.125: 5, 0.0625: 6, 3.0: 7, 6.0: 8}
    st.header("Choose Storage(GB)")
    option3 = st.selectbox(
        'Which Club do you like best?',
        storage.keys())
    'You selected: ', option3

    ram = {8: 0, 16: 1, 4: 2, 12: 3, 32: 4, 64: 5, 18: 6}
    st.header("Choose RAM")
    option4 = st.selectbox(
        'Which Club do you like best?',
        ram.keys())
    'You selected: ', option4

    size = {39.62: 0, 33.78: 1, 35.56: 2, 96.52: 3, 100.63: 4, 40.89: 5, 35.81: 6, 40.64: 7, 39.01: 8, 34.54: 9, 34.29: 10,
            38.1: 11, 38.0: 12, 29.46: 13, 17.78: 14, 43.94: 15, 26.67: 16, 34.04: 17, 33.02: 18, 35.0: 19, 41.15: 20,
            90.32: 21, 30.48: 22, 38.86: 23, 36.07: 24, 31.5: 25}
    st.header("Choose Screen Size")
    option5 = st.selectbox(
        'Which Club do you like best?',
        size.keys())
    'You selected: ', option5

    touch = {'True': 1, 'False': 0}
    st.header("Choose Touch Screen")
    option6 = st.selectbox(
        'Which Club do you like best?',
        touch.keys())
    'You selected: ', option6

    data = pd.DataFrame({'Brand': [brands[option]],
                         'Processor': [processor[option1]],
                         'Operating System': [op_system[option2]],
                         'Storage': [storage[option3]],
                         'RAM': [ram[option4]],
                         'Screen Size': [size[option5]],
                         'Touch Screen': [touch[option6]]})

    button_clicked = st.button("Предсказать")

    if button_clicked:
        with open('Models/rbf.pkl', 'rb') as file:
            model = pickle.load(file)

        st.header("SVC RBF:")
        nn_pred = round(model.predict(data)[0][0])

        st.write(f"{nn_pred}")
