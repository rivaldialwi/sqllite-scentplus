import streamlit as st
import app
import laporan

PAGES = {
    "Prediksi Sentimen": app,
    "Laporan Analisis Sentimen": laporan
}

st.sidebar.title('Menu')
selection = st.sidebar.radio("Silahkan Memilih Menu", list(PAGES.keys()))

page = PAGES[selection]
page.run()