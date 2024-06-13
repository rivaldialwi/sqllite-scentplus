import streamlit as st
import app
import laporan
import login
import manage_accounts
import edit

ADMIN_PAGES = {
    "Prediksi Sentimen": app,
    "Laporan Analisis Sentimen": laporan,
    "Mengelola Akun": manage_accounts,
}

USER_PAGES = {
    "Prediksi Sentimen": app,
    "Laporan Analisis Sentimen": laporan,
    "Edit Akun": edit,
}

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'role' not in st.session_state:
    st.session_state['role'] = None

if not st.session_state['logged_in']:
    login.login()
else:
    role = st.session_state['role']
    
    if role == 'admin':
        PAGES = ADMIN_PAGES
    elif role == 'user':
        PAGES = USER_PAGES
    
    st.sidebar.title('Menu')
    selection = st.sidebar.radio("Silahkan Memilih Menu", list(PAGES.keys()))
    page = PAGES[selection]
    page.run()