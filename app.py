import pandas as pd
import streamlit as st
import joblib
import nltk
import sqlite3
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Membaca model yang sudah dilatih
logreg_model = joblib.load("model100.pkl")

# Memuat TF-IDF Vectorizer yang sudah di-fit
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Fungsi untuk membersihkan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()  # Case folding
    words = word_tokenize(text)  # Tokenizing
    cleaned_words = [word for word in words if word not in stop_words]  # Stopword removal
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]  # Stemming
    return " ".join(stemmed_words)

# Fungsi untuk melakukan klasifikasi teks
def classify_text(input_text):
    # Membersihkan teks input
    cleaned_text = clean_text(input_text)
    # Mengubah teks input menjadi vektor fitur menggunakan TF-IDF
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    # Melakukan prediksi menggunakan model
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Fungsi untuk memasukkan data ke database
def insert_to_db(text, sentiment):
    conn = sqlite3.connect('db_scentplus.db')
    cursor = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''INSERT INTO riwayat (text, sentiment, date) VALUES (?, ?, ?)''', (text, sentiment, date))
    conn.commit()
    conn.close()

# Fungsi untuk mengambil data dari database
def fetch_data():
    conn = sqlite3.connect('db_scentplus.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT rowid AS id, text, sentiment, date FROM riwayat''')
    rows = cursor.fetchall()
    conn.close()
    return rows

# Fungsi untuk menjalankan aplikasi
def run():
    st.title("Aplikasi Analisis Sentimen scentplussss")
    input_text = st.text_input("Masukkan kalimat untuk analisis sentimen:")
    
    if 'data' not in st.session_state:
        st.session_state['data'] = fetch_data()

    if st.button("Analisis"):
        if input_text.strip() == "":
            st.error("Tolong masukkan sentimen terlebih dahulu.")
        else:
            result = classify_text(input_text)
            st.write("Hasil Analisis Sentimen:", result)
            insert_to_db(input_text, result)
            st.session_state['data'] = fetch_data()
    
    # Menampilkan data dari session state sebagai tabel
    data = st.session_state['data']
    if data:
        df = pd.DataFrame(data, columns=['id', 'text', 'sentiment', 'date'])
        
        # Tambahkan CSS untuk menyesuaikan header
        hide_dataframe_row_index = """
            <style>
            thead th {
                text-align: center;
            }
            </style>
        """
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        
        # Tampilkan tabel tanpa indeks
        st.write(df.to_html(index=False, escape=False), unsafe_allow_html=True)
    else:
        st.write("Tidak ada data yang tersedia.")

if __name__ == "__main__":
    run()