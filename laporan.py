import pandas as pd
import streamlit as st
import joblib
import nltk
import matplotlib.pyplot as plt
import plotly.express as px
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
from wordcloud import WordCloud
from collections import Counter

# Fungsi untuk mengunduh resource NLTK secara senyap
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Panggil fungsi unduh NLTK resource di awal
download_nltk_resources()

# Membaca model yang sudah dilatih dan TF-IDF Vectorizer
logreg_model = joblib.load("model100.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Daftar kata yang akan diganti atau dihapus
replace_words = {
    "yang": "",
    "nya": "",
    "kok": "",
    "sih": "",
    "ga": "tidak",
    "gak": "tidak",
    "tidakk": "tidak",
    "udah": "sudah",
    "ka": "kak",
    "kakk": "kak",
    "cewe": "cewek",
    "cew": "cewek",
    "cowo": "cowok",
    "cow": "cowok",
}

# Fungsi untuk mengganti atau menghapus kata-kata tertentu dalam teks
def replace_and_remove_words(text):
    words = text.split()
    replaced_words = [replace_words.get(word, word) for word in words]
    return " ".join(replaced_words)

# Fungsi untuk membuat word cloud
def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

# Fungsi untuk membersihkan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()  # Case folding
    words = word_tokenize(text)  # Tokenizing
    cleaned_words = [word for word in words if word not in stop_words]  # Stopword removal
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]  # Stemming
    replaced_text = replace_and_remove_words(" ".join(stemmed_words))  # Replace and remove words
    return replaced_text

# Fungsi untuk melakukan klasifikasi teks
def classify_text(input_text):
    # Membersihkan teks input
    cleaned_text = clean_text(input_text)
    # Mengubah teks input menjadi vektor fitur menggunakan TF-IDF
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    # Melakukan prediksi menggunakan model
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Fungsi untuk mengonversi DataFrame ke Excel
@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()  # Gunakan writer.close() untuk menyimpan file
    processed_data = output.getvalue()
    return processed_data

# Fungsi untuk menjalankan aplikasi
def run():
    st.title("Aplikasi Analisis Sentimen Scentplus")

    # Tab untuk dua fitur: Prediksi Sentimen dan Analisis Sentimen
    tab1, tab2 = st.tabs(["Prediksi Sentimen", "Analisis Sentimen"])

    with tab1:
        st.header("Unggah file untuk Prediksi Sentimen")
        uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"], key="file_uploader")

        if uploaded_file is not None:
            # Baca file Excel
            df = pd.read_excel(uploaded_file)
            
            # Periksa apakah kolom 'Text' ada di file yang diunggah
            if 'Text' in df.columns:
                # Inisialisasi TF-IDF Vectorizer dan fit_transform pada data teks
                X = df['Text'].apply(clean_text)
                X_tfidf = tfidf_vectorizer.transform(X)
                
                # Lakukan prediksi
                df['Human'] = logreg_model.predict(X_tfidf)
                
                # Tampilkan prediksi
                st.write(df)
                
                # Buat tombol unduh
                st.download_button(
                    label="Unduh file dengan prediksi",
                    data=convert_df_to_excel(df),
                    file_name="prediksi_sentimen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("File harus memiliki kolom 'Text'.")

    with tab2:
        st.header("Unggah file untuk Grafik dan Word Cloud Sentimen")
        uploaded_excel = st.file_uploader("Unggah file Excel", type=["xlsx"], key="file_uploader_analysis")

        if uploaded_excel is not None:
            # Baca file Excel
            df_excel = pd.read_excel(uploaded_excel)
            
            # Periksa apakah kolom 'Human' ada di file yang diunggah
            if 'Human' in df_excel.columns:
                # Hitung kemunculan setiap sentimen
                sentiment_counts = df_excel['Human'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                # Buat diagram batang menggunakan Plotly

                 # Buat diagram batang menggunakan Plotly
                color_discrete_map = {
                    'Negatif': 'red',
                    'Netral': 'gray',
                    'Positif': 'green'
                }

                fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                             labels={'Sentiment': 'Sentimen', 'Count': 'Jumlah'},
                             title='Distribusi Sentimen',
                             text='Count',
                             color_discrete_map=color_discrete_map)
                
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                
                st.plotly_chart(fig)
                
                # Hasilkan kata untuk setiap sentimen
                sentiments = df_excel['Human'].unique()
                for sentiment in sentiments:
                    sentiment_text = " ".join(df_excel[df_excel['Human'] == sentiment]['Text'])
                    sentiment_text_cleaned = clean_text(sentiment_text)
                    create_word_cloud(sentiment_text_cleaned, f'Word Cloud untuk Sentimen {sentiment}')
            else:
                st.error("File harus memiliki kolom 'Human'.")

if __name__ == "__main__":
    run()