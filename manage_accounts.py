import streamlit as st
import sqlite3

# Fungsi untuk membuat koneksi ke database SQLite
def create_connection():
    conn = sqlite3.connect('db_scentplus.db')
    return conn

# Fungsi untuk menambahkan pengguna baru ke dalam database
def create_user(email, username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user (email, username, password) VALUES (?, ?, ?)", (email, username, password))
    conn.commit()
    conn.close()

# Fungsi untuk membaca semua pengguna dari database
def read_users():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Fungsi untuk memperbarui data pengguna berdasarkan ID
def update_user(user_id, email, username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE user SET email = ?, username = ?, password = ? WHERE id = ?", (email, username, password, user_id))
    conn.commit()
    conn.close()

# Fungsi untuk menghapus pengguna berdasarkan ID
def delete_user(user_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

# Fungsi utama untuk mengelola akun pengguna
def manage_accounts():
    st.title("Mengelola Akun")

    # Menampilkan form untuk menambah pengguna baru
    st.subheader("Tambah Pengguna Baru")
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Tambah Pengguna"):
        create_user(email, username, password)
        st.success("Pengguna berhasil ditambahkan")

    # Menampilkan daftar pengguna
    st.subheader("Daftar Pengguna")
    users = read_users()
    
    # Membuat header tabel tanpa indeks
    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 2])
    col1.write("ID")
    col2.write("Email")
    col3.write("Username")
    col4.write("Password")
    col5.write("Action")

    for user in users:
        col1, col2, col3, col4, col5, col6 = st.columns([2, 3, 2, 2, 2, 2])
        col1.write(user[0])
        col2.write(user[1])
        col3.write(user[2])
        col4.write(user[3])
        with col5:
            if st.button("Update", key=f"update_{user[0]}"):
                update_modal(user)
        with col6:
            if st.button("Delete", key=f"delete_{user[0]}"):
                delete_modal(user)

# Fungsi untuk menampilkan modal update pengguna menggunakan experimental dialog
@st.experimental_dialog("Update Pengguna")
def update_modal(user):
    user_id, email, username, password = user
    email = st.text_input("Email", value=email)
    username = st.text_input("Username", value=username)
    password = st.text_input("Password", value=password, type="password")
    if st.button("Simpan Perubahan"):
        update_user(user_id, email, username, password)
        st.success(f"Pengguna {username} berhasil diperbarui")
        st.experimental_rerun()

# Fungsi untuk menampilkan modal konfirmasi hapus pengguna menggunakan experimental dialog
@st.experimental_dialog("Konfirmasi Hapus Pengguna")
def delete_modal(user):
    user_id, email, username, password = user
    st.warning(f"Apakah Anda yakin ingin menghapus pengguna {username}?")
    if st.button("Ya, Hapus"):
        delete_user(user_id)
        st.success(f"Pengguna {username} berhasil dihapus")
        st.experimental_rerun()

# Fungsi utama untuk menjalankan aplikasi
def run():
    manage_accounts()

# Menjalankan fungsi run() untuk memulai aplikasi
if __name__ == "__main__":
    run()