import streamlit as st
import sqlite3

def create_connection():
    conn = sqlite3.connect('db_scentplus.db')
    return conn

def get_user_data(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM user WHERE username = ? AND password = ?"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    conn.close()
    return result

def update_user_data(user_id, new_username, new_email, new_password):
    conn = create_connection()
    cursor = conn.cursor()
    query = "UPDATE user SET username = ?, email = ?, password = ? WHERE id = ?"
    cursor.execute(query, (new_username, new_email, new_password, user_id))
    conn.commit()
    conn.close()

def show_dialog(user_data):
    user_id, email, username, password = user_data
    st.session_state['edit_username'] = username
    st.session_state['edit_email'] = email
    st.session_state['edit_password'] = password

    @st.experimental_dialog("Edit User Data")
    def dialog():
        st.write("Edit your details below:")
        new_email = st.text_input("New Email", value=st.session_state['edit_email'])
        new_username = st.text_input("New Username", value=st.session_state['edit_username'])
        new_password = st.text_input("New Password", value=st.session_state['edit_password'], type="password")
        
        if st.button("Save"):
            update_user_data(user_id, new_username, new_email, new_password)
            st.success("Data updated successfully")
            st.experimental_rerun()

    dialog()

def run():
    st.title("Edit User Data")

    # Tambahkan tombol Logout
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.experimental_rerun()

    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        if 'username' not in st.session_state:
            st.error("No user data found. Please log in again.")
            return

        username = st.text_input("Username", value=st.session_state['username'], disabled=True)
        password = st.text_input("Password", value=st.session_state['password'], type="password")

        if st.button("Load Data"):
            user_data = get_user_data(username, st.session_state['password'])
            if user_data:
                show_dialog(user_data)
            else:
                st.error("Failed to load user data.")
    else:
        st.error("You need to log in first")

if __name__ == "__main__":
    run()