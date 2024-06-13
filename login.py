import streamlit as st
import sqlite3

def create_connection():
    conn = sqlite3.connect('db_scentplus.db')
    return conn

def validate_login(username, password, table):
    conn = create_connection()
    cursor = conn.cursor()
    query = f"SELECT * FROM {table} WHERE username = ? AND password = ?"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    conn.close()
    return result

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if validate_login(username, password, 'admin'):
            st.session_state['logged_in'] = True
            st.session_state['role'] = 'admin'
            st.experimental_rerun()
        elif validate_login(username, password, 'user'):
            st.session_state['logged_in'] = True
            st.session_state['role'] = 'user'
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")