import streamlit as st
from utils.auth import auth

def login_page():
    """Render the login page."""
    st.title("📦 Logistix Inventory System")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("static/login_illustration.png", width=400)
    
    with col2:
        st.header("Login")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            result = auth(username, password)
            
            if result == "empty":
                st.warning("Please enter both username and password.")
            elif result == "invalid":
                st.error("Invalid credentials. Please contact the system administrator.")
            elif result == "success":
                st.session_state.login_status = "logged_in"
                st.session_state.username = username
                st.rerun()