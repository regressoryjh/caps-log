# utils/auth.py
import os
import sqlite3

DB_PATH = os.path.join("data", "users.db")

def auth(username, password):
    """Autentikasi user sederhana"""
    if not username.strip() or not password.strip():
        return "empty"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username.strip(), password.strip())
        )
        return "success" if c.fetchone() else "invalid"
    except sqlite3.OperationalError:
        return "invalid"
    finally:
        conn.close()