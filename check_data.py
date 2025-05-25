# check_data.py
import sqlite3
import pandas as pd

def check_data():
    conn = sqlite3.connect("data/stock_data.db")
    df = pd.read_sql("SELECT * FROM stock_transactions LIMIT 5", conn)
    conn.close()
    
    print("Tipe data kolom:")
    print(df.dtypes)
    
    print("\nContoh data tanggal:")
    print(df['tanggal'].head())

if __name__ == "__main__":
    check_data()