# check_database.py
import sqlite3
import pandas as pd

def check_tables():
    """Cek tabel yang ada dalam database"""
    try:
        # Connect ke database
        conn = sqlite3.connect('data/stock_data.db')
        cursor = conn.cursor()
        
        # Query untuk melihat semua tabel
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("Database tidak memiliki tabel apapun!")
            return
            
        print("Tabel yang tersedia:")
        for table in tables:
            table_name = table[0]
            print(f"\n• Tabel: {table_name}")
            
            # Tampilkan 5 baris pertama dari setiap tabel
            try:
                df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
                print("\nContoh data:")
                print(df)
                
                # Tampilkan info kolom
                print("\nStruktur kolom:")
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"{col[1]} ({col[2]})")
                    
            except Exception as e:
                print(f"Gagal membaca tabel {table_name}: {str(e)}")
                
    except sqlite3.Error as e:
        print(f"Error koneksi database: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("Memulai pemeriksaan database...")
    check_tables()
    print("\nPemeriksaan selesai!")