# debug_init.py
from data_loader import initialize_data

print("Memulai inisialisasi data...")
if initialize_data():
    print("Inisialisasi berhasil!")
else:
    print("Inisialisasi gagal!")