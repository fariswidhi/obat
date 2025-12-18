import os
from dotenv import load_dotenv
from openai import OpenAI

import sys
import time

# Load environment variables from .env file
load_dotenv()

def get_user_input():
    # Cek apakah ada input dari argumen command line
    if len(sys.argv) > 1:
        # Menggabungkan semua argumen menjadi satu string
        return " ".join(sys.argv[1:])
    
    # Jika tidak ada argumen, minta input interaktif
    print("\nSilakan masukkan resep obat (akhiri dengan baris kosong/Enter 2x):")
    lines = []
    while True:
        try:
            line = input()
            if not line and lines:  # Berhenti jika baris kosong dan sudah ada isi
                break
            if line:
                lines.append(line)
        except EOFError:
            break
            
    if not lines:
        print("Input kosong. Menggunakan contoh default...")
        return """
R/ Furosemid XV/ ½ - 0 – 0
R/ Captopril 12,5 mg no XC/1-1-1
R/ Nitrokaf 2,5 mg no LX /1-0-1
"""
    return "\n".join(lines)

# Ambil API Key dari environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "ganti_dengan_api_key_anda_disini":
    print("Error: Harap set OPENAI_API_KEY di file .env terlebih dahulu.")
    exit(1)

client = OpenAI(api_key=api_key)

try:
    resep_text = get_user_input()
    print(f"\nMemproses resep:\n{resep_text}\n")
    print("Mengirim request ke OpenAI...")
    
    start_time = time.time()
    response = client.responses.create(
        model="gpt-4.1",              # pastikan sama dengan prompt config
        prompt={
            "id": "pmpt_69437afedb6c8190b6c6560913584516022ee7c443650bd6",
            "version": "1"
        },
        input=resep_text
    )
    end_time = time.time()
    execution_time = end_time - start_time

    # Note: response structure might vary depending on the library version/feature
    # User's code expects response.output_text
    print("\n--- Output ---")
    print(response.output_text)
    print(f"\nWaktu response: {execution_time:.2f} detik")

except Exception as e:
    print(f"\nTerjadi kesalahan: {e}")
