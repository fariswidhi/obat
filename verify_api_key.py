import os
from dotenv import load_dotenv
from openai import OpenAI

# Load env without override first to see what's physically there
with open('.env', 'r') as f:
    content = f.read().strip()
    print(f"Isi file .env mentah (50 karakter pertama): {content[:50]}...")

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

print(f"\nAPI Key terbaca: {'Ada' if api_key else 'Tidak ada'}")
if api_key:
    print(f"Panjang Key: {len(api_key)}")
    print(f"Prefix: {api_key[:10]}...")
    print(f"Suffix: ...{api_key[-5:]}")
    
    # Clean it just in case
    api_key = api_key.strip().strip("'").strip('"')
    print(f"Key setelah dibersihkan: {api_key[:10]}...{api_key[-5:]}")

    client = OpenAI(api_key=api_key)
    try:
        print("\nMencoba koneksi ke OpenAI (list models)...")
        client.models.list()
        print("✅ Koneksi BERHASIL! Key valid.")
    except Exception as e:
        print(f"❌ Koneksi GAGAL: {e}")
else:
    print("❌ API Key kosong.")
