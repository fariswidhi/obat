import streamlit as st
import time
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Page Configuration
st.set_page_config(
    page_title="AI Resep Obat Parser",
    page_icon="💊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-family: monospace;
    }
    .json-output {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to get API Key safely
def get_api_key():
    # 1. Cek Streamlit Secrets (Production)
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    
    # 2. Cek Environment Variable (.env Local)
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key != "ganti_dengan_api_key_anda_disini":
        return env_key
    
    return None

# Title and Description
st.title("💊 AI Resep Obat Parser")
st.markdown("Aplikasi ini menggunakan OpenAI untuk mengekstrak informasi terstruktur dari teks resep obat.")

# Sidebar for configuration
with st.sidebar:
    st.header("Konfigurasi")
    
    # API Key Handling
    api_key = get_api_key()
    
    if api_key:
        api_key = api_key.strip()
        st.success("✅ API Key terdeteksi (System/Secrets)")
    else:
        api_key = st.text_input("Masukkan OpenAI API Key Anda:", type="password")
        if not api_key:
            st.warning("⚠️ Harap masukkan API Key untuk melanjutkan.")
        else:
            api_key = api_key.strip()

    st.markdown("---")
    st.markdown("### Tentang")
    st.info("""
    Parser ini mendeteksi:
    - Nama Obat
    - Dosis
    - Jumlah (Quantity)
    - Instruksi Pemakaian
    """)

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Resep")
    default_resep = """R/ Furosemid XV/ ½ - 0 – 0
R/ Captopril 12,5 mg no XC/1-1-1
R/ Nitrokaf 2,5 mg no LX /1-0-1"""
    
    resep_input = st.text_area(
        "Masukkan teks resep di bawah ini:", 
        value=default_resep,
        height=300,
        placeholder="Ketik resep di sini..."
    )
    
    submit_btn = st.button("🔍 Proses Resep", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Ekstraksi")
    
    if submit_btn and resep_input:
        if not api_key:
            st.error("❌ API Key belum diatur!")
        else:
            try:
                # Initialize OpenAI Client
                client = OpenAI(api_key=api_key)
                
                with st.spinner('Sedang memproses resep dengan AI...'):
                    start_time = time.time()
                    
                    response = client.responses.create(
                        model="gpt-4.1",
                        prompt={
                            "id": "pmpt_69437afedb6c8190b6c6560913584516022ee7c443650bd6",
                            "version": "6",
                            "variables": {
                                "resep_input": resep_input
                            }
                        }
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                
                # Parse JSON output safely
                try:
                    json_output = json.loads(response.output_text)
                    st.json(json_output)
                except json.JSONDecodeError:
                    st.warning("⚠️ Output raw text (Gagal parsing JSON):")
                    st.text(response.output_text)
                
                # Show execution metrics
                st.success(f"✅ Selesai dalam {execution_time:.2f} detik")
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
    
    elif not submit_btn:
        st.info("👈 Klik tombol 'Proses Resep' untuk melihat hasil.")
