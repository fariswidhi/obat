import json
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Konfigurasi Model (Sesuaikan dengan yang dipakai saat generate embedding)
# Default sebelumnya: intfloat/multilingual-e5-small
MODEL_NAME = "intfloat/multilingual-e5-small"

def load_data(json_path):
    """Load data obat beserta embedding dari file JSON."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records.")
    
    # Konversi embedding ke numpy array untuk pencarian cepat
    embeddings = []
    valid_data = []
    for item in data:
        if item.get('embedding'):
            embeddings.append(item['embedding'])
            valid_data.append(item)
    
    return valid_data, np.array(embeddings, dtype=np.float32)

def cosine_similarity(query_vec, corpus_vecs):
    """Hitung cosine similarity antara query vector dan corpus vectors."""
    # Normalize vectors (jika belum ternormalisasi, tapi baiknya dinormalisasi)
    norm_query = np.linalg.norm(query_vec)
    norm_corpus = np.linalg.norm(corpus_vecs, axis=1)
    
    if norm_query == 0:
        return np.zeros(len(corpus_vecs))
        
    dot_products = np.dot(corpus_vecs, query_vec)
    similarities = dot_products / (norm_query * norm_corpus)
    return similarities

def parse_resep(resep_text):
    """
    Parse teks resep menjadi list item resep.
    Asumsi format: R/ Nama Obat ...
    """
    # Split berdasarkan "R/" atau baris baru
    # Regex menangkap "R/" di awal baris atau setelah newline, lalu mengambil teks sampai ketemu "R/" berikutnya atau akhir string
    items = re.split(r'(?:^|\n)\s*R/\s*', resep_text.strip())
    # Hapus item kosong (biasanya yang pertama sebelum R/ pertama)
    items = [item.strip() for item in items if item.strip()]
    return items

def search_obat(query_text, model, data, embeddings, top_k=3):
    """Cari obat yang paling mirip dengan query text."""
    # Generate embedding untuk query
    # Penting: Jika menggunakan model e5, query perlu prefix "query: "
    if "e5" in MODEL_NAME.lower():
        query_text = f"query: {query_text}"
    
    query_vec = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    
    # Hitung similarity
    scores = cosine_similarity(query_vec, embeddings)
    
    # Ambil top_k index terbaik
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        item = data[idx]
        results.append({
            'score': float(scores[idx]),
            'nama': item.get('nama'),
            'embedding_text': item.get('embedding_text'), # Teks asli yang di-embed
            'stok': item.get('stok_minimal'), # Info tambahan opsional
            'harga': item.get('hna')
        })
    return results

def main():
    json_file = "master_obat_with_embedding.json"
    
    # Load data
    try:
        data, embeddings = load_data(json_file)
    except FileNotFoundError:
        print(f"Error: File {json_file} tidak ditemukan.")
        return

    # Load Model
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Input Resep (Free Text)
    resep_input = """
    R/ Furosemid XV/ ½ - 0 – 0 
    R/ Captopril 12,5 mg no XC/1-1-1 
    R/ Nitrokaf 2,5 mg no LX /1-0-1 
    R/ spironolakton  25 mg no XXX/1-0-0 
    R/ Digoxin tab no XXX / ½ -0- 1/2 
    R/ Natto tab No X/1-0-1 
    R/ ISDN 5 mg No XXX/ k/p bila nyeri dada SL 
    R/ Neurodex tab no X/1-0-0 
    R/ B1 No XXX / 1-0-0
    """
    
    print("\n--- Menganalisis Resep ---")
    items = parse_resep(resep_input)
    
    for i, item_text in enumerate(items, 1):
        # Bersihkan teks resep untuk pencarian (hapus signa/aturan pakai yang mungkin membingungkan, atau biarkan saja)
        # Strategi: Ambil bagian awal yang kemungkinan besar nama obat
        # Contoh sederhana: ambil baris pertama atau sebelum angka jumlah (No XXX)
        # Tapi embedding semantic cukup pintar menangani noise. Kita coba cari full text dulu.
        
        print(f"\n#{i} Query: {item_text}")
        results = search_obat(item_text, model, data, embeddings, top_k=3)
        
        for res in results:
            print(f"   [{res['score']:.4f}] {res['nama']}")
            # print(f"       (Match source: {res['embedding_text']})")

if __name__ == "__main__":
    main()
