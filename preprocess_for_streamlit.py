import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

print("Memulai proses pra-pemrosesan untuk aplikasi Streamlit...")

# =====================================================================
# 1. Persiapan Direktori dan Data Awal
# =====================================================================

# Pastikan direktori 'data' ada
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Direktori '{output_dir}' telah dibuat.")

# Muat dataset yang sudah dibersihkan dari notebook Anda
# Pastikan file ini ada di path yang benar
try:
    df_cleaned = pd.read_parquet("data/app_reviews_cleaned.parquet")
    print("Dataset yang sudah dibersihkan berhasil dimuat.")
except FileNotFoundError:
    print(
        "Error: File 'app_reviews_cleaned.parquet' tidak ditemukan. Pastikan Anda sudah menjalankannya dari notebook analisis utama."
    )
    exit()

# Buang ulasan netral untuk pemodelan
df_model_data = df_cleaned[df_cleaned["sentiment"] != "Netral"].copy()
print(f"Total ulasan untuk diproses: {len(df_model_data)}")
print("-" * 50)


# =====================================================================
# 2. Membuat File Feature Importance untuk Setiap Aplikasi
# =====================================================================

print("Memulai pembuatan file 'feature importance' untuk setiap aplikasi...")
app_names = df_model_data["app_name"].unique()

for app_name in app_names:
    print(f"--> Memproses: {app_name.capitalize()}")

    # Filter data untuk aplikasi saat ini
    app_df = df_model_data[df_model_data["app_name"] == app_name]

    X_app = app_df["review_cleaned"].astype(str)
    y_app = app_df["sentiment"]

    # Latih model baru khusus untuk aplikasi ini
    app_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    app_model.fit(X_app, y_app)

    # Ekstrak feature importance
    vectorizer = app_model.named_steps["tfidf"]
    classifier = app_model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame(
        {"word": feature_names, "coefficient": coefficients}
    ).sort_values(by="coefficient", ascending=False)

    # Ambil 15 kata kunci teratas untuk masing-masing sentimen
    top_positive_keywords = coef_df.head(15).copy()
    top_positive_keywords["sentiment"] = "Positif"

    top_negative_keywords = (
        coef_df.tail(15).sort_values(by="coefficient", ascending=True).copy()
    )
    top_negative_keywords["sentiment"] = "Negatif"

    # Gabungkan menjadi satu DataFrame
    feature_importance_df = pd.concat(
        [top_positive_keywords, top_negative_keywords], ignore_index=True
    )

    # Simpan ke file parquet
    file_path = os.path.join(output_dir, f"feature_importance_{app_name}.parquet")
    feature_importance_df.to_parquet(file_path)
    print(f"    File '{file_path}' berhasil disimpan.")

print("-" * 50)
print("Semua file 'feature importance' telah berhasil dibuat.")


# =====================================================================
# 3. (Opsional) Jika Anda belum membuat file aspect_plot_df.parquet
# =====================================================================
# Kode ini diambil dari notebook analisis aspek Anda

print("\nMembuat file untuk plot analisis aspek...")
try:
    aspect_keywords = {
        "Aplikasi": [
            "aplikasi",
            "apk",
            "app",
            "update",
            "eror",
            "error",
            "lambat",
            "lemot",
            "boikot",
            "peta",
            "lokasi",
            "gps",
            "susah",
            "mudah",
            "uninstall",
            "bobrok",
            "notifikasi",
            "iklan",
            "sistem",
        ],
        "Harga": [
            "harga",
            "terjangkau",
            "tarif",
            "mahal",
            "murah",
            "promo",
            "diskon",
            "biaya",
            "ongkir",
            "poin",
            "poinnya",
        ],
        "Pengemudi": [
            "pengemudi",
            "driver",
            "drivernya",
            "ramah",
            "sopan",
            "kasar",
            "ugal",
            "baik",
            "batal",
            "cancel",
            "ngebut",
        ],
        "Layanan": [
            "layanan",
            "payah",
            "grab",
            "pertahankan",
            "cepat",
            "lama",
            "order",
            "jemput",
            "antar",
            "makanan",
            "gojek",
            "grab",
            "maxim",
            "indrive",
            "pesan",
            "pesanan",
            "gofood",
            "go food",
            "pelayanan",
            "cepat",
            "kasar",
            "pendukung",
            "sampah",
            "parah",
            "buruk",
            "terbaik",
            "best",
            "keren",
        ],
        "Customer Service": [
            "cs",
            "customer",
            "service",
            "bantuan",
            "pusat bantuan",
            "komplain",
            "laporan",
            "pengaduan",
            "respon",
            "solusi",
            "ganti rugi",
            "lambar",
            "balas",
        ],
    }

    def tag_aspect(review):
        found_aspects = []
        words = str(review).split()
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in words for keyword in keywords):
                found_aspects.append(aspect)
        if not found_aspects:
            return ["Umum"]
        return found_aspects

    df_aspect = df_model_data.copy()
    df_aspect["aspects"] = df_aspect["review_cleaned"].apply(tag_aspect)
    df_exploded = df_aspect.explode("aspects")
    aspect_summary = (
        df_exploded.groupby(["app_name", "aspects", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    aspect_percentage = aspect_summary.div(
        aspect_summary.sum(axis=1), axis=0
    ).reset_index()
    aspect_plot_df = aspect_percentage.melt(
        id_vars=["app_name", "aspects"],
        value_vars=["Positif", "Negatif"],
        var_name="sentiment",
        value_name="percentage",
    )

    aspect_file_path = os.path.join(output_dir, "aspect_plot_df.parquet")
    aspect_plot_df.to_parquet(aspect_file_path)
    print(f"File '{aspect_file_path}' berhasil disimpan.")

except Exception as e:
    print(f"Gagal membuat file analisis aspek: {e}")

print("\nPra-pemrosesan data untuk Streamlit selesai!")
