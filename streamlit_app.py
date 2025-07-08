import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Import SimpleImputer to handle missing values
from sklearn.impute import SimpleImputer

# --- Konfigurasi halaman ---
st.set_page_config(page_title="People Analytics Dashboard", layout="wide")

# --- Judul Utama ---
st.title("👥 People Analytics Dashboard")

# --- Generate Dummy Data (812 row) ---
np.random.seed(42)

jumlah_data = {
    "Sangat Baik": 609,
    "Baik": 122,
    "Cukup": 41,
    "Kurang": 24,
    "Sangat Kurang": 16
}

# List tetap
unit_list_real = [
    "Direktorat Anggaran Bidang Pembangunan Manusia dan Kebudayaan",
    "Direktorat Anggaran Bidang Perekonomian dan Kemaritiman",
    "Direktorat Anggaran Bidang Politik, Hukum, Pertahanan dan Keamanan, dan Bagian Anggaran Bendahara Umum Negara",
    "Direktorat Harmonisasi Peraturan Penganggaran",
    "Direktorat Penerimaan Negara Bukan Pajak Kementerian/Lembaga",
    "Direktorat Penerimaan Negara Bukan Pajak Sumber Daya Alam dan Kekayaan Negara Dipisahkan",
    "Direktorat Penyusunan Anggaran Pendapatan dan Belanja Negara",
    "Direktorat Potensi dan Pengawasan Penerimaan Negara Bukan Pajak",
    "Direktorat Sistem Penganggaran",
    "Sekretariat Direktorat Jenderal",
    "Tenaga Pengkaji Bidang Penerimaan Negara Bukan Pajak"
]
golongan_list_real = [
    "I/a", "I/b", "I/c", "I/d",
    "II/a", "II/b", "II/c", "II/d",
    "III/a", "III/b", "III/c", "III/d",
    "IV/a", "IV/b", "IV/c", "IV/d", "IV/e"
]

# Tambahkan 'Semua' di awal
unit_list = ["Semua"] + unit_list_real
golongan_list = ["Semua"] + golongan_list_real

tingkat_pendidikan_list = ["SMA", "D3", "S1", "S2", "S3"]
peringkat_list = ["Rendah", "Sedang", "Tinggi"]
ordered_labels = ["Sangat Kurang", "Kurang", "Cukup", "Baik", "Sangat Baik"]
color_map = {
    "Sangat Kurang": "#d62728",
    "Kurang": "#ff7f0e",
    "Cukup": "#bcbd22",
    "Baik": "#1f77b4",
    "Sangat Baik": "#2ca02c"
}

# Fungsi bantu: konversi golongan ke level angka
def golongan_to_level(gol):
    urutan = {
        "I/a": 1, "I/b": 2, "I/c": 3, "I/d": 4,
        "II/a": 5, "II/b": 6, "II/c": 7, "II/d": 8,
        "III/a": 9, "III/b": 10, "III/c": 11, "III/d": 12,
        "IV/a": 13, "IV/b": 14, "IV/c": 15, "IV/d": 16, "IV/e": 17
    }
    return urutan.get(gol, 1)

# Data generator
data = []
for label, count in jumlah_data.items():
    for _ in range(count):
        usia = np.random.randint(22, 60)
        masa_kerja = np.random.randint(1, 20)  # usia 22 = kerja max 1 tahun
        golongan = np.random.choice(golongan_list_real)
        golongan_level = golongan_to_level(golongan)

        # Rumus peringkat (maks 27)
        peringkat_nilai = int(
            0.5 * golongan_level +
            0.4 * (masa_kerja / 35 * 10) +
            0.1 * (usia / 60 * 10)
        )
        peringkat_nilai = max(1, min(27, round(peringkat_nilai)))

        row = {
            "Unit": np.random.choice(unit_list_real),
            "Golongan": golongan,
            "Total Jam Pelatihan": np.random.randint(10, 200),
            "Usia": usia,
            "Tingkat Pendidikan": np.random.choice(tingkat_pendidikan_list),
            "Peringkat": peringkat_nilai,
            "Masa Kerja": masa_kerja,
            "Prediksi Kinerja": label
        }
        data.append(row)

df = pd.DataFrame(data)
df["Prediksi Kinerja"] = pd.Categorical(df["Prediksi Kinerja"], categories=ordered_labels, ordered=True)

# --- Tab navigasi ---
tab1, tab2, tab3 = st.tabs(["📊 Prediksi Kinerja", "🔍 Klasterisasi Pegawai", "ℹ️ Rekomendasi Pelatihan"])

# ==================================
# 🔹 Tab 1: Prediksi Kinerja Pegawai
# ==================================
with tab1:
    st.header("📊 Prediksi Kinerja Pegawai")

    col1, col2 = st.columns(2)
    with col1:
        selected_unit = st.selectbox("Pilih Unit", unit_list)
    with col2:
        selected_golongan = st.selectbox("Pilih Golongan", golongan_list)

    # Terapkan filter
    filtered_df = df.copy()
    if selected_unit != "Semua":
        filtered_df = filtered_df[filtered_df["Unit"] == selected_unit]
    if selected_golongan != "Semua":
        filtered_df = filtered_df[filtered_df["Golongan"] == selected_golongan]

    # Tabel data
    st.markdown("### 📄 Tabel Hasil Prediksi")
    st.dataframe(filtered_df, use_container_width=True)

    # Distribusi bar chart dengan warna dan urutan
    st.markdown("### 📊 Distribusi Prediksi Kinerja")
    distribusi_filtered = filtered_df["Prediksi Kinerja"].value_counts().reindex(ordered_labels, fill_value=0).reset_index()
    distribusi_filtered.columns = ["Prediksi Kinerja", "Jumlah"]

    fig = px.bar(
        distribusi_filtered,
        x="Prediksi Kinerja",
        y="Jumlah",
        color="Prediksi Kinerja",
        color_discrete_map=color_map,
        category_orders={"Prediksi Kinerja": ordered_labels},
        title="Distribusi Kinerja Pegawai"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Unduh data
    with st.expander("⬇️ Download Data"):
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "data_kinerja.csv", "text/csv")

# ==================================
# 🔹 Tab 2: Visualisasi Data
# ==================================
with tab2:
    st.header("🔍 Klasterisasi Pegawai")

    st.markdown("""
    Tab ini menampilkan hasil segmentasi pegawai berdasarkan 6 atribut utama: **Usia**, **Golongan**, **Jabatan**, 
    **Tingkat Pendidikan**, **Masa Kerja**, dan **Total Jam Pelatihan**. 
    Proses klasterisasi menggunakan algoritma *K-Means* untuk mengelompokkan pegawai dengan karakteristik serupa.
    """)

    # Tambahkan kolom ID dan Jabatan (simulasi)
    df["ID Pegawai"] = ["PGW{:04d}".format(i) for i in range(1, len(df) + 1)]
    df["Jabatan"] = np.random.choice(["Staf", "Analis", "Koordinator", "Kasubdit", "Kabag", "Pengkaji"], size=len(df))

    # Fitur klasterisasi
    fitur_klaster = df[["Usia", "Masa Kerja", "Total Jam Pelatihan", "Peringkat"]].copy()

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    scaler = MinMaxScaler()
    fitur_scaled = scaler.fit_transform(fitur_klaster)

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df["Cluster"] = kmeans.fit_predict(fitur_scaled)

    # PCA untuk visualisasi 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(fitur_scaled)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # --- Tabel Hasil Klasterisasi ---
    st.subheader("📄 Tabel Hasil Klasterisasi Pegawai")
    col1, col2 = st.columns(2)
    with col1:
        selected_unit = st.selectbox("Filter Unit Kerja", ["Semua"] + unit_list_real, key="unit_filter_klaster")
    with col2:
        selected_golongan = st.selectbox("Filter Golongan", ["Semua"] + golongan_list_real, key="gol_filter_klaster")

    filtered_df = df.copy()
    if selected_unit != "Semua":
        filtered_df = filtered_df[filtered_df["Unit"] == selected_unit]
    if selected_golongan != "Semua":
        filtered_df = filtered_df[filtered_df["Golongan"] == selected_golongan]

    st.dataframe(filtered_df[[
        "ID Pegawai", "Usia", "Golongan", "Jabatan", "Tingkat Pendidikan", 
        "Masa Kerja", "Total Jam Pelatihan", "Cluster"
    ]], use_container_width=True)

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # --- Normalisasi dan klasterisasi dengan 4 cluster ---
    fitur_klaster = df[["Usia", "Masa Kerja", "Total Jam Pelatihan", "Peringkat"]].copy()
    scaler = MinMaxScaler()
    fitur_scaled = scaler.fit_transform(fitur_klaster)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df["Cluster"] = kmeans.fit_predict(fitur_scaled)

    # --- PCA untuk reduksi dimensi (2D scatter) ---
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(fitur_scaled)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # --- Skema warna yang berbeda jelas untuk 4 cluster ---
    color_map_cluster = {
        0: "#1f77b4",  # biru
        1: "#ff7f0e",  # oranye
        2: "#2ca02c",  # hijau
        3: "#d62728"   # merah
    }

    # --- Visualisasi PCA dengan warna & rasio sama ---
    st.subheader("📊 Visualisasi Klasterisasi Pegawai (PCA)")
    st.markdown("Setiap titik merepresentasikan satu pegawai, diwarnai berdasarkan klaster. Tooltip menampilkan informasi detail.")

    fig_pca = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color=df["Cluster"].astype(str),
        color_discrete_map={str(k): v for k, v in color_map_cluster.items()},
        hover_data=["ID Pegawai", "Usia", "Golongan", "Jabatan", "Tingkat Pendidikan", "Masa Kerja", "Total Jam Pelatihan"],
        title="Visualisasi Klaster Pegawai Berdasarkan PCA (4 Klaster)"
    )

    # Set rasio X dan Y sama
    # Plot dengan rasio 1:1 dan panjang pixel sumbu x = y
    fig_pca.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_pca.update_layout(
        autosize=False,
        width=600,
        height=600,  # agar kotak
        margin=dict(l=20, r=20, t=50, b=20)
    )
    # st.plotly_chart(fig_pca)


    # Tampilkan grafik
    st.plotly_chart(fig_pca, use_container_width=True)


    # --- Deskripsi Klaster ---
    st.subheader("📌 Karakteristik Umum Tiap Klaster")
    deskripsi_klaster = df.groupby("Cluster").agg({
        "Usia": "mean",
        "Masa Kerja": "mean",
        "Total Jam Pelatihan": "mean",
        "Peringkat": "mean"
    }).round(1).rename(columns={
        "Usia": "Rata-rata Usia",
        "Masa Kerja": "Rata-rata Masa Kerja",
        "Total Jam Pelatihan": "Rata-rata Jam Pelatihan",
        "Peringkat": "Rata-rata Peringkat"
    })

    pendidikan_utama = df.groupby("Cluster")["Tingkat Pendidikan"].agg(lambda x: x.value_counts().idxmax())
    jabatan_utama = df.groupby("Cluster")["Jabatan"].agg(lambda x: x.value_counts().idxmax())

    deskripsi_klaster["Tingkat Pendidikan Dominan"] = pendidikan_utama
    deskripsi_klaster["Jabatan Dominan"] = jabatan_utama

    st.dataframe(deskripsi_klaster.reset_index(), use_container_width=True)

# ==================================
# 🔹 Tab 3: Tentang Aplikasi
# ==================================
with tab3:
    st.header("🎓 Rekomendasi Pelatihan Pegawai")

    st.markdown("""
    Tab ini menyajikan daftar pelatihan yang direkomendasikan untuk setiap pegawai berdasarkan 
    kebutuhannya. Pengguna dapat memfilter berdasarkan **unit kerja**, serta melihat **skor relevansi** 
    terhadap pelatihan tersebut. Data ditampilkan dalam bentuk tabel, dan tersedia visualisasi 
    berupa grafik jumlah pelatihan yang direkomendasikan per unit kerja.
    """)

    # --- Simulasi daftar pelatihan ---
    pelatihan_list = [
        "Pelatihan Kepemimpinan",
        "Manajemen Proyek Pemerintahan",
        "Analisis Anggaran Berbasis Kinerja",
        "Data Analytics untuk Kebijakan",
        "Pengelolaan PNBP",
        "Digitalisasi Sistem Penganggaran"
    ]

    # Simulasi data rekomendasi: 1–3 pelatihan per pegawai
    rekomendasi_data = []
    for _, row in df.iterrows():
        n_rek = np.random.randint(1, 4)
        pelatihans = np.random.choice(pelatihan_list, n_rek, replace=False)
        for pel in pelatihans:
            rekomendasi_data.append({
                "ID Pegawai": row["ID Pegawai"],
                "Unit": row["Unit"],
                "Nama Pelatihan": pel,
                "Skor Relevansi": round(np.random.uniform(0.70, 1.00), 2)
            })

    df_rekom = pd.DataFrame(rekomendasi_data)

    # --- Filter Unit ---
    st.subheader("📋 Tabel Rekomendasi Pelatihan")
    selected_unit = st.selectbox("Filter Unit Kerja", ["Semua"] + unit_list_real, key="unit_filter_pelatihan_final")

    filtered_df_rekom = df_rekom.copy()
    if selected_unit != "Semua":
        filtered_df_rekom = filtered_df_rekom[filtered_df_rekom["Unit"] == selected_unit]

    # Tampilkan tabel rekomendasi
    st.dataframe(
        filtered_df_rekom[["ID Pegawai", "Unit", "Nama Pelatihan", "Skor Relevansi"]].sort_values(by="Skor Relevansi", ascending=False),
        use_container_width=True
    )

    # --- Visualisasi: Jumlah rekomendasi per unit ---
    st.subheader("📊 Jumlah Rekomendasi Pelatihan per Unit Kerja")
    chart_df = df_rekom.groupby("Unit")["Nama Pelatihan"].count().reset_index()
    chart_df.columns = ["Unit", "Jumlah Rekomendasi"]

    fig = px.bar(
        chart_df,
        x="Unit",
        y="Jumlah Rekomendasi",
        color="Unit",
        title="Distribusi Jumlah Pelatihan yang Direkomendasikan per Unit",
        text_auto=True
    )
    fig.update_layout(
        xaxis_title="Unit Kerja",
        yaxis_title="Jumlah Rekomendasi",
        showlegend=False,
        height=800  # <<— tinggi diperbesar dari default (~400)
    )

    st.plotly_chart(fig, use_container_width=True)


