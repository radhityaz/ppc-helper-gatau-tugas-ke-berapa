import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from matplotlib.ticker import MaxNLocator
import logging

logging.basicConfig(level=logging.ERROR)

# Set pengaturan halaman
st.set_page_config(
    page_title="Perencanaan Produksi PT LSP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Font agar mendukung teks bahasa Indonesia
plt.rcParams['font.family'] = 'DejaVu Sans'

# Fungsi untuk menginisialisasi API Gemini
@st.cache_resource
def inisialisasi_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        logging.error(f"Gagal menginisialisasi API Gemini: {e}")
        raise

# Inisialisasi model Gemini menggunakan secrets
try:
    # Coba ambil API key dari secrets.toml
    api_key = st.secrets["GEMINI_API_KEY"]
    model = inisialisasi_gemini(api_key)
    # Simpan model di session state agar tersedia di seluruh aplikasi
    st.session_state['gemini_model'] = model
except Exception as e:
    model = None
    st.sidebar.error(f"Error koneksi API: {e}")
    st.sidebar.warning("Pastikan file .streamlit/secrets.toml berisi GEMINI_API_KEY yang valid")

# Fungsi untuk interpretasi dengan AI - dipindahkan dan diganti dengan fungsi spesifik
def interpretasi_data_penjualan(model, data):
    """Interpretasi data penjualan menggunakan API Gemini."""
    if not model:
        return "Interpretasi AI tidak tersedia."

    prompt = f"""
    Saya memiliki data penjualan bulanan untuk PT LSP.

    Data:
    {data}

    Berikan interpretasi ringkas dalam 3-5 poin utama dalam bahasa Indonesia.
    Fokus pada tren, anomali, dan wawasan penting lainnya.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error menghasilkan interpretasi AI untuk data penjualan: {e}")
        return f"Error menghasilkan interpretasi AI untuk data penjualan: {e}"

def interpretasi_optimasi_parameter(model, metode, metrik_kesalahan, alpha_optimal, beta_optimal=None, nilai_kesalahan=None):
    """Interpretasi hasil optimasi parameter menggunakan API Gemini."""
    if not model:
        return "Interpretasi AI tidak tersedia."

    prompt = f"""
    Saya telah melakukan optimasi parameter untuk metode {metode}.
    Metrik kesalahan yang digunakan adalah {metrik_kesalahan}.
    Nilai parameter optimal adalah:
    Alpha: {alpha_optimal:.2f}
    """
    if beta_optimal is not None:
        prompt += f"\nBeta: {beta_optimal:.2f}"
    if nilai_kesalahan is not None:
        prompt += f"\nNilai {metrik_kesalahan} pada parameter optimal adalah: {nilai_kesalahan:.4f}"

    prompt += """

    Berikan interpretasi ringkas dalam 3-5 poin utama dalam bahasa Indonesia.
    Fokus pada signifikansi nilai parameter optimal dan implikasinya terhadap model peramalan.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error menghasilkan interpretasi AI untuk optimasi parameter {metode}: {e}")
        return f"Error menghasilkan interpretasi AI untuk optimasi parameter {metode}: {e}"

def interpretasi_hasil_peramalan(model, deskripsi_ramalan, metode_evaluasi):
    """Interpretasi hasil peramalan menggunakan API Gemini."""
    if not model:
        return "Interpretasi AI tidak tersedia."

    prompt = f"""
    Saya telah melakukan peramalan penjualan bulanan untuk PT LSP menggunakan metode {deskripsi_ramalan['metode_lebih_baik']}.
    Parameter optimal yang ditemukan adalah:
    Alpha: {deskripsi_ramalan['alpha_ses']:.2f} (untuk SES)
    Alpha: {deskripsi_ramalan['alpha_des']:.2f}, Beta: {deskripsi_ramalan['beta_des']:.2f} (untuk DES)
    Metrik kesalahan {metode_evaluasi} untuk metode terbaik adalah: {deskripsi_ramalan['kesalahan_terbaik']:.4f}.

    Berikut adalah beberapa data penjualan aktual terakhir dan hasil peramalan:
    Data Aktual (3 bulan terakhir): {deskripsi_ramalan['aktual'][-3:]}
    Ramalan SES (3 bulan terakhir): {deskripsi_ramalan['ramalan_ses'][-3:]}
    Ramalan DES (3 bulan terakhir): {deskripsi_ramalan['ramalan_des'][-3:]}

    Berikan interpretasi yang ringkas dan profesional dalam 3-5 poin utama dalam bahasa Indonesia.
    Fokus pada:
    - Seberapa baik model peramalan sesuai dengan data aktual.
    - Perbandingan antara metode SES dan DES.
    - Implikasi dari nilai metrik kesalahan.
    Mulai langsung dengan poin-poin tanpa pendahuluan.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error menghasilkan interpretasi AI untuk hasil peramalan: {e}")
        return f"Error menghasilkan interpretasi AI untuk hasil peramalan: {e}"

def interpretasi_peramalan_masa_depan(model, ramalan_ses_depan, ramalan_des_depan, metode_lebih_baik, penjualan_aktual_terakhir):
    """Interpretasi peramalan 3 bulan ke depan menggunakan API Gemini."""
    if not model:
        return "Interpretasi AI tidak tersedia."

    prompt = f"""
    Saya memiliki hasil peramalan penjualan untuk 3 bulan ke depan untuk PT LSP.
    Metode peramalan yang dianggap lebih baik adalah {metode_lebih_baik}.
    Penjualan aktual 3 bulan terakhir adalah: {penjualan_aktual_terakhir}
    Hasil peramalan untuk 3 bulan ke depan adalah:
    Ramalan SES: {ramalan_ses_depan}
    Ramalan DES: {ramalan_des_depan}

    Berikan interpretasi yang ringkas dan profesional dalam 3-5 poin utama dalam bahasa Indonesia.
    Fokus pada tren yang diprediksi dan implikasinya terhadap perencanaan produksi dan inventaris.
    Sebutkan juga metode mana yang lebih direkomendasikan berdasarkan hasil sebelumnya.
    Mulai langsung dengan poin-poin tanpa pendahuluan.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error menghasilkan interpretasi AI untuk peramalan masa depan: {e}")
        return f"Error menghasilkan interpretasi AI untuk peramalan masa depan: {e}"

def interpretasi_perbandingan_metode(model, tabel_perbandingan, metrik_utama, metode_lebih_baik, nilai_ses, nilai_des):
    """Interpretasi perbandingan metrik kesalahan antara metode SES dan DES."""
    if not model:
        return "Interpretasi AI tidak tersedia."

    prompt = f"""
    Saya telah membandingkan dua metode peramalan (SES dan DES) berdasarkan beberapa metrik kesalahan.
    Metrik utama yang digunakan untuk evaluasi adalah {metrik_utama}.
    Metode yang dianggap lebih baik berdasarkan metrik ini adalah {metode_lebih_baik}.
    Nilai metrik {metrik_utama} untuk SES adalah {nilai_ses:.4f} dan untuk DES adalah {nilai_des:.4f}.

    Berikut adalah tabel perbandingan metrik kesalahan:
    {tabel_perbandingan}

    Berikan interpretasi yang ringkas dan profesional dalam 3-5 poin utama dalam bahasa Indonesia.
    Fokus pada mengapa satu metode lebih baik dari yang lain berdasarkan metrik yang relevan.
    Sebutkan juga implikasi praktis dari perbedaan ini.
    Mulai langsung dengan poin-poin tanpa pendahuluan.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error menghasilkan interpretasi AI untuk perbandingan metode: {e}")
        return f"Error menghasilkan interpretasi AI untuk perbandingan metode: {e}"

def interpretasi_rekomendasi_akhir(model, nim, metrik_evaluasi, metode_lebih_baik, parameter_optimal, ramalan_masa_depan):
    """Interpretasi rekomendasi akhir berdasarkan hasil peramalan."""
    if not model:
        return "Interpretasi AI tidak tersedia."

    prompt = f"""
    Berdasarkan analisis peramalan penjualan yang dilakukan (NIM: {nim}),
    metrik evaluasi utama adalah {metrik_evaluasi}, dan metode peramalan yang lebih baik adalah {metode_lebih_baik}
    dengan parameter optimal: Alpha (SES) = {parameter_optimal['alpha_ses']:.2f}, Alpha (DES) = {parameter_optimal['alpha_des']:.2f}, Beta (DES) = {parameter_optimal['beta_des']:.2f}.
    Hasil peramalan 3 bulan ke depan adalah:
    SES: {ramalan_masa_depan['ses']}
    DES: {ramalan_masa_depan['des']}

    Berikan rekomendasi akhir yang ringkas dan praktis dalam 3-5 poin utama dalam bahasa Indonesia
    kepada PT LSP terkait perencanaan produksi berdasarkan hasil ini.
    Sertakan saran tentang metode peramalan yang sebaiknya digunakan dan perkiraan penjualan di masa depan.
    Mulai langsung dengan poin-poin tanpa pendahuluan.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error menghasilkan interpretasi AI untuk rekomendasi akhir: {e}")
        return f"Error menghasilkan interpretasi AI untuk rekomendasi akhir: {e}"

# Sidebar untuk pemilihan masalah
with st.sidebar:
    st.title("Perencanaan Produksi PT LSP")
    st.markdown("Solusi untuk masalah perencanaan produksi")

    # Pemilihan masalah
    masalah = st.radio(
        "Pilih Masalah:",
        ["1. Peramalan Penjualan",
         "2. Metode Winter",
         "3. Perencanaan Agregat",
         "4. Jadwal Produksi Induk",
         "5. Penyeimbangan Lini Perakitan"]
    )

# Masalah 1: Peramalan dengan Penghalusan Eksponensial
def selesaikan_masalah_1(nim, data_penjualan):
    # Hitung nilai awal berdasarkan NIM
    digit_6 = int(nim[-6]) if len(nim) >= 6 else 0
    digit_7 = int(nim[-5]) if len(nim) >= 5 else 0
    digit_8 = int(nim[-4]) if len(nim) >= 4 else 0

    B0 = 200 + (digit_6 + digit_7*10 + digit_8)
    T0 = 0

    # Tentukan metode evaluasi berdasarkan digit terakhir NIM
    digit_terakhir = int(nim[-1])
    if digit_terakhir in [0, 1, 2]:
        metode_evaluasi = "MFE"
    elif digit_terakhir in [3, 4, 5]:
        metode_evaluasi = "MAD"
    elif digit_terakhir in [6, 7]:
        metode_evaluasi = "MAPE"
    else:  # 8 atau 9
        metode_evaluasi = "MSE"

    # Implementasi SES (Single Exponential Smoothing)
    def ramalan_ses(data, alpha, B0):
        n = len(data)
        ramalan = np.zeros(n + 1)
        ramalan[0] = B0

        for t in range(n):
            ramalan[t+1] = alpha * data[t] + (1 - alpha) * ramalan[t]

        return ramalan

    # Implementasi DES (Double Exponential Smoothing)
    def ramalan_des(data, alpha, beta, B0, T0):
        n = len(data)
        ramalan = np.zeros(n + 1)
        level = np.zeros(n + 1)
        trend = np.zeros(n + 1)

        # Inisialisasi
        level[0] = B0
        trend[0] = T0
        ramalan[0] = level[0] + trend[0]

        for t in range(n):
            # Perbarui level dan trend
            level[t+1] = alpha * data[t] + (1 - alpha) * (level[t] + trend[t])
            trend[t+1] = beta * (level[t+1] - level[t]) + (1 - beta) * trend[t]

            # Buat ramalan
            ramalan[t+1] = level[t+1] + trend[t+1]

        return ramalan, level, trend

    # Hitung metrik kesalahan
    def hitung_metrik_kesalahan(aktual, ramalan):
        ramalan_satu_langkah = ramalan[1:len(aktual)+1]

        kesalahan = aktual - ramalan_satu_langkah
        kesalahan_absolut = np.abs(kesalahan)
        kesalahan_kuadrat = kesalahan ** 2
        kesalahan_persentase = np.abs(kesalahan / aktual) * 100

        metrik = {
            'MFE': np.mean(kesalahan),
            'MAD': np.mean(kesalahan_absolut),
            'MSE': np.mean(kesalahan_kuadrat),
            'MAPE': np.mean(kesalahan_persentase)
        }

        return metrik, kesalahan, kesalahan_absolut, kesalahan_kuadrat, kesalahan_persentase

    # Pencarian grid untuk parameter optimal
    def cari_parameter_optimal():
        alpha_list = np.arange(0.1, 1.0, 0.1)
        beta_list = np.arange(0.1, 1.0, 0.1)

        penjualan = data_penjualan['Penjualan'].values

        # Wadah untuk hasil
        hasil = {
            'SES': {'alpha_terbaik': 0, 'metrik_terbaik': float('inf'), 'ramalan': None, 'metrik': None},
            'DES': {'alpha_terbaik': 0, 'beta_terbaik': 0, 'metrik_terbaik': float('inf'),
                    'ramalan': None, 'level': None, 'trend': None, 'metrik': None}
        }

        semua_hasil = {
            'SES': {'alpha_list':[], 'metrik_list':[]},
            'DES': {'alpha_list':[], 'beta_list':[], 'metrik_list': []}}

        # Grid search untuk SES
        for alpha in alpha_list:
            ramalan = ramalan_ses(penjualan, alpha, B0)
            metrik, _, _, _, _ = hitung_metrik_kesalahan(penjualan, ramalan)

            semua_hasil['SES']['alpha_list'].append(alpha)
            semua_hasil['SES']['metrik_list'].append(metrik[metode_evaluasi])

            if metrik[metode_evaluasi] < hasil['SES']['metrik_terbaik']:
                hasil['SES']['alpha_terbaik'] = alpha
                hasil['SES']['metrik_terbaik'] = metrik[metode_evaluasi]
                hasil['SES']['ramalan'] = ramalan
                hasil['SES']['metrik'] = metrik

        # Grid search untuk DES
        for alpha in alpha_list:
            for beta in beta_list:
                ramalan, level, trend = ramalan_des(penjualan, alpha, beta, B0, T0)
                metrik, _, _, _, _ = hitung_metrik_kesalahan(penjualan, ramalan)

                semua_hasil['DES']['alpha_list'].append(alpha)
                semua_hasil['DES']['beta_list'].append(beta)
                semua_hasil['DES']['metrik_list'].append(metrik[metode_evaluasi])

                if metrik[metode_evaluasi] < hasil['DES']['metrik_terbaik']:
                    hasil['DES']['alpha_terbaik'] = alpha
                    hasil['DES']['beta_terbaik'] = beta
                    hasil['DES']['metrik_terbaik'] = metrik[metode_evaluasi]
                    hasil['DES']['ramalan'] = ramalan
                    hasil['DES']['level'] = level
                    hasil['DES']['trend'] = trend
                    hasil['DES']['metrik'] = metrik

        return hasil, semua_hasil

    # Jalankan pencarian parameter optimal
    hasil, semua_hasil = cari_parameter_optimal()

    # Hitung ramalan 3 bulan ke depan
    bulan_depan = [25, 26, 27]
    ramalan_ses_depan = np.ones(3) * hasil['SES']['ramalan'][-1]

    level_terakhir = hasil['DES']['level'][-1]
    trend_terakhir = hasil['DES']['trend'][-1]
    ramalan_des_depan = [level_terakhir + (i+1)*trend_terakhir for i in range(3)]

    # Tentukan metode yang lebih baik
    metode_lebih_baik = 'SES' if hasil['SES']['metrik_terbaik'] < hasil['DES']['metrik_terbaik'] else 'DES'

    # Kembalikan semua hasil
    return {
        'B0': B0,
        'T0': T0,
        'metode_evaluasi': metode_evaluasi,
        'hasil': hasil,
        'semua_hasil': semua_hasil,
        'bulan_depan': bulan_depan,
        'ramalan_ses_depan': ramalan_ses_depan,
        'ramalan_des_depan': ramalan_des_depan,
        'metode_lebih_baik': metode_lebih_baik
    }

# Masalah 2: Metode Winter
def selesaikan_masalah_2():
    st.header("Masalah 2: Metode Winter")
    st.info("Implementasi untuk Metode Winter akan ditambahkan di sini.")
    # Di sini Anda akan menambahkan logika dan UI untuk Metode Winter

# Masalah 3: Perencanaan Agregat
def selesaikan_masalah_3():
    st.header("Masalah 3: Perencanaan Agregat")
    st.info("Implementasi untuk Perencanaan Agregat akan ditambahkan di sini.")
    # Di sini Anda akan menambahkan logika dan UI untuk Perencanaan Agregat

# Masalah 4: Jadwal Produksi Induk
def selesaikan_masalah_4():
    st.header("Masalah 4: Jadwal Produksi Induk")
    st.info("Implementasi untuk Jadwal Produksi Induk akan ditambahkan di sini.")
    # Di sini Anda akan menambahkan logika dan UI untuk Jadwal Produksi Induk

# Masalah 5: Penyeimbangan Lini Perakitan
def selesaikan_masalah_5():
    st.header("Masalah 5: Penyeimbangan Lini Perakitan")
    st.info("Implementasi untuk Penyeimbangan Lini Perakitan akan ditambahkan di sini.")
    # Di sini Anda akan menambahkan logika dan UI untuk Penyeimbangan Lini Perakitan

# Aplikasi utama
def main():
    global model  # Deklarasi bahwa kita menggunakan variabel global model
    st.title("PT LSP Abang Dika - Perencanaan Produksi")

    # Ambil model Gemini (sudah diinisialisasi di awal)
    model = st.session_state.get('gemini_model') or model # Gunakan model yang sudah diinisialisasi

    if "1." in masalah:
        st.header("Masalah 1: Peramalan Penjualan")

        # Bagian input
        with st.expander("Parameter Input", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                nim = st.text_input("Masukkan NIM", "13423125")

            with col2:
                # Opsi untuk menggunakan data default atau mengunggah data kustom
                gunakan_default = st.checkbox("Gunakan data penjualan default", value=True)

            if gunakan_default:
                data_penjualan = pd.DataFrame({
                    'Bulan': list(range(1, 25)),
                    'Penjualan': [260, 210, 180, 255, 330, 395, 280, 215, 270, 130, 125, 90,
                                  140, 155, 195, 225, 250, 430, 510, 405, 375, 325, 295, 260]
                })
            else:
                uploaded_file = st.file_uploader("Unggah data penjualan CSV", type="csv")
                if uploaded_file is not None:
                    data_penjualan = pd.read_csv(uploaded_file)
                else:
                    st.warning("Silakan unggah file CSV atau pilih 'Gunakan data penjualan default'")
                    return

            st.dataframe(data_penjualan)

        # Tombol untuk memulai perhitungan
        if st.button("Selesaikan Masalah 1"):
            with st.spinner("Menghitung peramalan..."):
                # Dapatkan hasil
                p1_hasil = selesaikan_masalah_1(nim, data_penjualan)

                # 1. Tampilkan parameter dan hasil
                st.subheader("Parameter Model")
                col1, col2, col3 = st.columns(3)
                col1.metric("Level Awal (B₀)", p1_hasil['B0'])
                col2.metric("Trend Awal (T₀)", p1_hasil['T0'])
                col3.metric("Metode Evaluasi", p1_hasil['metode_evaluasi'])

                # 2. Tampilkan parameter optimal
                st.subheader("Parameter Optimal")
                col1, col2, col3 = st.columns(3)
                col1.metric("SES α", f"{p1_hasil['hasil']['SES']['alpha_terbaik']:.1f}")
                col2.metric("DES α", f"{p1_hasil['hasil']['DES']['alpha_terbaik']:.1f}")
                col3.metric("DES β", f"{p1_hasil['hasil']['DES']['beta_terbaik']:.1f}")
                
                # Tambahkan rumus matematika
                st.subheader("Rumus Peramalan")
                st.write("**Single Exponential Smoothing (SES)**")
                st.latex(r"F_{t+1} = \alpha Y_t + (1-\alpha)F_t")
                st.write("dimana:")
                st.write("- $F_{t+1}$ adalah ramalan untuk periode $t+1$")
                st.write("- $Y_t$ adalah nilai aktual pada periode $t$")
                st.write("- $F_t$ adalah ramalan untuk periode $t$")
                st.write("- $\\alpha$ adalah parameter penghalusan (0 < $\\alpha$ < 1)")
                
                st.write("**Double Exponential Smoothing (DES)**")
                st.latex(r"S_t = \alpha Y_t + (1-\alpha)(S_{t-1} + T_{t-1})")
                st.latex(r"T_t = \beta(S_t - S_{t-1}) + (1-\beta)T_{t-1}")
                st.latex(r"F_{t+1} = S_t + T_t")
                st.write("dimana:")
                st.write("- $S_t$ adalah nilai level pada periode $t$")
                st.write("- $T_t$ adalah nilai trend pada periode $t$")
                st.write("- $F_{t+1}$ adalah ramalan untuk periode $t+1$")
                st.write("- $Y_t$ adalah nilai aktual pada periode $t$")
                st.write("- $\\alpha$ dan $\\beta$ adalah parameter penghalusan (0 < $\\alpha,\\beta$ < 1)")

                # 3. Plot data penjualan
                st.subheader("Visualisasi Data Penjualan")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data_penjualan['Bulan'], data_penjualan['Penjualan'], marker='o', linestyle='-',
                        linewidth=2, markersize=8, color='blue')
                ax.axhline(y=p1_hasil['B0'], color='r', linestyle='--', alpha=0.5,
                           label=f'Level Awal (B₀ = {p1_hasil["B0"]})')

                # Tambahkan anotasi untuk poin penting
                ax.annotate('Penjualan Tertinggi', xy=(19, 510), xytext=(19, 550),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=12, fontweight='bold')
                ax.annotate('Penjualan Terendah', xy=(12, 90), xytext=(12, 40),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=12, fontweight='bold')

                ax.set_title('Data Penjualan Bulanan PT LSP Abang Dika', fontweight='bold')
                ax.set_xlabel('Bulan')
                ax.set_ylabel('Volume Penjualan')
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
                st.pyplot(fig)

                # Interpretasi data penjualan
                st.subheader("Interpretasi Data Penjualan")
                deskripsi_data_penjualan = {
                    "min": min(data_penjualan['Penjualan']),
                    "max": max(data_penjualan['Penjualan']),
                    "rata_rata": np.mean(data_penjualan['Penjualan']),
                    "bulan_tertinggi": data_penjualan['Bulan'][data_penjualan['Penjualan'].argmax()],
                    "bulan_terendah": data_penjualan['Bulan'][data_penjualan['Penjualan'].argmin()],
                    "level_awal": p1_hasil['B0']
                }
                with st.container():
                    if model:
                        st.markdown(interpretasi_data_penjualan(model, deskripsi_data_penjualan))
                    else:
                        st.warning("API Key belum dimasukkan atau gagal terhubung.")

                # 4. Plot hasil pencarian parameter untuk SES
                st.subheader("Optimasi Parameter SES")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(p1_hasil['semua_hasil']['SES']['alpha_list'],
                        p1_hasil['semua_hasil']['SES']['metrik_list'],
                        'o-', linewidth=2, markersize=8, color='blue')
                ax.axvline(x=p1_hasil['hasil']['SES']['alpha_terbaik'], color='red',
                            linestyle='--', label=f'α Optimal = {p1_hasil["hasil"]["SES"]["alpha_terbaik"]:.1f}')
                ax.set_title(f'Optimasi Parameter: {p1_hasil["metode_evaluasi"]} vs. Alpha (SES)', fontweight='bold')
                ax.set_xlabel('Alpha (α)')
                ax.set_ylabel(p1_hasil['metode_evaluasi'])
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

                # Interpretasi optimasi parameter SES
                st.subheader("Interpretasi Optimasi Parameter SES")
                deskripsi_opt_param_ses = {
                    "metode": "SES",
                    "metrik_kesalahan": p1_hasil['metode_evaluasi'],
                    "alpha_optimal": p1_hasil['hasil']['SES']['alpha_terbaik'],
                    "nilai_kesalahan": p1_hasil['hasil']['SES']['metrik_terbaik']
                }
                with st.container():
                    if model:
                        st.markdown(interpretasi_optimasi_parameter(model, **deskripsi_opt_param_ses))
                    else:
                        st.warning("API Key belum dimasukkan atau gagal terhubung.")

                # 5. Heatmap untuk parameter DES
                st.subheader("Optimasi Parameter DES")
                alpha_values = np.unique(p1_hasil['semua_hasil']['DES']['alpha_list'])
                beta_values = np.unique(p1_hasil['semua_hasil']['DES']['beta_list'])
                metric_grid = np.zeros((len(alpha_values), len(beta_values)))

                for i, alpha in enumerate(alpha_values):
                    for j, beta in enumerate(beta_values):
                        indices = [k for k in range(len(p1_hasil['semua_hasil']['DES']['alpha_list']))
                                   if p1_hasil['semua_hasil']['DES']['alpha_list'][k] == alpha and
                                   p1_hasil['semua_hasil']['DES']['beta_list'][k] == beta]

                        if indices:
                            metric_grid[i, j] = p1_hasil['semua_hasil']['DES']['metrik_list'][indices[0]]

                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(metric_grid, annot=True, fmt=".2f", cmap="viridis",
                            xticklabels=[f"{b:.1f}" for b in beta_values],
                            yticklabels=[f"{a:.1f}" for a in alpha_values],
                            ax=ax)

                opt_alpha_idx = np.where(alpha_values == p1_hasil['hasil']['DES']['alpha_terbaik'])[0][0]
                opt_beta_idx = np.where(beta_values == p1_hasil['hasil']['DES']['beta_terbaik'])[0][0]

                plt.plot(opt_beta_idx + 0.5, opt_alpha_idx + 0.5, 'r*', markersize=20,
                         label=f'Optimal (α={p1_hasil["hasil"]["DES"]["alpha_terbaik"]:.1f}, β={p1_hasil["hasil"]["DES"]["beta_terbaik"]:.1f})')

                plt.title(f'DES: {p1_hasil["metode_evaluasi"]} untuk Berbagai Nilai Alpha dan Beta', fontweight='bold')
                plt.xlabel('Beta (β)')
                plt.ylabel('Alpha (α)')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
                st.pyplot(fig)

                # Interpretasi heatmap DES
                st.subheader("Interpretasi Optimasi Parameter DES")
                deskripsi_opt_param_des = {
                    "metode": "DES",
                    "metrik_kesalahan": p1_hasil['metode_evaluasi'],
                    "alpha_optimal": p1_hasil['hasil']['DES']['alpha_terbaik'],
                    "beta_optimal": p1_hasil['hasil']['DES']['beta_terbaik'],
                    "nilai_kesalahan": p1_hasil['hasil']['DES']['metrik_terbaik']
                }
                with st.container():
                    if model:
                        st.markdown(interpretasi_optimasi_parameter(model, **deskripsi_opt_param_des))
                    else:
                        st.warning("API Key belum dimasukkan atau gagal terhubung.")

                # 6. Tampilkan hasil peramalan
                st.subheader("Hasil Peramalan vs Data Aktual")

                # Buat dataframe peramalan
                df_ramalan = pd.DataFrame({
                    'Bulan': range(1, 25),
                    'Aktual': data_penjualan['Penjualan'],
                    'Ramalan SES': p1_hasil['hasil']['SES']['ramalan'][1:25],
                    'Ramalan DES': p1_hasil['hasil']['DES']['ramalan'][1:25]
                })

                # Plot peramalan vs aktual
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_ramalan['Bulan'], df_ramalan['Aktual'], 'o-', linewidth=2,
                        markersize=8, color='blue', label='Data Aktual')
                ax.plot(df_ramalan['Bulan'], df_ramalan['Ramalan SES'], '--', linewidth=2,
                        color='green', label=f'SES (α={p1_hasil["hasil"]["SES"]["alpha_terbaik"]:.1f})')
                ax.plot(df_ramalan['Bulan'], df_ramalan['Ramalan DES'], ':', linewidth=2,
                        color='red', label=f'DES (α={p1_hasil["hasil"]["DES"]["alpha_terbaik"]:.1f}, β={p1_hasil["hasil"]["DES"]["beta_terbaik"]:.1f})')
                ax.set_title('Penjualan Aktual vs. Hasil Peramalan', fontweight='bold')
                ax.set_xlabel('Bulan')
                ax.set_ylabel('Penjualan')
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
                st.pyplot(fig)

                # Interpretasi hasil peramalan
                st.subheader("Interpretasi Hasil Peramalan")
                deskripsi_ramalan_ai = {
                    "aktual": data_penjualan['Penjualan'].tolist(),
                    "ramalan_ses": p1_hasil['hasil']['SES']['ramalan'].tolist(),
                    "ramalan_des": p1_hasil['hasil']['DES']['ramalan'].tolist(),
                    "kesalahan_ses": p1_hasil['hasil']['SES']['metrik'][p1_hasil['metode_evaluasi']],
                    "kesalahan_des": p1_hasil['hasil']['DES']['metrik'][p1_hasil['metode_evaluasi']],
                    "kesalahan_terbaik": min(p1_hasil['hasil']['SES']['metrik'][p1_hasil['metode_evaluasi']], p1_hasil['hasil']['DES']['metrik'][p1_hasil['metode_evaluasi']]),
                    "metode_lebih_baik": p1_hasil['metode_lebih_baik'],
                    "alpha_ses": p1_hasil['hasil']['SES']['alpha_terbaik'],
                    "alpha_des": p1_hasil['hasil']['DES']['alpha_terbaik'],
                    "beta_des": p1_hasil['hasil']['DES']['beta_terbaik']
                }
                with st.container():
                    if model:
                        st.markdown(interpretasi_hasil_peramalan(model, deskripsi_ramalan_ai, p1_hasil['metode_evaluasi']))
                    else:
                        st.warning("API Key belum dimasukkan atau gagal terhubung.")

                # 7. Tampilkan peramalan 3 bulan ke depan
                st.subheader("Peramalan 3 Bulan ke Depan")

                # Tabel peramalan masa depan
                df_masa_depan = pd.DataFrame({
                    'Bulan': p1_hasil['bulan_depan'],
                    'Ramalan SES': p1_hasil['ramalan_ses_depan'],
                    'Ramalan DES': p1_hasil['ramalan_des_depan']
                })
                st.dataframe(df_masa_depan)

                # Plot peramalan masa depan
                fig, ax = plt.subplots(figsize=(12, 6))
                # Data historis
                ax.plot(df_ramalan['Bulan'], df_ramalan['Aktual'], 'o-', linewidth=2,
                        markersize=8, color='blue', label='Data Aktual')
                ax.plot(df_ramalan['Bulan'], df_ramalan['Ramalan SES'], '--',
                        linewidth=2, color='green', label='SES Historis')
                ax.plot(df_ramalan['Bulan'], df_ramalan['Ramalan DES'], ':',
                        linewidth=2, color='red', label='DES Historis')

                # Data masa depan
                ax.plot(df_masa_depan['Bulan'], df_masa_depan['Ramalan SES'], 's--', linewidth=2,
                        markersize=10, color='green', label='SES Masa Depan')
                ax.plot(df_masa_depan['Bulan'], df_masa_depan['Ramalan DES'], 's:', linewidth=2,
                        markersize=10, color='red', label='DES Masa Depan')

                # Garis pemisah
                ax.axvline(x=24.5, color='black', linestyle='--', alpha=0.7, label='Batas Peramalan')

                # Anotasi
                ax.text(25.5, max(data_penjualan['Penjualan']), 'Periode\nMasa Depan',
                        fontsize=12, fontweight='bold')

                ax.set_title('Peramalan Penjualan Historis dan Masa Depan', fontweight='bold')
                ax.set_xlabel('Bulan')
                ax.set_ylabel('Penjualan')
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
                st.pyplot(fig)

                # Interpretasi peramalan masa depan
                st.subheader("Interpretasi Peramalan Masa Depan")
                # Konversi ke list jika belum berbentuk list
                ramalan_ses_depan = p1_hasil['ramalan_ses_depan']
                ramalan_des_depan = p1_hasil['ramalan_des_depan']
                
                if hasattr(ramalan_ses_depan, 'tolist'):
                    ramalan_ses_depan = ramalan_ses_depan.tolist()
                if hasattr(ramalan_des_depan, 'tolist'):
                    ramalan_des_depan = ramalan_des_depan.tolist()
                
                deskripsi_masa_depan_ai = {
                    "ramalan_ses_depan": ramalan_ses_depan,
                    "ramalan_des_depan": ramalan_des_depan,
                    "metode_lebih_baik": p1_hasil['metode_lebih_baik'],
                    "penjualan_aktual_terakhir": data_penjualan['Penjualan'].tolist()[-3:]
                }
                with st.container():
                    if model:
                        st.markdown(interpretasi_peramalan_masa_depan(model, **deskripsi_masa_depan_ai))
                    else:
                        st.warning("API Key belum dimasukkan atau gagal terhubung.")

                # 8. Perbandingan metrik kesalahan
                st.subheader("Perbandingan Metrik Kesalahan")

                metrik_untuk_dibandingkan = ['MFE', 'MAD', 'MSE', 'MAPE']
                nilai_ses = [p1_hasil['hasil']['SES']['metrik'][m] for m in metrik_untuk_dibandingkan]
                nilai_des = [p1_hasil['hasil']['DES']['metrik'][m] for m in metrik_untuk_dibandingkan]

                # Tabel perbandingan
                df_perbandingan = pd.DataFrame({
                    'Metrik': metrik_untuk_dibandingkan,
                    'SES': nilai_ses,
                    'DES': nilai_des,
                    'Metode Lebih Baik': ['SES' if s < d else 'DES' for s, d in zip(nilai_ses, nilai_des)]
                })
                st.dataframe(df_perbandingan)

                # Bagan batang perbandingan
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(metrik_untuk_dibandingkan))
                width = 0.35
                batang1 = ax.bar(x - width/2, nilai_ses, width, label='SES', color='green', alpha=0.7)
                batang2 = ax.bar(x + width/2, nilai_des, width, label='DES', color='red', alpha=0.7)

                # Tambahkan label nilai
                for bar in batang1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

                for bar in batang2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

                # Sorot metrik utama
                idx_utama = metrik_untuk_dibandingkan.index(p1_hasil['metode_evaluasi'])
                ax.axvline(x=idx_utama, color='blue', linestyle='--', alpha=0.3)

                # Tambahkan teks penjelasan untuk metrik utama
                ax.text(idx_utama, max(max(nilai_ses), max(nilai_des)) * 1.1,
                        f"Metrik Evaluasi Utama\n({p1_hasil['metode_evaluasi']})",
                        ha='center', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8))

                # Tambahkan anotasi untuk metode terbaik
                if nilai_ses[idx_utama] < nilai_des[idx_utama]:
                    ax.annotate('Metode Lebih Baik', xy=(idx_utama - width/2, nilai_ses[idx_utama]),
                                xytext=(idx_utama - width/2, nilai_ses[idx_utama] * 0.7),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                                fontsize=12, fontweight='bold', ha='center')
                else:
                    ax.annotate('Metode Lebih Baik', xy=(idx_utama + width/2, nilai_des[idx_utama]),
                                xytext=(idx_utama + width/2, nilai_des[idx_utama] * 0.7),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                                fontsize=12, fontweight='bold', ha='center')

                ax.set_title('Perbandingan Metrik Kesalahan', fontweight='bold')
                ax.set_ylabel('Nilai Kesalahan')
                ax.set_xticks(x)
                ax.set_xticklabels(metrik_untuk_dibandingkan)
                ax.legend()
                st.pyplot(fig)

                # Interpretasi perbandingan metode
                st.subheader("Interpretasi Perbandingan Metode")
                deskripsi_perbandingan_ai = {
                    "tabel_perbandingan": df_perbandingan.to_string(),
                    "metrik_utama": p1_hasil['metode_evaluasi'],
                    "metode_lebih_baik": p1_hasil['metode_lebih_baik'],
                    "nilai_ses": nilai_ses[idx_utama],
                    "nilai_des": nilai_des[idx_utama]
                }
                with st.container():
                    if model:
                        st.markdown(interpretasi_perbandingan_metode(model, **deskripsi_perbandingan_ai))
                    else:
                        st.warning("API Key belum dimasukkan atau gagal terhubung.")

                # 9. Kesimpulan akhir
                st.subheader("Kesimpulan")

                kesimpulan = f"""
                ### Berdasarkan metrik kesalahan {p1_hasil['metode_evaluasi']} (ditentukan oleh NIM: {nim}):

                1. **Nilai parameter optimal** adalah:
                    - SES: α = {p1_hasil['hasil']['SES']['alpha_terbaik']:.1f}
                    - DES: α = {p1_hasil['hasil']['DES']['alpha_terbaik']:.1f}, β = {p1_hasil['hasil']['DES']['beta_terbaik']:.1f}

                2. **Peramalan 3 bulan ke depan** adalah:
                    - Bulan 25: SES = {p1_hasil['ramalan_ses_depan'][0]:.2f}, DES = {p1_hasil['ramalan_des_depan'][0]:.2f}
                    - Bulan 26: SES = {p1_hasil['ramalan_ses_depan'][1]:.2f}, DES = {p1_hasil['ramalan_des_depan'][1]:.2f}
                    - Bulan 27: SES = {p1_hasil['ramalan_ses_depan'][2]:.2f}, DES = {p1_hasil['ramalan_des_depan'][2]:.2f}

                3. **Metode peramalan yang lebih baik** adalah **{p1_hasil['metode_lebih_baik']}** dengan nilai {p1_hasil['metode_evaluasi']} lebih rendah sebesar {min(p1_hasil['hasil']['SES']['metrik_terbaik'], p1_hasil['hasil']['DES']['metrik_terbaik']):.4f}
                """

                st.markdown(kesimpulan)

                # Interpretasi akhir dengan AI
                st.subheader("Rekomendasi Akhir")

                # Konversi ke list jika belum berbentuk list
                ramalan_ses_depan = p1_hasil['ramalan_ses_depan']
                ramalan_des_depan = p1_hasil['ramalan_des_depan']
                
                if hasattr(ramalan_ses_depan, 'tolist'):
                    ramalan_ses_depan = ramalan_ses_depan.tolist()
                if hasattr(ramalan_des_depan, 'tolist'):
                    ramalan_des_depan = ramalan_des_depan.tolist()
                
                deskripsi_akhir_ai = {
                    "nim": nim,
                    "metrik_evaluasi": p1_hasil['metode_evaluasi'],
                    "metode_lebih_baik": p1_hasil['metode_lebih_baik'],
                    "parameter_optimal": {
                        "alpha_ses": p1_hasil['hasil']['SES']['alpha_terbaik'],
                        "alpha_des": p1_hasil['hasil']['DES']['alpha_terbaik'],
                        "beta_des": p1_hasil['hasil']['DES']['beta_terbaik']
                    },
                    "ramalan_masa_depan": {
                        "ses": ramalan_ses_depan,
                        "des": ramalan_des_depan
                    }
                }

                with st.container():
                    if model:
                        st.markdown(interpretasi_rekomendasi_akhir(model, **deskripsi_akhir_ai))
                    else:
                        st.warning("API Key belum dimasukkan atau gagal terhubung.")

    elif "2." in masalah:
        selesaikan_masalah_2()

    elif "3." in masalah:
        selesaikan_masalah_3()

    elif "4." in masalah:
        selesaikan_masalah_4()

    elif "5." in masalah:
        selesaikan_masalah_5()

if __name__ == "__main__":
    main()
