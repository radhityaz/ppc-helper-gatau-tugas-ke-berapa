import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from matplotlib.ticker import MaxNLocator
import logging
import networkx as nx

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
# Masalah 2: Metode Winter
def selesaikan_masalah_2():
    st.header("Masalah 2: Metode Winter")
    
    # Fungsi interpretasi khusus untuk Winter's Method
    def interpretasi_winter_method(model, hasil):
        if not model:
            return "Interpretasi AI tidak tersedia."
            
        prompt = f"""
        Saya telah menganalisis data permintaan musiman menggunakan Metode Winter (Triple Exponential Smoothing).
        
        Hasil analisis:
        - Faktor musiman: {hasil['faktor_musiman']}
        - Level yang diperbarui: {hasil['level']}
        - Trend yang diperbarui: {hasil['trend']}
        - Peramalan untuk kuartal berikutnya: {hasil['ramalan']}
        
        Berikan interpretasi yang ringkas dan profesional dalam 3-5 poin utama dalam bahasa Indonesia.
        Fokus pada:
        - Pengaruh faktor musiman terhadap permintaan
        - Dampak lonjakan permintaan terhadap perkiraan masa depan
        - Implikasi bisnis dari hasil peramalan
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error menghasilkan interpretasi Winter Method: {e}")
            return f"Error menghasilkan interpretasi: {e}"
    
    # UI dan Implementasi untuk Winter's Method
    with st.expander("Parameter Input", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            nim = st.text_input("Masukkan NIM (Masalah 2)", "13423125")
            
            # Hitung x berdasarkan NIM
            digit = [int(d) for d in nim[-3:]]
            x = sum(digit)
            st.info(f"Nilai x berdasarkan NIM: {x}")
            
        with col2:
            # Parameter penghalusan
            alpha = st.slider("Parameter α (Level)", 0.0, 1.0, 0.25, 0.01)
            beta = st.slider("Parameter β (Trend)", 0.0, 1.0, 0.12, 0.01)
            gamma = st.slider("Parameter γ (Seasonal)", 0.0, 1.0, 0.08, 0.01)
        
        # Input data permintaan
        st.subheader("Data Permintaan Kuartalan")
        col1, col2, col3, col4 = st.columns(4)
        
        # Tahun 2
        st.write("Tahun 2")
        t2q1 = col1.number_input("Q1 Tahun 2", value=14*x, step=1.0)
        t2q2 = col2.number_input("Q2 Tahun 2", value=28*x, step=1.0)
        t2q3 = col3.number_input("Q3 Tahun 2", value=65*x, step=1.0)
        t2q4 = col4.number_input("Q4 Tahun 2", value=50*x, step=1.0)
        
        # Tahun 3
        st.write("Tahun 3")
        t3q1 = col1.number_input("Q1 Tahun 3", value=20*x, step=1.0)
        t3q2 = col2.number_input("Q2 Tahun 3", value=40*x, step=1.0)
        t3q3 = col3.number_input("Q3 Tahun 3", value=80*x, step=1.0)
        t3q4 = col4.number_input("Q4 Tahun 3", value=55*x, step=1.0)
        
        # Lonjakan Q1 Tahun 4
        st.write("Tahun 4")
        t4q1 = st.number_input("Lonjakan Q1 Tahun 4", value=275, step=1.0)
    
    # Tambahkan rumus matematika Winter's Method
    st.subheader("Rumus Winter's Method (Triple Exponential Smoothing)")
    st.write("Winter's Method menggunakan tiga persamaan penghalusan untuk level, trend, dan faktor musiman:")
    
    # Level
    st.latex(r"S_t = \alpha \frac{Y_t}{I_{t-L}} + (1-\alpha)(S_{t-1} + T_{t-1})")
    # Trend
    st.latex(r"T_t = \beta(S_t - S_{t-1}) + (1-\beta)T_{t-1}")
    # Seasonal
    st.latex(r"I_t = \gamma\frac{Y_t}{S_t} + (1-\gamma)I_{t-L}")
    # Forecast
    st.latex(r"F_{t+m} = (S_t + mT_t)I_{t-L+m}")
    
    st.write("dimana:")
    st.write("- $S_t$ adalah komponen level pada waktu $t$")
    st.write("- $T_t$ adalah komponen trend pada waktu $t$")
    st.write("- $I_t$ adalah indeks musiman pada waktu $t$")
    st.write("- $Y_t$ adalah nilai aktual pada waktu $t$")
    st.write("- $F_{t+m}$ adalah peramalan untuk $m$ periode ke depan")
    st.write("- $L$ adalah panjang musiman (4 kuartal dalam kasus ini)")
    st.write("- $\\alpha$, $\\beta$, $\\gamma$ adalah parameter penghalusan")
    
    # Tombol untuk menjalankan perhitungan
    if st.button("Selesaikan Masalah 2"):
        with st.spinner("Menghitung dengan Metode Winter..."):
            # Implementasi Metode Winter
            
            # Membuat array data
            data_tahun2 = [t2q1, t2q2, t2q3, t2q4]
            data_tahun3 = [t3q1, t3q2, t3q3, t3q4]
            
            # Menghitung nilai awal
            seasons = 4  # 4 kuartal per tahun
            
            # Menghitung nilai awal intercept dan slope (pendekatan regresi sederhana)
            y_avg_year2 = sum(data_tahun2) / seasons
            y_avg_year3 = sum(data_tahun3) / seasons
            
            intercept = y_avg_year2
            slope = (y_avg_year3 - y_avg_year2) / seasons
            
            # Menghitung faktor musiman awal
            seasonal_indices = []
            
            # Rata-rata per kuartal
            for i in range(seasons):
                seasonal_avg = (data_tahun2[i] + data_tahun3[i]) / 2
                seasonal_indices.append(seasonal_avg / ((y_avg_year2 + y_avg_year3) / 2))
            
            # Normalisasi faktor musiman agar jumlahnya = seasons
            sum_indices = sum(seasonal_indices)
            normalized_indices = [idx * (seasons / sum_indices) for idx in seasonal_indices]
            
            # Data historis dan lonjakan
            all_data = data_tahun2 + data_tahun3 + [t4q1]
            
            # Menghitung perbaruan setelah lonjakan Q1
            l_t4q1 = alpha * (t4q1 / normalized_indices[0]) + (1 - alpha) * (intercept + slope)
            b_t4q1 = beta * (l_t4q1 - intercept) + (1 - beta) * slope
            s_t4q1 = gamma * (t4q1 / l_t4q1) + (1 - gamma) * normalized_indices[0]
            
            # Perbarui faktor musiman
            updated_indices = normalized_indices.copy()
            updated_indices[0] = s_t4q1
            
            # Peramalan untuk 3 kuartal berikutnya (Q2, Q3, Q4 Tahun 4)
            forecasts = []
            for m in range(1, 4):
                idx = m % seasons
                forecast = (l_t4q1 + m * b_t4q1) * updated_indices[idx]
                forecasts.append(forecast)
                
            # Hasil untuk ditampilkan
            hasil = {
                'nilai_x': x,
                'intercept_awal': intercept,
                'slope_awal': slope,
                'faktor_musiman': normalized_indices,
                'level': l_t4q1,
                'trend': b_t4q1,
                'faktor_musiman_diperbarui': updated_indices,
                'ramalan': forecasts
            }
            
            # Tampilkan hasil
            st.subheader("Komponen Model Awal")
            col1, col2 = st.columns(2)
            col1.metric("Intercept Awal", f"{intercept:.2f}")
            col2.metric("Slope Awal", f"{slope:.2f}")
            
            # Tampilkan faktor musiman
            st.subheader("Faktor Musiman Awal")
            seasons_names = ["Q1", "Q2", "Q3", "Q4"]
            df_musiman = pd.DataFrame({
                'Kuartal': seasons_names,
                'Faktor Musiman': normalized_indices
            })
            st.dataframe(df_musiman)
            
            # Plot faktor musiman awal
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(seasons_names, normalized_indices, color='skyblue')
            ax.axhline(y=1.0, color='red', linestyle='--', label='Rata-rata')
            ax.set_title('Faktor Musiman Awal', fontweight='bold')
            ax.set_ylabel('Faktor')
            ax.set_ylim(0, max(normalized_indices) * 1.2)
            ax.legend()
            st.pyplot(fig)
            
            # Tampilkan hasil perbaruan setelah lonjakan Q1
            st.subheader("Hasil Perbaruan Setelah Lonjakan Q1 Tahun 4")
            col1, col2, col3 = st.columns(3)
            col1.metric("Level Diperbarui", f"{l_t4q1:.2f}")
            col2.metric("Trend Diperbarui", f"{b_t4q1:.2f}")
            col3.metric("Faktor Q1 Diperbarui", f"{s_t4q1:.2f}")
            
            # Tampilkan peramalan untuk Q2-Q4 Tahun 4
            st.subheader("Peramalan Tahun 4")
            df_ramalan = pd.DataFrame({
                'Kuartal': ["Q2", "Q3", "Q4"],
                'Peramalan': forecasts
            })
            st.dataframe(df_ramalan)
            
            # Plot data historis dan peramalan
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Data quarters
            all_quarters = ["T2Q1", "T2Q2", "T2Q3", "T2Q4", 
                           "T3Q1", "T3Q2", "T3Q3", "T3Q4", 
                           "T4Q1", "T4Q2", "T4Q3", "T4Q4"]
            
            all_values = all_data + forecasts
            
            # Warna berbeda untuk data historis, lonjakan, dan peramalan
            colors = ['blue']*8 + ['red'] + ['green']*3
            labels = ['Data Historis']*8 + ['Lonjakan']*1 + ['Peramalan']*3
            
            # Plot data
            for i, (q, v, c, l) in enumerate(zip(all_quarters, all_values, colors, labels)):
                if i == 0:
                    ax.scatter(q, v, color=c, label='Data Historis')
                elif i == 8:
                    ax.scatter(q, v, color=c, label='Lonjakan')
                elif i == 9:
                    ax.scatter(q, v, color=c, label='Peramalan')
                else:
                    ax.scatter(q, v, color=c)
            
            # Garis untuk menghubungkan titik-titik
            ax.plot(all_quarters[:9], all_values[:9], 'b-')
            ax.plot(all_quarters[8:], all_values[8:], 'g--')
            
            # Tambahkan garis vertikal untuk memisahkan data historis dan peramalan
            ax.axvline(x=8.5, color='black', linestyle='--', alpha=0.7)
            
            ax.set_title('Data Historis dan Peramalan', fontweight='bold')
            ax.set_ylabel('Permintaan')
            ax.set_xticks(range(len(all_quarters)))
            ax.set_xticklabels(all_quarters, rotation=45)
            ax.legend()
            
            st.pyplot(fig)
            
            # Interpretasi dengan API
            st.subheader("Interpretasi Hasil Metode Winter")
            with st.container():
                if model:
                    st.markdown(interpretasi_winter_method(model, hasil))
                else:
                    st.warning("API Key belum dimasukkan atau gagal terhubung.")

# Masalah 3: Perencanaan Agregat
# Masalah 3: Perencanaan Agregat
def selesaikan_masalah_3():
    st.header("Masalah 3: Perencanaan Agregat")
    
    # Fungsi interpretasi khusus untuk Perencanaan Agregat
    def interpretasi_perencanaan_agregat(model, hasil):
        if not model:
            return "Interpretasi AI tidak tersedia."
            
        prompt = f"""
        Saya telah melakukan perencanaan agregat untuk PT LScream menggunakan mixed strategy.
        
        Hasil analisis:
        - Total biaya: {hasil['total_biaya']} juta Rupiah
        - Rincian produksi: {hasil['rincian_produksi']}
        - Persediaan: {hasil['persediaan']}
        - Biaya per komponen: {hasil['biaya_komponen']}
        
        Berikan interpretasi yang ringkas dan profesional dalam 3-5 poin utama dalam bahasa Indonesia.
        Fokus pada:
        - Strategi produksi yang direkomendasikan
        - Trade-off antara komponen biaya
        - Implikasi terhadap pengelolaan tenaga kerja dan persediaan
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error menghasilkan interpretasi Perencanaan Agregat: {e}")
            return f"Error menghasilkan interpretasi: {e}"
    
    # Rumus matematika dan model perencanaan agregat
    st.subheader("Model Perencanaan Agregat dengan Mixed Strategy")
    st.write("Mixed strategy dalam perencanaan agregat menggabungkan beberapa strategi untuk meminimalkan total biaya:")
    
    st.latex(r"Minimize\ Z = \sum_{t=1}^{T}(C_r \cdot R_t + C_o \cdot O_t + C_i \cdot I_t + C_{ru} \cdot RU_t + C_{rd} \cdot RD_t + C_s \cdot S_t)")
    
    st.write("dengan batasan:")
    st.latex(r"R_t + O_t + I_{t-1} - I_t = D_t \quad \forall t=1,...,T")
    st.latex(r"R_t \leq M \cdot WF_t \quad \forall t=1,...,T")
    st.latex(r"O_t \leq N \cdot WF_t \quad \forall t=1,...,T")
    st.latex(r"RU_t \geq WF_t - WF_{t-1} \quad \forall t=1,...,T")
    st.latex(r"RD_t \geq WF_{t-1} - WF_t \quad \forall t=1,...,T")
    
    with st.expander("Data Permintaan & Parameter", expanded=True):
        # Tabel permintaan default
        permintaan_default = {
            'Minggu': [1, 2, 3, 4, 5, 6],
            'Vanila': [180, 120, 210, 180, 95, 140],
            'Coklat': [180, 110, 200, 420, 110, 90],
            'Stroberi': [90, 70, 190, 100, 95, 70]
        }
        
        # Tampilkan tabel permintaan dengan kemungkinan untuk diedit
        st.subheader("Data Permintaan (Kiloliter)")
        edited_df = st.data_editor(pd.DataFrame(permintaan_default), use_container_width=True)
        
        # Parameter filling rate
        st.subheader("Filling Rates (kiloliter/jam)")
        col1, col2, col3 = st.columns(3)
        fill_vanila = col1.number_input("Filling Rate Vanila", value=7.5, step=0.1)
        fill_coklat = col2.number_input("Filling Rate Coklat", value=8.1, step=0.1)
        fill_stroberi = col3.number_input("Filling Rate Stroberi", value=5.2, step=0.1)
        fill_agregat = st.number_input("Filling Rate Agregat", value=9.0, step=0.1)
        
        # Parameter biaya
        st.subheader("Parameter Biaya (Rp juta)")
        col1, col2 = st.columns(2)
        
        with col1:
            biaya_prod = st.number_input("Biaya Produksi", value=15.0, step=1.0)
            inv_cost = st.number_input("Inventory Cost (per jam per minggu)", value=2.0, step=0.1)
            reg_cost = st.number_input("Regular Time Cost (per jam)", value=6.0, step=0.1)
            ot_cost = st.number_input("Overtime Cost (per jam)", value=10.0, step=0.1)
        
        with col2:
            ramp_up = st.number_input("Ramping Up Cost (per jam)", value=5.0, step=0.1)
            ramp_down = st.number_input("Ramping Down Cost (per jam)", value=8.0, step=0.1)
            setup_vanila = st.number_input("Setup Cost Vanila", value=10.0, step=1.0)
            setup_coklat = st.number_input("Setup Cost Coklat", value=15.0, step=1.0)
            setup_stroberi = st.number_input("Setup Cost Stroberi", value=20.0, step=1.0)
        
        # Jam kerja
        st.subheader("Jam Kerja")
        col1, col2, col3 = st.columns(3)
        reg_hours = col1.number_input("Jam Kerja Regular (jam/hari)", value=8, step=1)
        ot_hours = col2.number_input("Jam Kerja Overtime (jam/hari)", value=2, step=1)
        work_days = col3.number_input("Hari Kerja per Minggu", value=5, step=1)
        
        # Pilihan metode
        solution_method = st.radio(
            "Pilih Metode Penyelesaian:",
            ["Basic (Trial and Error)", "Advanced (Linear Programming)"]
        )
    
    # Tombol untuk menjalankan perhitungan
    if st.button("Selesaikan Masalah 3"):
        with st.spinner("Menghitung perencanaan agregat..."):
            # Fungsi perencanaan agregat metode advanced (Linear Programming)
            def perencanaan_agregat_advanced():
                try:
                    import pulp as pl
                except ImportError:
                    st.error("Library PuLP tidak ditemukan. Silakan install dengan 'pip install pulp' atau gunakan metode Basic.")
                    return None
                
                # Ambil data dari editor
                permintaan_vanila = edited_df['Vanila'].tolist()
                permintaan_coklat = edited_df['Coklat'].tolist()
                permintaan_stroberi = edited_df['Stroberi'].tolist()
                
                # Hitung jam kerja tersedia per minggu
                regular_hours_per_week = reg_hours * work_days
                overtime_hours_per_week = ot_hours * work_days
                
                # Konversi permintaan ke jam produksi
                jam_vanila = [d / fill_vanila for d in permintaan_vanila]
                jam_coklat = [d / fill_coklat for d in permintaan_coklat]
                jam_stroberi = [d / fill_stroberi for d in permintaan_stroberi]
                
                # Total jam produksi yang dibutuhkan per minggu
                total_jam = [v + c + s for v, c, s in zip(jam_vanila, jam_coklat, jam_stroberi)]
                
                # Paramater model
                weeks = range(len(total_jam))
                
                # Buat model LP
                model = pl.LpProblem("Aggregate_Planning", pl.LpMinimize)
                
                # Variabel keputusan
                reg_prod = {w: pl.LpVariable(f"RegularProduction_{w}", lowBound=0) for w in weeks}
                ot_prod = {w: pl.LpVariable(f"OvertimeProduction_{w}", lowBound=0) for w in weeks}
                inventory = {w: pl.LpVariable(f"Inventory_{w}", lowBound=0) for w in weeks}
                backlog = {w: pl.LpVariable(f"Backlog_{w}", lowBound=0) for w in weeks}
                
                # Variabel untuk ramping
                ramp_up_var = {w: pl.LpVariable(f"RampUp_{w}", lowBound=0) for w in weeks[1:]}
                ramp_down_var = {w: pl.LpVariable(f"RampDown_{w}", lowBound=0) for w in weeks[1:]}
                
                # Variabel setup (biner)
                setup_v = {w: pl.LpVariable(f"SetupVanilla_{w}", cat=pl.LpBinary) for w in weeks}
                setup_c = {w: pl.LpVariable(f"SetupChocolate_{w}", cat=pl.LpBinary) for w in weeks}
                setup_s = {w: pl.LpVariable(f"SetupStrawberry_{w}", cat=pl.LpBinary) for w in weeks}
                
                # Fungsi tujuan: minimasi total biaya
                model += (
                    pl.lpSum([reg_prod[w] * reg_cost for w in weeks]) +  # Regular production cost
                    pl.lpSum([ot_prod[w] * ot_cost for w in weeks]) +    # Overtime production cost
                    pl.lpSum([inventory[w] * inv_cost for w in weeks]) + # Inventory holding cost
                    pl.lpSum([ramp_up_var[w] * ramp_up for w in weeks[1:]]) +  # Ramping up cost
                    pl.lpSum([ramp_down_var[w] * ramp_down for w in weeks[1:]]) +  # Ramping down cost
                    pl.lpSum([setup_v[w] * setup_vanila for w in weeks]) +  # Setup cost vanilla
                    pl.lpSum([setup_c[w] * setup_coklat for w in weeks]) +  # Setup cost chocolate
                    pl.lpSum([setup_s[w] * setup_stroberi for w in weeks])  # Setup cost strawberry
                )
                
                # Batasan kapasitas
                for w in weeks:
                    model += reg_prod[w] <= regular_hours_per_week  # Regular capacity constraint
                    model += ot_prod[w] <= overtime_hours_per_week  # Overtime capacity constraint
                
                # Batasan keseimbangan inventori
                for w in weeks:
                    if w == 0:
                        # Minggu pertama tidak ada inventori dan backlog awal
                        model += reg_prod[w] + ot_prod[w] - inventory[w] + backlog[w] == total_jam[w]
                    else:
                        # Minggu berikutnya memperhitungkan inventori dan backlog dari minggu sebelumnya
                        model += reg_prod[w] + ot_prod[w] + inventory[w-1] - inventory[w] - backlog[w-1] + backlog[w] == total_jam[w]
                
                # Batasan ramping up/down
                for w in weeks[1:]:
                    model += ramp_up_var[w] - ramp_down_var[w] == reg_prod[w] - reg_prod[w-1]  # Definisi ramping
                
                # Batasan setup - M adalah angka yang cukup besar (Big-M method)
                M = max(jam_vanila + jam_coklat + jam_stroberi) * 10
                for w in weeks:
                    if jam_vanila[w] > 0:
                        model += jam_vanila[w] <= M * setup_v[w]  # Jika ada produksi vanilla, setup = 1
                    if jam_coklat[w] > 0:
                        model += jam_coklat[w] <= M * setup_c[w]  # Jika ada produksi coklat, setup = 1
                    if jam_stroberi[w] > 0:
                        model += jam_stroberi[w] <= M * setup_s[w]  # Jika ada produksi stroberi, setup = 1
                
                # Selesaikan model
                model.solve(pl.PULP_CBC_CMD(msg=False))
                
                if pl.LpStatus[model.status] != 'Optimal':
                    st.warning(f"Solver tidak menemukan solusi optimal. Status: {pl.LpStatus[model.status]}")
                    return None
                
                # Ekstrak hasil
                regular_used = [pl.value(reg_prod[w]) for w in weeks]
                overtime_used = [pl.value(ot_prod[w]) for w in weeks]
                inventory_levels = [pl.value(inventory[w]) for w in weeks]
                backlog_levels = [pl.value(backlog[w]) for w in weeks]
                
                # Hitung komponen biaya
                regular_cost = sum(regular_used) * reg_cost
                overtime_cost = sum(overtime_used) * ot_cost
                inventory_cost = sum(inventory_levels) * inv_cost
                
                ramping_cost = sum(pl.value(ramp_up_var[w]) * ramp_up + 
                                pl.value(ramp_down_var[w]) * ramp_down 
                                for w in weeks[1:])
                    
                setup_cost = sum(pl.value(setup_v[w]) * setup_vanila +
                                pl.value(setup_c[w]) * setup_coklat +
                                pl.value(setup_s[w]) * setup_stroberi 
                                for w in weeks)
                
                # Total biaya
                total_biaya = regular_cost + overtime_cost + inventory_cost + ramping_cost + setup_cost
                
                return {
                    'total_biaya': total_biaya,
                    'rincian_produksi': {
                        'regular_hours': regular_used,
                        'overtime_hours': overtime_used,
                        'total_required_hours': total_jam
                    },
                    'persediaan': inventory_levels,
                    'backlog': backlog_levels,
                    'biaya_komponen': {
                        'regular_cost': regular_cost,
                        'overtime_cost': overtime_cost,
                        'inventory_cost': inventory_cost,
                        'ramping_cost': ramping_cost,
                        'setup_cost': setup_cost
                    },
                    'status': pl.LpStatus[model.status]
                }
            
            # Fungsi perencanaan agregat metode basic (Trial and Error)
            def perencanaan_agregat_basic():
                # Ambil data dari editor
                permintaan_vanila = edited_df['Vanila'].tolist()
                permintaan_coklat = edited_df['Coklat'].tolist()
                permintaan_stroberi = edited_df['Stroberi'].tolist()
                
                # Hitung jam kerja tersedia per minggu
                regular_hours_per_week = reg_hours * work_days
                overtime_hours_per_week = ot_hours * work_days
                total_hours_per_week = regular_hours_per_week + overtime_hours_per_week
                
                # Konversi permintaan ke jam produksi
                jam_vanila = [d / fill_vanila for d in permintaan_vanila]
                jam_coklat = [d / fill_coklat for d in permintaan_coklat]
                jam_stroberi = [d / fill_stroberi for d in permintaan_stroberi]
                
                # Total jam produksi yang dibutuhkan per minggu
                total_jam = [v + c + s for v, c, s in zip(jam_vanila, jam_coklat, jam_stroberi)]
                
                # Inisialisasi variabel tracking
                regular_used = []
                overtime_used = []
                inventory = [0]  # Mulai dengan persediaan 0
                backlog = [0]    # Backlog awal 0
                
                # Strategi mixed - pendekatan yang ditingkatkan
                for i, jam in enumerate(total_jam):
                    # Jam yang tersedia di minggu ini
                    available_reg = regular_hours_per_week
                    available_ot = overtime_hours_per_week
                    
                    # Inventori dari minggu sebelumnya
                    prev_inv = inventory[-1]
                    prev_backlog = backlog[-1]
                    
                    # Kebutuhan produksi setelah memperhitungkan inventori dan backlog
                    net_req = jam + prev_backlog - prev_inv
                    
                    # Perbaikan strategi mixed untuk lebih mengoptimalkan biaya
                    # Kita akan coba lebih "look-ahead" dengan melihat kebutuhan 2 minggu ke depan
                    # Jika minggu depan kebutuhan lebih tinggi, kita akan produksi lebih banyak sekarang
                    # untuk mengurangi overtime di masa depan
                    if i < len(total_jam) - 1 and total_jam[i+1] > total_jam[i] and net_req < available_reg:
                        # Ada kapasitas ekstra dan minggu depan lebih tinggi
                        future_need = min(total_jam[i+1], available_reg - net_req)
                        extra_prod = min(future_need, available_reg - net_req)
                        
                        # Tapi jangan terlalu banyak extra jika biaya inventori mahal
                        if inv_cost < ot_cost * 0.5:  # Heuristik: Jika inv cost < 50% overtime cost
                            extra_prod = max(0, extra_prod)
                        else:
                            extra_prod = 0
                    else:
                        extra_prod = 0
                    
                    if net_req <= 0:
                        # Persediaan berlebih, tidak perlu produksi
                        reg_used = 0
                        ot_used = 0
                        curr_inv = abs(net_req)
                        curr_backlog = 0
                    elif net_req <= available_reg:
                        # Cukup dengan regular time + mungkin extra untuk future
                        reg_used = net_req + extra_prod
                        ot_used = 0
                        curr_inv = extra_prod
                        curr_backlog = 0
                    elif net_req <= available_reg + available_ot:
                        # Perlu regular + sebagian overtime
                        reg_used = available_reg
                        ot_used = net_req - available_reg
                        curr_inv = 0
                        curr_backlog = 0
                    else:
                        # Gunakan semua kapasitas + backlog
                        reg_used = available_reg
                        ot_used = available_ot
                        curr_inv = 0
                        curr_backlog = net_req - (available_reg + available_ot)
                    
                    regular_used.append(reg_used)
                    overtime_used.append(ot_used)
                    inventory.append(curr_inv)
                    backlog.append(curr_backlog)
                
                # Hapus inventori dan backlog awal
                inventory = inventory[1:]
                backlog = backlog[1:]
                
                # Hitung biaya-biaya
                # 1. Biaya regular time
                regular_cost = sum(regular_used) * reg_cost
                
                # 2. Biaya overtime
                overtime_cost = sum(overtime_used) * ot_cost
                
                # 3. Biaya inventory
                inventory_cost = sum(inventory) * inv_cost
                
                # 4. Biaya ramping (perubahan level produksi)
                ramping_cost = 0
                for i in range(1, len(regular_used)):
                    if regular_used[i] > regular_used[i-1]:
                        # Ramping up
                        ramping_cost += (regular_used[i] - regular_used[i-1]) * ramp_up
                    else:
                        # Ramping down
                        ramping_cost += (regular_used[i-1] - regular_used[i]) * ramp_down
                
                # 5. Biaya setup (diasumsikan setup sekali per minggu untuk tiap jenis)
                # Jika ada produksi minimal 1 jam, ada setup
                setup_cost = 0
                for i in range(len(permintaan_vanila)):
                    if jam_vanila[i] > 0:
                        setup_cost += setup_vanila
                    if jam_coklat[i] > 0:
                        setup_cost += setup_coklat
                    if jam_stroberi[i] > 0:
                        setup_cost += setup_stroberi
                
                # Total biaya
                total_biaya = regular_cost + overtime_cost + inventory_cost + ramping_cost + setup_cost
                
                return {
                    'total_biaya': total_biaya,
                    'rincian_produksi': {
                        'regular_hours': regular_used,
                        'overtime_hours': overtime_used,
                        'total_required_hours': total_jam
                    },
                    'persediaan': inventory,
                    'backlog': backlog,
                    'biaya_komponen': {
                        'regular_cost': regular_cost,
                        'overtime_cost': overtime_cost,
                        'inventory_cost': inventory_cost,
                        'ramping_cost': ramping_cost,
                        'setup_cost': setup_cost
                    }
                }
            
            # Pilih metode yang akan digunakan
            if solution_method == "Advanced (Linear Programming)":
                try:
                    import pulp
                    hasil = perencanaan_agregat_advanced()
                    if hasil is None:  # Jika linear programming gagal
                        st.warning("Metode Linear Programming gagal. Menggunakan metode Trial and Error sebagai fallback.")
                        hasil = perencanaan_agregat_basic()
                except ImportError:
                    st.warning("Library PuLP tidak ditemukan. Menggunakan metode Trial and Error sebagai fallback.")
                    hasil = perencanaan_agregat_basic()
            else:
                hasil = perencanaan_agregat_basic()
            
            # Tampilkan hasil
            st.subheader("Hasil Perencanaan Agregat")
            
            # Metrik ringkasan
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Biaya", f"Rp {hasil['total_biaya']:.2f} Juta")
            col2.metric("Total Jam Regular", f"{sum(hasil['rincian_produksi']['regular_hours']):.2f} jam")
            col3.metric("Total Jam Overtime", f"{sum(hasil['rincian_produksi']['overtime_hours']):.2f} jam")
            
            # Tabel perencanaan detail
            st.subheader("Rencana Produksi Mingguan")
            df_rencana = pd.DataFrame({
                'Minggu': range(1, 7),
                'Permintaan Total (kl)': [sum(x) for x in zip(edited_df['Vanila'], edited_df['Coklat'], edited_df['Stroberi'])],
                'Kebutuhan Jam': hasil['rincian_produksi']['total_required_hours'],
                'Jam Regular': hasil['rincian_produksi']['regular_hours'],
                'Jam Overtime': hasil['rincian_produksi']['overtime_hours'],
                'Inventori (jam)': hasil['persediaan'],
                'Backlog (jam)': hasil['backlog']
            })
            st.dataframe(df_rencana)
            
            # Visualisasi permintaan vs kapasitas
            st.subheader("Permintaan vs Kapasitas")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            weeks = range(1, 7)
            capacity_reg = [reg_hours * work_days] * 6
            capacity_total = [reg_hours * work_days + ot_hours * work_days] * 6
            
            ax.bar(weeks, hasil['rincian_produksi']['total_required_hours'], color='blue', alpha=0.7, label='Kebutuhan Jam')
            ax.plot(weeks, capacity_reg, 'r--', linewidth=2, label=f'Kapasitas Regular ({reg_hours * work_days} jam)')
            ax.plot(weeks, capacity_total, 'g--', linewidth=2, label=f'Kapasitas Total ({reg_hours * work_days + ot_hours * work_days} jam)')
            
            ax.set_xlabel('Minggu')
            ax.set_ylabel('Jam')
            ax.set_title('Permintaan vs Kapasitas Produksi')
            ax.legend()
            ax.grid(alpha=0.3)
            
            st.pyplot(fig)
            
            # Visualisasi rencana produksi aktual
            st.subheader("Rencana Produksi Aktual")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot kebutuhan jam
            ax.bar(weeks, hasil['rincian_produksi']['total_required_hours'], color='blue', alpha=0.3, label='Kebutuhan Total')
            
            # Plot jam regular
            ax.bar(weeks, hasil['rincian_produksi']['regular_hours'], color='green', alpha=0.7, label='Regular Time')
            
            # Plot jam overtime (di atas regular)
            bottom_ot = hasil['rincian_produksi']['regular_hours']
            ax.bar(weeks, hasil['rincian_produksi']['overtime_hours'], bottom=bottom_ot, color='orange', alpha=0.7, label='Overtime')
            
            # Plot inventori sebagai garis
            ax_inv = ax.twinx()
            ax_inv.plot(weeks, hasil['persediaan'], 'r-o', linewidth=2, label='Inventori')
            ax_inv.set_ylabel('Inventori (jam)', color='r')
            
            ax.set_xlabel('Minggu')
            ax.set_ylabel('Jam Produksi')
            ax.set_title('Rencana Produksi Aktual per Minggu')
            
            # Gabungkan legend dari kedua axis
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_inv.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Visualisasi komponen biaya
            st.subheader("Komponen Biaya")
            biaya_labels = ['Regular', 'Overtime', 'Inventory', 'Ramping', 'Setup']
            biaya_values = [
                hasil['biaya_komponen']['regular_cost'], 
                hasil['biaya_komponen']['overtime_cost'],
                hasil['biaya_komponen']['inventory_cost'],
                hasil['biaya_komponen']['ramping_cost'],
                hasil['biaya_komponen']['setup_cost']
            ]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(biaya_labels, biaya_values, color=['blue', 'orange', 'green', 'red', 'purple'])
            
            # Tambahkan label nilai di atas setiap batang
            for i, v in enumerate(biaya_values):
                ax.text(i, v + 1, f'{v:.1f}', ha='center')
                
            ax.set_ylabel('Biaya (Rp Juta)')
            ax.set_title('Komponen Biaya Perencanaan Agregat')
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            
            # Pie chart komposisi biaya
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(biaya_values, labels=biaya_labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('Proporsi Komponen Biaya')
            
            st.pyplot(fig)
            
            # Dashboard Ringkasan
            st.subheader("Dashboard Ringkasan Hasil")
            col1, col2 = st.columns(2)
            
            with col1:
                # Metrik-metrik utama
                st.metric("Utilisasi Kapasitas Regular", 
                          f"{sum(hasil['rincian_produksi']['regular_hours']) / (reg_hours * work_days * 6) * 100:.1f}%")
                st.metric("Utilisasi Kapasitas Total", 
                          f"{(sum(hasil['rincian_produksi']['regular_hours']) + sum(hasil['rincian_produksi']['overtime_hours'])) / ((reg_hours + ot_hours) * work_days * 6) * 100:.1f}%")
                st.metric("Rata-rata Inventori", f"{sum(hasil['persediaan']) / len(hasil['persediaan']):.2f} jam")
            
            with col2:
                # Metrik lain
                total_produksi = sum([sum(x) for x in zip(edited_df['Vanila'], edited_df['Coklat'], edited_df['Stroberi'])])
                st.metric("Biaya per Unit", f"Rp {hasil['total_biaya'] / total_produksi:.2f} Juta/kl")
                st.metric("Backlog Maksimum", f"{max(hasil['backlog']):.2f} jam")
                st.metric("Minggu dengan Kebutuhan Tertinggi", 
                          f"Minggu {hasil['rincian_produksi']['total_required_hours'].index(max(hasil['rincian_produksi']['total_required_hours']))+1} ({max(hasil['rincian_produksi']['total_required_hours']):.2f} jam)")
            
            # Interpretasi dengan API
            st.subheader("Interpretasi Perencanaan Agregat")
            with st.container():
                if model:
                    st.markdown(interpretasi_perencanaan_agregat(model, hasil))
                else:
                    st.warning("API Key belum dimasukkan atau gagal terhubung.")
            
            # Perbedaan antara basic dan advanced (jika menggunakan advanced)
            if solution_method == "Advanced (Linear Programming)":
                st.subheader("Mengapa Linear Programming Lebih Baik?")
                st.write("""
                Linear Programming memberikan solusi yang optimal secara global dengan mempertimbangkan seluruh horizon perencanaan sekaligus. 
                Beberapa keunggulan termasuk:
                
                1. **Optimasi Global**: Mempertimbangkan semua periode sekaligus, bukan hanya keputusan lokal per minggu
                2. **Keseimbangan Biaya**: Secara matematis menyeimbangkan semua komponen biaya (produksi, inventori, setup, ramping)
                3. **Look-ahead Capability**: Secara otomatis memperhitungkan permintaan masa depan dalam keputusan saat ini
                4. **Batching Optimal**: Menentukan kapan batching produksi (memproduksi lebih untuk inventori) lebih ekonomis
                
                Metode ini umumnya menghasilkan penghematan 5-15% dibandingkan pendekatan heuristik seperti trial-and-error.
                """)
# Masalah 4: Jadwal Produksi Induk
# Masalah 4: Jadwal Produksi Induk (MPS) dengan Metode Bitran Hax
# Masalah 4: Jadwal Produksi Induk (MPS) dengan Metode Bitran Hax
# Terintegrasi Winter's Method dengan MPS
def selesaikan_masalah_4():
    st.header("Masalah 4: Jadwal Produksi Induk (MPS) dengan Winter's Method")
    
    # Pilihan metode
    metode_options = ["Bitran Hax Standard", "Winter-MPS Terintegrasi", "Hybrid Approach"]
    metode_selected = st.radio("Pilih Metode MPS:", metode_options)
    
    if metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"]:
        st.info(f"Menggunakan pendekatan {metode_selected} yang menggabungkan peramalan Winter's Method dengan MPS")
    
    # Tambahkan rumus dan penjelasan
    with st.expander("Penjelasan Metode", expanded=True):
        st.subheader("Metode Bitran Hax Standard")
        st.write("Metode Bitran Hax adalah pendekatan dua tahap untuk membuat jadwal produksi induk:")
        
        st.latex(r"\min \sum_{i \in F} C_i Y_i")
        st.latex(r"\text{s.t. } \sum_{i \in F} P_i \leq P^T")
        st.latex(r"L_i \leq P_i \leq U_i, \forall i \in F")
        st.latex(r"Y_i = \begin{cases} 1 & \text{jika } P_i > 0 \\ 0 & \text{jika } P_i = 0 \end{cases}")
        
        st.subheader("Winter's Method (Triple Exponential Smoothing)")
        st.write("Winter's Method menggunakan tiga persamaan penghalusan untuk meramalkan permintaan dengan pola musiman:")
        
        # Level
        st.latex(r"S_t = \alpha \frac{Y_t}{I_{t-L}} + (1-\alpha)(S_{t-1} + T_{t-1})")
        # Trend
        st.latex(r"T_t = \beta(S_t - S_{t-1}) + (1-\beta)T_{t-1}")
        # Seasonal
        st.latex(r"I_t = \gamma\frac{Y_t}{S_t} + (1-\gamma)I_{t-L}")
        # Forecast
        st.latex(r"F_{t+m} = (S_t + mT_t)I_{t-L+m}")
        
        st.subheader("Pendekatan Winter-MPS Terintegrasi")
        st.write("""
        Pendekatan Winter-MPS Terintegrasi mengintegrasikan Winter's Method langsung ke dalam proses MPS dengan cara:
        
        1. **Peramalan Multi-Periode**: Menggunakan Winter's Method untuk meramalkan permintaan selama horizon perencanaan MPS.
        
        2. **Perhitungan Dynamic Safety Stock**: Safety stock disesuaikan berdasarkan komponen musiman (seasonal) dan variabilitas dari Winter's Method.
        
        3. **Kapasitas Adaptif**: Alokasi kapasitas produksi dimodifikasi berdasarkan ekspektasi musiman.
        
        4. **Net Requirement dengan Seasonal Adjustment**: Perhitungan kebutuhan bersih (Net Requirement) memperhitungkan indeks musiman.
        """)
        
        st.subheader("Perhitungan Dynamic Safety Stock")
        st.latex(r"SS_i = z \times \sigma_i \times \sqrt{L} \times \frac{I_{t+L}}{I_{avg}}")
        st.write("dimana:")
        st.write("- $SS_i$ = Safety stock untuk item $i$")
        st.write("- $z$ = Faktor service level (umumnya 1.65 untuk 95% service level)")
        st.write("- $\\sigma_i$ = Standar deviasi permintaan item $i$")
        st.write("- $L$ = Lead time")
        st.write("- $I_{t+L}$ = Indeks musiman saat lead time")
        st.write("- $I_{avg}$ = Rata-rata indeks musiman")
    
    # Parameter Winter's Method
    if metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"]:
        st.subheader("Parameter Winter's Method")
        col1, col2, col3 = st.columns(3)
        alpha_w = col1.slider("α (Level)", 0.0, 1.0, 0.25, 0.01)
        beta_w = col2.slider("β (Trend)", 0.0, 1.0, 0.12, 0.01)
        gamma_w = col3.slider("γ (Seasonal)", 0.0, 1.0, 0.08, 0.01)
        
        col1, col2 = st.columns(2)
        season_length = col1.number_input("Panjang Musiman", value=3, min_value=2, max_value=12, step=1)
        confidence_level = col2.slider("Service Level (%)", 90.0, 99.9, 95.0, 0.1)
        
        # Convert confidence level to z-score (common approximation)
        z_score = 1.65  # Default for 95%
        if confidence_level >= 99.0:
            z_score = 2.33
        elif confidence_level >= 98.0:
            z_score = 2.05
        elif confidence_level >= 97.0:
            z_score = 1.88
        elif confidence_level >= 96.0:
            z_score = 1.75
        elif confidence_level >= 95.0:
            z_score = 1.65
        elif confidence_level >= 90.0:
            z_score = 1.28
    
    # Input data
    with st.expander("Data Produksi dan Permintaan", expanded=True):
        # Input rencana produksi agregat
        st.subheader("Rencana Produksi Agregat 2025 (ribu unit)")
        col1, col2, col3, col4 = st.columns(4)
        prod_jan = col1.number_input("Januari", value=98.3, step=0.1)
        prod_feb = col2.number_input("Februari", value=97.0, step=0.1)
        prod_mar = col3.number_input("Maret", value=105.3, step=0.1)
        prod_apr = col4.number_input("April", value=114.1, step=0.1)
        
        # Data historis untuk Winter's Method
        if metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"]:
            st.subheader("Data Historis untuk Peramalan (ribu unit)")
            col1, col2, col3 = st.columns(3)
            hist_oct = col1.number_input("Oktober 2024", value=90.5, step=0.1)
            hist_nov = col2.number_input("November 2024", value=95.2, step=0.1)
            hist_dec = col3.number_input("Desember 2024", value=102.1, step=0.1)
        
        # Data default untuk faktor konversi dan safety stock
        data_produk_default = {
            'Famili': ['Roti Hamburger', 'Roti Hamburger', 'Roti Hamburger', 
                      'Roti Hotdog', 'Roti Hotdog',
                      'Roti Bagel', 'Roti Bagel'],
            'Item': ['White', 'Wijen', 'WholeGrain', 'Plain', 'Garlic', 'Savory', 'Sweet'],
            'Konversi': [0.2, 0.3, 0.4, 0.15, 0.2, 0.1, 0.15],
            'Safety Stock': [120, 60, 45, 90, 80, 75, 50],
            'StdDev': [15, 12, 8, 14, 10, 9, 11]  # Standar deviasi permintaan untuk perhitungan safety stock
        }
        
        # Tabel kebijakan produksi
        st.subheader("Kebijakan Produksi")
        df_kebijakan = pd.DataFrame(data_produk_default)
        edited_kebijakan = st.data_editor(df_kebijakan, use_container_width=True)
        
        # Data perkiraan permintaan
        data_permintaan_default = {
            'Item': ['White', 'Wijen', 'WholeGrain', 'Plain', 'Garlic', 'Savory', 'Sweet'],
            'Inventory Des': [335.2, 148.7, 57.5, 280.4, 96.2, 101, 121],
            'Jan': [210.6, 84.0, 15.4, 175.2, 27.2, 20.7, 65.3],
            'Feb': [172.0, 74.8, 16.0, 198.0, 17.6, 22.8, 71.8],
            'Mar': [255.2, 88.8, 16.8, 178.0, 22.0, 29.6, 93.4],
            'Apr': [360.0, 57.6, 19.2, 128.0, 36.0, 44.4, 140.1]
        }
        
        # Data historis untuk item (untuk peramalan)
        if metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"]:
            hist_permintaan_default = {
                'Item': ['White', 'Wijen', 'WholeGrain', 'Plain', 'Garlic', 'Savory', 'Sweet'],
                'Oct': [190.2, 75.6, 14.0, 160.7, 25.0, 18.5, 60.1],
                'Nov': [200.5, 80.1, 14.8, 168.3, 26.3, 19.6, 62.8],
                'Dec': [205.8, 82.3, 15.1, 172.4, 26.8, 20.2, 64.0]
            }
            st.subheader("Data Historis Permintaan per Item (ribu unit)")
            df_hist_permintaan = pd.DataFrame(hist_permintaan_default)
            edited_hist_permintaan = st.data_editor(df_hist_permintaan, use_container_width=True)
        
        # Tabel perkiraan permintaan
        st.subheader("Perkiraan Permintaan (ribu unit)")
        df_permintaan = pd.DataFrame(data_permintaan_default)
        edited_permintaan = st.data_editor(df_permintaan, use_container_width=True)
        
        # Biaya setup
        st.subheader("Biaya Setup (Miliar Rupiah)")
        col1, col2, col3 = st.columns(3)
        setup_hamburger = col1.number_input("Roti Hamburger", value=9, step=1)
        setup_hotdog = col2.number_input("Roti Hotdog", value=4, step=1)
        setup_bagel = col3.number_input("Roti Bagel", value=6, step=1)
        
        # Parameter batas atas
        st.subheader("Parameter Batas Atas")
        n_param = st.number_input("Nilai N untuk batas atas", value=2, min_value=1, max_value=4, step=1)
    
    # Tombol untuk menjalankan perhitungan
    if st.button("Selesaikan Masalah 4"):
        with st.spinner("Menghitung Jadwal Produksi Induk..."):
            # 1. Mengumpulkan data dari input
            items = edited_kebijakan['Item'].tolist()
            families = edited_kebijakan['Famili'].tolist()
            konversi = edited_kebijakan['Konversi'].tolist()
            safety_stock_original = edited_kebijakan['Safety Stock'].tolist()
            std_dev = edited_kebijakan['StdDev'].tolist()
            
            # Inventori awal dan permintaan
            inv_awal = edited_permintaan['Inventory Des'].tolist()
            permintaan_jan_original = edited_permintaan['Jan'].tolist()
            permintaan_feb = edited_permintaan['Feb'].tolist()
            permintaan_mar = edited_permintaan['Mar'].tolist()
            
            # Buat salinan permintaan yang dapat dimodifikasi
            permintaan_jan = permintaan_jan_original.copy()
            safety_stock = safety_stock_original.copy()
            
            # Proses Winter's Method jika diperlukan
            seasonal_indices = None
            forecasted_permintaan = None
            dynamic_safety_stock = None
            
            if metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"]:
                st.subheader("Hasil Peramalan Winter's Method")
                
                # Implementasi Winter's method
                def winters_method(series, alpha, beta, gamma, season_length, forecast_horizon=4):
                    """
                    Implementasi Triple Exponential Smoothing (Winter's Method)
                    """
                    n = len(series)
                    if n <= 2 * season_length:
                        return series[-1] * np.ones(forecast_horizon), None, None, None
                    
                    # Inisialisasi
                    result = np.zeros(n + forecast_horizon)
                    smooth = np.zeros(n)
                    trend = np.zeros(n)
                    seasonal = np.zeros(n + forecast_horizon)
                    
                    # Inisialisasi nilai awal
                    smooth[0] = series[0]
                    trend[0] = (series[season_length] - series[0]) / season_length
                    
                    # Inisialisasi seasonal indices
                    season_averages = [0] * season_length
                    for i in range(season_length):
                        for j in range(n // season_length):
                            if i + j * season_length < n:
                                season_averages[i] += series[i + j * season_length]
                        season_averages[i] /= (n // season_length)
                    
                    overall_average = sum(season_averages) / season_length
                    
                    for i in range(season_length):
                        seasonal[i] = season_averages[i] / overall_average if overall_average != 0 else 1.0
                    
                    # Peramalan untuk data historis
                    for i in range(1, n):
                        s_idx = i % season_length
                        if i - season_length >= 0:
                            smooth[i] = alpha * (series[i] / seasonal[i - season_length]) + (1 - alpha) * (smooth[i-1] + trend[i-1])
                            trend[i] = beta * (smooth[i] - smooth[i-1]) + (1 - beta) * trend[i-1]
                            seasonal[i] = gamma * (series[i] / smooth[i]) + (1 - gamma) * seasonal[i - season_length]
                            result[i] = (smooth[i-1] + trend[i-1]) * seasonal[i - season_length]
                        else:
                            smooth[i] = alpha * series[i] + (1 - alpha) * (smooth[i-1] + trend[i-1])
                            trend[i] = beta * (smooth[i] - smooth[i-1]) + (1 - beta) * trend[i-1]
                            seasonal[i] = seasonal[s_idx]  # Use initial seasonal indices
                            result[i] = smooth[i-1] + trend[i-1]
                    
                    # Peramalan untuk periode ke depan
                    for i in range(n, n + forecast_horizon):
                        m = i - n + 1
                        s_idx = (i % season_length)
                        if i - season_length >= 0:
                            seasonal[i] = seasonal[i - season_length]
                        else:
                            seasonal[i] = seasonal[s_idx]
                        result[i] = (smooth[n-1] + m * trend[n-1]) * seasonal[i]
                    
                    return result[n:n + forecast_horizon], smooth[n-1], trend[n-1], seasonal[n:n + forecast_horizon]
                
                # Peramalan untuk data agregat
                hist_agregat = [hist_oct, hist_nov, hist_dec, prod_jan]
                forecast_agregat, level_agregat, trend_agregat, seasonal_indices_agregat = winters_method(
                    np.array(hist_agregat), 
                    alpha_w, 
                    beta_w, 
                    gamma_w, 
                    int(season_length)
                )
                
                # Visualisasi peramalan agregat
                st.subheader("Peramalan Produksi Agregat")
                
                # Data historis dan peramalan
                periods = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
                historical_values = hist_agregat
                forecast_values = list(hist_agregat) + list(forecast_agregat)
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(periods[:4], historical_values, 'o-', color='blue', label='Data Historis')
                ax.plot(periods[3:], forecast_values[3:], 's--', color='red', label='Peramalan')
                
                # Tambahkan titik pertemuan
                ax.plot(periods[3], historical_values[3], 'o', color='purple', markersize=10, label='Titik Pertemuan')
                
                # Tambahkan label nilai
                for i, period in enumerate(periods[:4]):
                    ax.text(period, historical_values[i] + 2, f'{historical_values[i]:.1f}', ha='center')
                
                for i, period in enumerate(periods[3:]):
                    ax.text(period, forecast_values[i+3] + 2, f'{forecast_values[i+3]:.1f}', ha='center')
                
                ax.set_title('Peramalan Produksi Agregat dengan Winter\'s Method')
                ax.set_xlabel('Periode')
                ax.set_ylabel('Produksi (ribu unit)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # Penjelasan metrik peramalan
                st.subheader("Metrik Peramalan")
                col1, col2, col3 = st.columns(3)
                col1.metric("Level Akhir", f"{level_agregat:.2f}")
                col2.metric("Trend Akhir", f"{trend_agregat:.2f}")
                col3.metric("Indeks Musiman Feb", f"{seasonal_indices_agregat[0]:.2f}")
                
                # Peramalan untuk setiap item
                if "edited_hist_permintaan" in locals():
                    # Ambil data historis item
                    forecasted_permintaan = {}
                    seasonal_indices = {}
                    dynamic_safety_stock = {}
                    
                    for i, item in enumerate(items):
                        hist_item = [
                            edited_hist_permintaan.loc[edited_hist_permintaan['Item'] == item, 'Oct'].values[0],
                            edited_hist_permintaan.loc[edited_hist_permintaan['Item'] == item, 'Nov'].values[0],
                            edited_hist_permintaan.loc[edited_hist_permintaan['Item'] == item, 'Dec'].values[0],
                            edited_permintaan.loc[edited_permintaan['Item'] == item, 'Jan'].values[0]
                        ]
                        
                        # Jalankan Winter's method untuk item
                        forecast_item, level_item, trend_item, seasonal_item = winters_method(
                            np.array(hist_item), 
                            alpha_w, 
                            beta_w, 
                            gamma_w, 
                            int(season_length)
                        )
                        
                        forecasted_permintaan[item] = forecast_item
                        seasonal_indices[item] = seasonal_item
                        
                        # Hitung dynamic safety stock berdasarkan seasonal index
                        lead_time = 1  # Asumsi lead time 1 bulan
                        seasonal_factor = seasonal_item[0] / np.mean(seasonal_item)
                        dynamic_ss = z_score * std_dev[i] * np.sqrt(lead_time) * seasonal_factor
                        dynamic_safety_stock[item] = dynamic_ss
                    
                    # Tabel hasil peramalan item
                    forecast_data = []
                    for i, item in enumerate(items):
                        forecast_data.append({
                            'Item': item,
                            'Permintaan Asli Jan': permintaan_jan_original[i],
                            'Peramalan Jan': forecasted_permintaan[item][0],
                            'Peramalan Feb': forecasted_permintaan[item][1],
                            'Peramalan Mar': forecasted_permintaan[item][2],
                            'Safety Stock Asli': safety_stock_original[i],
                            'Safety Stock Dinamis': dynamic_safety_stock[item]
                        })
                    
                    forecast_df = pd.DataFrame(forecast_data)
                    st.dataframe(forecast_df)
                    
                    # Modifikasi permintaan dan safety stock berdasarkan pendekatan yang dipilih
                    if metode_selected == "Winter-MPS Terintegrasi":
                        # Gunakan hasil peramalan sebagai permintaan
                        for i, item in enumerate(items):
                            permintaan_jan[i] = forecasted_permintaan[item][0]
                            safety_stock[i] = dynamic_safety_stock[item]
                        
                        st.success("Menggunakan hasil peramalan Winter's Method untuk permintaan Januari dan dynamic safety stock")
                    
                    elif metode_selected == "Hybrid Approach":
                        # Gunakan rata-rata peramalan dan permintaan asli
                        for i, item in enumerate(items):
                            permintaan_jan[i] = (permintaan_jan_original[i] + forecasted_permintaan[item][0]) / 2
                            safety_stock[i] = (safety_stock_original[i] + dynamic_safety_stock[item]) / 2
                        
                        st.success("Menggunakan pendekatan hybrid (rata-rata peramalan dan permintaan asli)")
            
            # Biaya setup per famili
            setup_costs = {
                'Roti Hamburger': setup_hamburger,
                'Roti Hotdog': setup_hotdog,
                'Roti Bagel': setup_bagel
            }
            
            # 2. Implementasi Metode Bitran Hax
            # Langkah 1: Hitung batas atas dan bawah untuk setiap produk
            
            # Fungsi untuk menghitung net requirement
            def hitung_net_req(inv_awal, permintaan, ss):
                net_req = max(0, permintaan + ss - inv_awal)
                return net_req
            
            # Hitung batas bawah (NetReq untuk Jan)
            batas_bawah = []
            for i in range(len(items)):
                net_req_jan = hitung_net_req(inv_awal[i], permintaan_jan[i], safety_stock[i])
                
                # Jika menggunakan seasonal index dan terintegrasi
                if metode_selected == "Winter-MPS Terintegrasi" and seasonal_indices:
                    # Sesuaikan batas bawah berdasarkan seasonal index
                    seasonal_factor = seasonal_indices[items[i]][0]
                    net_req_jan = net_req_jan * seasonal_factor
                
                batas_bawah.append(net_req_jan)
            
            # Hitung batas atas (NetReq untuk Jan + Feb atau lebih)
            batas_atas = []
            for i in range(len(items)):
                # Batas atas dengan N=n_param bulan ke depan
                if metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"] and forecasted_permintaan:
                    # Gunakan hasil peramalan untuk batas atas
                    total_demand = 0
                    for j in range(min(n_param, 3)):  # Batasi pada 3 bulan ke depan
                        total_demand += forecasted_permintaan[items[i]][j]
                else:
                    total_demand = permintaan_jan[i]
                    
                    if n_param >= 2:
                        total_demand += permintaan_feb[i]
                        
                    if n_param >= 3:
                        total_demand += permintaan_mar[i]
                
                net_req_total = hitung_net_req(inv_awal[i], total_demand, safety_stock[i])
                batas_atas.append(net_req_total)
            
            # Langkah 2: Agregasi batas bawah dan atas per famili
            unique_families = list(set(families))
            family_lb = {fam: 0 for fam in unique_families}
            family_ub = {fam: 0 for fam in unique_families}
            
            for i, fam in enumerate(families):
                family_lb[fam] += batas_bawah[i]
                family_ub[fam] += batas_atas[i]
            
            # Langkah 3: Alokasi kapasitas agregat ke famili (Tahap 1 Bitran Hax)
            # Pendekatan prioritas berdasarkan biaya setup
            
            # Buat daftar famili berdasarkan prioritas (biaya setup tertinggi)
            priority_families = sorted(unique_families, key=lambda x: setup_costs[x], reverse=True)
            
            # Alokasi kapasitas tersisa ke famili
            remaining_capacity = prod_jan * 1000  # konversi dari ribu unit ke unit
            family_alloc = {fam: 0 for fam in unique_families}
            
            # Pertama, penuhi semua batas bawah
            for fam in priority_families:
                family_alloc[fam] = family_lb[fam]
                remaining_capacity -= family_lb[fam]
            
            # Kemudian, prioritaskan alokasi ke batas atas mulai dari prioritas tertinggi
            for fam in priority_families:
                additional_alloc = min(remaining_capacity, family_ub[fam] - family_lb[fam])
                family_alloc[fam] += additional_alloc
                remaining_capacity -= additional_alloc
            
            # Langkah 4: Alokasi dari famili ke item (Tahap 2 Bitran Hax)
            # Proportional allocation based on conversion rates
            item_alloc = {}
            
            for i, item in enumerate(items):
                fam = families[i]
                
                # Pastikan batas bawah item terpenuhi
                item_alloc[item] = batas_bawah[i]
                
                # Jika masih ada kapasitas tersisa untuk famili, alokasikan berdasarkan proporsi
                remaining_family_cap = family_alloc[fam] - sum([batas_bawah[j] for j, f in enumerate(families) if f == fam])
                
                if remaining_family_cap > 0:
                    # Hitung proporsi konversi untuk semua item dalam famili
                    family_items_idx = [j for j, f in enumerate(families) if f == fam]
                    family_konversi = [konversi[j] for j in family_items_idx]
                    total_konversi = sum(family_konversi)
                    
                    # Alokasikan berdasarkan proporsi
                    item_konversi_proportion = konversi[i] / total_konversi
                    
                    # Jika menggunakan Winter, sesuaikan alokasi berdasarkan seasonal index
                    if metode_selected == "Winter-MPS Terintegrasi" and seasonal_indices:
                        seasonal_factor = seasonal_indices[item][0] / sum([seasonal_indices[items[j]][0] for j in family_items_idx])
                        item_konversi_proportion = item_konversi_proportion * seasonal_factor
                    
                    item_additional = min(
                        remaining_family_cap * item_konversi_proportion,
                        batas_atas[i] - batas_bawah[i]
                    )
                    item_alloc[item] += item_additional
            
            # Langkah 5: Hitung nilai akhir MPS dan inventori akhir
            mps_januari = {}
            inv_akhir_jan = {}
            
            for i, item in enumerate(items):
                # Pembulatan ke atas untuk memastikan safety stock terpenuhi
                mps_januari[item] = round(item_alloc[item])
                
                # Hitung inventori akhir
                inv_akhir_jan[item] = inv_awal[i] + mps_januari[item] - permintaan_jan[i]
            
            # Evaluasi hasil
            # Cek apakah safety stock terpenuhi
            safety_stock_terpenuhi = {}
            for i, item in enumerate(items):
                safety_stock_terpenuhi[item] = inv_akhir_jan[item] >= safety_stock[i]
            
            # Hitung total biaya setup
            total_setup_cost = sum([setup_costs[fam] for fam in unique_families if sum([mps_januari[item] for i, item in enumerate(items) if families[i] == fam]) > 0])
            
            # Hitung kebutuhan kapasitas
            kebutuhan_kapasitas = sum([mps_januari[item] * konversi[i] for i, item in enumerate(items)])
            
            # Kumpulkan hasil untuk interpretasi
            hasil = {
                'mps_januari': mps_januari,
                'safety_stock_terpenuhi': safety_stock_terpenuhi,
                'setup_cost': total_setup_cost,
                'kebutuhan_kapasitas': kebutuhan_kapasitas,
                'inv_akhir_jan': inv_akhir_jan,
                'metode': metode_selected,
                'used_winters': metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"]
            }
            
            # Tampilkan hasil
            st.subheader(f"Jadwal Produksi Induk Januari 2025 ({metode_selected})")
            
            # Tabel MPS
            mps_data = []
            for i, item in enumerate(items):
                mps_data.append({
                    'Famili': families[i],
                    'Item': item,
                    'Konversi': konversi[i],
                    'Safety Stock': safety_stock[i],
                    'Inventori Awal': inv_awal[i],
                    'Permintaan': permintaan_jan[i],
                    'Net Requirement': batas_bawah[i],
                    'Batas Atas': batas_atas[i],
                    'MPS Januari': mps_januari[item],
                    'Inventori Akhir': inv_akhir_jan[item],
                    'SS Terpenuhi': 'Ya' if safety_stock_terpenuhi[item] else 'Tidak'
                })
            
            mps_df = pd.DataFrame(mps_data)
            st.dataframe(mps_df)
            
            # Jika menggunakan Winter, bandingkan dengan MPS standard
            if metode_selected in ["Winter-MPS Terintegrasi", "Hybrid Approach"] and "permintaan_jan_original" in locals():
                st.subheader("Perbandingan MPS dengan Metode Standar")
                
                # Hitung MPS standard untuk perbandingan
                # Penggunaan fungsi sebelumnya, dengan data asli
                batas_bawah_std = []
                for i in range(len(items)):
                    net_req_jan_std = hitung_net_req(inv_awal[i], permintaan_jan_original[i], safety_stock_original[i])
                    batas_bawah_std.append(net_req_jan_std)
                
                # Hitung batas atas standard
                batas_atas_std = []
                for i in range(len(items)):
                    # Batas atas dengan N=n_param bulan ke depan
                    total_demand_std = permintaan_jan_original[i]
                    
                    if n_param >= 2:
                        total_demand_std += permintaan_feb[i]
                        
                    if n_param >= 3:
                        total_demand_std += permintaan_mar[i]
                    
                    net_req_total_std = hitung_net_req(inv_awal[i], total_demand_std, safety_stock_original[i])
                    batas_atas_std.append(net_req_total_std)
                
                # Alokasi famili standard
                family_lb_std = {fam: 0 for fam in unique_families}
                family_ub_std = {fam: 0 for fam in unique_families}
                
                for i, fam in enumerate(families):
                    family_lb_std[fam] += batas_bawah_std[i]
                    family_ub_std[fam] += batas_atas_std[i]
                
                # Alokasi kapasitas
                remaining_capacity_std = prod_jan * 1000
                family_alloc_std = {fam: 0 for fam in unique_families}
                
                for fam in priority_families:
                    family_alloc_std[fam] = family_lb_std[fam]
                    remaining_capacity_std -= family_lb_std[fam]
                
                for fam in priority_families:
                    additional_alloc_std = min(remaining_capacity_std, family_ub_std[fam] - family_lb_std[fam])
                    family_alloc_std[fam] += additional_alloc_std
                    remaining_capacity_std -= additional_alloc_std
                
                # Alokasi ke item
                item_alloc_std = {}
                
                for i, item in enumerate(items):
                    fam = families[i]
                    
                    item_alloc_std[item] = batas_bawah_std[i]
                    
                    remaining_family_cap_std = family_alloc_std[fam] - sum([batas_bawah_std[j] for j, f in enumerate(families) if f == fam])
                    
                    if remaining_family_cap_std > 0:
                        family_items_idx = [j for j, f in enumerate(families) if f == fam]
                        family_konversi = [konversi[j] for j in family_items_idx]
                        total_konversi = sum(family_konversi)
                        
                        item_konversi_proportion = konversi[i] / total_konversi
                        item_additional_std = min(
                            remaining_family_cap_std * item_konversi_proportion,
                            batas_atas_std[i] - batas_bawah_std[i]
                        )
                        item_alloc_std[item] += item_additional_std
                
                # Hitung MPS standard dan inventori akhir
                mps_januari_std = {}
                inv_akhir_jan_std = {}
                
                for i, item in enumerate(items):
                    mps_januari_std[item] = round(item_alloc_std[item])
                    inv_akhir_jan_std[item] = inv_awal[i] + mps_januari_std[item] - permintaan_jan_original[i]
                
                # Safety stock terpenuhi standard
                safety_stock_terpenuhi_std = {}
                for i, item in enumerate(items):
                    safety_stock_terpenuhi_std[item] = inv_akhir_jan_std[item] >= safety_stock_original[i]
                
                # Biaya setup standard
                total_setup_cost_std = sum([setup_costs[fam] for fam in unique_families if sum([mps_januari_std[item] for i, item in enumerate(items) if families[i] == fam]) > 0])
                
                # Kebutuhan kapasitas standard
                kebutuhan_kapasitas_std = sum([mps_januari_std[item] * konversi[i] for i, item in enumerate(items)])
                
                # Tabel perbandingan
                comparison_data = []
                for i, item in enumerate(items):
                    comparison_data.append({
                        'Item': item,
                        'MPS Standard': mps_januari_std[item],
                        f'MPS {metode_selected}': mps_januari[item],
                        'Perbedaan': mps_januari[item] - mps_januari_std[item],
                        'Inv. Akhir Standard': inv_akhir_jan_std[item],
                        f'Inv. Akhir {metode_selected}': inv_akhir_jan[item],
                        'SS Terpenuhi Standard': 'Ya' if safety_stock_terpenuhi_std[item] else 'Tidak',
                        f'SS Terpenuhi {metode_selected}': 'Ya' if safety_stock_terpenuhi[item] else 'Tidak'
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Metrik perbandingan
                st.subheader("Metrik Perbandingan")
                col1, col2 = st.columns(2)
                col1.metric("Total Produksi Standard", f"{sum(mps_januari_std.values()):.1f} unit")
                col2.metric("Total Produksi " + metode_selected, f"{sum(mps_januari.values()):.1f} unit")
                
                col1, col2 = st.columns(2)
                col1.metric("Total Biaya Setup Standard", f"Rp {total_setup_cost_std} Miliar")
                col2.metric("Total Biaya Setup " + metode_selected, f"Rp {total_setup_cost} Miliar")
                
                col1, col2 = st.columns(2)
                col1.metric("Kebutuhan Kapasitas Standard", f"{kebutuhan_kapasitas_std:.2f} unit agregat")
                col2.metric("Kebutuhan Kapasitas " + metode_selected, f"{kebutuhan_kapasitas:.2f} unit agregat")
                
                # Cek service level
                items_safety_std = sum([1 for item in items if safety_stock_terpenuhi_std[item]])
                items_safety = sum([1 for item in items if safety_stock_terpenuhi[item]])
                
                col1, col2 = st.columns(2)
                col1.metric("Service Level Standard", f"{items_safety_std / len(items) * 100:.1f}%")
                col2.metric("Service Level " + metode_selected, f"{items_safety / len(items) * 100:.1f}%")
                
                # Visualisasi perbandingan
                st.subheader("Visualisasi Perbandingan MPS")
                
                # Bar chart untuk perbandingan MPS
                fig, ax = plt.subplots(figsize=(12, 8))
                
                x = range(len(items))
                width = 0.35
                
                bars1 = ax.bar([p - width/2 for p in x], [mps_januari_std[item] for item in items], width, label='MPS Standard')
                bars2 = ax.bar([p + width/2 for p in x], [mps_januari[item] for item in items], width, label=f'MPS {metode_selected}')
                
                ax.set_xticks(x)
                ax.set_xticklabels(items, rotation=45)
                ax.set_title(f'Perbandingan MPS Standard vs {metode_selected}')
                ax.set_ylabel('Jumlah Produksi')
                ax.legend()
                
                st.pyplot(fig)
            
            # Tampilkan metrik ringkasan
            st.subheader("Metrik Ringkasan")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Produksi", f"{sum(mps_januari.values()):.1f} unit")
            col2.metric("Total Biaya Setup", f"Rp {total_setup_cost} Miliar")
            col3.metric("Kebutuhan Kapasitas", f"{kebutuhan_kapasitas:.2f} unit agregat")
            
            # Visualisasi perbandingan inventory vs safety stock
            st.subheader("Inventori Akhir vs Safety Stock")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = range(len(items))
            width = 0.35
            
            # Plot inventori akhir
            bars1 = ax.bar([p - width/2 for p in x], [inv_akhir_jan[item] for item in items], width, color='blue', label='Inventori Akhir')
            
            # Plot safety stock
            bars2 = ax.bar([p + width/2 for p in x], safety_stock, width, color='red', label='Safety Stock')
            
            # Tambahkan label nilai
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center')
                
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center')
            
            ax.set_title('Inventori Akhir vs Safety Stock Januari 2025')
            ax.set_ylabel('Jumlah')
            ax.set_xticks(x)
            ax.set_xticklabels(items, rotation=45)
            ax.legend()
            
            st.pyplot(fig)
            
            # Interpretasi keseluruhan dan rekomendasi
            st.subheader("Interpretasi Jadwal Produksi Induk")
            with st.container():
                if model:
                    st.markdown(interpretasi_mps(model, hasil))
                else:
                    st.warning("API Key belum dimasukkan atau gagal terhubung.")

# Enhanced interpretasi_mps function untuk penjelasan mengenai integrasi Winter's Method
def interpretasi_mps(model, hasil):
    """Interpretasi hasil MPS menggunakan API Gemini."""
    if not model:
        return "Interpretasi AI tidak tersedia."
        
    # Check which type of dictionary is being passed
    metode = hasil.get('metode', 'Bitran Hax Standard')
    metode_text = f"menggunakan {metode}"
    
    prompt = f"""
    Saya telah mengembangkan Jadwal Produksi Induk (MPS) untuk LSP Bakery {metode_text}.
    
    Hasil analisis:
    - MPS untuk bulan Januari: {str(hasil.get('mps_januari', {}))}
    - Pemenuhan kebutuhan safety stock: {str(hasil.get('safety_stock_terpenuhi', {}))}
    - Biaya setup: {hasil.get('setup_cost', 0)} miliar Rupiah
    - Kebutuhan kapasitas: {hasil.get('kebutuhan_kapasitas', 0)} unit produksi
    
    Berikan interpretasi yang ringkas dan profesional dalam 5-7 poin utama dalam bahasa Indonesia.
    Fokus pada:
    - Alokasi produksi antar keluarga dan item produk
    - Efisiensi dalam penggunaan sumber daya dan kapasitas
    - Implikasi terhadap inventory management dan customer service level
    - Keuntungan menggunakan pendekatan {metode} dibandingkan dengan metode standard
    - Implikasi terhadap perencanaan produksi jangka panjang
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error menghasilkan interpretasi MPS: {e}")
        return f"Error menghasilkan interpretasi: {e}"

    st.header("Masalah 4: Jadwal Produksi Induk (MPS)")
    
    # Fungsi interpretasi khusus untuk MPS
    def interpretasi_mps(model, hasil):
        if not model:
            return "Interpretasi AI tidak tersedia."
            
        prompt = f"""
        Saya telah mengembangkan Jadwal Produksi Induk (MPS) untuk LSP Bakery menggunakan metode Bitran Hax.
        
        Hasil analisis:
        - MPS untuk bulan Januari: {hasil['mps_januari']}
        - Pemenuhan kebutuhan safety stock: {hasil['safety_stock_terpenuhi']}
        - Biaya setup: {hasil['setup_cost']} miliar Rupiah
        - Kebutuhan kapasitas: {hasil['kebutuhan_kapasitas']} unit produksi
        
        Berikan interpretasi yang ringkas dan profesional dalam 3-5 poin utama dalam bahasa Indonesia.
        Fokus pada:
        - Alokasi produksi antar keluarga dan item produk
        - Efisiensi dalam penggunaan sumber daya dan kapasitas
        - Implikasi terhadap inventory management dan customer service level
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error menghasilkan interpretasi MPS: {e}")
            return f"Error menghasilkan interpretasi: {e}"
    
    # Tambahkan rumus dan penjelasan - PERBAIKAN FORMAT LATEX
    st.subheader("Metode Bitran Hax untuk Master Production Schedule (MPS)")
    st.write("Metode Bitran Hax adalah pendekatan dua tahap untuk membuat jadwal produksi induk:")
    
    st.write("**Tahap 1: Alokasi Kapasitas ke Keluarga Produk**")
    # Gunakan rumus individual untuk menghindari masalah format
    st.latex(r"\min \sum_{i \in F} C_i Y_i")
    st.latex(r"\text{s.t. } \sum_{i \in F} P_i \leq P^T")
    st.latex(r"L_i \leq P_i \leq U_i, \forall i \in F")
    st.latex(r"Y_i = \begin{cases} 1 & \text{jika } P_i > 0 \\ 0 & \text{jika } P_i = 0 \end{cases}")
    
    st.write("**Tahap 2: Disagregasi ke Item-Item Produk**")
    st.latex(r"L_{ij} \leq X_{ij} \leq U_{ij}, \forall i \in F, j \in J_i")
    st.latex(r"\sum_{j \in J_i} X_{ij} = P_i, \forall i \in F")
    
    st.write("dimana:")
    st.write("- $F$ = Himpunan keluarga produk")
    st.write("- $J_i$ = Himpunan item dalam keluarga $i$")
    st.write("- $P_i$ = Jumlah produksi keluarga $i$")
    st.write("- $P^T$ = Total kapasitas produksi")
    st.write("- $L_i, U_i$ = Batas bawah dan atas produksi keluarga $i$")
    st.write("- $L_{ij}, U_{ij}$ = Batas bawah dan atas produksi item $j$ dalam keluarga $i$")
    st.write("- $X_{ij}$ = Jumlah produksi item $j$ dalam keluarga $i$")
    st.write("- $C_i$ = Biaya setup keluarga $i$")
    st.write("- $Y_i$ = Variabel biner untuk setup keluarga $i$")
    
    with st.expander("Data Produksi dan Permintaan", expanded=True):
        # Input rencana produksi agregat
        st.subheader("Rencana Produksi Agregat 2025 (ribu unit)")
        col1, col2, col3, col4 = st.columns(4)
        prod_jan = col1.number_input("Januari", value=98.3, step=0.1)
        prod_feb = col2.number_input("Februari", value=97.0, step=0.1)
        prod_mar = col3.number_input("Maret", value=105.3, step=0.1)
        prod_apr = col4.number_input("April", value=114.1, step=0.1)
        
        # Data default untuk faktor konversi dan safety stock
        data_produk_default = {
            'Famili': ['Roti Hamburger', 'Roti Hamburger', 'Roti Hamburger', 
                      'Roti Hotdog', 'Roti Hotdog',
                      'Roti Bagel', 'Roti Bagel'],
            'Item': ['White', 'Wijen', 'WholeGrain', 'Plain', 'Garlic', 'Savory', 'Sweet'],
            'Konversi': [0.2, 0.3, 0.4, 0.15, 0.2, 0.1, 0.15],
            'Safety Stock': [120, 60, 45, 90, 80, 75, 50]
        }
        
        # Tabel kebijakan produksi
        st.subheader("Kebijakan Produksi")
        df_kebijakan = pd.DataFrame(data_produk_default)
        edited_kebijakan = st.data_editor(df_kebijakan, use_container_width=True)
        
        # Data perkiraan permintaan
        data_permintaan_default = {
            'Item': ['White', 'Wijen', 'WholeGrain', 'Plain', 'Garlic', 'Savory', 'Sweet'],
            'Inventory Des': [335.2, 148.7, 57.5, 280.4, 96.2, 101, 121],
            'Jan': [210.6, 84.0, 15.4, 175.2, 27.2, 20.7, 65.3],
            'Feb': [172.0, 74.8, 16.0, 198.0, 17.6, 22.8, 71.8],
            'Mar': [255.2, 88.8, 16.8, 178.0, 22.0, 29.6, 93.4],
            'Apr': [360.0, 57.6, 19.2, 128.0, 36.0, 44.4, 140.1]
        }
        
        # Tabel perkiraan permintaan
        st.subheader("Perkiraan Permintaan (ribu unit)")
        df_permintaan = pd.DataFrame(data_permintaan_default)
        edited_permintaan = st.data_editor(df_permintaan, use_container_width=True)
        
        # Biaya setup
        st.subheader("Biaya Setup (Miliar Rupiah)")
        col1, col2, col3 = st.columns(3)
        setup_hamburger = col1.number_input("Roti Hamburger", value=9, step=1)
        setup_hotdog = col2.number_input("Roti Hotdog", value=4, step=1)
        setup_bagel = col3.number_input("Roti Bagel", value=6, step=1)
        
        # Parameter batas atas
        st.subheader("Parameter Batas Atas")
        n_param = st.number_input("Nilai N untuk batas atas", value=2, min_value=1, max_value=4, step=1)
    
    # Tombol untuk menjalankan perhitungan
    if st.button("Selesaikan Masalah 4"):
        with st.spinner("Menghitung Jadwal Produksi Induk..."):
            # 1. Mengumpulkan data dari input
            items = edited_kebijakan['Item'].tolist()
            families = edited_kebijakan['Famili'].tolist()
            konversi = edited_kebijakan['Konversi'].tolist()
            safety_stock = edited_kebijakan['Safety Stock'].tolist()
            
            # Inventori awal dan permintaan
            inv_awal = edited_permintaan['Inventory Des'].tolist()
            permintaan_jan = edited_permintaan['Jan'].tolist()
            permintaan_feb = edited_permintaan['Feb'].tolist()
            permintaan_mar = edited_permintaan['Mar'].tolist()
            
            # Biaya setup per famili
            setup_costs = {
                'Roti Hamburger': setup_hamburger,
                'Roti Hotdog': setup_hotdog,
                'Roti Bagel': setup_bagel
            }
            
            # 2. Implementasi Metode Bitran Hax
            # Langkah 1: Hitung batas atas dan bawah untuk setiap produk
            
            # Fungsi untuk menghitung net requirement
            def hitung_net_req(inv_awal, permintaan, ss):
                net_req = max(0, permintaan + ss - inv_awal)
                return net_req
            
            # Hitung batas bawah (NetReq untuk Jan)
            batas_bawah = []
            for i in range(len(items)):
                net_req_jan = hitung_net_req(inv_awal[i], permintaan_jan[i], safety_stock[i])
                batas_bawah.append(net_req_jan)
            
            # Hitung batas atas (NetReq untuk Jan + Feb atau lebih)
            batas_atas = []
            for i in range(len(items)):
                # Batas atas dengan N=n_param bulan ke depan
                total_demand = permintaan_jan[i]
                
                if n_param >= 2:
                    total_demand += permintaan_feb[i]
                    
                if n_param >= 3:
                    total_demand += permintaan_mar[i]
                
                net_req_total = hitung_net_req(inv_awal[i], total_demand, safety_stock[i])
                batas_atas.append(net_req_total)
            
            # Langkah 2: Agregasi batas bawah dan atas per famili
            unique_families = list(set(families))
            family_lb = {fam: 0 for fam in unique_families}
            family_ub = {fam: 0 for fam in unique_families}
            
            for i, fam in enumerate(families):
                family_lb[fam] += batas_bawah[i]
                family_ub[fam] += batas_atas[i]
            
            # Langkah 3: Alokasi kapasitas agregat ke famili (Tahap 1 Bitran Hax)
            # Pendekatan prioritas berdasarkan biaya setup
            
            # Buat daftar famili berdasarkan prioritas (biaya setup tertinggi)
            priority_families = sorted(unique_families, key=lambda x: setup_costs[x], reverse=True)
            
            # Alokasi kapasitas tersisa ke famili
            remaining_capacity = prod_jan * 1000  # konversi dari ribu unit ke unit
            family_alloc = {fam: 0 for fam in unique_families}
            
            # Pertama, penuhi semua batas bawah
            for fam in priority_families:
                family_alloc[fam] = family_lb[fam]
                remaining_capacity -= family_lb[fam]
            
            # Kemudian, prioritaskan alokasi ke batas atas mulai dari prioritas tertinggi
            for fam in priority_families:
                additional_alloc = min(remaining_capacity, family_ub[fam] - family_lb[fam])
                family_alloc[fam] += additional_alloc
                remaining_capacity -= additional_alloc
            
            # Langkah 4: Alokasi dari famili ke item (Tahap 2 Bitran Hax)
            # Proportional allocation based on conversion rates
            item_alloc = {}
            
            for i, item in enumerate(items):
                fam = families[i]
                
                # Pastikan batas bawah item terpenuhi
                item_alloc[item] = batas_bawah[i]
                
                # Jika masih ada kapasitas tersisa untuk famili, alokasikan berdasarkan proporsi
                remaining_family_cap = family_alloc[fam] - sum([batas_bawah[j] for j, f in enumerate(families) if f == fam])
                
                if remaining_family_cap > 0:
                    # Hitung proporsi konversi untuk semua item dalam famili
                    family_items_idx = [j for j, f in enumerate(families) if f == fam]
                    family_konversi = [konversi[j] for j in family_items_idx]
                    total_konversi = sum(family_konversi)
                    
                    # Alokasikan berdasarkan proporsi
                    item_konversi_proportion = konversi[i] / total_konversi
                    item_additional = min(
                        remaining_family_cap * item_konversi_proportion,
                        batas_atas[i] - batas_bawah[i]
                    )
                    item_alloc[item] += item_additional
            
            # Langkah 5: Hitung nilai akhir MPS dan inventori akhir
            mps_januari = {}
            inv_akhir_jan = {}
            
            for i, item in enumerate(items):
                # Pembulatan ke atas untuk memastikan safety stock terpenuhi
                mps_januari[item] = round(item_alloc[item])
                
                # Hitung inventori akhir
                inv_akhir_jan[item] = inv_awal[i] + mps_januari[item] - permintaan_jan[i]
            
            # Evaluasi hasil
            # Cek apakah safety stock terpenuhi
            safety_stock_terpenuhi = {}
            for i, item in enumerate(items):
                safety_stock_terpenuhi[item] = inv_akhir_jan[item] >= safety_stock[i]
            
            # Hitung total biaya setup
            total_setup_cost = sum([setup_costs[fam] for fam in unique_families if sum([mps_januari[item] for i, item in enumerate(items) if families[i] == fam]) > 0])
            
            # Hitung kebutuhan kapasitas
            kebutuhan_kapasitas = sum([mps_januari[item] * konversi[i] for i, item in enumerate(items)])
            
            # Kumpulkan hasil untuk interpretasi
            hasil = {
                'mps_januari': mps_januari,
                'safety_stock_terpenuhi': safety_stock_terpenuhi,
                'setup_cost': total_setup_cost,
                'kebutuhan_kapasitas': kebutuhan_kapasitas,
                'inv_akhir_jan': inv_akhir_jan
            }
            
            # Tampilkan hasil
            st.subheader("Jadwal Produksi Induk Januari 2025")
            
            # Tabel MPS
            mps_data = []
            for i, item in enumerate(items):
                mps_data.append({
                    'Famili': families[i],
                    'Item': item,
                    'Konversi': konversi[i],
                    'Safety Stock': safety_stock[i],
                    'Inventori Awal': inv_awal[i],
                    'Net Requirement': batas_bawah[i],
                    'Batas Atas': batas_atas[i],
                    'MPS Januari': mps_januari[item],
                    'Inventori Akhir': inv_akhir_jan[item],
                    'SS Terpenuhi': 'Ya' if safety_stock_terpenuhi[item] else 'Tidak'
                })
            
            mps_df = pd.DataFrame(mps_data)
            st.dataframe(mps_df)
            
            # Tampilkan metrik ringkasan
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Produksi", f"{sum(mps_januari.values()):.1f} unit")
            col2.metric("Total Biaya Setup", f"Rp {total_setup_cost} Miliar")
            col3.metric("Kebutuhan Kapasitas", f"{kebutuhan_kapasitas:.2f} unit agregat")
            
            # Visualisasi MPS per item
            st.subheader("MPS Januari 2025 per Item")
            
            # Bar chart untuk MPS
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Kelompokkan berdasarkan famili
            family_items = {}
            for fam in unique_families:
                family_items[fam] = [i for i, f in enumerate(families) if f == fam]
            
            # Plot dengan warna berbeda per famili
            colors = ['blue', 'green', 'orange']
            bar_positions = list(range(len(items)))
            
            for i, fam in enumerate(unique_families):
                fam_positions = [bar_positions[idx] for idx in family_items[fam]]
                fam_items = [items[idx] for idx in family_items[fam]]
                fam_mps = [mps_januari[item] for item in fam_items]
                
                ax.bar(fam_positions, fam_mps, color=colors[i % len(colors)], label=fam)
                
                # Tambahkan label nilai
                for j, pos in enumerate(fam_positions):
                    ax.text(pos, fam_mps[j] + 5, f'{int(fam_mps[j])}', ha='center')
            
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(items, rotation=45)
            ax.set_title('Master Production Schedule Januari 2025')
            ax.set_ylabel('Jumlah Produksi')
            ax.legend()
            st.pyplot(fig)
            
            # Interpretasi hasil visualisasi MPS
            st.markdown("### Interpretasi Visualisasi MPS")
            vis_mps_desc = {
                "distribusi_produksi": {fam: sum([mps_januari[items[i]] for i in idx]) for fam, idx in family_items.items()},
                "item_terbanyak": max(mps_januari.items(), key=lambda x: x[1])[0],
                "item_tersedikit": min(mps_januari.items(), key=lambda x: x[1])[0]
            }
            if model:
                st.markdown(interpretasi_mps(model, vis_mps_desc))
            else:
                st.warning("API Key belum dimasukkan atau gagal terhubung.")
            
            # Visualisasi perbandingan inventory vs safety stock
            st.subheader("Inventori Akhir vs Safety Stock")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = range(len(items))
            width = 0.35
            
            # Plot inventori akhir
            bars1 = ax.bar([p - width/2 for p in x], [inv_akhir_jan[item] for item in items], width, color='blue', label='Inventori Akhir')
            
            # Plot safety stock
            bars2 = ax.bar([p + width/2 for p in x], safety_stock, width, color='red', label='Safety Stock')
            
            # Tambahkan label nilai
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center')
                
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center')
            
            ax.set_title('Inventori Akhir vs Safety Stock Januari 2025')
            ax.set_ylabel('Jumlah')
            ax.set_xticks(x)
            ax.set_xticklabels(items, rotation=45)
            ax.legend()
            
            st.pyplot(fig)
            
            # Interpretasi hasil inventori
            st.markdown("### Interpretasi Inventori vs Safety Stock")
            inv_desc = {
                "safety_stock_terpenuhi": all(safety_stock_terpenuhi.values()),
                "item_tidak_terpenuhi": [item for item, terpenuhi in safety_stock_terpenuhi.items() if not terpenuhi],
                "total_inventori": sum(inv_akhir_jan.values()),
                "ratio_inventori_safety": sum(inv_akhir_jan.values()) / sum(safety_stock)
            }
            if model:
                st.markdown(interpretasi_mps(model, inv_desc))
            else:
                st.warning("API Key belum dimasukkan atau gagal terhubung.")
            
            # Visualisasi alokasi produksi per famili
            st.subheader("Alokasi Produksi per Famili")
            
            family_prod = {fam: sum([mps_januari[items[i]] for i in family_items[fam]]) for fam in unique_families}
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(list(family_prod.values()), labels=list(family_prod.keys()), autopct='%1.1f%%')
            ax.set_title('Proporsi Produksi per Famili')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            st.pyplot(fig)
            
            # Interpretasi distribusi produksi per famili
            st.markdown("### Interpretasi Distribusi Produksi per Famili")
            fam_dist_desc = {
                "distribusi": family_prod,
                "famili_terbesar": max(family_prod.items(), key=lambda x: x[1])[0],
                "proporsi_terbesar": max(family_prod.values()) / sum(family_prod.values()) * 100,
                "biaya_setup": {fam: setup_costs[fam] for fam in family_prod.keys()}
            }
            if model:
                st.markdown(interpretasi_mps(model, fam_dist_desc))
            else:
                st.warning("API Key belum dimasukkan atau gagal terhubung.")
            
            # Dashboard ringkasan
            st.subheader("Dashboard Ringkasan MPS")
            
            # Perbandingan kapasitas agregat vs alokasi
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Kapasitas agregat
            ax.bar(0, prod_jan * 1000, color='green', alpha=0.7, label='Kapasitas Agregat Jan')
            
            # Alokasi
            ax.bar(1, sum(mps_januari.values()), color='blue', alpha=0.7, label='Total MPS Jan')
            
            # Tambahkan label nilai
            ax.text(0, prod_jan * 1000 + 1000, f'{int(prod_jan * 1000)}', ha='center')
            ax.text(1, sum(mps_januari.values()) + 1000, f'{int(sum(mps_januari.values()))}', ha='center')
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Kapasitas Agregat', 'Total MPS'])
            ax.set_title('Kapasitas Agregat vs Total MPS Januari 2025')
            ax.set_ylabel('Unit')
            ax.legend()
            
            st.pyplot(fig)
            
            # Interpretasi keseluruhan dan rekomendasi
            st.subheader("Interpretasi Jadwal Produksi Induk")
            with st.container():
                if model:
                    st.markdown(interpretasi_mps(model, hasil))
                else:
                    st.warning("API Key belum dimasukkan atau gagal terhubung.")
# Masalah 5: Penyeimbangan Lini Perakitan
# Masalah 5: Penyeimbangan Lini Perakitan
def selesaikan_masalah_5():
    st.header("Masalah 5: Penyeimbangan Lini Perakitan")
    
    # Fungsi interpretasi untuk line balancing
    def interpretasi_line_balancing(model, hasil):
        if not model:
            return "Interpretasi AI tidak tersedia."
            
        prompt = f"""
        Saya telah melakukan analisis penyeimbangan lini perakitan untuk PT Jason.
        
        Hasil analisis:
        - Jumlah workstation: {hasil['jumlah_workstation']}
        - Efisiensi lini: {hasil['efisiensi']}
        - Perbandingan metode: {hasil['perbandingan']}
        - Penugasan task: {hasil['penugasan']}
        
        Berikan interpretasi yang ringkas dan profesional dalam 3-5 poin utama dalam bahasa Indonesia.
        Fokus pada:
        - Metode yang paling efisien dan alasannya
        - Implikasi terhadap produktivitas dan utilisasi sumber daya
        - Rekomendasi terbaik untuk PT Jason
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error menghasilkan interpretasi Line Balancing: {e}")
            return f"Error menghasilkan interpretasi: {e}"
    
    # Rumus dan penjelasan
    st.subheader("Penyeimbangan Lini Perakitan (Assembly Line Balancing)")
    st.write("Penyeimbangan lini adalah proses membagi tugas ke stasiun kerja untuk meminimalkan waktu idle dan memaksimalkan efisiensi.")
    
    # Rumus perhitungan
    st.write("**Perhitungan Utama:**")
    st.latex(r"\text{Jumlah workstation minimal} = \left\lceil \frac{\sum_{i=1}^{n} t_i}{c} \right\rceil")
    st.latex(r"\text{Efisiensi lini} = \frac{\sum_{i=1}^{n} t_i}{m \times c} \times 100\%")
    st.latex(r"\text{Balance delay} = \frac{m \times c - \sum_{i=1}^{n} t_i}{m \times c} \times 100\%")
    
    st.write("dimana:")
    st.write("- $t_i$ = Waktu tugas $i$")
    st.write("- $c$ = Waktu siklus")
    st.write("- $m$ = Jumlah workstation")
    st.write("- $n$ = Jumlah tugas")
    
    st.subheader("Metode Penyeimbangan Lini")
    st.write("1. **Kilbridge-Weston**: Mengelompokkan tugas berdasarkan tingkat predesesor (level/kolom)")
    st.write("2. **Helgeson-Birnie (RPW)**: Mengurutkan tugas berdasarkan bobot posisional")
    st.write("3. **Regional Approach**: Mengelompokkan tugas berdasarkan kedekatan fisik/fungsi")
    st.write("4. **Largest Candidate Rule**: Mengurutkan tugas berdasarkan waktu pengerjaan terbesar")
    
    # Input data untuk diagram jaringan
    with st.expander("Data Task dan Precedence", expanded=True):
        st.subheader("Diagram Jaringan Tugas")
        
        # Input data waktu tugas
        st.write("Waktu Task (detik)")
        col1, col2, col3, col4 = st.columns(4)
        waktu_a = col1.number_input("Task A", value=5, step=1)
        waktu_b = col2.number_input("Task B", value=4, step=1)
        waktu_c = col3.number_input("Task C", value=6, step=1)
        waktu_d = col4.number_input("Task D", value=4, step=1)
        
        col1, col2, col3 = st.columns(3)
        waktu_e = col1.number_input("Task E", value=5, step=1)
        waktu_f = col2.number_input("Task F", value=4, step=1)
        waktu_g = col3.number_input("Task G", value=6, step=1)
        
        # Parameter produksi
        st.subheader("Parameter Produksi")
        col1, col2 = st.columns(2)
        target_produksi = col1.number_input("Target Produksi (produk/menit)", value=6, step=1)
        waktu_siklus = col2.number_input("Waktu Siklus (detik/produk)", value=10, step=1)
    
    # Visualisasi diagram jaringan
    st.subheader("Visualisasi Diagram Jaringan")
    
    # Membuat diagram jaringan secara manual

    
    G = nx.DiGraph()
    
    # Tambahkan node dengan label waktu
    tasks = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    waktu_tasks = {
        'A': waktu_a,
        'B': waktu_b,
        'C': waktu_c,
        'D': waktu_d,
        'E': waktu_e,
        'F': waktu_f,
        'G': waktu_g
    }
    
    # Tambahkan node dengan label waktu
    for task, waktu in waktu_tasks.items():
        G.add_node(task, time=waktu)
    
    # Tambahkan edge berdasarkan hubungan preseden yang diberikan
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('A', 'D')
    G.add_edge('B', 'E')
    G.add_edge('C', 'E')
    G.add_edge('C', 'F')
    G.add_edge('D', 'F')
    G.add_edge('E', 'G')
    G.add_edge('F', 'G')
    
    # Plot diagram jaringan
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Layout untuk network diagram
    pos = {
        'A': (1, 2),
        'B': (2, 3),
        'C': (2, 2),
        'D': (2, 1),
        'E': (3, 2.5),
        'F': (3, 1.5),
        'G': (4, 2)
    }
    
    # Gambar node
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
    
    # Gambar edge dengan label waktu
    edge_labels = {(u, v): G.nodes[v]['time'] for u, v in G.edges()}
    nx.draw_networkx_edges(G, pos, arrows=True, width=2, edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
    
    # Tambahkan label node dengan waktu
    labels = {node: f"{node}\n({G.nodes[node]['time']}s)" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)
    
    # Tampilkan diagram
    plt.axis('off')
    st.pyplot(fig)
    
    # Tombol untuk menjalankan perhitungan
    if st.button("Selesaikan Masalah 5"):
        with st.spinner("Menghitung penyeimbangan lini..."):
            # Ekstrak data dari network diagram untuk perhitungan
            task_times = {task: G.nodes[task]['time'] for task in tasks}
            
            # Mendapatkan preseden untuk setiap task
            predecessors = {task: list(G.predecessors(task)) for task in tasks}
            
            # Menghitung total waktu tugas
            total_task_time = sum(task_times.values())
            
            # Teoritis jumlah workstation minimal
            min_workstations = int(np.ceil(total_task_time / waktu_siklus))
            
            # 1. Metode Kilbridge-Weston
            # Mengelompokkan tugas berdasarkan level/kolom
            levels = {}
            current_level = ['A']  # Task tanpa preseden
            assigned_tasks = set(current_level)
            level_num = 0
            
            while assigned_tasks != set(tasks):
                levels[level_num] = current_level.copy()
                next_level = []
                
                for task in current_level:
                    for successor in G.successors(task):
                        if all(pred in assigned_tasks for pred in predecessors[successor]):
                            if successor not in next_level and successor not in assigned_tasks:
                                next_level.append(successor)
                
                current_level = next_level
                assigned_tasks.update(next_level)
                level_num += 1
                
                if not next_level:  # Untuk menghindari infinite loop
                    break
                    
            levels[level_num] = current_level.copy()
            
            # Menerapkan metode Kilbridge-Weston
            kw_workstations = []
            kw_current_station = []
            kw_current_time = 0
            
            # Flatten tasks by level and sort by time within level
            kw_ordered_tasks = []
            for level_idx in range(level_num + 1):
                level_tasks = levels.get(level_idx, [])
                level_tasks.sort(key=lambda t: task_times[t], reverse=True)
                kw_ordered_tasks.extend(level_tasks)
            
            # Assign tasks to workstations
            for task in kw_ordered_tasks:
                if kw_current_time + task_times[task] <= waktu_siklus:
                    kw_current_station.append(task)
                    kw_current_time += task_times[task]
                else:
                    kw_workstations.append((kw_current_station.copy(), kw_current_time))
                    kw_current_station = [task]
                    kw_current_time = task_times[task]
            
            if kw_current_station:
                kw_workstations.append((kw_current_station, kw_current_time))
            
            # Menghitung efisiensi Kilbridge-Weston
            kw_num_stations = len(kw_workstations)
            kw_efficiency = (total_task_time / (kw_num_stations * waktu_siklus)) * 100
            
            # 2. Metode Helgeson-Birnie (Ranked Positional Weight)
            # Menghitung bobot posisional untuk setiap tugas
            rpw = {}
            for task in tasks:
                # Bobot tugas itu sendiri
                weight = task_times[task]
                
                # Tambahkan bobot successor
                visited = set()
                successors_to_process = list(G.successors(task))
                
                while successors_to_process:
                    succ = successors_to_process.pop(0)
                    if succ not in visited:
                        weight += task_times[succ]
                        visited.add(succ)
                        successors_to_process.extend([s for s in G.successors(succ) if s not in visited])
                
                rpw[task] = weight
            
            # Urutkan tugas berdasarkan RPW (dari terbesar ke terkecil)
            rpw_ordered_tasks = sorted(tasks, key=lambda t: rpw[t], reverse=True)
            
            # Assign tugas ke workstation
            hb_workstations = []
            hb_current_station = []
            hb_current_time = 0
            
            for task in rpw_ordered_tasks:
                # Periksa preseden
                if not all(pred in [t for station, _ in hb_workstations for t in station] + hb_current_station for pred in predecessors[task]):
                    continue
                    
                if hb_current_time + task_times[task] <= waktu_siklus:
                    hb_current_station.append(task)
                    hb_current_time += task_times[task]
                else:
                    hb_workstations.append((hb_current_station.copy(), hb_current_time))
                    hb_current_station = [task]
                    hb_current_time = task_times[task]
            
            if hb_current_station:
                hb_workstations.append((hb_current_station, hb_current_time))
            
            # Menghitung efisiensi Helgeson-Birnie
            hb_num_stations = len(hb_workstations)
            hb_efficiency = (total_task_time / (hb_num_stations * waktu_siklus)) * 100
            
            # 3. Metode Regional Approach
            # Mengelompokkan tugas berdasarkan region (simplifikasi: A+B+C, D+E, F+G)
            regions = {
                0: ['A', 'B', 'C'],
                1: ['D', 'E'],
                2: ['F', 'G']
            }
            
            # Urutkan tugas berdasarkan region dan waktu
            ra_ordered_tasks = []
            for region_idx in range(3):
                region_tasks = regions.get(region_idx, [])
                region_tasks.sort(key=lambda t: task_times[t], reverse=True)
                ra_ordered_tasks.extend(region_tasks)
            
            # Assign tugas ke workstation
            ra_workstations = []
            ra_current_station = []
            ra_current_time = 0
            
            # Kumpulkan tugas yang sudah diassign
            assigned = set()
            
            for task in ra_ordered_tasks:
                # Periksa preseden
                if not all(pred in assigned for pred in predecessors[task]):
                    continue
                    
                if ra_current_time + task_times[task] <= waktu_siklus:
                    ra_current_station.append(task)
                    ra_current_time += task_times[task]
                    assigned.add(task)
                else:
                    ra_workstations.append((ra_current_station.copy(), ra_current_time))
                    ra_current_station = [task]
                    ra_current_time = task_times[task]
                    assigned.add(task)
            
            if ra_current_station:
                ra_workstations.append((ra_current_station, ra_current_time))
            
            # Menghitung efisiensi Regional Approach
            ra_num_stations = len(ra_workstations)
            ra_efficiency = (total_task_time / (ra_num_stations * waktu_siklus)) * 100
            
            # 4. Metode Largest Candidate Rule
            # Urutkan tugas berdasarkan waktu (dari terbesar ke terkecil)
            lcr_ordered_tasks = sorted(tasks, key=lambda t: task_times[t], reverse=True)
            
            # Assign tugas ke workstation
            lcr_workstations = []
            lcr_current_station = []
            lcr_current_time = 0
            
            # Kumpulkan tugas yang sudah diassign
            assigned = set()
            
            for _ in range(len(tasks)):  # Loop sampai semua tugas terassign
                for task in lcr_ordered_tasks:
                    if task in assigned:
                        continue
                        
                    # Periksa preseden
                    if not all(pred in assigned for pred in predecessors[task]):
                        continue
                        
                    if lcr_current_time + task_times[task] <= waktu_siklus:
                        lcr_current_station.append(task)
                        lcr_current_time += task_times[task]
                        assigned.add(task)
                
                # Jika ada tugas yang diassign atau semua tugas sudah selesai
                if lcr_current_station or len(assigned) == len(tasks):
                    if lcr_current_station:  # Skip stasiun kosong
                        lcr_workstations.append((lcr_current_station.copy(), lcr_current_time))
                    lcr_current_station = []
                    lcr_current_time = 0
                
                # Keluar jika semua tugas sudah diassign
                if len(assigned) == len(tasks):
                    break
            
            # Menghitung efisiensi Largest Candidate Rule
            lcr_num_stations = len(lcr_workstations)
            lcr_efficiency = (total_task_time / (lcr_num_stations * waktu_siklus)) * 100
            
            # Menentukan metode terbaik
            efficiencies = {
                'Kilbridge-Weston': kw_efficiency,
                'Helgeson-Birnie': hb_efficiency,
                'Regional Approach': ra_efficiency,
                'Largest Candidate Rule': lcr_efficiency
            }
            
            best_method = max(efficiencies, key=efficiencies.get)
            
            # Mapping untuk jumlah workstation
            num_stations = {
                'Kilbridge-Weston': kw_num_stations,
                'Helgeson-Birnie': hb_num_stations,
                'Regional Approach': ra_num_stations,
                'Largest Candidate Rule': lcr_num_stations
            }
            
            # Mapping untuk workstation assignments
            workstation_assignments = {
                'Kilbridge-Weston': kw_workstations,
                'Helgeson-Birnie': hb_workstations,
                'Regional Approach': ra_workstations,
                'Largest Candidate Rule': lcr_workstations
            }
            
            # Hasil untuk interpretasi
            hasil = {
                'jumlah_workstation': num_stations,
                'efisiensi': efficiencies,
                'perbandingan': {
                    'best_method': best_method,
                    'best_efficiency': efficiencies[best_method]
                },
                'penugasan': workstation_assignments
            }
            
            # Tampilkan hasil
            st.subheader("Hasil Penyeimbangan Lini")
            
            # Tampilkan informasi dasar
            st.write(f"**Total waktu tugas:** {total_task_time} detik")
            st.write(f"**Waktu siklus:** {waktu_siklus} detik")
            st.write(f"**Jumlah workstation teoritis minimal:** {min_workstations}")
            
            # Tampilkan perbandingan metode
            comp_data = []
            for method, eff in efficiencies.items():
                comp_data.append({
                    'Metode': method,
                    'Jumlah Workstation': num_stations[method],
                    'Efisiensi (%)': f"{eff:.2f}%",
                    'Balance Delay (%)': f"{100 - eff:.2f}%"
                })
                
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df)
            
            # Highlight metode terbaik
            st.success(f"**Metode terbaik: {best_method}** dengan efisiensi {efficiencies[best_method]:.2f}% dan {num_stations[best_method]} workstation")
            
            # Visualisasi perbandingan efisiensi
            fig, ax = plt.subplots(figsize=(10, 6))
            
            methods = list(efficiencies.keys())
            effs = [efficiencies[m] for m in methods]
            
            bars = ax.bar(methods, effs, color=['blue', 'green', 'orange', 'red'])
            
            # Highlight metode terbaik
            best_idx = methods.index(best_method)
            bars[best_idx].set_color('gold')
            
            # Tambahkan label nilai
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center')
            
            ax.set_title('Perbandingan Efisiensi Metode Penyeimbangan Lini')
            ax.set_ylabel('Efisiensi (%)')
            ax.set_ylim(0, 110)  # Set y-axis to have some space above the bars
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Visualisasi perbandingan jumlah workstation
            fig, ax = plt.subplots(figsize=(10, 6))
            
            stations = [num_stations[m] for m in methods]
            
            bars = ax.bar(methods, stations, color=['blue', 'green', 'orange', 'red'])
            
            # Highlight metode terbaik
            best_idx = methods.index(best_method)
            bars[best_idx].set_color('gold')
            
            # Tambahkan label nilai
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center')
            
            ax.axhline(y=min_workstations, color='black', linestyle='--', label=f'Teoritis Minimal ({min_workstations})')
            
            ax.set_title('Perbandingan Jumlah Workstation')
            ax.set_ylabel('Jumlah Workstation')
            ax.set_ylim(0, max(stations) + 1)  # Set y-axis to have some space above the bars
            
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tampilkan detail hasil tiap metode
            st.subheader("Detail Penugasan Workstation")
            
            # Pilih metode untuk ditampilkan
            selected_method = st.selectbox("Pilih metode untuk melihat detail:", methods)
            
            # Tampilkan workstation assignments untuk metode yang dipilih
            st.write(f"**Detail Workstations untuk {selected_method}:**")
            
            ws_assignments = workstation_assignments[selected_method]
            for i, (tasks, time) in enumerate(ws_assignments):
                st.write(f"Workstation {i+1}: {', '.join(tasks)} (Total waktu: {time} detik, Idle: {waktu_siklus - time} detik)")
            
            # Visualisasi workload distribution
            st.subheader("Visualisasi Distribusi Beban Kerja")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ws_labels = [f'WS {i+1}' for i in range(len(ws_assignments))]
            ws_times = [time for _, time in ws_assignments]
            idle_times = [waktu_siklus - time for _, time in ws_assignments]
            
            # Plot beban kerja
            ax.bar(ws_labels, ws_times, color='blue', label='Waktu Kerja')
            
            # Plot idle time
            ax.bar(ws_labels, idle_times, bottom=ws_times, color='red', alpha=0.7, label='Waktu Idle')
            
            # Tambahkan garis untuk cycle time
            ax.axhline(y=waktu_siklus, color='green', linestyle='--', label=f'Waktu Siklus ({waktu_siklus} detik)')
            
            # Tambahkan label nilai
            for i, (work_time, idle_time) in enumerate(zip(ws_times, idle_times)):
                ax.text(i, work_time/2, f'{work_time}s', ha='center', va='center', color='white', fontweight='bold')
                if idle_time > 0:
                    ax.text(i, work_time + idle_time/2, f'{idle_time}s', ha='center', va='center')
            
            ax.set_title(f'Distribusi Beban Kerja untuk {selected_method}')
            ax.set_ylabel('Waktu (detik)')
            ax.set_ylim(0, waktu_siklus * 1.2)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretasi dengan API
            st.subheader("Interpretasi Penyeimbangan Lini")
            with st.container():
                if model:
                    st.markdown(interpretasi_line_balancing(model, hasil))
                else:
                    st.warning("API Key belum dimasukkan atau gagal terhubung.")

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
