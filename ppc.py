import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Tugas PPC",
    page_icon="📊",
    layout="wide"
)

# Header aplikasi
st.title("Aplikasi Otomatisasi Perhitungan Tugas PPC")
st.write("Aplikasi ini membantu menyelesaikan 5 soal tugas PPC dengan perhitungan otomatis dan visualisasi lengkap")

# Sidebar untuk navigasi
st.sidebar.header("Navigasi")
page = st.sidebar.radio(
    "Pilih Soal",
    ["Input NIM", "Soal 1: Exponential Smoothing", "Soal 2: Winter's Method", 
     "Soal 3: Aggregate Planning", "Soal 4: Master Production Schedule", 
     "Soal 5: Line Balancing"]
)

# Area untuk input NIM
if page == "Input NIM":
    st.header("Input Data NIM")
    
    nim = st.text_input("Masukkan NIM Anda:", "")
    
    if nim and len(nim) >= 8:
        # Ekstrak informasi dari NIM
        digit_6 = int(nim[-6]) if len(nim) >= 6 else 0
        digit_7_8 = int(nim[-2:]) if len(nim) >= 2 else 0
        digit_last = int(nim[-1])
        three_last_digits = sum(int(d) for d in nim[-3:])
        
        # Hitung B0 untuk Soal 1
        B0 = 200 + (digit_6 + digit_7_8)
        
        # Tentukan metode evaluasi berdasarkan digit terakhir
        if digit_last in [0, 1, 2]:
            eval_method = "MFE"
        elif digit_last in [3, 4, 5]:
            eval_method = "MAD"
        elif digit_last in [6, 7]:
            eval_method = "MAPE"
        else:  # 8 or 9
            eval_method = "MSE"
        
        # Tampilkan parameter yang dihitung dari NIM
        st.success(f"NIM berhasil diproses: {nim}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Parameter untuk Soal 1:**")
            st.write(f"B₀ = 200 + ({digit_6} + {digit_7_8}) = {B0}")
            st.write(f"T₀ = 0")
            st.write(f"Metode Evaluasi: {eval_method}")
        
        with col2:
            st.info("**Parameter untuk Soal 2:**")
            st.write(f"Nilai x = {three_last_digits}")
        
        # Simpan nilai dalam session state untuk digunakan di halaman lain
        st.session_state['nim'] = nim
        st.session_state['B0'] = B0
        st.session_state['eval_method'] = eval_method
        st.session_state['x'] = three_last_digits
        
        st.success("Data berhasil disimpan! Silakan pilih soal yang ingin diselesaikan di sidebar.")
    else:
        st.warning("Silakan masukkan NIM yang valid (minimal 8 digit).")
        
# Function untuk membuat table lengkap SES dengan perhitungan detail
def calculate_ses(sales, alpha, B0):
    forecasts = [B0]
    errors = []
    abs_errors = []
    squared_errors = []
    percent_errors = []
    
    for i in range(len(sales)):
        # Hitung error
        error = sales[i] - forecasts[i]
        errors.append(error)
        abs_errors.append(abs(error))
        squared_errors.append(error ** 2)
        percent_errors.append(abs(error) / sales[i] * 100 if sales[i] != 0 else 0)
        
        # Hitung forecast untuk periode berikutnya
        forecast = alpha * sales[i] + (1 - alpha) * forecasts[i]
        forecasts.append(forecast)
    
    # Hapus forecast terakhir (untuk periode n+1)
    forecasts_result = forecasts[:-1]
    
    # Hitung metrik evaluasi
    mfe = sum(errors) / len(errors)
    mad = sum(abs_errors) / len(abs_errors)
    mse = sum(squared_errors) / len(squared_errors)
    mape = sum(percent_errors) / len(percent_errors)
    
    forecast_next_3_periods = [forecasts[-1]] * 3
    
    return forecasts_result, errors, abs_errors, squared_errors, percent_errors, mfe, mad, mse, mape, forecast_next_3_periods

# Function untuk membuat table lengkap DES dengan perhitungan detail
def calculate_des(sales, alpha, beta, B0, T0=0):
    levels = [B0]
    trends = [T0]
    forecasts = [B0]
    errors = []
    abs_errors = []
    squared_errors = []
    percent_errors = []
    
    for i in range(len(sales)):
        # Hitung error
        error = sales[i] - forecasts[i]
        errors.append(error)
        abs_errors.append(abs(error))
        squared_errors.append(error ** 2)
        percent_errors.append(abs(error) / sales[i] * 100 if sales[i] != 0 else 0)
        
        # Hitung level
        level = alpha * sales[i] + (1 - alpha) * (levels[i] + trends[i])
        levels.append(level)
        
        # Hitung trend
        trend = beta * (levels[i+1] - levels[i]) + (1 - beta) * trends[i]
        trends.append(trend)
        
        # Hitung forecast untuk periode berikutnya
        forecast = levels[i+1] + trends[i+1]
        forecasts.append(forecast)
    
    # Hapus level, trend, dan forecast terakhir (untuk periode n+1)
    levels_result = levels[:-1]
    trends_result = trends[:-1]
    forecasts_result = forecasts[:-1]
    
    # Hitung metrik evaluasi
    mfe = sum(errors) / len(errors)
    mad = sum(abs_errors) / len(abs_errors)
    mse = sum(squared_errors) / len(squared_errors)
    mape = sum(percent_errors) / len(percent_errors)
    
    # Forecast untuk 3 periode berikutnya
    forecast_next_3_periods = [
        levels[-1] + (m+1) * trends[-1] for m in range(3)
    ]
    
    return levels_result, trends_result, forecasts_result, errors, abs_errors, squared_errors, percent_errors, mfe, mad, mse, mape, forecast_next_3_periods

# Soal 1: Exponential Smoothing
if page == "Soal 1: Exponential Smoothing":
    st.header("Soal 1: PT LSP Abang Dika - Exponential Smoothing")
    
    if 'nim' not in st.session_state:
        st.warning("Silakan input NIM terlebih dahulu di halaman 'Input NIM'")
    else:
        nim = st.session_state['nim']
        B0 = st.session_state['B0']
        eval_method = st.session_state['eval_method']
        
        st.subheader("Data dan Parameter")
        st.write(f"NIM: {nim}")
        st.write(f"B₀ = {B0}")
        st.write(f"T₀ = 0")
        st.write(f"Metode Evaluasi: {eval_method}")
        
        # Data penjualan
        months = list(range(1, 25))
        sales = [260, 210, 180, 255, 330, 395, 280, 215, 270, 130, 125, 90,
                140, 155, 195, 225, 250, 430, 510, 405, 375, 325, 295, 260]
        
        # Tampilkan data penjualan dalam tabel
        sales_df = pd.DataFrame({
            'Bulan': months,
            'Penjualan': sales
        })
        st.write("Data Penjualan:")
        st.dataframe(sales_df)
        
        # Penjelasan metode
        with st.expander("Penjelasan Metode Exponential Smoothing"):
            st.markdown("""
            ### Single Exponential Smoothing (SES)
            
            Metode SES digunakan untuk data tanpa tren dan musiman dengan rumus:
            
            $F_{t+1} = \\alpha Y_t + (1-\\alpha)F_t$
            
            dimana:
            - $F_{t+1}$ adalah peramalan untuk periode t+1
            - $Y_t$ adalah nilai aktual pada periode t
            - $F_t$ adalah peramalan pada periode t
            - $\\alpha$ adalah parameter pemulusan (0 < $\\alpha$ < 1)
            
            ### Double Exponential Smoothing (DES)
            
            Metode DES digunakan untuk data dengan tren dengan rumus:
            
            $S_t = \\alpha Y_t + (1-\\alpha)(S_{t-1} + T_{t-1})$
            
            $T_t = \\beta(S_t - S_{t-1}) + (1-\\beta)T_{t-1}$
            
            $F_{t+m} = S_t + mT_t$
            
            dimana:
            - $S_t$ adalah nilai level pada periode t
            - $T_t$ adalah nilai tren pada periode t
            - $\\alpha$ dan $\\beta$ adalah parameter pemulusan (0 < $\\alpha$, $\\beta$ < 1)
            - $m$ adalah jumlah periode ke depan yang diramalkan
            """)
        
        # Cari parameter optimal untuk SES
        st.subheader("1a) Single Exponential Smoothing (SES)")
        
        # Uji beberapa nilai alpha dan pilih yang terbaik
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
        ses_results = []
        
        for alpha in alphas:
            forecasts, errors, abs_errors, squared_errors, percent_errors, mfe, mad, mse, mape, _ = calculate_ses(sales, alpha, B0)
            ses_results.append({
                'alpha': alpha,
                'MFE': mfe,
                'MAD': mad,
                'MSE': mse,
                'MAPE': mape
            })
        
        # Tentukan alpha optimal berdasarkan metode evaluasi
        if eval_method == "MFE":
            optimal_alpha_ses = min(ses_results, key=lambda x: abs(x['MFE']))['alpha']
        elif eval_method == "MAD":
            optimal_alpha_ses = min(ses_results, key=lambda x: x['MAD'])['alpha']
        elif eval_method == "MAPE":
            optimal_alpha_ses = min(ses_results, key=lambda x: x['MAPE'])['alpha']
        else:  # MSE
            optimal_alpha_ses = min(ses_results, key=lambda x: x['MSE'])['alpha']
        
        st.write(f"Pengujian beberapa nilai α:")
        ses_results_df = pd.DataFrame(ses_results).set_index('alpha')
        st.dataframe(ses_results_df)
        
        st.success(f"Nilai α optimal untuk SES: {optimal_alpha_ses}")
        
        # Hitung SES dengan parameter optimal
        ses_forecasts, ses_errors, ses_abs_errors, ses_squared_errors, ses_percent_errors, ses_mfe, ses_mad, ses_mse, ses_mape, ses_forecast_next = calculate_ses(sales, optimal_alpha_ses, B0)
        
        # Tampilkan perhitungan lengkap SES
        st.write("Perhitungan Lengkap SES dengan α Optimal:")
        ses_detail_df = pd.DataFrame({
            'Bulan': months,
            'Penjualan Aktual': sales,
            'Peramalan SES': ses_forecasts,
            'Error': ses_errors,
            'Absolute Error': ses_abs_errors,
            'Squared Error': ses_squared_errors,
            'Percentage Error (%)': ses_percent_errors
        })
        st.dataframe(ses_detail_df)
        
        # Cari parameter optimal untuk DES
        st.subheader("1a) Double Exponential Smoothing (DES)")
        
        # Uji beberapa kombinasi alpha dan beta
        betas = [0.1, 0.2, 0.3]
        des_results = []
        
        for alpha in alphas:
            for beta in betas:
                levels, trends, forecasts, errors, abs_errors, squared_errors, percent_errors, mfe, mad, mse, mape, _ = calculate_des(sales, alpha, beta, B0)
                des_results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'MFE': mfe,
                    'MAD': mad,
                    'MSE': mse,
                    'MAPE': mape
                })
        
        # Tentukan alpha dan beta optimal berdasarkan metode evaluasi
        if eval_method == "MFE":
            optimal_des = min(des_results, key=lambda x: abs(x['MFE']))
        elif eval_method == "MAD":
            optimal_des = min(des_results, key=lambda x: x['MAD'])
        elif eval_method == "MAPE":
            optimal_des = min(des_results, key=lambda x: x['MAPE'])
        else:  # MSE
            optimal_des = min(des_results, key=lambda x: x['MSE'])
        
        optimal_alpha_des = optimal_des['alpha']
        optimal_beta_des = optimal_des['beta']
        
        st.write(f"Pengujian beberapa kombinasi α dan β:")
        des_results_df = pd.DataFrame(des_results)
        st.dataframe(des_results_df)
        
        st.success(f"Nilai α optimal untuk DES: {optimal_alpha_des}")
        st.success(f"Nilai β optimal untuk DES: {optimal_beta_des}")
        
        # Hitung DES dengan parameter optimal
        des_levels, des_trends, des_forecasts, des_errors, des_abs_errors, des_squared_errors, des_percent_errors, des_mfe, des_mad, des_mse, des_mape, des_forecast_next = calculate_des(sales, optimal_alpha_des, optimal_beta_des, B0)
        
        # Tampilkan perhitungan lengkap DES
        st.write("Perhitungan Lengkap DES dengan α dan β Optimal:")
        des_detail_df = pd.DataFrame({
            'Bulan': months,
            'Penjualan Aktual': sales,
            'Level (S)': des_levels,
            'Trend (T)': des_trends,
            'Peramalan DES': des_forecasts,
            'Error': des_errors,
            'Absolute Error': des_abs_errors,
            'Squared Error': des_squared_errors,
            'Percentage Error (%)': des_percent_errors
        })
        st.dataframe(des_detail_df)
        
        # Tampilkan peramalan 3 bulan ke depan
        st.subheader("Hasil Peramalan 3 Bulan Ke Depan")
        
        future_months = [25, 26, 27]
        future_forecasts_df = pd.DataFrame({
            'Bulan': future_months,
            'Peramalan SES': ses_forecast_next,
            'Peramalan DES': des_forecast_next
        })
        st.dataframe(future_forecasts_df)
        
        # Visualisasi hasil peramalan
        st.subheader("Visualisasi Hasil Peramalan")
        
        # Gabungkan data aktual dengan peramalan
        plot_data = pd.DataFrame({
            'Bulan': months + future_months,
            'Aktual': sales + [None] * 3,
            'SES': ses_forecasts + ses_forecast_next,
            'DES': des_forecasts + des_forecast_next
        })
        
        # Buat grafik interaktif
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_data['Bulan'], y=plot_data['Aktual'], 
                                mode='lines+markers', name='Penjualan Aktual'))
        fig.add_trace(go.Scatter(x=plot_data['Bulan'], y=plot_data['SES'], 
                                mode='lines', name=f'SES (α={optimal_alpha_ses})'))
        fig.add_trace(go.Scatter(x=plot_data['Bulan'], y=plot_data['DES'], 
                                mode='lines', name=f'DES (α={optimal_alpha_des}, β={optimal_beta_des})'))
        
        fig.update_layout(
            title='Perbandingan Penjualan Aktual dengan Hasil Peramalan',
            xaxis_title='Bulan',
            yaxis_title='Penjualan',
            legend_title='Keterangan',
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Perbandingan metode dan evaluasi
        st.subheader("1b) Evaluasi dan Perbandingan Metode")
        
        comparison_df = pd.DataFrame({
            'Metode': ['SES', 'DES'],
            'Parameter': [f'α = {optimal_alpha_ses}', f'α = {optimal_alpha_des}, β = {optimal_beta_des}'],
            'MFE': [ses_mfe, des_mfe],
            'MAD': [ses_mad, des_mad],
            'MSE': [ses_mse, des_mse],
            'MAPE (%)': [ses_mape, des_mape]
        })
        
        st.dataframe(comparison_df.set_index('Metode'))
        
        # Tentukan metode terbaik berdasarkan metode evaluasi
        if eval_method == "MFE":
            better_method = "DES" if abs(des_mfe) < abs(ses_mfe) else "SES"
            metric_val = f"MFE: {des_mfe:.2f} vs {ses_mfe:.2f}"
        elif eval_method == "MAD":
            better_method = "DES" if des_mad < ses_mad else "SES"
            metric_val = f"MAD: {des_mad:.2f} vs {ses_mad:.2f}"
        elif eval_method == "MAPE":
            better_method = "DES" if des_mape < ses_mape else "SES"
            metric_val = f"MAPE: {des_mape:.2f}% vs {ses_mape:.2f}%"
        else:  # MSE
            better_method = "DES" if des_mse < ses_mse else "SES"
            metric_val = f"MSE: {des_mse:.2f} vs {ses_mse:.2f}"
        
        st.success(f"Berdasarkan metode evaluasi {eval_method}, metode yang lebih baik adalah **{better_method}** ({metric_val}).")
        
        # Fitur ekspor hasil ke Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sales_df.to_excel(writer, sheet_name='Data Penjualan', index=False)
            ses_detail_df.to_excel(writer, sheet_name='SES Detail', index=False)
            des_detail_df.to_excel(writer, sheet_name='DES Detail', index=False)
            future_forecasts_df.to_excel(writer, sheet_name='Peramalan 3 Bulan', index=False)
            comparison_df.to_excel(writer, sheet_name='Perbandingan', index=False)
        
        st.download_button(
            label="Download Hasil Perhitungan Soal 1 (Excel)",
            data=output.getvalue(),
            file_name=f"Hasil_Soal1_NIM_{nim}.xlsx",
            mime="application/vnd.ms-excel"
        )

# Soal 2: Winter's Method
if page == "Soal 2: Winter's Method":
    st.header("Soal 2: PT LSP Beauty - Winter's Method")
    
    if 'nim' not in st.session_state:
        st.warning("Silakan input NIM terlebih dahulu di halaman 'Input NIM'")
    else:
        nim = st.session_state['nim']
        x = st.session_state['x']
        
        st.subheader("Data dan Parameter")
        st.write(f"NIM: {nim}")
        st.write(f"Nilai x = {x}")
        
        # Hitung data permintaan
        periods = [1, 2, 3, 4]
        base_demand_year1 = [8, 18, 45, 35]
        base_demand_year2 = [14, 28, 65, 50]
        base_demand_year3 = [20, 40, 80, 55]
        
        demand_year1 = [val * x for val in base_demand_year1]
        demand_year2 = [val * x for val in base_demand_year2]
        demand_year3 = [val * x for val in base_demand_year3]
        
        # Buat dataframe permintaan
        demand_df = pd.DataFrame({
            'Kuartal': periods * 3,
            'Tahun': [1] * 4 + [2] * 4 + [3] * 4,
            'Permintaan': demand_year1 + demand_year2 + demand_year3
        })
        
        # Tampilkan tabel permintaan
        st.write("Data Permintaan Serum Wajah:")
        demand_pivot = demand_df.pivot(index='Kuartal', columns='Tahun', values='Permintaan')
        st.dataframe(demand_pivot)
        
        # Penjelasan Winter's Method
        with st.expander("Penjelasan Winter's Method"):
            st.markdown("""
            ### Winter's Method (Holt-Winters)
            
            Winter's Method atau Holt-Winters adalah metode peramalan time series yang dapat menangani data dengan tren dan pola musiman. Metode ini menggunakan tiga persamaan smoothing:
            
            1. **Level/Intercept (a)**: Nilai dasar data yang diperbarui
            2. **Trend/Slope (b)**: Perubahan dari periode ke periode
            3. **Seasonal Factors (S)**: Pengaruh musiman
            
            **Rumus Perhitungan**:
            
            1. Level: $L_t = \\alpha(Y_t / S_{t-s}) + (1-\\alpha)(L_{t-1} + T_{t-1})$
            2. Trend: $T_t = \\beta(L_t - L_{t-1}) + (1-\\beta)T_{t-1}$
            3. Seasonal: $S_t = \\gamma(Y_t / L_t) + (1-\\gamma)S_{t-s}$
            4. Forecast: $F_{t+m} = (L_t + mT_t)S_{t+m-s}$
            
            dimana:
            - $\\alpha$, $\\beta$, $\\gamma$ adalah parameter pemulusan (0 < $\\alpha$, $\\beta$, $\\gamma$ < 1)
            - $s$ adalah panjang musiman (4 untuk data kuartalan)
            - $m$ adalah jumlah periode ke depan yang diramalkan
            """)
        
        st.subheader("2a) Nilai Awal Winter's Method")
        
        # Hitung rata-rata untuk setiap kuartal (Tahun 2 dan 3)
        avg_q1 = (demand_year2[0] + demand_year3[0]) / 2
        avg_q2 = (demand_year2[1] + demand_year3[1]) / 2
        avg_q3 = (demand_year2[2] + demand_year3[2]) / 2
        avg_q4 = (demand_year2[3] + demand_year3[3]) / 2
        
        avg_all = (avg_q1 + avg_q2 + avg_q3 + avg_q4) / 4
        
        st.write("Rata-rata Setiap Kuartal (Tahun 2 dan 3):")
        avg_df = pd.DataFrame({
            'Kuartal': ['Q1', 'Q2', 'Q3', 'Q4', 'Keseluruhan'],
            'Rata-rata': [avg_q1, avg_q2, avg_q3, avg_q4, avg_all]
        })
        st.dataframe(avg_df)
        
        # Hitung faktor musiman awal
        sf_q1 = avg_q1 / avg_all
        sf_q2 = avg_q2 / avg_all
        sf_q3 = avg_q3 / avg_all
        sf_q4 = avg_q4 / avg_all
        
        seasonal_df = pd.DataFrame({
            'Kuartal': ['Q1', 'Q2', 'Q3', 'Q4'],
            'Faktor Musiman': [sf_q1, sf_q2, sf_q3, sf_q4]
        })
        
        st.write("Faktor Musiman Awal:")
        st.dataframe(seasonal_df)
        
        # Hitung data deseasonalized
        deseasonal_data = []
        
        for i in range(4):
            deseasonal_data.append(demand_year2[i] / [sf_q1, sf_q2, sf_q3, sf_q4][i])
            
        for i in range(4):
            deseasonal_data.append(demand_year3[i] / [sf_q1, sf_q2, sf_q3, sf_q4][i])
        
        # Regresi linear untuk mencari intercept dan slope
        periods = list(range(1, 9))
        periods_mean = np.mean(periods)
        deseasonal_mean = np.mean(deseasonal_data)
        
        numerator = sum([(periods[i] - periods_mean) * (deseasonal_data[i] - deseasonal_mean) for i in range(8)])
        denominator = sum([(periods[i] - periods_mean) ** 2 for i in range(8)])
        
        slope = numerator / denominator
        intercept = deseasonal_mean - slope * periods_mean
        
        st.write("Data Deseasonalized dan Perhitungan Regresi Linear:")
        deseasonal_df = pd.DataFrame({
            'Periode': periods,
            'Tahun-Kuartal': ['T2Q1', 'T2Q2', 'T2Q3', 'T2Q4', 'T3Q1', 'T3Q2', 'T3Q3', 'T3Q4'],
            'Permintaan': demand_year2 + demand_year3,
            'Faktor Musiman': [sf_q1, sf_q2, sf_q3, sf_q4] * 2,
            'Deseasonalized': deseasonal_data
        })
        st.dataframe(deseasonal_df)
        
        st.write(f"Persamaan Regresi Linear: Y = {intercept:.2f} + {slope:.2f}X")
        st.write(f"Intercept (a): {intercept:.2f}")
        st.write(f"Slope (b): {slope:.2f}")
        
        # Visualisasi data deseasonalized dan garis regresi
        fig = px.scatter(deseasonal_df, x='Periode', y='Deseasonalized', 
                         text='Tahun-Kuartal', title='Data Deseasonalized dan Garis Regresi')
        
        # Add regression line
        reg_y = [intercept + slope * x for x in periods]
        fig.add_trace(go.Scatter(x=periods, y=reg_y, mode='lines', name='Garis Regresi'))
        
        fig.update_layout(
            xaxis_title='Periode',
            yaxis_title='Nilai Deseasonalized',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("2b) Pembaruan Estimasi untuk Kuartal 1 Tahun 4")
        
        alpha = 0.25
        beta = 0.12
        gamma = 0.08
        
        st.write(f"Parameter: α = {alpha}, β = {beta}, γ = {gamma}")
        
        # Initial values at end of year 3
        level_last = intercept + 8 * slope  # Level at the end of period 8
        trend_last = slope
        seasonal_factors = [sf_q1, sf_q2, sf_q3, sf_q4]
        
        st.write(f"Level Akhir Tahun 3 (L₈): {level_last:.2f}")
        st.write(f"Trend Akhir Tahun 3 (T₈): {trend_last:.2f}")
        
        # New demand in Q1 Year 4
        new_demand = 275
        
        st.write(f"Permintaan Aktual Q1 Tahun 4: {new_demand}")
        
        # Update level, trend, and seasonal factors
        level_new = alpha * (new_demand / seasonal_factors[0]) + (1 - alpha) * (level_last + trend_last)
        trend_new = beta * (level_new - level_last) + (1 - beta) * trend_last
        seasonal_new = gamma * (new_demand / level_new) + (1 - gamma) * seasonal_factors[0]
        
        updated_seasonal_factors = [seasonal_new, seasonal_factors[1], seasonal_factors[2], seasonal_factors[3]]
        
        # Tampilkan detail perhitungan
        st.write("Detail Perhitungan:")
        st.latex(f"L_9 = {alpha} \\times (\\frac{{{new_demand}}}{{{seasonal_factors[0]:.3f}}}) + (1-{alpha}) \\times ({level_last:.2f} + {trend_last:.2f})")
        st.latex(f"L_9 = {alpha} \\times {new_demand/seasonal_factors[0]:.2f} + {1-alpha} \\times {level_last + trend_last:.2f}")
        st.latex(f"L_9 = {alpha * (new_demand/seasonal_factors[0]):.2f} + {(1-alpha) * (level_last + trend_last):.2f} = {level_new:.2f}")
        
        st.latex(f"T_9 = {beta} \\times ({level_new:.2f} - {level_last:.2f}) + (1-{beta}) \\times {trend_last:.2f}")
        st.latex(f"T_9 = {beta} \\times {level_new - level_last:.2f} + {1-beta} \\times {trend_last:.2f}")
        st.latex(f"T_9 = {beta * (level_new - level_last):.2f} + {(1-beta) * trend_last:.2f} = {trend_new:.2f}")
        
        st.latex(f"S_9 = {gamma} \\times (\\frac{{{new_demand}}}{{{level_new:.2f}}}) + (1-{gamma}) \\times {seasonal_factors[0]:.3f}")
        st.latex(f"S_9 = {gamma} \\times {new_demand/level_new:.3f} + {1-gamma} \\times {seasonal_factors[0]:.3f}")
        st.latex(f"S_9 = {gamma * (new_demand/level_new):.3f} + {(1-gamma) * seasonal_factors[0]:.3f} = {seasonal_new:.3f}")
        
        st.success(f"Level Baru (L₉): {level_new:.2f}")
        st.success(f"Trend Baru (T₉): {trend_new:.2f}")
        st.success(f"Faktor Musiman Baru (S₉): {seasonal_new:.3f}")
        
        st.subheader("2c) Peramalan untuk Kuartal 2, 3, dan 4 Tahun 4")
        
        # Forecast for remaining quarters
        forecast_q2 = (level_new + 1 * trend_new) * seasonal_factors[1]
        forecast_q3 = (level_new + 2 * trend_new) * seasonal_factors[2]
        forecast_q4 = (level_new + 3 * trend_new) * seasonal_factors[3]
        
        st.write("Detail Perhitungan Peramalan:")
        st.latex(f"F_{10} = ({level_new:.2f} + 1 \\times {trend_new:.2f}) \\times {seasonal_factors[1]:.3f}")
        st.latex(f"F_{10} = {level_new + trend_new:.2f} \\times {seasonal_factors[1]:.3f} = {forecast_q2:.2f}")
        
        st.latex(f"F_{11} = ({level_new:.2f} + 2 \\times {trend_new:.2f}) \\times {seasonal_factors[2]:.3f}")
        st.latex(f"F_{11} = {level_new + 2*trend_new:.2f} \\times {seasonal_factors[2]:.3f} = {forecast_q3:.2f}")
        
        st.latex(f"F_{12} = ({level_new:.2f} + 3 \\times {trend_new:.2f}) \\times {seasonal_factors[3]:.3f}")
        st.latex(f"F_{12} = {level_new + 3*trend_new:.2f} \\times {seasonal_factors[3]:.3f} = {forecast_q4:.2f}")
        
        # Tampilkan hasil peramalan dalam tabel
        forecast_df = pd.DataFrame({
            'Kuartal Tahun 4': ['Q2', 'Q3', 'Q4'],
            'Peramalan': [forecast_q2, forecast_q3, forecast_q4]
        })
        
        st.success("Hasil Peramalan untuk Sisa Kuartal Tahun 4:")
        st.dataframe(forecast_df)
        
        # Visualisasi hasil peramalan
        history_data = demand_year1 + demand_year2 + demand_year3 + [new_demand]
        quarters = ['T1Q1', 'T1Q2', 'T1Q3', 'T1Q4',
                   'T2Q1', 'T2Q2', 'T2Q3', 'T2Q4',
                   'T3Q1', 'T3Q2', 'T3Q3', 'T3Q4',
                   'T4Q1']
        
        forecast_data = [None] * 13 + [forecast_q2, forecast_q3, forecast_q4]
        forecast_quarters = quarters + ['T4Q2', 'T4Q3', 'T4Q4']
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'Kuartal': forecast_quarters,
            'Aktual': history_data + [None] * 3,
            'Peramalan': forecast_data
        })
        
        fig = px.line(plot_df, x='Kuartal', y=['Aktual', 'Peramalan'], 
                     title='Permintaan Aktual dan Peramalan',
                     markers=True)
        
        fig.update_layout(
            xaxis_title='Kuartal',
            yaxis_title='Permintaan',
            legend_title='Keterangan'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fitur ekspor hasil ke Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            demand_pivot.to_excel(writer, sheet_name='Data Permintaan')
            deseasonal_df.to_excel(writer, sheet_name='Perhitungan Awal', index=False)
            pd.DataFrame({'Parameter': ['Level Baru (L₉)', 'Trend Baru (T₉)', 'Faktor Musiman Baru (S₉)'],
                         'Nilai': [level_new, trend_new, seasonal_new]}).to_excel(writer, sheet_name='Pembaruan Estimasi', index=False)
            forecast_df.to_excel(writer, sheet_name='Hasil Peramalan', index=False)
        
        st.download_button(
            label="Download Hasil Perhitungan Soal 2 (Excel)",
            data=output.getvalue(),
            file_name=f"Hasil_Soal2_NIM_{nim}.xlsx",
            mime="application/vnd.ms-excel"
        )

# Soal 3: Aggregate Planning
if page == "Soal 3: Aggregate Planning":
    st.header("Soal 3: PT LScream - Aggregate Planning")
    
    # Data permintaan dan parameter
    st.subheader("Data Permintaan")
    
    # Data permintaan per produk
    demand_data = {
        'Minggu': [1, 2, 3, 4, 5, 6],
        'Vanila': [180, 120, 210, 180, 95, 140],
        'Coklat': [180, 110, 200, 420, 110, 90],
        'Stroberi': [90, 70, 190, 100, 95, 70]
    }
    
    demand_df = pd.DataFrame(demand_data)
    st.dataframe(demand_df)
    
    # Filling rates
    filling_rates = {
        'Vanila': 7.5,
        'Coklat': 8.1,
        'Stroberi': 5.2
    }
    
    filling_df = pd.DataFrame({
        'Produk': list(filling_rates.keys()),
        'Filling Rate (kL/jam)': list(filling_rates.values())
    })
    
    st.write("Filling Rates:")
    st.dataframe(filling_df)
    
    # Biaya dan parameter lain
    st.subheader("Data Biaya dan Parameter Produksi")
    
    cost_data = {
        'Parameter': ['Filling rate agregat', 'Biaya', 'Inventory cost', 'Regular time cost', 
                     'Overtime cost', 'Ramping up cost', 'Ramping down cost'],
        'Nilai': ['9 kiloliter/jam', '15 Rp juta/jam', '2 Rp juta/jam/minggu', 
                 '6 Rp juta/jam', '10 Rp juta/jam', '5 Rp juta/jam', '8 Rp juta/jam']
    }
    
    cost_df = pd.DataFrame(cost_data)
    st.dataframe(cost_df)
    
    # Setup costs
    setup_data = {
        'Produk': ['Eskrim Vanila', 'Eskrim Coklat', 'Eskrim Stroberi'],
        'Biaya Setup (Rp juta)': [10, 15, 20]
    }
    
    setup_df = pd.DataFrame(setup_data)
    st.dataframe(setup_df)
    
    # Jam kerja
    work_data = {
        'Parameter': ['Jam kerja regular time', 'Jam kerja overtime', 'Hari kerja'],
        'Nilai': ['8 jam/hari', '2 jam/hari', '5 hari/minggu']
    }
    
    work_df = pd.DataFrame(work_data)
    st.dataframe(work_df)
    
    # Penjelasan Aggregate Planning
    with st.expander("Penjelasan Aggregate Planning"):
        st.markdown("""
        ### Aggregate Planning
        
        Aggregate Planning adalah perencanaan produksi jangka menengah (6-18 bulan) yang bertujuan untuk menentukan tingkat produksi, tenaga kerja, dan persediaan yang optimal untuk memenuhi permintaan dengan biaya minimal.
        
        ### Mixed Strategy
        
        Mixed Strategy adalah pendekatan dalam aggregate planning yang mengkombinasikan beberapa strategi seperti:
        1. **Chase Strategy**: Menyesuaikan tingkat produksi mengikuti permintaan
        2. **Level Strategy**: Mempertahankan tingkat produksi konstan
        3. **Hybrid Strategy**: Kombinasi dari keduanya
        
        Komponen biaya yang diperhitungkan meliputi:
        - Regular time cost
        - Overtime cost
        - Inventory holding cost
        - Ramping up/down cost (biaya perubahan tingkat produksi)
        - Subcontracting cost
        
        Tujuannya adalah memilih kombinasi yang menghasilkan total biaya minimum.
        """)
    
    st.subheader("Pembuatan Rencana Agregat")
    
    # Hitung total permintaan per minggu
    demand_df['Total Permintaan (kL)'] = demand_df['Vanila'] + demand_df['Coklat'] + demand_df['Stroberi']
    
    # Konversi ke jam produksi dengan filling rate agregat
    filling_rate_agregat = 9  # kiloliter/jam
    demand_df['Kebutuhan Jam Produksi'] = demand_df['Total Permintaan (kL)'] / filling_rate_agregat
    
    st.write("Permintaan Total dan Kebutuhan Jam Produksi:")
    st.dataframe(demand_df)
    
    # Kapasitas produksi tersedia
    regular_time = 8 * 5  # 8 jam/hari * 5 hari/minggu = 40 jam/minggu
    overtime = 2 * 5      # 2 jam/hari * 5 hari/minggu = 10 jam/minggu
    total_capacity = regular_time + overtime  # 50 jam/minggu
    
    st.write(f"Jam Regular Time per minggu: {regular_time} jam")
    st.write(f"Jam Overtime per minggu: {overtime} jam")
    st.write(f"Total kapasitas per minggu: {total_capacity} jam")
    
    # Mixed Strategy (Trial and Error)
    st.subheader("Mixed Strategy (Trial and Error)")
    
    # Definisikan strategi mixed
    mixed_strategy = {
        'Minggu': list(range(1, 7)),
        'Permintaan (jam)': demand_df['Kebutuhan Jam Produksi'].tolist(),
        'Regular Time': [40, 40, 40, 40, 40, 33.33],
        'Overtime': [10, 0, 10, 10, 0, 0],
        'Subkontrak': [0, 0, 10, 27.78, 0, 0],
        'Inventory': [0, 6.67, 0, 0, 6.67, 0],
        'Ramping': ['Up', 'Down', 'Up', 'Konstan', 'Down', 'Down']
    }
    
    mixed_df = pd.DataFrame(mixed_strategy)
    st.dataframe(mixed_df)
    
    # Hitung biaya
    regular_cost = sum(mixed_df['Regular Time']) * 6  # 6 juta/jam
    overtime_cost = sum(mixed_df['Overtime']) * 10    # 10 juta/jam
    subcontract_cost = sum(mixed_df['Subkontrak']) * 15  # 15 juta/jam
    inventory_cost = sum(mixed_df['Inventory']) * 2    # 2 juta/jam/minggu
    
    # Hitung biaya ramping up/down
    ramping_up_count = mixed_df['Ramping'].value_counts().get('Up', 0)
    ramping_down_count = mixed_df['Ramping'].value_counts().get('Down', 0)
    
    ramping_up_cost = ramping_up_count * 5    # 5 juta/jam
    ramping_down_cost = ramping_down_count * 8  # 8 juta/jam
    
    total_cost = regular_cost + overtime_cost + subcontract_cost + inventory_cost + ramping_up_cost + ramping_down_cost
    
    # Tampilkan perhitungan biaya
    st.subheader("Perhitungan Biaya")
    
    cost_calculation = {
        'Komponen Biaya': ['Regular Time', 'Overtime', 'Subkontrak', 'Inventory', 
                          'Ramping Up', 'Ramping Down', 'Total'],
        'Detail Perhitungan': [
            f"{sum(mixed_df['Regular Time'])} jam × 6 juta",
            f"{sum(mixed_df['Overtime'])} jam × 10 juta",
            f"{sum(mixed_df['Subkontrak'])} jam × 15 juta",
            f"{sum(mixed_df['Inventory'])} jam × 2 juta",
            f"{ramping_up_count} kali × 5 juta",
            f"{ramping_down_count} kali × 8 juta",
            "Jumlah seluruh biaya"
        ],
        'Biaya (Rp juta)': [
            regular_cost,
            overtime_cost,
            subcontract_cost,
            inventory_cost,
            ramping_up_cost,
            ramping_down_cost,
            total_cost
        ]
    }
    
    cost_df = pd.DataFrame(cost_calculation)
    st.dataframe(cost_df)
    
    st.success(f"Total Biaya Mixed Strategy: Rp {total_cost:.2f} juta")
    
    # Visualisasi rencana agregat
    fig = go.Figure()
    
    # Demand line
    fig.add_trace(go.Scatter(
        x=mixed_df['Minggu'], 
        y=mixed_df['Permintaan (jam)'],
        mode='lines+markers',
        name='Permintaan'
    ))
    
    # Regular time
    fig.add_trace(go.Bar(
        x=mixed_df['Minggu'],
        y=mixed_df['Regular Time'],
        name='Regular Time',
        marker_color='blue'
    ))
    
    # Overtime
    fig.add_trace(go.Bar(
        x=mixed_df['Minggu'],
        y=mixed_df['Overtime'],
        name='Overtime',
        marker_color='green'
    ))
    
    # Subcontract
    fig.add_trace(go.Bar(
        x=mixed_df['Minggu'],
        y=mixed_df['Subkontrak'],
        name='Subkontrak',
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Rencana Produksi Agregat',
        xaxis_title='Minggu',
        yaxis_title='Jam',
        barmode='stack',
        hovermode='x'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Fitur ekspor hasil ke Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        demand_df.to_excel(writer, sheet_name='Data Permintaan', index=False)
        mixed_df.to_excel(writer, sheet_name='Mixed Strategy', index=False)
        cost_df.to_excel(writer, sheet_name='Perhitungan Biaya', index=False)
    
    st.download_button(
        label="Download Hasil Perhitungan Soal 3 (Excel)",
        data=output.getvalue(),
        file_name="Hasil_Soal3_Aggregate_Planning.xlsx",
        mime="application/vnd.ms-excel"
    )

# Soal 4: Master Production Schedule
if page == "Soal 4: Master Production Schedule":
    st.header("Soal 4: LSP Bakery - Master Production Schedule")
    
    # Data produksi agregat
    st.subheader("Rencana Produksi Agregat 2025")
    
    prod_data = {
        'Bulan': ['Jan', 'Feb', 'Mar', 'Apr'],
        'Jumlah Produksi (ribu unit)': [98.3, 97.0, 105.3, 114.1]
    }
    
    prod_df = pd.DataFrame(prod_data)
    st.dataframe(prod_df)
    
    # Kebijakan produksi
    st.subheader("Kebijakan Produksi")
    
    policy_data = {
        'Famili': ['Roti Hamburger', 'Roti Hamburger', 'Roti Hamburger', 
                  'Roti Hotdog', 'Roti Hotdog', 'Roti Bagel', 'Roti Bagel'],
        'Item': ['White', 'Wijen', 'WholeGrain', 'Plain', 'Garlic', 'Savory', 'Sweet'],
        'Konversi item ke agregat': [0.2, 0.3, 0.4, 0.15, 0.2, 0.1, 0.15],
        'Safety Stock': [120, 60, 45, 90, 80, 75, 50]
    }
    
    policy_df = pd.DataFrame(policy_data)
    st.dataframe(policy_df)
    
    # Permintaan dan inventory
    st.subheader("Perkiraan Permintaan dan Posisi Inventory")
    
    inventory_data = {
        'Famili': ['Roti Hamburger', 'Roti Hamburger', 'Roti Hamburger', 
                  'Roti Hotdog', 'Roti Hotdog', 'Roti Bagel', 'Roti Bagel'],
        'Item': ['White', 'Wijen', 'WholeGrain', 'Plain', 'Garlic', 'Savory', 'Sweet'],
        'Inventory Akhir Des': [335.2, 148.7, 57.5, 280.4, 96.2, 101, 121],
        'Jan': [210.6, 84.0, 15.4, 175.2, 27.2, 20.7, 65.3],
        'Feb': [172.0, 74.8, 16.0, 198.0, 17.6, 22.8, 71.8],
        'Mar': [255.2, 88.8, 16.8, 178.0, 22.0, 29.6, 93.4],
        'Apr': [360.0, 57.6, 19.2, 128.0, 36.0, 44.4, 140.1]
    }
    
    demand_inv_df = pd.DataFrame(inventory_data)
    st.dataframe(demand_inv_df)
    
    # Biaya setup
    setup_cost_data = {
        'Famili': ['Roti Hamburger', 'Roti Hotdog', 'Roti Bagel'],
        'Biaya Setup (Miliar Rp)': [9, 4, 6]
    }
    
    setup_cost_df = pd.DataFrame(setup_cost_data)
    st.dataframe(setup_cost_df)
    
    # Penjelasan Metode Bitran Hax
    with st.expander("Penjelasan Metode Bitran Hax"):
        st.markdown("""
        ### Master Production Schedule dengan Metode Bitran Hax
        
        Metode Bitran Hax adalah pendekatan untuk menghasilkan jadwal produksi induk (MPS) dari rencana agregat dengan mempertimbangkan biaya setup dan konversi unit.
        
        **Langkah-langkah Metode Bitran Hax:**
        
        1. **Hitung kebutuhan bersih** untuk setiap item:
           - Kebutuhan Bersih = Permintaan - (Inventory - Safety Stock)
           
        2. **Hitung batas atas** untuk keluarga produk dengan periode N:
           - Batas Atas = Total Permintaan N periode - (Total Inventory - Total Safety Stock)
           
        3. **Tentukan keluarga mana yang diproduksi** berdasarkan rasio kebutuhan/biaya setup
        
        4. **Alokasi produksi agregat** ke item berdasarkan prioritas
        
        5. **Hasilkan Master Production Schedule**
        """)
    
    st.subheader("Perhitungan Master Production Schedule dengan Metode Bitran Hax")
    
    # Perhitungan kebutuhan bersih untuk setiap item
    st.write("1. Kebutuhan Bersih untuk Setiap Item")
    
    # Hitung kebutuhan bersih untuk Januari
    net_req_data = []
    
    for i in range(len(demand_inv_df)):
        item = demand_inv_df.iloc[i]
        demand_jan = item['Jan']
        inv_dec = item['Inventory Akhir Des']
        ss = policy_df.iloc[i]['Safety Stock']
        
        net_req = demand_jan - (inv_dec - ss)
        
        net_req_data.append({
            'Famili': item['Famili'],
            'Item': item['Item'],
            'Permintaan Jan': demand_jan,
            'Inventory Akhir Des': inv_dec,
            'Safety Stock': ss,
            'Perhitungan': f"{demand_jan} - ({inv_dec} - {ss})",
            'Kebutuhan Bersih': net_req
        })
    
    net_req_df = pd.DataFrame(net_req_data)
    st.dataframe(net_req_df)
    
    # Perhitungan batas atas untuk keluarga produk (N=2)
    st.write("2. Batas Atas untuk Keluarga Produk (N=2)")
    
    # Agregasi berdasarkan famili
    hamburger_data = net_req_df[net_req_df['Famili'] == 'Roti Hamburger']
    hotdog_data = net_req_df[net_req_df['Famili'] == 'Roti Hotdog']
    bagel_data = net_req_df[net_req_df['Famili'] == 'Roti Bagel']
    
    # Jumlah permintaan untuk 2 periode
    hamburger_demand_jan_feb = sum(demand_inv_df[demand_inv_df['Famili'] == 'Roti Hamburger']['Jan'] + 
                                   demand_inv_df[demand_inv_df['Famili'] == 'Roti Hamburger']['Feb'])
    hotdog_demand_jan_feb = sum(demand_inv_df[demand_inv_df['Famili'] == 'Roti Hotdog']['Jan'] + 
                               demand_inv_df[demand_inv_df['Famili'] == 'Roti Hotdog']['Feb'])
    bagel_demand_jan_feb = sum(demand_inv_df[demand_inv_df['Famili'] == 'Roti Bagel']['Jan'] + 
                              demand_inv_df[demand_inv_df['Famili'] == 'Roti Bagel']['Feb'])
    
    # Total inventory dan safety stock
    hamburger_inv = sum(hamburger_data['Inventory Akhir Des'])
    hamburger_ss = sum(hamburger_data['Safety Stock'])
    
    hotdog_inv = sum(hotdog_data['Inventory Akhir Des'])
    hotdog_ss = sum(hotdog_data['Safety Stock'])
    
    bagel_inv = sum(bagel_data['Inventory Akhir Des'])
    bagel_ss = sum(bagel_data['Safety Stock'])
    
    # Hitung batas atas
    hamburger_upper = hamburger_demand_jan_feb - (hamburger_inv - hamburger_ss)
    hotdog_upper = hotdog_demand_jan_feb - (hotdog_inv - hotdog_ss)
    bagel_upper = bagel_demand_jan_feb - (bagel_inv - bagel_ss)
    
    upper_bound_data = [
        {
            'Famili': 'Roti Hamburger',
            'Total Permintaan Jan-Feb': hamburger_demand_jan_feb,
            'Total Inventory': hamburger_inv,
            'Total Safety Stock': hamburger_ss,
            'Perhitungan': f"{hamburger_demand_jan_feb} - ({hamburger_inv} - {hamburger_ss})",
            'Batas Atas': hamburger_upper
        },
        {
            'Famili': 'Roti Hotdog',
            'Total Permintaan Jan-Feb': hotdog_demand_jan_feb,
            'Total Inventory': hotdog_inv,
            'Total Safety Stock': hotdog_ss,
            'Perhitungan': f"{hotdog_demand_jan_feb} - ({hotdog_inv} - {hotdog_ss})",
            'Batas Atas': hotdog_upper
        },
        {
            'Famili': 'Roti Bagel',
            'Total Permintaan Jan-Feb': bagel_demand_jan_feb,
            'Total Inventory': bagel_inv,
            'Total Safety Stock': bagel_ss,
            'Perhitungan': f"{bagel_demand_jan_feb} - ({bagel_inv} - {bagel_ss})",
            'Batas Atas': bagel_upper
        }
    ]
    
    upper_bound_df = pd.DataFrame(upper_bound_data)
    st.dataframe(upper_bound_df)
    
    # Rasio kebutuhan/biaya setup
    st.write("3. Rasio Kebutuhan/Biaya Setup")
    
    # Hitung rasio
    setup_ratio_data = []
    
    for i in range(len(upper_bound_df)):
        famili = upper_bound_df.iloc[i]['Famili']
        upper_bound = upper_bound_df.iloc[i]['Batas Atas']
        setup_cost = setup_cost_df[setup_cost_df['Famili'] == famili]['Biaya Setup (Miliar Rp)'].values[0]
        
        if upper_bound > 0:
            ratio = upper_bound / setup_cost
        else:
            ratio = 0
        
        setup_ratio_data.append({
            'Famili': famili,
            'Batas Atas': upper_bound,
            'Biaya Setup (Miliar Rp)': setup_cost,
            'Perhitungan': f"{upper_bound} / {setup_cost}",
            'Rasio': ratio
        })
    
    setup_ratio_df = pd.DataFrame(setup_ratio_data)
    st.dataframe(setup_ratio_df)
    
    # Tentukan keluarga yang diproduksi
    selected_families = setup_ratio_df[setup_ratio_df['Batas Atas'] > 0].sort_values('Rasio', ascending=False)
    st.write("Berdasarkan rasio, keluarga yang diproduksi (Batas Atas > 0, urut berdasarkan rasio):")
    st.dataframe(selected_families[['Famili', 'Rasio']])
    
    # Alokasi produksi
    st.write("4. Alokasi Produksi Agregat ke Item")
    
    # Filter hanya item dengan kebutuhan bersih positif dari keluarga terpilih
    positive_items = net_req_df[
        (net_req_df['Kebutuhan Bersih'] > 0) & 
        (net_req_df['Famili'].isin(selected_families['Famili']))
    ]
    
    if not positive_items.empty:
        # Konversi ke unit agregat
        allocation_data = []
        
        for _, item in positive_items.iterrows():
            famili = item['Famili']
            item_name = item['Item']
            net_req = item['Kebutuhan Bersih']
            
            # Ambil faktor konversi
            conversion = policy_df[(policy_df['Famili'] == famili) & 
                                  (policy_df['Item'] == item_name)]['Konversi item ke agregat'].values[0]
            
            # Konversi ke unit agregat
            aggregate_units = net_req * conversion
            
            allocation_data.append({
                'Famili': famili,
                'Item': item_name,
                'Kebutuhan Bersih (ribu unit)': net_req,
                'Faktor Konversi': conversion,
                'Unit Agregat (ribu unit)': aggregate_units
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df)
        
        # Master Production Schedule
        st.subheader("Master Production Schedule untuk Januari 2025")
        
        mps_data = []
        
        for _, item in positive_items.iterrows():
            famili = item['Famili']
            item_name = item['Item']
            production = item['Kebutuhan Bersih']
            
            if production > 0:
                mps_data.append({
                    'Famili': famili,
                    'Item': item_name,
                    'Jumlah Produksi (ribu unit)': production
                })
        
        if mps_data:
            mps_df = pd.DataFrame(mps_data)
            st.dataframe(mps_df)
        else:
            st.write("Tidak ada item yang memerlukan produksi pada bulan Januari 2025.")
    else:
        st.write("Tidak ada item dengan kebutuhan bersih positif dari keluarga terpilih.")
        
        # Master Production Schedule (kosong)
        st.subheader("Master Production Schedule untuk Januari 2025")
        st.write("Tidak ada item yang memerlukan produksi pada bulan Januari 2025.")
    
    # Fitur ekspor hasil ke Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        prod_df.to_excel(writer, sheet_name='Produksi Agregat', index=False)
        policy_df.to_excel(writer, sheet_name='Kebijakan Produksi', index=False)
        demand_inv_df.to_excel(writer, sheet_name='Permintaan & Inventory', index=False)
        net_req_df.to_excel(writer, sheet_name='Kebutuhan Bersih', index=False)
        upper_bound_df.to_excel(writer, sheet_name='Batas Atas', index=False)
        setup_ratio_df.to_excel(writer, sheet_name='Rasio Setup', index=False)
        if 'allocation_df' in locals():
            allocation_df.to_excel(writer, sheet_name='Alokasi Produksi', index=False)
        if 'mps_df' in locals() and not mps_df.empty:
            mps_df.to_excel(writer, sheet_name='MPS', index=False)
    
    st.download_button(
        label="Download Hasil Perhitungan Soal 4 (Excel)",
        data=output.getvalue(),
        file_name="Hasil_Soal4_MPS.xlsx",
        mime="application/vnd.ms-excel"
    )

# Soal 5: Line Balancing
if page == "Soal 5: Line Balancing":
    st.header("Soal 5: PT Jason - Line Balancing")
    
    # Data task dan diagram jaringan
    st.subheader("Diagram Jaringan Proses Perakitan")
    
    # Tampilkan gambar diagram jaringan
    st.markdown("""
    ```
          4
          ↑
          B
         ↗ ↘
        ↗    ↘ 5
    5  ↗      ↘
      ↗        ↘
     A          E
      ↘        ↗
    6  ↘      ↗ 6
        ↘    ↗
         ↘  ↗
          C → G
           ↘ ↗
          4 ↓ 4
            D → F
    ```
    """)
    
    # Data task
    task_data = {
        'Task': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Waktu (detik)': [5, 4, 6, 4, 5, 4, 6],
        'Predecessors': ['-', 'A', 'A', 'A', 'B,C', 'D', 'E,F']
    }
    
    task_df = pd.DataFrame(task_data)
    st.dataframe(task_df)
    
    # Data permintaan dan waktu siklus
    st.write("Permintaan: 6 produk/menit")
    st.write("Waktu siklus: 10 detik/produk")
    
    total_task_time = sum(task_df['Waktu (detik)'])
    min_stations = math.ceil(total_task_time / 10)
    
    st.write(f"Total waktu task: {total_task_time} detik")
    st.write(f"Jumlah workstation minimum: {min_stations}")
    
    # Penjelasan metode-metode line balancing
    with st.expander("Penjelasan Metode-Metode Line Balancing"):
        st.markdown("""
        ### Metode Line Balancing
        
        Line Balancing adalah proses penyeimbangan beban kerja pada setiap workstation dalam lini perakitan untuk meminimalkan idle time dan memaksimalkan efisiensi.
        
        ### 1. Kilbridge-Weston Heuristic
        
        Metode ini mengelompokkan task berdasarkan posisinya dalam diagram jaringan precedence (kolom). Task dipilih berdasarkan kolom, dimulai dari kolom pertama.
        
        ### 2. Helgeson-Birnie Method (Largest Candidate Rule)
        
        Metode ini memilih task berdasarkan waktu terbesar terlebih dahulu, dengan tetap memperhatikan precedence.
        
        ### 3. Regional Approach
        
        Metode ini mengelompokkan task berdasarkan region dalam diagram jaringan. Task dalam region yang sama cenderung ditempatkan dalam workstation yang sama.
        
        ### 4. Largest Candidate Rule
        
        Mirip dengan Helgeson-Birnie, metode ini memprioritaskan task dengan waktu terbesar, namun lebih menekankan pada precedence relationship.
        
        ### Evaluasi Efisiensi
        
        Efisiensi lini dihitung dengan rumus:
        
        $Efisiensi = \\frac{\\sum \text{Waktu Task}}{\\text{Jumlah Workstation} \\times \\text{Waktu Siklus}} \\times 100\\%$
        """)
    
    # Implementasi metode-metode line balancing
    st.subheader("5a) Pembagian Workstation dengan Berbagai Metode")
    
    # 1. Metode Kilbridge-Weston
    st.write("### 1. Kilbridge-Weston Heuristic")
    
    kilbridge_columns = {
        1: ['A'],
        2: ['B', 'C', 'D'],
        3: ['E', 'F'],
        4: ['G']
    }
    
    kw_stations = []
    current_station = []
    current_time = 0
    
    # Iterasi melalui kolom
    for col in kilbridge_columns:
        # Urutkan task dalam kolom berdasarkan waktu terbesar
        sorted_tasks = sorted(kilbridge_columns[col], 
                             key=lambda x: task_df[task_df['Task'] == x]['Waktu (detik)'].values[0],
                             reverse=True)
        
        for task in sorted_tasks:
            task_time = task_df[task_df['Task'] == task]['Waktu (detik)'].values[0]
            
            # Cek apakah task bisa ditambahkan ke workstation saat ini
            if current_time + task_time <= 10:
                current_station.append(task)
                current_time += task_time
            else:
                # Simpan workstation saat ini dan buat yang baru
                kw_stations.append({
                    'Workstation': len(kw_stations) + 1,
                    'Tasks': current_station.copy(),
                    'Total Waktu': current_time
                })
                current_station = [task]
                current_time = task_time
    
    # Tambahkan workstation terakhir jika ada
    if current_station:
        kw_stations.append({
            'Workstation': len(kw_stations) + 1,
            'Tasks': current_station.copy(),
            'Total Waktu': current_time
        })
    
    # Tampilkan hasil Kilbridge-Weston
    kw_df = pd.DataFrame(kw_stations)
    kw_df['Tasks'] = kw_df['Tasks'].apply(lambda x: ', '.join(x))
    st.dataframe(kw_df)
    
    # 2. Metode Helgeson-Birnie
    st.write("### 2. Helgeson-Birnie Method")
    
    # Urutkan task berdasarkan waktu terbesar
    sorted_tasks = task_df.sort_values('Waktu (detik)', ascending=False)['Task'].tolist()
    
    # Fungsi untuk memeriksa apakah task dapat ditambahkan (predecessors sudah selesai)
    def check_predecessors(task, completed_tasks):
        predecessors = task_df[task_df['Task'] == task]['Predecessors'].values[0]
        if predecessors == '-':
            return True
        for pred in predecessors.split(','):
            if pred not in completed_tasks:
                return False
        return True
    
    hb_stations = []
    current_station = []
    current_time = 0
    completed_tasks = []
    
    # Coba tempatkan setiap task
    remaining_tasks = sorted_tasks.copy()
    
    while remaining_tasks:
        task_added = False
        
        for task in remaining_tasks:
            # Cek apakah predecessors sudah selesai
            if check_predecessors(task, completed_tasks):
                task_time = task_df[task_df['Task'] == task]['Waktu (detik)'].values[0]
                
                # Cek apakah task bisa ditambahkan ke workstation saat ini
                if current_time + task_time <= 10:
                    current_station.append(task)
                    current_time += task_time
                    completed_tasks.append(task)
                    remaining_tasks.remove(task)
                    task_added = True
                    break
        
        # Jika tidak ada task yang bisa ditambahkan, buat workstation baru
        if not task_added:
            if current_station:
                hb_stations.append({
                    'Workstation': len(hb_stations) + 1,
                    'Tasks': current_station.copy(),
                    'Total Waktu': current_time
                })
                current_station = []
                current_time = 0
            # Jika tidak ada workstation yang bisa dibuat, maka ada task yang tidak bisa ditempatkan
            else:
                break
    
    # Tambahkan workstation terakhir jika ada
    if current_station:
        hb_stations.append({
            'Workstation': len(hb_stations) + 1,
            'Tasks': current_station.copy(),
            'Total Waktu': current_time
        })
    
    # Tampilkan hasil Helgeson-Birnie
    hb_df = pd.DataFrame(hb_stations)
    hb_df['Tasks'] = hb_df['Tasks'].apply(lambda x: ', '.join(x))
    st.dataframe(hb_df)
    
    # 3. Regional Approach
    st.write("### 3. Regional Approach")
    
    # Definisikan region
    regions = {
        1: ['A'],
        2: ['B', 'C', 'D'],
        3: ['E', 'F'],
        4: ['G']
    }
    
    ra_stations = []
    current_station = []
    current_time = 0
    completed_tasks = []
    
    # Iterasi melalui region
    for region in regions:
        # Coba tempatkan setiap task dalam region
        for task in regions[region]:
            # Cek apakah predecessors sudah selesai
            if check_predecessors(task, completed_tasks):
                task_time = task_df[task_df['Task'] == task]['Waktu (detik)'].values[0]
                
                # Cek apakah task bisa ditambahkan ke workstation saat ini
                if current_time + task_time <= 10:
                    current_station.append(task)
                    current_time += task_time
                    completed_tasks.append(task)
                else:
                    # Simpan workstation saat ini dan buat yang baru
                    ra_stations.append({
                        'Workstation': len(ra_stations) + 1,
                        'Tasks': current_station.copy(),
                        'Total Waktu': current_time
                    })
                    current_station = [task]
                    current_time = task_time
                    completed_tasks.append(task)
    
    # Tambahkan workstation terakhir jika ada
    if current_station:
        ra_stations.append({
            'Workstation': len(ra_stations) + 1,
            'Tasks': current_station.copy(),
            'Total Waktu': current_time
        })
    
    # Tampilkan hasil Regional Approach
    ra_df = pd.DataFrame(ra_stations)
    ra_df['Tasks'] = ra_df['Tasks'].apply(lambda x: ', '.join(x))
    st.dataframe(ra_df)
    
    # 4. Largest Candidate Rule
    st.write("### 4. Largest Candidate Rule")
    
    # Urutkan task berdasarkan waktu terbesar
    sorted_tasks = task_df.sort_values('Waktu (detik)', ascending=False)['Task'].tolist()
    
    lcr_stations = []
    current_station = []
    current_time = 0
    completed_tasks = []
    
    # Coba tempatkan setiap task
    remaining_tasks = sorted_tasks.copy()
    
    while remaining_tasks:
        task_added = False
        
        for task in remaining_tasks:
            # Cek apakah predecessors sudah selesai
            if check_predecessors(task, completed_tasks):
                task_time = task_df[task_df['Task'] == task]['Waktu (detik)'].values[0]
                
                # Cek apakah task bisa ditambahkan ke workstation saat ini
                if current_time + task_time <= 10:
                    current_station.append(task)
                    current_time += task_time
                    completed_tasks.append(task)
                    remaining_tasks.remove(task)
                    task_added = True
        
        # Jika tidak ada task yang bisa ditambahkan, buat workstation baru
        if not task_added:
            if current_station:
                lcr_stations.append({
                    'Workstation': len(lcr_stations) + 1,
                    'Tasks': current_station.copy(),
                    'Total Waktu': current_time
                })
                current_station = []
                current_time = 0
            # Jika tidak ada workstation yang bisa dibuat, maka ada task yang tidak bisa ditempatkan
            else:
                break
    
    # Tambahkan workstation terakhir jika ada
    if current_station:
        lcr_stations.append({
            'Workstation': len(lcr_stations) + 1,
            'Tasks': current_station.copy(),
            'Total Waktu': current_time
        })
    
    # Tampilkan hasil Largest Candidate Rule
    lcr_df = pd.DataFrame(lcr_stations)
    lcr_df['Tasks'] = lcr_df['Tasks'].apply(lambda x: ', '.join(x))
    st.dataframe(lcr_df)
    
    # Perbandingan efisiensi dan jumlah workstation
    st.subheader("5b) Perbandingan Metode Line Balancing")
    
    # Hitung efisiensi untuk setiap metode
    kw_efficiency = (total_task_time / (len(kw_stations) * 10)) * 100
    hb_efficiency = (total_task_time / (len(hb_stations) * 10)) * 100
    ra_efficiency = (total_task_time / (len(ra_stations) * 10)) * 100
    lcr_efficiency = (total_task_time / (len(lcr_stations) * 10)) * 100
    
    comparison_data = {
        'Metode': ['Kilbridge-Weston', 'Helgeson-Birnie', 'Regional Approach', 'Largest Candidate Rule'],
        'Jumlah Workstation': [len(kw_stations), len(hb_stations), len(ra_stations), len(lcr_stations)],
        'Efisiensi (%)': [kw_efficiency, hb_efficiency, ra_efficiency, lcr_efficiency]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    
    # Tentukan metode terbaik
    best_method = comparison_df.loc[comparison_df['Efisiensi (%)'].idxmax()]['Metode']
    max_efficiency = comparison_df['Efisiensi (%)'].max()
    min_stations = comparison_df['Jumlah Workstation'].min()
    
    best_methods = comparison_df[comparison_df['Jumlah Workstation'] == min_stations].sort_values('Efisiensi (%)', ascending=False)
    
    if not best_methods.empty:
        final_best = best_methods.iloc[0]['Metode']
        st.success(f"Metode terbaik adalah **{final_best}** dengan {min_stations} workstation dan efisiensi {best_methods.iloc[0]['Efisiensi (%)']:.2f}%.")
    else:
        st.success(f"Metode terbaik adalah **{best_method}** dengan efisiensi {max_efficiency:.2f}%.")
    
    # Visualisasi perbandingan metode
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison_df['Metode'],
        y=comparison_df['Efisiensi (%)'],
        name='Efisiensi (%)',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Scatter(
        x=comparison_df['Metode'],
        y=comparison_df['Jumlah Workstation'],
        name='Jumlah Workstation',
        marker_color='red',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Perbandingan Metode Line Balancing',
        xaxis_title='Metode',
        yaxis_title='Efisiensi (%)',
        yaxis2=dict(
            title='Jumlah Workstation',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Fitur ekspor hasil ke Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        task_df.to_excel(writer, sheet_name='Data Task', index=False)
        kw_df.to_excel(writer, sheet_name='Kilbridge-Weston', index=False)
        hb_df.to_excel(writer, sheet_name='Helgeson-Birnie', index=False)
        ra_df.to_excel(writer, sheet_name='Regional Approach', index=False)
        lcr_df.to_excel(writer, sheet_name='Largest Candidate Rule', index=False)
        comparison_df.to_excel(writer, sheet_name='Perbandingan', index=False)
    
    st.download_button(
        label="Download Hasil Perhitungan Soal 5 (Excel)",
        data=output.getvalue(),
        file_name="Hasil_Soal5_Line_Balancing.xlsx",
        mime="application/vnd.ms-excel"
    )

# Informasi tambahan di sidebar
with st.sidebar:
    st.subheader("Informasi Aplikasi")
    st.write("Aplikasi ini mengotomatisasi perhitungan untuk 5 soal tugas PPC.")
    st.write("Mulai dengan memasukkan NIM pada halaman 'Input NIM'.")
    
    st.subheader("Metode yang Digunakan")
    st.write("Soal 1: Single dan Double Exponential Smoothing")
    st.write("Soal 2: Winter's Method")
    st.write("Soal 3: Aggregate Planning dengan Mixed Strategy")
    st.write("Soal 4: Master Production Schedule dengan Metode Bitran Hax")
    st.write("Soal 5: Line Balancing dengan 4 metode berbeda")