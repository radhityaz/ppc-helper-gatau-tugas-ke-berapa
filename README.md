# PT LSP - Perencanaan Produksi

Aplikasi Streamlit untuk perencanaan produksi PT LSP dengan fitur peramalan penjualan menggunakan metode penghalusan eksponensial (SES dan DES) dan interpretasi AI dengan Gemini.

## Fitur

- Peramalan penjualan dengan metode SES (Single Exponential Smoothing) dan DES (Double Exponential Smoothing)
- Optimasi parameter alpha dan beta
- Visualisasi data penjualan dan hasil peramalan
- Interpretasi hasil dengan AI (Gemini)
- Perbandingan metrik kesalahan (MFE, MAD, MSE, MAPE)
- Peramalan untuk 3 bulan ke depan

## Instalasi

1. Clone repository ini:
   ```
   git clone https://github.com/username/pt-lsp-perencanaan-produksi.git
   cd pt-lsp-perencanaan-produksi
   ```

2. Instal dependensi:
   ```
   pip install -r requirements.txt
   ```

3. Setup API Key Gemini:
   - Buat folder `.streamlit` di direktori proyek jika belum ada
   - Buat file `secrets.toml` di dalam folder `.streamlit`
   - Tambahkan API key Gemini Anda ke file tersebut:
     ```
     GEMINI_API_KEY = "YOUR_API_KEY_HERE"
     ```

## Penggunaan

Jalankan aplikasi dengan perintah:
```
streamlit run ppc.py
```

## Catatan Penting

- File `.streamlit/secrets.toml` sudah ditambahkan ke `.gitignore` dan tidak akan di-push ke GitHub
- Pastikan Anda tidak meng-commit file yang berisi API key atau informasi sensitif lainnya
- Setiap pengguna yang mengkloning repository ini perlu membuat file `secrets.toml` mereka sendiri dengan API key mereka

## Masalah yang Tersedia

1. Peramalan Penjualan - Menggunakan metode SES dan DES
2. Metode Winter - (Coming soon)
3. Perencanaan Agregat - (Coming soon)
4. Jadwal Produksi Induk - (Coming soon)
5. Penyeimbangan Lini Perakitan - (Coming soon)
