# Panduan Lengkap: Mendapatkan API Keys (GRATIS)

## Yang Perlu Anda Lakukan SEKARANG

### ✅ Step 1: Password PostgreSQL Anda

Edit file `.env` di baris ke-6:
```
DB_PASSWORD=YOUR_POSTGRES_PASSWORD_HERE
```
Ganti dengan password PostgreSQL yang Anda gunakan saat install.

Kalau lupa password PostgreSQL:
1. Buka pgAdmin
2. Atau cek di PostgreSQL service properties
3. Atau reset password via command line

---

## API Keys (Bisa dikerjakan sambil data collection berjalan)

### 🇮🇩 1. BPS API Key (GRATIS - Untuk Data Demografi)

**Waktu: ~5 menit**

1. **Kunjungi**: https://webapi.bps.go.id/developer/

2. **Klik "Daftar"** (pojok kanan atas)

3. **Isi Form Registrasi**:
   - Nama lengkap
   - Email (gunakan email aktif)
   - Institusi/Perusahaan: (nama company/pribadi)
   - Nomor HP

4. **Verifikasi Email**:
   - Check inbox email Anda
   - Klik link verifikasi dari BPS

5. **Login & Dapatkan API Key**:
   - Login ke dashboard
   - Menu "API Key" atau "Developer"
   - Copy API key yang muncul

6. **Tambahkan ke `.env`**:
   ```
   BPS_API_KEY=key_yang_anda_dapat_dari_bps
   ```

**Quota**: Unlimited, gratis selamanya!

---

### 🌍 2. Google Places API (GRATIS $200/bulan)

**Waktu: ~10 menit**

1. **Kunjungi**: https://console.cloud.google.com/

2. **Login/Daftar** dengan Google Account

3. **Buat Project Baru**:
   - Klik "Select a project" di navbar
   - Klik "NEW PROJECT"
   - Nama project: `jakarta-poi-mvp`
   - Klik "CREATE"

4. **Enable Places API**:
   - Di search bar, cari "Places API"
   - Klik "Places API"
   - Klik "ENABLE"

5. **Buat API Key**:
   - Menu kiri: APIs & Services → Credentials
   - Klik "+ CREATE CREDENTIALS"
   - Pilih "API key"
   - Copy API key yang muncul

6. **Restrict API Key (Penting!)** - Agar aman:
   - Klik ikon edit di API key yang baru dibuat
   - Application restrictions: None (untuk development)
   - API restrictions:
     - Pilih "Restrict key"
     - Centang "Places API"
   - Save

7. **Setup Billing** (Perlu kartu kredit, tapi TIDAK akan dicharge):
   - Menu kiri: Billing
   - Klik "Link a billing account" atau "Add billing account"
   - Isi data kartu kredit
   - **PENTING**: Set budget alert di $10 untuk keamanan

8. **Tambahkan ke `.env`**:
   ```
   GOOGLE_PLACES_API_KEY=key_yang_anda_dapat_dari_google
   ```

**Quota**: $200 GRATIS per bulan
- Text Search: ~$32/1000 requests
- Dengan $200 kredit = ~6,000 free searches/month
- Cukup untuk testing dan development!

---

### 🗺️ 3. Foursquare Places API (GRATIS 10,000 calls/month)

**Waktu: ~5 menit**

1. **Kunjungi**: https://foursquare.com/developers/apps

2. **Sign Up / Login**:
   - Bisa pakai Google/GitHub/Email

3. **Create New App**:
   - Klik "Create a new app"
   - App name: `Jakarta POI MVP`
   - Website URL: https://localhost (untuk development)
   - Accept terms & Create

4. **Get API Key**:
   - Masuk ke app yang baru dibuat
   - Copy "API Key" yang muncul
   - (Bukan "Secret"!)

5. **Tambahkan ke `.env`**:
   ```
   FOURSQUARE_API_KEY=key_yang_anda_dapat_dari_foursquare
   ```

**Quota**: 10,000 requests GRATIS per bulan!

---

## Testing API Keys

Setelah semua API key diisi di `.env`, test koneksi:

```bash
# Test BPS API
python -c "from src.data.collect_bps import BPSCollector; c = BPSCollector(); print('BPS OK' if c.api_key else 'BPS Not Configured')"

# Test Google Places
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Google OK' if os.getenv('GOOGLE_PLACES_API_KEY') else 'Google Not Configured')"

# Test Foursquare
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Foursquare OK' if os.getenv('FOURSQUARE_API_KEY') else 'Foursquare Not Configured')"
```

---

## Alternatif: Mulai Tanpa API Keys

Jika mau mulai cepat tanpa API keys dulu:

```bash
# Collect OSM data saja (tidak perlu API key)
python src/data/collect_osm.py

# Training data bisa dicari alternatif atau manual scraping
```

API keys bisa ditambahkan belakangan untuk data tambahan!

---

## Troubleshooting

### "Invalid API Key" - BPS
- Pastikan sudah verifikasi email
- Login ulang ke dashboard BPS
- Generate ulang API key kalau perlu

### "Billing not enabled" - Google
- Harus setup billing (perlu kartu kredit)
- Tapi TIDAK akan dicharge selama masih di bawah $200/month
- Set budget alert untuk keamanan

### "Rate limit exceeded" - Foursquare
- Free tier: 10k calls/month
- Reset setiap bulan
- Kalau habis, tunggu reset atau upgrade

### "API Key not found in .env"
- Pastikan .env file ada di root directory project
- Pastikan API key tidak ada spasi atau quotes
- Format: `API_KEY=value_tanpa_quotes`

---

## Summary

**Yang WAJIB sekarang**:
- ✅ PostgreSQL password di `.env`

**Yang bisa sambil jalan**:
- ⏳ BPS API (~5 menit)
- ⏳ Google Places (~10 menit, perlu kartu kredit)
- ⏳ Foursquare (~5 menit)

**Total waktu**: ~20-30 menit untuk semua API keys

**Total biaya**: **$0** (semua gratis!)

Data collection bisa mulai dengan OSM dulu (tidak perlu API key), API keys bisa ditambahkan sambil jalan!
