# Dashboard Deteksi Kesiapan Panen Maggot

Dashboard Streamlit ini memakai model YOLO11 pada `yolo_maggot/weights/best.pt` untuk mendeteksi tiga fase maggot dan menyimpulkan kesiapan panen.

## Menjalankan aplikasi

Disarankan memakai Python 3.10–3.12 agar instalasi PyTorch/Ultralytics berjalan mulus.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Masukkan gambar melalui unggahan file atau kamera langsung. Dashboard akan menampilkan bounding box, jumlah objek, confidence, komposisi fase, kesimpulan, dan berkas CSV yang dapat diunduh.

## Menggunakan kamera komputer atau HP

Pilih **Kamera langsung** pada bagian sumber foto, izinkan akses kamera pada browser, arahkan kamera ke maggot, kemudian ambil foto. Kamera yang digunakan mengikuti perangkat yang membuka dashboard:

- dashboard dibuka di komputer → memakai webcam komputer;
- dashboard dibuka di HP → memakai kamera HP.

Untuk mengakses dashboard dari HP pada jaringan Wi-Fi yang sama, jalankan:

```powershell
streamlit run app.py --server.address 0.0.0.0
```

Kemudian buka alamat jaringan yang ditampilkan Streamlit pada browser HP. Akses kamera melalui alamat jaringan lokal dapat dibatasi oleh browser karena bukan koneksi HTTPS. Untuk penggunaan HP yang stabil, jalankan dashboard pada domain/deployment HTTPS. Jika izin kamera diblokir, gunakan pilihan **Unggah gambar** sebagai alternatif.

## Logika kesimpulan

Persentase dihitung dari jumlah confidence per kelas yang dinormalisasi terhadap total confidence semua deteksi:

- `Bayi_Maggot > 60%` → **belum siap panen**
- `Remaja_Maggot > 60%` → **siap panen**
- `Dewasa_Maggot > 60%` → **melewati siap panen**
- tidak ada kelas `> 60%` → **belum dapat disimpulkan**

Ambang, confidence minimum, IoU, dan ukuran inferensi dapat diubah melalui sidebar.

## Kamera BARDI melalui ODM/RTSP

Mode **Kamera BARDI (RTSP/ODM)** ditujukan untuk aplikasi yang dijalankan secara lokal pada laptop yang satu jaringan Wi-Fi dengan kamera. ODM digunakan untuk menemukan kamera dan URL stream; dashboard membaca URL RTSP tersebut secara langsung.

Salin contoh konfigurasi berikut menjadi `.streamlit/secrets.toml`:

```toml
[bardi]
rtsp_url = "rtsp://192.168.1.100:8554/Streaming/Channels/101"
username = ""
password = ""
```

Isi username dan password hanya jika autentikasi RTSP diaktifkan. Jangan commit `secrets.toml` ke GitHub. Jalankan aplikasi lokal, pilih sumber **Kamera BARDI (RTSP/ODM)**, lalu tekan **Ambil gambar dari BARDI**.

Streamlit Community Cloud tidak dapat mengakses alamat IP privat `192.168.x.x`. Untuk membuka dashboard lokal dari HP dengan HTTPS, gunakan tunnel yang mengarah ke Streamlit lokal; proses pengambilan RTSP dan YOLO tetap berjalan pada laptop.

## Deployment ke Streamlit Community Cloud

1. Unggah repository ini ke GitHub. Pastikan `app.py`, `requirements.txt`, dan `yolo_maggot/weights/best.pt` ikut terunggah.
2. Buka [share.streamlit.io](https://share.streamlit.io) dan hubungkan akun GitHub.
3. Pilih **Create app** lalu isi repository, branch `main`, dan entrypoint `app.py`.
4. Pada **Advanced settings**, gunakan Python 3.12.
5. Pilih **Deploy**. Alamat yang dihasilkan memakai HTTPS sehingga akses kamera HP dapat meminta izin secara normal.
