# Sistem Deteksi dan Penghitung Pengunjung Perpustakaan Real-Time Menggunakan YOLOv8 dan Python

Proyek ini merupakan sistem deteksi dan penghitung jumlah pengunjung perpustakaan menggunakan **YOLOv8**, **OpenCV**, dan **Python**. Sistem dapat memproses input berupa gambar, folder berisi banyak gambar, maupun video. Objek yang dideteksi hanya manusia, yaitu class `person` pada dataset COCO.

## Deskripsi Proyek

Sistem ini dibuat untuk mendeteksi keberadaan manusia pada gambar atau video perpustakaan, kemudian menghitung jumlah pengunjung yang terlihat pada setiap frame. Hasil deteksi ditampilkan dalam bentuk bounding box beserta teks jumlah pengunjung saat ini.

Pada input video, sistem akan membaca video frame demi frame, menjalankan deteksi manusia menggunakan model YOLOv8, lalu menampilkan jumlah pengunjung secara real-time pada tampilan video dan terminal.

## Fitur Utama

- Deteksi manusia menggunakan YOLOv8.
- Menggunakan model pretrained `yolov8n.pt`.
- Hanya mendeteksi class `person` dengan class ID `0`.
- Mendukung input:
  - Gambar tunggal.
  - Banyak gambar dalam satu folder.
  - Video.
- Menampilkan bounding box pada setiap orang yang terdeteksi.
- Menampilkan teks:

```text
Jumlah Pengunjung Saat Ini: [angka]
```

## 📂 Struktur Folder

```text
project/
├── deteksi_pengunjung.py
├── README.md
├── dataset_gambar/
│   ├── perpus1.jpg
│   ├── perpus2.jpg
│   └── perpus3.png
├── video/
│   └── perpustakaan.mp4
└── hasil_gambar/
    ├── hasil_perpus1.jpg
    ├── hasil_perpus2.jpg
    ├── hasil_perpus3.png
    └── hasil_video/
        └── hasil_pengunjung.mp4
```

## Instalasi

```bash
pip install ultralytics opencv-python
```

## Menjalankan Program

1. Menjalankan Deteksi pada 1 Gambar

```bash
python deteksi_pengunjung.py --input dataset_gambar\nama_gambar.jpg --output hasil_gambar
```

2. Menjalankan Deteksi pada Folder Gambar

```bash
python deteksi_pengunjung.py --input dataset_gambar --output hasil_gambar
```

3. Menjalankan Deteksi pada Video

```bash
python deteksi_pengunjung.py --input dataset_video\nama_video.mp4 --output hasil_pengunjung.mp4
```

## Alur Program

```markdown
## 🔄 Alur Program

> 📥 **Input** gambar/video
> 
> 🔽
> 
> 🧠 **Load model** YOLOv8
> 
> 🔽
> 
> 🔍 **Deteksi objek** person
> 
> 🔽
> 
> 🎯 **Filter class** person
> 
> 🔽
> 
> 🔢 **Hitung jumlah** bounding box
> 
> 🔽
> 
> 📺 **Tampilkan** bounding box & jumlah pengunjung
> 
> 🔽
> 
> 💾 **Simpan** hasil output
```
