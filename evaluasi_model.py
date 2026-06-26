"""
evaluasi_model.py
Evaluasi Precision & Recall model YOLOv8n untuk deteksi pengunjung perpustakaan.
"""

from ultralytics import YOLO
import cv2
import os
import csv

# =========================================================
# 1. ISI INI DENGAN DATA ASLI ANDA (ground truth manual)
# =========================================================
# key = nama file gambar, value = jumlah orang SEBENARNYA (hitung manual)
GROUND_TRUTH = {
    "perpus1.jpg": 2,   # ganti None dengan jumlah orang asli, misal: 4
    "perpus2.png": 4,
    "perpus3.jpg": 2,
    "perpus4.jpg": 6,
    "perpus5.jpg": 13,
}

INPUT_DIR = "dataset_gambar"
OUTPUT_DIR = "hasil_gambar"
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.4   # ambang batas confidence deteksi person
PERSON_CLASS_ID = 0    # class 'person' pada COCO

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Validasi: pastikan ground truth sudah diisi
    for fname, gt in GROUND_TRUTH.items():
        if gt is None:
            raise ValueError(
                f"Ground truth untuk '{fname}' belum diisi. "
                f"Hitung manual jumlah orang pada gambar tersebut lalu isi di GROUND_TRUTH."
            )

    model = YOLO(MODEL_PATH)
    rows = []
    total_tp = total_fp = total_fn = 0

    for fname, gt_count in GROUND_TRUTH.items():
        img_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(img_path):
            print(f"[SKIP] File tidak ditemukan: {img_path}")
            continue

        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        # filter hanya class 'person'
        person_boxes = [b for b in result.boxes if int(b.cls[0]) == PERSON_CLASS_ID]
        detected_count = len(person_boxes)

        # --- Catatan penting (jujur soal limitasi) ---
        # Karena tidak ada anotasi bounding box ground truth (IoU-based),
        # TP/FP/FN di sini dihitung secara SEDERHANA berbasis selisih jumlah
        # (count-based), bukan matching IoU per box. Ini valid untuk skenario
        # "counting" tapi BUKAN evaluasi mAP standar.
        if detected_count <= gt_count:
            tp = detected_count
            fn = gt_count - detected_count
            fp = 0
        else:
            tp = gt_count
            fn = 0
            fp = detected_count - gt_count

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # simpan gambar hasil dengan bounding box + teks jumlah
        annotated = result.plot()
        cv2.putText(
            annotated,
            f"Jumlah Pengunjung Saat Ini: {detected_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        out_path = os.path.join(OUTPUT_DIR, f"hasil_{fname}")
        cv2.imwrite(out_path, annotated)

        rows.append({
            "nama_gambar": fname,
            "ground_truth": gt_count,
            "terdeteksi": detected_count,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        })

        print(f"{fname}: GT={gt_count}, Terdeteksi={detected_count}, "
              f"TP={tp}, FP={fp}, FN={fn}")

    # --- Hitung Precision & Recall keseluruhan ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    print("\n=== HASIL EVALUASI TOTAL ===")
    print(f"Total TP : {total_tp}")
    print(f"Total FP : {total_fp}")
    print(f"Total FN : {total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")

    # simpan ke CSV biar bisa dilampirkan di laporan
    with open("evaluasi_hasil.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nama_gambar", "ground_truth", "terdeteksi", "TP", "FP", "FN"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({})
        writer.writerow({"nama_gambar": "TOTAL", "TP": total_tp, "FP": total_fp, "FN": total_fn})
        writer.writerow({"nama_gambar": "Precision", "ground_truth": f"{precision:.3f}"})
        writer.writerow({"nama_gambar": "Recall", "ground_truth": f"{recall:.3f}"})

    print("\nDetail tersimpan di evaluasi_hasil.csv")
    print(f"Gambar hasil deteksi tersimpan di folder '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
