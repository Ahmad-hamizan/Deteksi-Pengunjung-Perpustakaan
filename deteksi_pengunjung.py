import cv2
import argparse
import os
from ultralytics import YOLO


# =========================
# Konfigurasi dasar program
# =========================

PERSON_CLASS_ID = 0          # Di dataset COCO, class ID 0 adalah "person"
CONFIDENCE_THRESHOLD = 0.4   # Ambang batas confidence deteksi
MODEL_PATH = "yolov8n.pt"    # Model YOLOv8 nano, ringan dan cocok untuk demo real-time


def is_image_file(file_path):
    """
    Mengecek apakah input adalah file gambar berdasarkan ekstensi.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ext = os.path.splitext(file_path)[1].lower()
    return ext in image_extensions


def draw_text_with_background(frame, text, position=(20, 40)):
    """
    Menampilkan teks jumlah pengunjung di pojok kiri atas
    dengan background hitam agar mudah dibaca.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    # Menghitung ukuran teks
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size

    x, y = position

    # Membuat background hitam di belakang teks
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        (0, 0, 0),
        -1
    )

    # Menulis teks berwarna hijau
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),
        font_thickness,
        cv2.LINE_AA
    )


def detect_people_in_frame(model, frame):
    """
    Melakukan deteksi manusia pada satu frame.
    Fungsi ini hanya mengambil class 'person' atau class ID 0.
    """
    results = model.predict(
        frame,
        classes=[PERSON_CLASS_ID],          # Hanya deteksi manusia
        conf=CONFIDENCE_THRESHOLD,          # Confidence minimal
        verbose=False                       # Supaya terminal tidak terlalu ramai
    )

    # Ambil hasil prediksi pertama karena input hanya satu frame
    result = results[0]

    # Jika tidak ada bounding box, kembalikan list kosong
    if result.boxes is None:
        return []

    detections = []

    # Loop semua bounding box hasil deteksi
    for box in result.boxes:
        # Koordinat bounding box dalam format x1, y1, x2, y2
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Confidence deteksi
        confidence = float(box.conf[0].cpu().numpy())

        # Class ID hasil deteksi
        class_id = int(box.cls[0].cpu().numpy())

        # Filter tambahan agar benar-benar hanya class person
        if class_id == PERSON_CLASS_ID:
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence
            })

    return detections


def annotate_frame(frame, detections):
    """
    Menggambar bounding box YOLO dan teks jumlah pengunjung pada frame.
    """
    visitor_count = len(detections)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]

        # Gambar bounding box warna hijau
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label pada bounding box
        label = f"Person {confidence:.2f}"

        # Background kecil untuk label
        cv2.rectangle(
            frame,
            (x1, y1 - 25),
            (x1 + 130, y1),
            (0, 255, 0),
            -1
        )

        # Teks label
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Teks utama jumlah pengunjung di pojok kiri atas
    counter_text = f"Jumlah Pengunjung Saat Ini: {visitor_count}"
    draw_text_with_background(frame, counter_text, position=(20, 40))

    return frame, visitor_count


def process_image(model, input_path, output_path):
    """
    Memproses input berupa gambar.
    """
    frame = cv2.imread(input_path)

    if frame is None:
        print(f"[ERROR] Gambar tidak dapat dibaca: {input_path}")
        return

    # Deteksi manusia pada gambar
    detections = detect_people_in_frame(model, frame)

    # Gambar bounding box dan jumlah pengunjung
    annotated_frame, visitor_count = annotate_frame(frame, detections)

    # Simpan hasil gambar
    cv2.imwrite(output_path, annotated_frame)

    print(f"[INFO] Jumlah Pengunjung Saat Ini: {visitor_count}")
    print(f"[INFO] Hasil gambar disimpan ke: {output_path}")

    # Tampilkan hasil gambar
    cv2.imshow("Hasil Deteksi Pengunjung", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def process_image_folder(model, input_folder, output_folder):
    """
    Memproses banyak gambar dari satu folder.
    Semua hasil deteksi akan disimpan ke folder output.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()

        # Lewati file yang bukan gambar
        if ext not in image_extensions:
            continue

        input_path = os.path.join(input_folder, filename)

        # Nama output, contoh: hasil_foto1.jpg
        output_filename = f"hasil_{filename}"
        output_path = os.path.join(output_folder, output_filename)

        frame = cv2.imread(input_path)

        if frame is None:
            print(f"[WARNING] Gagal membaca gambar: {input_path}")
            continue

        detections = detect_people_in_frame(model, frame)
        annotated_frame, visitor_count = annotate_frame(frame, detections)

        cv2.imwrite(output_path, annotated_frame)

        print(f"{filename}: Jumlah Pengunjung Saat Ini = {visitor_count}")

    print(f"[INFO] Semua hasil disimpan di folder: {output_folder}")



def process_video(model, input_path, output_path):
    """
    Memproses input berupa video.
    Program membaca video frame demi frame, mendeteksi manusia,
    menghitung jumlah pengunjung pada frame saat ini,
    lalu menyimpan video hasilnya.
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"[ERROR] Video tidak dapat dibuka: {input_path}")
        return

    # Ambil informasi dasar video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Jika FPS gagal dibaca, gunakan nilai default 30
    if fps == 0:
        fps = 30

    # Codec untuk menyimpan output video MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Membuat objek VideoWriter untuk menyimpan hasil video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    last_reported_second = -1

    print("[INFO] Memproses video...")
    print("[INFO] Tekan tombol 'q' untuk keluar lebih cepat.")

    while True:
        ret, frame = cap.read()

        # Jika frame sudah habis, hentikan loop
        if not ret:
            break

        # Hitung waktu video saat ini dalam detik
        current_second = int(frame_index / fps)

        # Deteksi manusia pada frame saat ini
        detections = detect_people_in_frame(model, frame)

        # Gambar bounding box dan jumlah pengunjung
        annotated_frame, visitor_count = annotate_frame(frame, detections)

        # Menampilkan jumlah pengunjung di terminal setiap detik video
        if current_second != last_reported_second:
            print(f"Detik ke-{current_second}: Jumlah Pengunjung Saat Ini = {visitor_count}")
            last_reported_second = current_second

        # Simpan frame yang sudah dianotasi ke video output
        out.write(annotated_frame)

        # Tampilkan frame secara real-time
        cv2.imshow("Sistem Deteksi dan Penghitung Pengunjung", annotated_frame)

        # Jika user menekan tombol q, keluar dari program
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    # Bersihkan resource
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Selesai. Video output disimpan ke: {output_path}")


def main():
    """
    Fungsi utama program.
    Mengatur argument input dan output dari terminal.
    """
    parser = argparse.ArgumentParser(
        description="Sistem Deteksi dan Penghitung Pengunjung Perpustakaan Menggunakan YOLOv8"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path file input, bisa gambar atau video. Contoh: data/perpustakaan.mp4"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path file output. Contoh: hasil/output.mp4 atau hasil/output.jpg"
    )

    args = parser.parse_args()

    input_path = args.input

    # Jika user tidak memberi nama output, buat otomatis
    if args.output is None:
        if is_image_file(input_path):
            output_path = "output_deteksi_pengunjung.jpg"
        else:
            output_path = "output_deteksi_pengunjung.mp4"
    else:
        output_path = args.output

    # Load model YOLOv8
    # Jika file yolov8n.pt belum ada, Ultralytics akan mengunduh otomatis
    print("[INFO] Memuat model YOLOv8...")
    model = YOLO(MODEL_PATH)

    # Cek apakah input gambar atau video
    # Jika input adalah folder, proses semua gambar di dalam folder
    if os.path.isdir(input_path):
        output_folder = args.output if args.output is not None else "hasil_gambar"
        process_image_folder(model, input_path, output_folder)

    # Jika input adalah satu file gambar
    elif is_image_file(input_path):
        process_image(model, input_path, output_path)

    # Jika input adalah video
    else:
        process_video(model, input_path, output_path)

if __name__ == "__main__":
    main()
