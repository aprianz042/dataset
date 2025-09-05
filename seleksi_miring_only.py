import os
import cv2
import glob
import shutil
import mediapipe as mp
import math

# Konfigurasi folder
source_dir = "FINAL/3_dataset_affectnet_rafdb_seleksi_landmark"
target_dir = "FINAL/5_dataset_affectnet_rafdb_seleksi_wajah_miring_diatas_03"

# Inisialisasi MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Format file yang didukung
valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

# Ambil semua file gambar dari subfolder
image_paths = glob.glob(os.path.join(source_dir, "*", "*.*"))

# Fungsi untuk menghitung yaw (rotasi kepala sekitar sumbu vertikal) dengan nilai absolut
def calculate_yaw(landmarks):
    # Ambil titik landmark untuk mata kiri, mata kanan, dan hidung
    left_eye = landmarks[33]  # Landmark untuk mata kiri
    right_eye = landmarks[263]  # Landmark untuk mata kanan
    nose = landmarks[1]  # Landmark untuk hidung

    # Hitung jarak horizontal antara mata kiri dan hidung
    left_eye_to_nose = abs(left_eye.x - nose.x)
    right_eye_to_nose = abs(right_eye.x - nose.x)

    # Jika jarak kiri ke hidung dan kanan ke hidung sama, yaw = 0 (kepala lurus)
    if left_eye_to_nose == right_eye_to_nose:
        yaw = 0.0
    else:
        # Jika ada perbedaan, hitung yaw berdasarkan selisih jarak
        yaw = math.atan2(right_eye_to_nose - left_eye_to_nose, 1)  # Membandingkan perbedaan jarak

    # Pastikan yaw selalu positif
    return abs(yaw)

# Fungsi untuk menghitung roll (rotasi kepala sekitar sumbu horizontal) dengan nilai absolut
def calculate_roll(landmarks):
    # Ambil titik landmark untuk mata kiri, mata kanan, dan hidung
    left_eye = landmarks[33]  # Landmark untuk mata kiri
    right_eye = landmarks[263]  # Landmark untuk mata kanan
    nose = landmarks[1]  # Landmark untuk hidung

    # Hitung perbedaan y antara mata kiri dan mata kanan
    eye_diff_y = right_eye.y - left_eye.y

    # Hitung perbedaan x antara mata kiri dan hidung
    nose_eye_diff_x = nose.x - left_eye.x

    # Perhitungan roll, dengan menggunakan perbandingan y (vertikal) antara mata kiri dan mata kanan
    # serta perbandingan x (horizontal) antara mata kiri dan hidung
    roll = math.atan2(eye_diff_y, nose_eye_diff_x)

    # Pastikan roll selalu positif
    return abs(roll)

# Threshold untuk yaw dan roll
min_yaw = 0.2

yaw_threshold = 0.3  # Threshold untuk yaw
roll_threshold = 0.5  # Threshold untuk roll

for img_path in image_paths:
    if not img_path.lower().endswith(valid_exts):
        continue

    try:
        # Ambil label dan nama file
        parts = img_path.split(os.sep)
        emotion_label = parts[-2]
        filename = parts[-1]

        # Baca dan konversi gambar ke RGB
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️ Tidak bisa dibaca: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Deteksi landmark wajah
        result = face_mesh.process(img_rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            yaw = calculate_yaw(landmarks)
            roll = calculate_roll(landmarks)

            # Cek apakah yaw atau roll terlalu besar
            #if yaw > min_yaw and yaw < yaw_threshold and roll < roll_threshold:
            #if yaw > min_yaw and yaw < yaw_threshold and roll < roll_threshold:
            if yaw > yaw_threshold and roll < roll_threshold:
                # Buat folder target
                target_folder = os.path.join(target_dir, emotion_label)
                os.makedirs(target_folder, exist_ok=True)

                # Salin file ke folder baru
                target_path = os.path.join(target_folder, filename)
                shutil.copy2(img_path, target_path)
                print(f"✔️ Kepala miring (Yaw: {yaw:.2f}, Roll: {roll:.2f}), disalin: {target_path}")
            else:
                print(f"❌ Kepala tidak miring: {img_path} (Yaw: {yaw:.2f}, Roll: {roll:.2f})")

        else:
            print(f"❌ Tidak ada wajah: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
