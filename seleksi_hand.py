import os
import cv2
import glob
import shutil
import mediapipe as mp

# Konfigurasi folder
source_dir = "db_faces_undersampled"
target_dir = "db_hands"

# Inisialisasi MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1)

# Format file yang didukung
valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

# Ambil semua file gambar dari subfolder
image_paths = glob.glob(os.path.join(source_dir, "*", "*.*"))

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

        # Deteksi tangan
        result = hands.process(img_rgb)

        # Cek apakah tangan terdeteksi
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
            # Buat folder target
            target_folder = os.path.join(target_dir, emotion_label)
            os.makedirs(target_folder, exist_ok=True)

            # Salin file ke folder baru
            target_path = os.path.join(target_folder, filename)
            shutil.copy2(img_path, target_path)
            print(f"✔️ Tangan terdeteksi, disalin: {target_path}")
        else:
            print(f"❌ Tidak ada tangan: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
