import os
import cv2
import glob
import shutil
import mediapipe as mp
from frontal import *

# Konfigurasi folder
source_dir = "SAMPLE/before"
target_dir = "SAMPLE/6_hand_fail"

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
        img_rgb = cv2.imread(img_path)
        
        points_, _ = get_face_mesh_3d(img_rgb)
        pitch, roll, yaw = compute_head_angle(points_)  
       
        img_roll, face_detected = correct_roll(img_rgb, roll, yaw)
        _, list_hand = landmark_tangan(img_roll) 
        sum_of_hand = len(list_hand)

        # Cek apakah tangan terdeteksi
        if sum_of_hand == 0:  
            target_folder = os.path.join(target_dir, emotion_label)
            os.makedirs(target_folder, exist_ok=True)

            target_path = os.path.join(target_folder, filename)
            shutil.copy2(img_path, target_path)
            print(f"❌ Tangan tidak terdeteksi, disalin: {target_path}")
            
        else:
            print(f"✔️ ada tangan: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
