import os
import cv2
import glob
import shutil
import mediapipe as mp
import math

# Konfigurasi folder
source_dir = "FINAL/3_dataset_affectnet_rafdb_seleksi_landmark"
target_dir = "FINAL/4_dataset_affectnet_rafdb_seleksi_wajah_lurus"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

image_paths = glob.glob(os.path.join(source_dir, "*", "*.*"))

def calculate_yaw(landmarks):
    left_eye = landmarks[33]  # Landmark untuk mata kiri
    right_eye = landmarks[263]  # Landmark untuk mata kanan
    nose = landmarks[1]  # Landmark untuk hidung

    left_eye_to_nose = abs(left_eye.x - nose.x)
    right_eye_to_nose = abs(right_eye.x - nose.x)

    if left_eye_to_nose == right_eye_to_nose:
        yaw = 0.0
    else:
        yaw = math.atan2(right_eye_to_nose - left_eye_to_nose, 1)  # Membandingkan perbedaan jarak
    return abs(yaw)

def calculate_roll(landmarks):
    left_eye = landmarks[33]  # Landmark untuk mata kiri
    right_eye = landmarks[263]  # Landmark untuk mata kanan
    nose = landmarks[1]  # Landmark untuk hidung
    eye_diff_y = right_eye.y - left_eye.y
    nose_eye_diff_x = nose.x - left_eye.x
    roll = math.atan2(eye_diff_y, nose_eye_diff_x)
    return abs(roll)

# Threshold untuk yaw dan roll
yaw_threshold = 0.1  # Threshold untuk yaw
roll_threshold = 0.1  # Threshold untuk roll

for img_path in image_paths:
    if not img_path.lower().endswith(valid_exts):
        continue
    try:
        parts = img_path.split(os.sep)
        emotion_label = parts[-2]
        filename = parts[-1]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️ Tidak bisa dibaca: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            yaw = calculate_yaw(landmarks)
            roll = calculate_roll(landmarks)
            if abs(yaw) < yaw_threshold and abs(roll) < roll_threshold:
                target_folder = os.path.join(target_dir, emotion_label)
                os.makedirs(target_folder, exist_ok=True)                

                target_path = os.path.join(target_folder, filename)
                shutil.copy2(img_path, target_path)
                print(f"✔️ Kepala urus (Yaw: {yaw:.2f}, Roll: {roll:.2f}), disalin: {target_path}")
            else:
                print(f"❌ Kepala miring: {img_path} (Yaw: {yaw:.2f}, Roll: {roll:.2f})")
        else:
            print(f"❌ Tidak ada wajah: {img_path}")
    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
