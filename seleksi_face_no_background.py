import os
import cv2
import glob
import shutil
import mediapipe as mp
import math

#from frontalization import *
from frontal import *

source_dir = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring"
target_dir = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring_no_backgrond"

valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
image_paths = glob.glob(os.path.join(source_dir, "*", "*.*"))

for img_path in image_paths:
    if not img_path.lower().endswith(valid_exts):
        continue
    try:
        parts = img_path.split(os.sep)
        emotion_label = parts[-2]
        filename = parts[-1]

        img = cv2.imread(img_path)
        _, face_ori, _ = half_flip(img)
        #cek = cek_landmark_wajah(face_ori)

        #if cek is not None:
        if face_ori is not None:
            target_folder = os.path.join(target_dir, emotion_label)
            os.makedirs(target_folder, exist_ok=True)

            target_path = os.path.join(target_folder, filename)
            cv2.imwrite(target_path, face_ori) 
            print(f"✔️ Proses berhasil: {target_path}")
        else:
            print(f"❌ Tidak ada wajah: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
