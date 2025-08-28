import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
import glob

from frontalization import *

# Konfigurasi folder
source_dir = "FINAL/6_dataset_affectnet_rafdb_seleksi_wajah_lurus_no_backgrond"

valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

image_paths = glob.glob(os.path.join(source_dir, "*", "*.*"))
print(image_paths)

for img_path in image_paths:
    try:
        # Ambil label dan nama file
        parts = img_path.split(os.sep)
        emotion_label = parts[-2]
        filename = parts[-1]

        files_ = cv2.imread(img_path)
        a, b = face_landmark(files_)
        if not b.face_landmarks:
            target_folder = os.path.join(source_dir, emotion_label)            

            target = os.path.join(target_folder, filename)
            os.remove(target)  
            print(f"✔️ Delete : {img_path}")
        
        else:
            print(f"❌ Pass: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
