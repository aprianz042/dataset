import os
import cv2
import glob
from frontalization import *

source_dir = "FINAL/5_dataset_affectnet_rafdb_seleksi_wajah_miring"
target_dir = "FINAL/11_dataset_affectnet_rafdb_seleksi_wajah_miring_frontal"

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, _, flip_output = half_flip(img)
                
        if flip_output is not None:
            output_ = cv2.cvtColor(flip_output, cv2.COLOR_BGR2RGB)
            target_folder = os.path.join(target_dir, emotion_label)
            os.makedirs(target_folder, exist_ok=True)

            target_path = os.path.join(target_folder, filename)
            cv2.imwrite(target_path, output_) 
            print(f"✔️✔️✔️✔️✔️✔️ Proses berhasil: {target_path}")
        else:
            print(f"❌ Tidak ada wajah: {img_path}")

    except Exception as e:
        print(f"❌❌❌ Error proses {img_path} - {e}")
