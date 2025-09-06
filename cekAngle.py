import os
import cv2
import glob
import mediapipe as mp

#from frontalization import *
from frontal import *

source_dir = "SAMPLE/miring"
valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
image_paths = glob.glob(os.path.join(source_dir, "*", "*.*"))

roll_ = []
pitch_ = []
yaw_ = []

for img_path in image_paths:
    if not img_path.lower().endswith(valid_exts):
        continue
    try:
        parts = img_path.split(os.sep)
        emotion_label = parts[-2]
        filename = parts[-1]

        img = cv2.imread(img_path)
        new_height = 500                                          # ukuran tinggi image (sesuaikan)
        (h, w) = img.shape[:2]
        aspect_ratio = w / h
        new_width = int(new_height * aspect_ratio)
        images = cv2.resize(img, (new_width, new_height))      # resize tinggi image ke ukuran baru

        points_, _ = get_face_mesh_3d(images)
        pitch, roll, yaw = compute_head_angle(points_)  

        if roll:
            print(f'{img_path} - roll : {roll}, pitch : {pitch}, yaw : {yaw}')
            roll_.append(roll)
            pitch_.append(pitch)
            yaw_.append(yaw)
        else:
            print(f"❌ Tidak ada wajah: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")

roll_maks = max(roll_)
pitch_maks = max(pitch_)
yaw_maks = max(yaw_)
print(f'roll maksimum : {roll_maks}')
print(f'pitch maksimum : {pitch_maks}')
print(f'yaw maksimum : {yaw_maks}')
