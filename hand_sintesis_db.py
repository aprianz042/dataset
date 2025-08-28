import cv2
import mediapipe as mp
import numpy as np
import os
import random

def hand_random():
    tangan_list = ["kanan", "kiri"]
    gambar_list = ["hand1.png", "hand2.png", "hand3.png"]
    tangan = random.choice(tangan_list)
    gambar = random.choice(gambar_list)
    path = f"hand/{tangan}/{gambar}"
    return tangan, path


# === Konfigurasi ===
input_root = "FINAL/3_dataset_affectnet_rafdb_seleksi_landmark"
output_root = "FINAL/4_dataset_affectnet_rafdb_seleksi_wajah_hand_sintesis"

# Inisialisasi MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# === Fungsi Overlay Transparan ===
def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]
    if x >= bw or y >= bh:
        return background
    if x + ow > bw:
        ow = bw - x
        overlay = overlay[:, :ow]
    if y + oh > bh:
        oh = bh - y
        overlay = overlay[:oh]
    if x < 0:
        overlay = overlay[:, -x:]
        ow += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        oh += y
        y = 0
    if overlay.shape[2] < 4:
        return background
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    roi = background[y:y + oh, x:x + ow]
    blended = (1.0 - mask) * roi + mask * overlay_img
    background[y:y + oh, x:x + ow] = blended.astype(np.uint8)
    return background

# === Proses Semua Subfolder dan Gambar ===
for subfolder in os.listdir(input_root):
    subfolder_path = os.path.join(input_root, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    output_subfolder_path = os.path.join(output_root, subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)

    for filename in os.listdir(subfolder_path):
        tangan, hand_image_path = hand_random()
        hand_img = cv2.imread(hand_image_path, cv2.IMREAD_UNCHANGED)

        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        input_img_path = os.path.join(subfolder_path, filename)
        image = cv2.imread(input_img_path)

        if image is None:
            print(f"❌ Gagal membaca {input_img_path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            ih, iw, _ = image.shape
            for face_landmarks in results.multi_face_landmarks:
                # Ukur lebar wajah
                cheek_left = face_landmarks.landmark[234]
                cheek_right = face_landmarks.landmark[454]
                x1, y1 = int(cheek_left.x * iw), int(cheek_left.y * ih)
                x2, y2 = int(cheek_right.x * iw), int(cheek_right.y * ih)
                face_width = np.linalg.norm([x2 - x1, y2 - y1])

                # Resize tangan
                desired_width = int(face_width * 0.5 + 10)
                scale_ratio = desired_width / hand_img.shape[1]
                resized_hand = cv2.resize(hand_img, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

                # Roll kepala (dahi ke dagu)
                top = face_landmarks.landmark[10]
                bottom = face_landmarks.landmark[152]
                x_top = int(top.x * iw)
                y_top = int(top.y * ih)
                x_bottom = int(bottom.x * iw)
                y_bottom = int(bottom.y * ih)
                dx = x_bottom - x_top
                dy = y_bottom - y_top
                angle = 90 - np.degrees(np.arctan2(dy, dx))

                # Rotasi tangan
                hh, ww = resized_hand.shape[:2]
                M = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle, 1)
                rotated_hand = cv2.warpAffine(resized_hand, M, (ww, hh),
                                              flags=cv2.INTER_AREA,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0, 0))

                # Posisi wajah (bawah hidung)
                center = face_landmarks.landmark[4]
                px = int(center.x * iw)
                py = int(center.y * ih)

                if tangan == "kanan":  
                    offset_x = int(px - rotated_hand.shape[1] * 0.5 - 30)
                else:
                    offset_x = int(px - rotated_hand.shape[1] * 0.5 + 30) 
                offset_y = int(py - rotated_hand.shape[0] * 0.5)

                # Tempel tangan
                image = overlay_transparent(image, rotated_hand, offset_x, offset_y)

        # Simpan hasil
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_hand{ext}"
        output_img_path = os.path.join(output_subfolder_path, output_filename)
        cv2.imwrite(output_img_path, image)
        print(f"{output_img_path} --> OK.")

print("✅ Selesai proses semua gambar.")
