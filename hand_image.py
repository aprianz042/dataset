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

# ==== Konfigurasi ====
input_image_path = 'db_head_straight/angry/angry_0012.jpg'        # Gambar wajah input
#input_image_path = 'db_head_not_straight/angry/angry_3093.jpg'
output_image_path = 'output.png'                                  # Gambar hasil setelah tangan ditempel
tangan, hand_image_path = hand_random()                     
    
# Load gambar input dan tangan
image = cv2.imread(input_image_path)
hand_img = cv2.imread(hand_image_path, cv2.IMREAD_UNCHANGED)

# Inisialisasi MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Fungsi overlay transparan
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

# Proses gambar dengan MediaPipe
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

if results.multi_face_landmarks:
    ih, iw, _ = image.shape
    for face_landmarks in results.multi_face_landmarks:
        # --- Ukur lebar wajah dari pipi kiri ke kanan ---
        cheek_left = face_landmarks.landmark[234]
        cheek_right = face_landmarks.landmark[454]
        x1, y1 = int(cheek_left.x * iw), int(cheek_left.y * ih)
        x2, y2 = int(cheek_right.x * iw), int(cheek_right.y * ih)
        face_width = np.linalg.norm([x2 - x1, y2 - y1])

        # --- Resize gambar tangan berdasarkan lebar wajah ---
        desired_width = int(face_width * 0.5 + 10)
        scale_ratio = desired_width / hand_img.shape[1]
        resized_hand = cv2.resize(hand_img, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

        # --- Hitung roll kepala (kemiringan dahi ke dagu) ---
        top = face_landmarks.landmark[10]   # dahi
        bottom = face_landmarks.landmark[152]  # dagu
        x_top = int(top.x * iw)
        y_top = int(top.y * ih)
        x_bottom = int(bottom.x * iw)
        y_bottom = int(bottom.y * ih)

        dx = x_bottom - x_top
        dy = y_bottom - y_top
        #angle = np.degrees(np.arctan2(dy, dx)) - 90  # dikurangi 90 agar referensinya vertikal
        angle = 90 - np.degrees(np.arctan2(dy, dx))

        # --- Rotasi gambar tangan ---
        hh, ww = resized_hand.shape[:2]
        M = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle, 1)
        rotated_hand = cv2.warpAffine(
            resized_hand, M, (ww, hh),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # --- Posisi tengah wajah (bawah hidung) ---
        center_landmark = face_landmarks.landmark[4]
        px = int(center_landmark.x * iw)
        py = int(center_landmark.y * ih)

        # --- Offset posisi tangan (atur sesuai kebutuhan) ---
        if tangan == "kanan":  # geser kiri/kanan
            offset_x = int(px - rotated_hand.shape[1] * 0.5 - 35)
        else:
            offset_x = int(px - rotated_hand.shape[1] * 0.5 + 35) 
        offset_y = int(py - rotated_hand.shape[0] * 0.5) # geser atas/bawah

        # --- Tempel tangan ke gambar wajah ---
        image = overlay_transparent(image, rotated_hand, offset_x, offset_y)

# Simpan & tampilkan hasil
#cv2.imwrite(output_image_path, image)
cv2.imshow("Hasil Tempel Tangan dengan Rotasi", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
