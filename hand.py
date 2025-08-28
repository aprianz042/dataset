import cv2
import mediapipe as mp
import numpy as np

# Load gambar tangan (RGBA)
hand_img = cv2.imread('hand1.png', cv2.IMREAD_UNCHANGED)

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Fungsi overlay gambar transparan ke frame
def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    if x >= bw or y >= bh:
        return background

    # Potong overlay jika keluar dari kanan/bawah
    if x + ow > bw:
        ow = bw - x
        overlay = overlay[:, :ow]
    if y + oh > bh:
        oh = bh - y
        overlay = overlay[:oh]

    # Potong jika koordinat negatif (keluar kiri/atas)
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

# Mulai webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        ih, iw, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            # Ambil landmark pipi kiri dan kanan untuk mengukur lebar wajah
            cheek_left = face_landmarks.landmark[234]
            cheek_right = face_landmarks.landmark[454]

            x1, y1 = int(cheek_left.x * iw), int(cheek_left.y * ih)
            x2, y2 = int(cheek_right.x * iw), int(cheek_right.y * ih)

            # Hitung lebar wajah
            face_width = np.linalg.norm([x2 - x1, y2 - y1])

            # Tentukan lebar tangan berdasarkan lebar wajah (skalanya bisa diubah)
            
            desired_width = int(face_width * 0.5)  #ukuran tanggannya
            
            scale_ratio = desired_width / hand_img.shape[1]
            resized_hand = cv2.resize(
                hand_img, None,
                fx=scale_ratio, fy=scale_ratio,
                interpolation=cv2.INTER_AREA
            )

            # Landmark tengah wajah (bagian bawah hidung)
            center_landmark = face_landmarks.landmark[4]
            px = int(center_landmark.x * iw)
            py = int(center_landmark.y * ih)

            # Hitung posisi tangan supaya agak ke bawah dan tengah
            offset_x = int(px - resized_hand.shape[1] * 0.5 - 60) #geser kiri kanan
            offset_y = int(py - resized_hand.shape[0] * 0.3) #geser atas bawah

            # Tempel gambar tangan ke frame
            frame = overlay_transparent(frame, resized_hand, offset_x, offset_y)

    cv2.imshow("Tangan di Tengah Wajah", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
