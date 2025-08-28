import cv2
import mediapipe as mp
import numpy as np

# Load gambar kacamata (harus RGBA untuk transparansi)
glasses_img = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

def overlay_transparent(background, overlay, x, y):
    """Tempelkan overlay RGBA ke frame dengan alpha blending."""
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    if x >= bw or y >= bh:
        return background

    # Batasi ukuran
    if x + ow > bw:
        overlay = overlay[:, :bw - x]
    if y + oh > bh:
        overlay = overlay[:bh - y, :]

    if overlay.shape[2] < 4:
        return background

    # Pisahkan channel
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y + oh, x:x + ow] = (1.0 - mask) * background[y:y + oh, x:x + ow] + mask * overlay_img
    return background.astype(np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        ih, iw, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            # Ambil 2 titik ujung mata kiri dan kanan
            left_eye_outer = face_landmarks.landmark[263]  # ujung luar mata kiri
            right_eye_outer = face_landmarks.landmark[33]  # ujung luar mata kanan

            x1 = int(left_eye_outer.x * iw)
            y1 = int(left_eye_outer.y * ih)
            x2 = int(right_eye_outer.x * iw)
            y2 = int(right_eye_outer.y * ih)

            # Hitung tengah dan sudut rotasi
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            glasses_width = int(1.2 * np.linalg.norm([x2 - x1, y2 - y1]))
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            #angle = np.degrees(np.arctan2(y1 - y2, x2 - x1))


            # Resize gambar kacamata
            scale = glasses_width / glasses_img.shape[1]
            resized_glasses = cv2.resize(glasses_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Rotasi kacamata
            (h, w) = resized_glasses.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            rotated_glasses = cv2.warpAffine(resized_glasses, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

            # Flip vertikal jika terbalik
            rotated_glasses = cv2.flip(rotated_glasses, 0)  # 0 = vertical flip

            # Hitung posisi tempel
            top_left_x = int(center_x - rotated_glasses.shape[1] / 2)
            top_left_y = int(center_y - rotated_glasses.shape[0] / 2)

            # Tempelkan ke frame
            frame = overlay_transparent(frame, rotated_glasses, top_left_x, top_left_y)

    cv2.imshow("Kacamata Sintesis", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
