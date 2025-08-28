import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

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
            # Ambil titik-titik penting untuk masker (misal: dagu & pipi)
            mask_indices = [46, 53, 65, 55, 8, 
                            285, 295, 282, 283, 
                            276, 353, 265, 261, 
                            448, 449, 450, 451, 
                            452, 453, 6, 233, 
                            232, 231, 230, 229, 
                            228, 31, 35, 124]  # Kiri ke kanan + dagu

            mask_points = []
            for idx in mask_indices:
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)
                mask_points.append((x, y))

            # Gambar masker sebagai polygon (biru transparan)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(mask_points, dtype=np.int32)], (0, 0, 0))  # Warna masker

            alpha = 1  # Transparansi masker
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow('Masker Tanpa Landmark', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
