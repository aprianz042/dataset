import cv2
import mediapipe as mp
import math

# Inisialisasi MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Fungsi untuk menghitung yaw (rotasi kepala sekitar sumbu vertikal)
def calculate_yaw(landmarks):
    # Ambil titik landmark untuk mata kiri, mata kanan, dan hidung
    left_eye = landmarks[33]  # Landmark untuk mata kiri
    right_eye = landmarks[263]  # Landmark untuk mata kanan
    nose = landmarks[1]  # Landmark untuk hidung

    # Hitung jarak horizontal antara mata kiri dan hidung
    left_eye_to_nose = abs(left_eye.x - nose.x)
    right_eye_to_nose = abs(right_eye.x - nose.x)

    # Jika jarak kiri ke hidung dan kanan ke hidung sama, yaw = 0 (kepala lurus)
    if left_eye_to_nose == right_eye_to_nose:
        yaw = 0.0
    else:
        # Jika ada perbedaan, hitung yaw berdasarkan selisih jarak
        yaw = math.atan2(right_eye_to_nose - left_eye_to_nose, 1)  # Membandingkan perbedaan jarak

    # Pastikan yaw selalu positif
    return abs(yaw)

# Fungsi untuk menghitung roll (rotasi kepala sekitar sumbu horizontal)
def calculate_roll(landmarks):
    # Ambil titik landmark untuk mata kiri, mata kanan, dan hidung
    left_eye = landmarks[33]  # Landmark untuk mata kiri
    right_eye = landmarks[263]  # Landmark untuk mata kanan
    nose = landmarks[1]  # Landmark untuk hidung

    # Hitung perbedaan y antara mata kiri dan mata kanan
    eye_diff_y = right_eye.y - left_eye.y

    # Hitung perbedaan x antara mata kiri dan hidung
    nose_eye_diff_x = nose.x - left_eye.x

    # Perhitungan roll, dengan menggunakan perbandingan y (vertikal) antara mata kiri dan mata kanan
    # serta perbandingan x (horizontal) antara mata kiri dan hidung
    roll = math.atan2(eye_diff_y, nose_eye_diff_x)

    # Pastikan roll selalu positif
    return abs(roll)

# Mengakses webcam
cap = cv2.VideoCapture(0)  # 0 berarti webcam default, ganti dengan angka lain jika kamu punya lebih dari 1 webcam

if not cap.isOpened():
    print("⚠️ Tidak dapat mengakses webcam")
    exit()

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame dari webcam.")
        break

    # Konversi frame ke RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi landmark wajah
    result = face_mesh.process(img_rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        yaw = calculate_yaw(landmarks)
        roll = calculate_roll(landmarks)

        # Menampilkan nilai yaw dan roll
        yaw_deg = math.degrees(yaw)  # Mengonversi yaw dari radian ke derajat
        roll_deg = math.degrees(roll)  # Mengonversi roll dari radian ke derajat
        cv2.putText(frame, f"Yaw: {yaw_deg:.2f} deg", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Roll: {roll_deg:.2f} deg", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Tampilkan frame dengan nilai yaw dan roll
    cv2.imshow('Webcam - Yaw and Roll Detection', frame)

    # Keluar dari loop dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
