import cv2
import numpy as np
import mediapipe as mp
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Inisialisasi Mediapipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=0.1, circle_radius=0.1)

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Ambil daftar koneksi untuk Tesselation, Contours, dan Irises
connections = {
    "tesselation": (mp_face_mesh.FACEMESH_TESSELATION, 'cyan'),
    "contours": (mp_face_mesh.FACEMESH_CONTOURS, 'red'),
    "irises": (mp_face_mesh.FACEMESH_IRISES, 'magenta')
}

# Atur linewidth untuk setiap koneksi
linewidths = {
    "tesselation": 0.5,  # Lebih tipis
    "contours": 2.5,  # Medium
    "irises": 1.0  # Lebih tebal
}

def draw_face_mesh(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        annotated_image = image
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
    return annotated_image

def get_face_mesh_3d(image):
    """ Deteksi wajah dan ambil landmark 3D """
    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        print("Wajah tidak ditemukan!")
        return None, None

    face_landmarks = results.multi_face_landmarks[0]
    points = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark])
    
    return points, face_landmarks

def resize_width(img, new_width):
    """Resize gambar dengan lebar tetap dan tinggi otomatis."""
    height, width = img.shape[:2]
    aspect_ratio = height / width  # Hitung rasio tinggi terhadap lebar
    new_height = int(new_width * aspect_ratio)  # Hitung tinggi otomatis
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_img

def compute_roll_angle(points):
    """ Hitung sudut roll berdasarkan garis antara kedua mata """
    LEFT_EYE_IDX = 33  # Landmark mata kiri
    RIGHT_EYE_IDX = 263  # Landmark mata kanan

    left_eye = points[LEFT_EYE_IDX]
    right_eye = points[RIGHT_EYE_IDX]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    roll_angle = np.degrees(np.arctan2(dy, dx))
    return roll_angle

def compute_yaw_angle(points):
    """ Hitung sudut yaw berdasarkan posisi hidung dan dagu """
    NOSE_IDX = 1  # Landmark hidung (pusat wajah)
    CHIN_IDX = 152  # Landmark dagu

    nose = points[NOSE_IDX]
    chin = points[CHIN_IDX]

    dx = chin[0] - nose[0]
    dz = chin[2] - nose[2]

    yaw_angle = np.degrees(np.arctan2(dx, dz))  # Sudut yaw dari sumbu Z
    return yaw_angle

def compute_pitch_angle(points):
    """ Hitung sudut pitch berdasarkan posisi hidung dan dagu """
    NOSE_IDX = 1  # Landmark hidung (pusat wajah)
    CHIN_IDX = 152  # Landmark dagu

    nose = points[NOSE_IDX]
    chin = points[CHIN_IDX]

    dy = chin[1] - nose[1]
    dz = chin[2] - nose[2]

    pitch_angle = np.degrees(np.arctan2(dy, dz))  # Sudut pitch dari sumbu Z
    return pitch_angle

def rotate(points, angle, axis='z'):
    """ Rotasi titik 3D berdasarkan sumbu (z = roll, y = yaw, x = pitch) """
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'z':  # Rotasi Roll
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    elif axis == 'y':  # Rotasi Yaw
        R = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis == 'x':  # Rotasi Pitch
        R = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    else:
        raise ValueError("Axis harus 'z' (roll), 'y' (yaw), atau 'x' (pitch)")

    # Pusatkan titik sebelum rotasi
    center = np.mean(points, axis=0)
    rotated_points = np.dot(points - center, R.T) + center

    return rotated_points

def convert_to_2d_xz(points):
    """ Konversi dari 3D ke 2D dengan menghilangkan koordinat Z """    
    points_2d = points[:, [0, 2]]  # Ambil hanya X dan Z
    points_2d[:, 1] *= -1  # Balik sumbu Z agar orientasi wajah benar
    return points_2d

def convert_to_2d_xy(points):
    """ Konversi dari 3D ke 2D dengan menghilangkan koordinat Z """
    points_2d_xy = points[:, :2]  # Ambil hanya X dan Y
    return points_2d_xy

def plot_3d_face(points, title="3D Face Mesh"):
    """ Plot wajah 3D dengan warna sesuai Mediapipe """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot titik-titik wajah
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c='black', s=2)  # Titik landmark

    # Plot koneksi antar titik dengan warna sesuai Mediapipe
    for name, (conn, color) in connections.items():
        lw = linewidths.get(name, 1.0)
        segments = [(points[start], points[end]) for start, end in conn]
        line_collection = Line3DCollection(segments, color=color, linewidths=lw, alpha=0.9)
        ax.add_collection3d(line_collection)

    ax.view_init(elev=10, azim=90)  # Tampilan samping
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)
    plt.show()

def plot_2d_face(points_2d, title="2D Face Mesh"):
    """ Plot wajah dalam bentuk 2D setelah koreksi """
    fig, ax = plt.subplots(figsize=(6, 6))

    x, y = points_2d[:, 0], points_2d[:, 1]
    ax.scatter(x, y, c='black', s=2)  # Titik landmark

    # Plot koneksi antar titik dengan warna sesuai Mediapipe
    for name, (conn, color) in connections.items():
        lw = linewidths.get(name, 1.0)
        for start, end in conn:
            ax.plot([x[start], x[end]], [y[start], y[end]], color=color, linewidth=lw)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Sesuai dengan sistem koordinat gambar
    plt.show()


def save_2d_face(points_2d, task, filename="face_2d_centered.png", image_size=(128, 128), padding=5):    
    mp_face_mesh = mp.solutions.face_mesh
    connections = {
        "tesselation": mp_face_mesh.FACEMESH_TESSELATION,
        "contours": mp_face_mesh.FACEMESH_CONTOURS,
        "irises": mp_face_mesh.FACEMESH_IRISES
    }
    color_map = {
        "tesselation": (0,0,0),
        "contours": (0, 0, 0),  
        "irises": (255, 0, 0)  
    }
    linewidths = {
        "tesselation":1,
        "contours": 1,
        "irises": 1
    }

    canvas = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255  

    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)

    scale_x = (image_size[0] - 2 * padding) / (max_x - min_x + 1e-6)
    scale_y = (image_size[1] - 2 * padding) / (max_y - min_y + 1e-6)
    scale = min(scale_x, scale_y)

    points_scaled = (points_2d - [min_x, min_y]) * scale

    face_center_x = (np.max(points_scaled[:, 0]) + np.min(points_scaled[:, 0])) / 2
    face_center_y = (np.max(points_scaled[:, 1]) + np.min(points_scaled[:, 1])) / 2

    frame_center_x, frame_center_y = image_size[0] // 2, image_size[1] // 2

    offset_x = frame_center_x - face_center_x
    offset_y = frame_center_y - face_center_y

    points_centered = points_scaled + [offset_x, offset_y]
    points_centered = points_centered.astype(int)

    for name, conn in connections.items():
        color = color_map[name]  # Ambil warna dari color_map
        thickness = linewidths[name]  # Ambil ketebalan dari linewidths
        for start, end in conn:  # Iterasi langsung tanpa unpacking tambahan
            pt1 = tuple(points_centered[start])
            pt2 = tuple(points_centered[end])
            cv2.line(canvas, pt1, pt2, color, thickness)

    #for x, y in points_centered:
    #    cv2.circle(canvas, (x, y), 1, (0, 0, 0), -1)
    
    if task == 'simpan':  
        cv2.imwrite(filename, canvas)
        print(f"Gambar 2D wajah tersimpan sebagai {filename}")
        return None
    else:
        return canvas



def show_images(images, titles=None):
    n = len(images)  # Jumlah gambar
    cols = 4  # Jumlah kolom
    rows = (n + cols - 1) // cols  # Hitung jumlah baris yang diperlukan
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    # Jika hanya satu baris, jadikan axes array satu dimensi
    if rows == 1:
        axes = np.array(axes).reshape(1, -1)
    for i, img in enumerate(images):
        row, col = divmod(i, cols)  # Hitung posisi
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col].axis("off")
        if titles:
            axes[row, col].set_title(titles[i])
    # Sembunyikan subplot kosong jika jumlah gambar tidak pas
    for j in range(i + 1, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()

def rgb2mesh(imgs):
    #image = cv2.imread("uji_mata/uji4.jpg")
    image = imgs
    image = resize_width(image, 1024)
    points_3d, face_landmarks = get_face_mesh_3d(image)
    
    if points_3d is not None and face_landmarks is not None:
    #if points_3d is not None:
        
        #annotated_image = draw_face_mesh(image)
        #result_img.append(annotated_image)
        #result_lbl.append('gambar asli')
    
        #asli
        #points_2d_xy = convert_to_2d_xy(points_3d)
        #landmark_asli = save_2d_face(points_2d_xy, 'ubah')
        #result_img.append(landmark_asli)
        #result_lbl.append('landmark asli')
        
        # Koreksi Roll
        roll_angle = compute_roll_angle(points_3d)
        points_3d = rotate(points_3d, -roll_angle, axis='z')  
        #points_2d_xy = convert_to_2d_xy(points_3d)
        #plot_2d_face(points_2d_xy, title="Koreksi Roll")
        #roll = save_2d_face(points_2d_xy, 'ubah')
        #result_img.append(roll)
        #result_lbl.append('koreksi roll')
    
        # Koreksi Yaw
        yaw_angle = compute_yaw_angle(points_3d)
        points_3d = rotate(points_3d, -yaw_angle, axis='y')  

        final = convert_to_2d_xy(points_3d)
        #plot_2d_face(points_2d_xy, title="Koreksi Yaw")
        #yaw = save_2d_face(points_2d, 'ubah')
        #result_img.append(yaw)
        #result_lbl.append('koreksi yaw')
                     
        # Koreksi Pitch
        #pitch_angle = compute_pitch_angle(points_3d)
        #points_3d = rotate(points_3d, -pitch_angle, axis='x')  
        
        #points_2d_xy = convert_to_2d_xz(points_3d)
        #plot_2d_face(points_2d_xy, title="Koreksi Pitch / Final 2D Face Mesh")
    
        #plot_3d_face(points_3d, title="After Corrections (3D)")
        return final
    else:
        return None


result_img = []
result_lbl = []
# Load gambar
#image = cv2.imread("uji4.jpg")
#image_files = glob.glob("kepala_miring/*.jpg") 

DATASET_PATH = 'FINAL/13_dataset_affectnet_rafdb_seleksi_wajah_miring_diatas_03_frontal'
#LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABELS = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

output_folder = 'FINAL/13_dataset_affectnet_rafdb_seleksi_wajah_miring_diatas_03_frontal_mesh'
os.makedirs(output_folder, exist_ok=True)

z = 0
for label in LABELS:
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder_path):
        continue  # Skip jika folder tidak ada
        
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_mesh = rgb2mesh(image)
        if img_mesh is None:
            continue

        #Panggil fungsi untuk menyimpan gambar setelah koreksi pitch
        path_temp = os.path.join(output_folder, label)
        if not os.path.exists(path_temp):
            os.makedirs(path_temp, exist_ok=True)
            
        #face_filename = os.path.join(path_temp, f'face_{str(z).zfill(8)}.jpg')
        face_filename = os.path.join(path_temp, img_name)
        save_2d_face(img_mesh, 'simpan', face_filename)
        z+=1

#show_images(result_img, titles=[f"{a}" for a in result_lbl])
print('Selesai')
