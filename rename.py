import os

# Folder utama yang berisi subfolder label emosi
base_dir = "db_faces"

# Format file yang akan diproses
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

# Iterasi setiap subfolder (label)
for emotion_label in os.listdir(base_dir):
    emotion_path = os.path.join(base_dir, emotion_label)

    if not os.path.isdir(emotion_path):
        continue

    # Ambil semua file valid
    files = [f for f in os.listdir(emotion_path) if f.lower().endswith(valid_exts)]
    files.sort()  # Urutkan biar konsisten

    for idx, old_name in enumerate(files):
        _, ext = os.path.splitext(old_name)
        new_name = f"{emotion_label}_{idx+1:04d}{ext.lower()}"

        old_path = os.path.join(emotion_path, old_name)
        new_path = os.path.join(emotion_path, new_name)

        os.rename(old_path, new_path)
        print(f"✔️ {old_name} → {new_name}")
