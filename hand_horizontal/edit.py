from PIL import Image

# Buka gambar asli
folder = "kiri/"
gambar = "hand1"
gmb = folder + gambar + ".png"
img = Image.open(gmb)  # ukuran misal 300x200 px
width, height = img.size

# Tambahkan padding 10px
padding = 50
new_width = width + padding
new_height = height + padding

# Buat canvas baru transparan
canvas = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

# Tempelkan gambar di tengah canvas dengan padding
canvas.paste(img, (padding // 2, padding // 2))

# Simpan hasilnya
hasil = gambar + "_.png"
canvas.save(hasil, "PNG")

print("Berhasil membuat canvas baru:", canvas.size)
