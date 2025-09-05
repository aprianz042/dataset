from rgb2mesh import rgb2mesh, show_images, save_2d_face
import cv2
import matplotlib.pyplot as plt

path = "FINAL/4_dataset_affectnet_rafdb_seleksi_wajah_lurus/angry/angry_0888.jpg"
path = cv2.imread(path)
img_mesh = rgb2mesh(path)

out = save_2d_face(img_mesh, 'bukan', "apalah")

plt.imshow(out)
plt.axis('off')  # Optional, to hide the axis
plt.show()