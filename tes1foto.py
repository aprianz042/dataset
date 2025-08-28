import cv2
from frontalization import *

path = "FINAL/6_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis/angry/angry_0911.jpg"
#path = "FINAL/6_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis/angry/angry_0026.jpg"
#path = "FINAL/5_dataset_affectnet_rafdb_seleksi_wajah_miring/angry/angry_0350.jpg"
#path = "miring.jpg"
path = cv2.imread(path)
path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)

flow, ori, output = half_flip(path)
show_images_grid(flow)

#output, _ = hand_detection(path)
#show_img(output)

