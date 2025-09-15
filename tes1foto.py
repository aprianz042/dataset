import cv2
#from frontalization import *
from frontal import *

path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/angry/angry_6771.jpg"
path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/surprise/surprise_0038.jpg"


#path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/neutral/neutral_0425.jpg"
#path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/neutral/neutral_0303.jpg"
path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/happy/happy_1400.jpg"

#path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/disgust/disgust_2859.jpg"
path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/fear/fear_1617.jpg"
#path = "SAMPLE/2_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis/angry/angry_1378.jpg"

path = cv2.imread(path)
#path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)

flow, output = half_flip(path)
show_images_grid(flow)

#output, _ = hand_detection(path)
#show_img(output)

