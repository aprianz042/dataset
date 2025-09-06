import cv2
#from frontalization import *
from frontal import *

#path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/disgust/disgust_2859.jpg"
#path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/fear/fear_2073.jpg"
path = "SAMPLE/3_dataset_affectnet_rafdb_seleksi_wajah_miring/fear/fear_3783.jpg"


path = cv2.imread(path)
#path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)

flow, output = half_flip(path)
show_images_grid(flow)

#output, _ = hand_detection(path)
#show_img(output)

