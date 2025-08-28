import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat

def tangan(image): 
    model_path = 'model/hand_model.task'

    options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                    running_mode=VisionRunningMode.IMAGE,
                                    min_hand_detection_confidence=0.2)
        
    with HandLandmarker.create_from_options(options) as landmarker:           
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = Image(ImageFormat.SRGB, image_rgb)
        results = landmarker.detect(mp_image)
        if results.hand_landmarks:
            for landmarks in results.hand_landmarks:
                for landmark in landmarks:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Gambar titik dengan warna hijau
        return image

def face_landmark(image):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    base_options = python.BaseOptions(model_asset_path='model/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, 
                                           output_face_blendshapes=True, 
                                           output_facial_transformation_matrixes=True, 
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    detection_result = detector.detect(img)
    return None, detection_result

def landmark_wajah(face):
    face_lm, lm = face_landmark(face)
    if len(lm.face_landmarks):
        return face_lm, lm
    else:      
        return None, None

def correct_roll(image):
    h, w, _ = image.shape
    img_rgb = image
    _, lm = landmark_wajah(img_rgb)
    if lm is None:  
        return image
    else:
        for face_landmarks in lm.face_landmarks:
            left_eye = np.array([face_landmarks[33].x * w, face_landmarks[33].y * h])
            right_eye = np.array([face_landmarks[263].x * w, face_landmarks[263].y * h])
            dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            #corrected_image = cv2.warpAffine(image, M, (w, h))
            corrected_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            return corrected_image

def main_():    
    image = cv2.imread("FINAL/6_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis/angry/angry_0888.jpg")
    print(image.dtype)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image = correct_roll(image)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = tangan(image)
    return output


gam = main_()
cv2.imshow('Hand Landmarks', gam)
cv2.waitKey(0)
cv2.destroyAllWindows()

