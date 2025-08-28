import os
import random

from typing import Tuple, Union
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def hand_random():
    tangan_list = ["kanan", "kiri"]
    gambar_list = ["hand1.png", "hand2.png", "hand3.png", "hand4.png"]
    tangan = random.choice(tangan_list)
    gambar = random.choice(gambar_list)
    path = f"hand/{tangan}/{gambar}"
    return tangan, path

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

connections = {
    "tesselation": (mp.solutions.face_mesh.FACEMESH_TESSELATION, 'cyan'),
    "contours": (mp.solutions.face_mesh.FACEMESH_CONTOURS, 'red'),
    "irises": (mp.solutions.face_mesh.FACEMESH_IRISES, 'magenta')
}

linewidths = {
    "tesselation": 0.5,  # Lebih tipis
    "contours": 2.5,  # Medium
    "irises": 1.0  # Lebih tebal
}

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(image,detection_result) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw_face_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

def face_landmark(image):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    base_options = python.BaseOptions(model_asset_path='model/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, 
                                           output_face_blendshapes=True, 
                                           output_facial_transformation_matrixes=True, 
                                           min_face_detection_confidence=0.3,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    detection_result = detector.detect(img)
    annotated_image = draw_face_landmarks_on_image(img.numpy_view(), detection_result)
    return annotated_image, detection_result

def landmark_wajah(face):
    face_lm, lm = face_landmark(face)
    if len(lm.face_landmarks):
        return face_lm, lm
    else:
        stat = 'gagal'
        return face, stat

def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]
    if x >= bw or y >= bh:
        return background
    if x + ow > bw:
        ow = bw - x
        overlay = overlay[:, :ow]
    if y + oh > bh:
        oh = bh - y
        overlay = overlay[:oh]
    if x < 0:
        overlay = overlay[:, -x:]
        ow += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        oh += y
        y = 0
    if overlay.shape[2] < 4:
        return background
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    roi = background[y:y + oh, x:x + ow]
    blended = (1.0 - mask) * roi + mask * overlay_img
    background[y:y + oh, x:x + ow] = blended.astype(np.uint8)
    return background


input_root = "FINAL/4_dataset_affectnet_rafdb_seleksi_wajah_lurus"
output_root = "FINAL/6_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis"

for subfolder in os.listdir(input_root):
    subfolder_path = os.path.join(input_root, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    output_subfolder_path = os.path.join(output_root, subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)

    for filename in os.listdir(subfolder_path):
        tangan, hand_image_path = hand_random()
        hand_img = cv2.imread(hand_image_path, cv2.IMREAD_UNCHANGED)

        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        input_img_path = os.path.join(subfolder_path, filename)
        image = cv2.imread(input_img_path)

        if image is None:
            print(f"❌ Gagal membaca {input_img_path}")
            continue

        try:
            _, land_mark = face_landmark(image)
            lm = land_mark.face_landmarks[0]
            if lm:
                ih, iw, _ = image.shape    
                # Ukur lebar wajah
                cheek_left = lm[234]
                cheek_right = lm[454]
                x1, y1 = int(cheek_left.x * iw), int(cheek_left.y * ih)
                x2, y2 = int(cheek_right.x * iw), int(cheek_right.y * ih)
                face_width = np.linalg.norm([x2 - x1, y2 - y1])

                # Resize tangan
                desired_width = int(face_width * 0.5 + 10)
                scale_ratio = desired_width / hand_img.shape[1]
                resized_hand = cv2.resize(hand_img, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

                # Roll kepala (dahi ke dagu)
                top = lm[10]
                bottom = lm[152]
                x_top = int(top.x * iw)
                y_top = int(top.y * ih)
                x_bottom = int(bottom.x * iw)
                y_bottom = int(bottom.y * ih)
                dx = x_bottom - x_top
                dy = y_bottom - y_top
                angle = 90 - np.degrees(np.arctan2(dy, dx))

                # Rotasi tangan
                hh, ww = resized_hand.shape[:2]
                M = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle, 1)
                rotated_hand = cv2.warpAffine(resized_hand, M, (ww, hh),
                                            flags=cv2.INTER_AREA,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0, 0))

                # Posisi wajah (bawah hidung)
                center = lm[4]
                px = int(center.x * iw)
                py = int(center.y * ih)

                if tangan == "kanan":  
                    offset_x = int(px - rotated_hand.shape[1] * 0.5 - 28)
                else:
                    offset_x = int(px - rotated_hand.shape[1] * 0.5 + 28) 
                offset_y = int(py - rotated_hand.shape[0] * 0.5)

                # Tempel tangan
                image = overlay_transparent(image, rotated_hand, offset_x, offset_y)

            # Simpan hasil
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}{ext}"
            output_img_path = os.path.join(output_subfolder_path, output_filename)
            cv2.imwrite(output_img_path, image)
            print(f"{output_img_path} --> OK.")
        
        except Exception as e:
            print(f"❌ Error proses {input_img_path} - {e}")

print("✅ Selesai proses semua gambar.")
