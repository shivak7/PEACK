import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
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
    text_y = int(min(y_coordinates) * height) - 50
    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                2, (255,0,0), 1, cv2.LINE_AA)

  return annotated_image

model_path = '../../../Data/Trackers/MediaPipe/pose_landmarker_heavy.task'
#model_path = '../../../Data/Trackers/MediaPipe/hand_landmarker.task'
model_file = open(model_path, "rb")
model_data = model_file.read()
model_file.close()

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_tracking_confidence = 0.75,
    min_pose_detection_confidence= 0.75)


# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_buffer=model_data),
#     running_mode=VisionRunningMode.IMAGE,
#     num_hands=2)


#FilePath = '/Users/shiva/Downloads/BT045 Pre-test.MP4'    # MacOS @ BNI
FilePath = 'C:\\Users\\shiva\\Downloads\\BT036 Pre-test.MP4'  # Windows @ Home
cap = cv2.VideoCapture(FilePath)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps
time_Stamp = 0

with PoseLandmarker.create_from_options(options) as landmarker:
#with HandLandmarker.create_from_options(options) as landmarker:

#base_options = python.BaseOptions(model_asset_path=FilePath)
#options = vision.HandLandmarkerOptions(base_options=base_options,
#                                       num_hands=2)
#detector = vision.HandLandmarker.create_from_options(options)


  while True:

      ret, cv_frame = cap.read()

      if not ret:
          print('End of file.')
          break
      
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_frame)
      pose_landmarker_result = landmarker.detect(mp_image)
      time_Stamp = time_Stamp + int(dt*1000)

      annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
      #annotated_image = draw_hand_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
      cv2.imshow('Frame', annotated_image)

      if cv2.waitKey(1) == ord('q'):
          break

  cap.release()
cv2.destroyAllWindows()