import cv2
import numpy as np                                                               
from skimage import transform as tf

import dlib
import mediapipe as mp


def extract_landmarks_dlib(video_path, detector, predictor, display = False):
    cap = cv2.VideoCapture(video_path)
    video_landmarks = []
    while True:
        ret, image = cap.read()
        if ret == False:
            break
        # Convert the image color to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect the face
        rects = detector(gray, 1)
        # Detect landmarks for each face
        for rect in rects:
            # Get the landmark points
            shape = predictor(gray, rect)
            # Convert it to the NumPy Array
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)
            shape = shape_np

            video_landmarks.append(shape_np)

            if display:
                # Display the landmarks
                for i, (x, y) in enumerate(shape):
                    # Draw the circle to mark the keypoint
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                cv2.imshow('Landmark Detection', image)
                if cv2.waitKey(10) == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()

    return video_landmarks

def extract_landmarks_mediapipe(video_path, mp_face_mesh, mp_drawing, mp_drawing_styles, display = False):
    data = cv2.VideoCapture(video_path)
    video_landmarks = []
    while True:
        success, img = data.read()
        if not success:
            # break out of the loop if there are no frames to read
            break
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.multi_face_landmarks:
                continue
            annotated_image = img.copy()
            for face_landmarks in results.multi_face_landmarks:
                if display:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())

                    # Display the image
                    cv2.imshow('Landmark Detection', annotated_image)
                    cv2.waitKey(20)

    return None


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped


# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)
