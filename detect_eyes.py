import cv2
import numpy as np
from glob import glob


face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

height = 720
width = 1280
position = 1

image_paths = glob(f"captured_images/{position}/*.jpg")

flag = False
for path in image_paths:
    captured_image = cv2.imread(path, 0)
    faces = face_cascade.detectMultiScale(captured_image)
    if len(faces) == 0:
        print(f"{path.split('/')[-1]} is skipped by face")
        cv2.imwrite(f'skipped_face/{position}/{path.split("/")[-1]}', captured_image)
        continue

    eye_mask = np.full((height, width), 1, dtype=np.uint8)

    for x, y, w, h in faces:
        face_gray = captured_image[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        if len(eyes) != 2:
            print(f"{path.split('/')[-1]} is skipped by eyes")
            cv2.imwrite(f'skipped_eyes/{position}/{path.split("/")[-1]}', face_gray)
            flag = True
            break

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(eye_mask, (x + ex, y + ey), (x + ex + ew, y + ey + eh),(255), cv2.FILLED)

    if flag:
        flag = False
        continue

    eyes_image = cv2.bitwise_and(captured_image, eye_mask)
    cv2.imwrite(f'face_images/{position}/{path.split("/")[-1]}', face_gray)
    cv2.imwrite(f'eye_images/{position}/{path.split("/")[-1]}', eyes_image)