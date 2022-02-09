from sklearn.preprocessing import LabelEncoder
import numpy as np
import mediapipe as mp
import cv2

BATCH_SIZE = 128
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)


def process_image_data(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgRGB)

    angles = None
    if not res.multi_hand_landmarks is None:
        connections = np.array(list(mpHands.HAND_CONNECTIONS))
        landmarks = []
        angles = []

        handLms = res.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark):
            h, w, c = img.shape
            x, y = lm.x, lm.y
            x, y = int(x*w), int(y*h)
            landmarks.append([lm.x*w, lm.y*h])

        landmarks = np.array(landmarks)

        for i in connections:
            for j in connections:
                u = landmarks[i[1]] - landmarks[i[0]]
                v = landmarks[j[1]] - landmarks[j[0]]
                dot_product = u[0]*v[0]+u[1]*v[1]
                norm = np.linalg.norm(u) * np.linalg.norm(v)
                if np.any(np.isnan([np.arccos(dot_product / norm)])) :
                    angles.append(0)
                else:
                    angles.append(np.arccos(dot_product / norm))
        angles = np.array(angles)
        
    if angles is None:
        angles = np.zeros((441,))

    angles[np.isnan(angles).astype(int)] = 0
    angles = np.round(angles, 4)
    angles[angles < 10**-4] = 0
    angles[angles > 22/7] = 22/7
    angles = np.array(angles).reshape(-1, 441)

    return angles
