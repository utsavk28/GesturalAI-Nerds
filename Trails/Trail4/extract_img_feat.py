import os
import numpy as np
import mediapipe as mp
import cv2

img = cv2.imread('./A_10.jpg')
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
img = cv2.flip(img, 1)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res = hands.process(imgRGB)

# print(res)

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
        # if id == 8 :
        #     cv2.circle(img,(x,y),10,(255,0,0),-1)
        cv2.putText(img,str(id), (x, y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255))
        
    landmarks = np.array(landmarks)
    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    for i in connections:
        for j in connections:
            print(i, j)
            u = landmarks[i[1]] - landmarks[i[0]]
            v = landmarks[j[1]] - landmarks[j[0]]
            print(u,v)
            dot_product = u[0]*v[0]+u[1]*v[1]
            norm = np.linalg.norm(u) * np.linalg.norm(v)
            # print(dot_product,norm)
            print(np.arccos(dot_product / norm))
            # print(dot_product/norm, np.arccos(dot_product/norm))
            angles.append(np.arccos(dot_product / norm))

    # print(landmarks)
    # print(connections)
    # print(sorted(angles))

cv2.imshow('Img', img)
cv2.waitKey(0)
