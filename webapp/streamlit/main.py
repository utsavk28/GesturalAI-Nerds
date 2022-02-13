import os
import numpy as np
import cv2
import sklearn
from tensorflow import keras
import streamlit as st

labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
          15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

# @st.cache


def load_model():
    return keras.models.load_model("./resnet_model.h5")


def main():
    st.write("# GesturalAI 🤖")

    model = load_model()

    run = st.checkbox('Run')

    FRAMEWINDOW = st.image([])
    final = []
    letters = []
    cap = cv2.VideoCapture(0)
    while run:
        isTrue, frame = cap.read()

        if not isTrue:
            break

        img = frame[:200, :200]
        img = cv2.flip(img, 1)

        img = cv2.resize(img, (100, 100,))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).reshape((1, 100, 100, 3))
        res = model.predict(img)[0]
        n = np.argmax(res)

        letters.append(labels[n])
        letters = letters[-10:]
        addInFinal = True
        for l in letters:
            if l != labels[n]:
                addInFinal = False
                break

        if addInFinal:
            if len(final) == 0:
                final.append(labels[n])
            elif final[-1] != labels[n]:
                final.append(labels[n])

        h, w, c = frame.shape
        print_text = ''.join(list(filter(lambda x: x != 'nothing' and x != 'del', list(
            map(lambda x: '_' if x == 'space' else x, final)))))
        cv2.rectangle(frame, (0, 0), (200, 200), (225, 0, 255), 2)
        cv2.putText(frame, f"{print_text}", (w//2-len(print_text)*10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAMEWINDOW.image(frame)
        print(labels[n])


if __name__ == "__main__":
    main()
