import os
import random
import time
import cv2
import numpy as np


def rename_file(dist_lable: str):
    count = 0
    file_list = os.listdir("croppedData")
    for i in range(file_list.__len__()):
        if file_list[i].endswith(".jpg"):

            src = "croppedData/" + file_list[i]
            dst = "croppedData/" + dist_lable + count.__str__() + ".jpg"
            os.rename(src, dst)
            print(src + " rename to " + dst)
            count += 1


if __name__ == '__main__':
    rename_file('temp')

    facenet = cv2.dnn.readNet('models/model.prototxt', 'models/model.caffemodel')

    cap = cv2.VideoCapture(0)
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(405, 405), mean=(104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            face = frame[y1:y2, x1:x2]
            face = face/256

            if (x2 >= w or y2 >= h):
                continue
            if (x1<=0 or y1<=0):
                continue

            face_input = cv2.resize(face,(200, 200))
            face_input = np.expand_dims(face_input, axis=0)
            face_input = np.array(face_input)

            color = (255, 255, 255)

            file_list = os.listdir("croppedData")

            cropped_data_path = "croppedData/temp" + random.randrange(0, 999999).__str__() + ".jpg"
            height_dist = (y2-y1)//2
            crop = frame[y1: y2 - height_dist, x1: x2]
            try:
                cv2.imwrite(cropped_data_path, crop)
            except Exception as e:
                print(e)

            cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)

        cv2.imshow('masktest', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    rename_file('crop')
