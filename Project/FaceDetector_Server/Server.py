import socket
import cv2
import numpy as np
import os
import random
import cv2


# socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


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
    facenet = cv2.dnn.readNet('models/model.prototxt', 'models/model.caffemodel')

    HOST = '0.0.0.0'
    PORT = 8808

    # TCP 사용
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    # 서버의 아이피와 포트번호 지정
    s.bind((HOST, PORT))
    print('Socket bind complete')
    # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
    s.listen(10)
    print('Socket now listening')

    # 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
    conn, addr = s.accept()

    while True:
        # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype='uint8')

        # data를 디코딩한다.
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
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
            face = face / 256

            if (x2 >= w or y2 >= h):
                continue
            if (x1 <= 0 or y1 <= 0):
                continue

            face_input = cv2.resize(face, (200, 200))
            face_input = np.expand_dims(face_input, axis=0)
            face_input = np.array(face_input)

            color = (255, 255, 255)

            file_list = os.listdir("croppedData")

            cropped_data_path = "croppedData/temp" + random.randrange(0, 999999).__str__() + ".jpg"
            height_dist = (y2 - y1) // 2
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
