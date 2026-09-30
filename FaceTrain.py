import cv2
import os

cap = cv2.VideoCapture(0)

def get_face():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (720, 300), (1200, 800), (0, 0, 255), 2)
    face = cv2.cvtColor(frame[300:800, 720:1200, :], cv2.COLOR_BGR2GRAY)
    return frame, face

if __name__ == "__main__":
    members = [
        '6752301255_Phongdanai',
        '6752300194_Teerapat',
        '6752301271_Tawaikiar',
        '6752300001_Friend4',
        '6752300002_Friend5'
    ]
    current_member = members[1]
    name = 'data/' + current_member
    i = 1
    os.mkdir(name)
    while True:
        frame, face = get_face()
        cv2.imshow('frame', frame)
        cv2.imshow('face', face)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f'{name}/{i}.jpg', face)
            print(f"บันทึกภาพที่: {i}")
            i += 1
