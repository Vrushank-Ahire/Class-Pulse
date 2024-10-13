import cv2
from mtcnn import MTCNN

def detect_faces(image_path):
    img = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 2)
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return faces