import os
import shutil

# Create project structure
def create_project_structure():
    base_dir = 'attendance-system'
    dirs = [
        '',
        'data',
        'models',
        'src',
        'reports'
    ]
    for dir in dirs:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)

# Create requirements.txt
def create_requirements():
    requirements = '''
opencv-python
tensorflow
torch
dlib
scikit-learn
pandas
SQLAlchemy
mtcnn
facenet-pytorch
flask
'''
    with open('attendance-system/requirements.txt', 'w') as f:
        f.write(requirements.strip())

# Create detection.py
def create_detection_py():
    code = '''
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
'''
    with open('attendance-system/src/detection.py', 'w') as f:
        f.write(code.strip())

# Create recognition.py
def create_recognition_py():
    code = '''
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np

model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess(image):
    # Implement preprocessing steps (resize, normalize, etc.)
    return image

def get_face_embedding(image):
    img_cropped = preprocess(image)  # Preprocess image (crop, resize)
    embedding = model(img_cropped.unsqueeze(0))  # Get embedding
    return embedding

def compare_embeddings(embedding1, embedding2, threshold=0.5):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < threshold
'''
    with open('attendance-system/src/recognition.py', 'w') as f:
        f.write(code.strip())

# Create database.py
def create_database_py():
    code = '''
from sqlalchemy import create_engine, Column, Integer, String, Date, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    embedding = Column(String)  # Save as JSON or serialized object

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer)
    date = Column(Date)
    status = Column(Boolean)

engine = create_engine('sqlite:///attendance.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
'''
    with open('attendance-system/src/database.py', 'w') as f:
        f.write(code.strip())

# Create attendance.py
def create_attendance_py():
    code = '''
from datetime import date
from recognition import get_face_embedding, compare_embeddings
from database import Student, Attendance, session
from detection import detect_faces

def mark_attendance(image_path):
    faces = detect_faces(image_path)
    for face in faces:
        embedding = get_face_embedding(face['image'])
        students = session.query(Student).all()
        for student in students:
            if compare_embeddings(embedding, student.embedding):
                attendance = Attendance(student_id=student.id, date=date.today(), status=True)
                session.add(attendance)
                session.commit()
                print(f"Attendance marked for student {student.name}")
                break
        else:
            print("Unknown face detected")

if __name__ == "__main__":
    mark_attendance("path/to/classroom/image.jpg")
'''
    with open('attendance-system/src/attendance.py', 'w') as f:
        f.write(code.strip())

# Create report_generation.py
def create_report_generation_py():
    code = '''
import pandas as pd
from database import Attendance, session

def generate_report():
    attendance_data = session.query(Attendance).all()
    df = pd.DataFrame([(att.student_id, att.date, att.status) for att in attendance_data],
                      columns=['Student ID', 'Date', 'Status'])
    df.to_csv('reports/attendance_report.csv', index=False)
    print("Attendance report generated: reports/attendance_report.csv")

if __name__ == "__main__":
    generate_report()
'''
    with open('attendance-system/src/report_generation.py', 'w') as f:
        f.write(code.strip())

# Create api.py
def create_api_py():
    code = '''
from flask import Flask, jsonify
from database import Attendance, session

app = Flask(__name__)

@app.route('/attendance', methods=['GET'])
def get_attendance():
    data = session.query(Attendance).all()
    attendance_list = [{"student_id": att.student_id, "date": str(att.date), "status": att.status} for att in data]
    return jsonify(attendance_list)

if __name__ == '__main__':
    app.run(debug=True)
'''
    with open('attendance-system/src/api.py', 'w') as f:
        f.write(code.strip())

# Main function to create all files
def create_attendance_system():
    create_project_structure()
    create_requirements()
    create_detection_py()
    create_recognition_py()
    create_database_py()
    create_attendance_py()
    create_report_generation_py()
    create_api_py()
    print("Attendance system files created successfully!")

# Run the script
if __name__ == "__main__":
    create_attendance_system()