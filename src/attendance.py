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