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