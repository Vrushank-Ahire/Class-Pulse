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