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