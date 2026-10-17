import tkinter as tk
from tkinter import *
import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time

# Window is our Main frame of system
window = tk.Tk()
window.title("FAMS-Face Recognition Based Attendance Management System")
window.geometry('1280x720')
window.configure(background='grey80')

# GUI for manually fill attendance
rtsp_url = "rtsp://192.168.1.10:8080/h264.sdp"


def manually_fill():
    global sb
    sb = tk.Tk()
    sb.title("Enter subject name...")
    sb.geometry('580x320')
    sb.configure(background='grey80')


def err_screen_for_subject():
    def ec_delete():
        ec.destroy()

    global ec
    ec = tk.Tk()
    ec.geometry('300x100')
    ec.title('Warning!!')
    ec.configure(background='snow')
    Label(ec, text='Please enter your subject name!!!', fg='red', bg='white', font=('times', 16, ' bold ')).pack()
    Button(ec, text='OK', command=ec_delete, fg="black", bg="lawn green", width=9, height=1, activebackground="Red",
           font=('times', 15, ' bold ')).place(x=90, y=50)


def fill_attendance():
    ts = time.time()
    Date = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

    # Create table for Attendance date_for_DB
    date_for_DB = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')

    global subb
    subb = SUB_ENTRY.get()

    DB_table_name = str(subb + "_" + Date + "_Time_" + timeStamp.replace(':', '_'))

    import pymysql.connections
    try:
        global cursor
        connection = pymysql.connect(host='localhost', user='root', password='', db='manually_fill_attendance')
        cursor = connection.cursor()
    except Exception as e:
        print(e)

    sql = "CREATE TABLE " + DB_table_name + """ (ID INT NOT NULL AUTO_INCREMENT, ENROLLMENT varchar(100) NOT NULL, NAME VARCHAR(50) NOT NULL, DATE VARCHAR(20) NOT NULL, TIME VARCHAR(20) NOT NULL, PRIMARY KEY (ID) ); """

    try:
        cursor.execute(sql)
    except Exception as ex:
        print(ex)

    if subb == '':
        err_screen_for_subject()
    else:
        sb.destroy()

        MFW = tk.Tk()
        MFW.title("Manually attendance of " + str(subb))
        MFW.geometry('880x470')
        MFW.configure(background='grey80')

        def del_errsc2():
            errsc2.destroy()

        def err_screen1():
            global errsc2
            errsc2 = tk.Tk()
            errsc2.geometry('330x100')
            errsc2.title('Warning!!')
            errsc2.configure(background='grey80')
            Label(errsc2, text='Please enter Student & Enrollment!!!', fg='black', bg='white',
                  font=('times', 16, ' bold ')).pack()
            Button(errsc2, text='OK', command=del_errsc2, fg="black", bg="lawn green", width=9, height=1,
                   activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

        def testVal(inStr, acttyp):
            if acttyp == '1':
                if not inStr.isdigit():
                    return False
            return True

        ENR = tk.Label(MFW, text="Enter Enrollment", width=15, height=2, fg="black", bg="grey", font=('times', 15))
        ENR.place(x=30, y=100)

        STU_NAME = tk.Label(MFW, text="Enter Student name", width=15, height=2, fg="black", bg="grey",
                            font=('times', 15))
        STU_NAME.place(x=30, y=200)

        global ENR_ENTRY
        ENR_ENTRY = tk.Entry(MFW, width=20, validate='key', bg="white", fg="black", font=('times', 23))
        ENR_ENTRY['validatecommand'] = (ENR_ENTRY.register(testVal), '%P', '%d')
        ENR_ENTRY.place(x=290, y=105)

        STUDENT_ENTRY = tk.Entry(MFW, width=20, bg="white", fg="black", font=('times', 23))
        STUDENT_ENTRY.place(x=290, y=205)

        def enter_data_DB():
            ENROLLMENT = ENR_ENTRY.get()
            STUDENT = STUDENT_ENTRY.get()
            if ENROLLMENT == '' or STUDENT == '':
                err_screen1()
            else:
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Insert_data = "INSERT INTO " + DB_table_name + " (ID,ENROLLMENT,NAME,DATE,TIME) VALUES (0, %s, %s, %s,%s)"
                VALUES = (str(ENROLLMENT), str(STUDENT), str(Date), str(timeStamp))
                try:
                    cursor.execute(Insert_data, VALUES)
                except Exception as e:
                    print(e)
                ENR_ENTRY.delete(first=0, last=22)
                STUDENT_ENTRY.delete(first=0, last=22)

        def create_csv():
            cursor.execute("select * from " + DB_table_name + ";")
            csv_name = f'D:/Project/Face-Recognition-Attendance-System-main/Attendance/Manually Attendance/{DB_table_name}.csv'
            with open(csv_name, "w") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([i[0] for i in cursor.description])  # write headers
                csv_writer.writerows(cursor)

            Notifi.configure(text="CSV created Successfully", bg="Green", fg="white", width=33,
                             font=('times', 19, 'bold'))
            Notifi.place(x=180, y=380)

        Notifi = tk.Label(MFW)

        clear_enroll = tk.Button(MFW, text="Clear", command=lambda: ENR_ENTRY.delete(0, 'end'), fg="white", bg="black",
                                 width=10)
        clear_enroll.place(x=690, y=100)

        clear_student = tk.Button(MFW, text="Clear", command=lambda: STUDENT_ENTRY.delete(0, 'end'), fg="white",
                                  bg="black", width=10)
        clear_student.place(x=690, y=200)

        DATA_SUB = tk.Button(MFW, text="Enter Data", command=enter_data_DB, fg="black", bg="SkyBlue1", width=20,
                             height=2)
        DATA_SUB.place(x=170, y=300)

        MAKE_CSV = tk.Button(MFW, text="Convert to CSV", command=create_csv, fg="black", bg="SkyBlue1", width=20,
                             height=2)
        MAKE_CSV.place(x=570, y=300)

        MFW.mainloop()


SUB = tk.Label(sb, text="Enter Subject : ", width=15, height=2, fg="black", bg="grey80", font=('times', 15, ' bold '))
SUB.place(x=30, y=100)

global SUB_ENTRY
SUB_ENTRY = tk.Entry(sb, width=20, bg="white", fg="black", font=('times', 23))
SUB_ENTRY.place(x=250, y=105)

fill_manual_attendance = tk.Button(sb, text="Fill Attendance", command=fill_attendance, fg="black", bg="SkyBlue1",
                                   width=20, height=2)
fill_manual_attendance.place(x=250, y=160)

sb.mainloop()


# For clear textbox functions
def clear():
    txt.delete(first=0, last=22)


def clear1():
    txt2.delete(first=0, last=22)


def take_img():
    l1 = txt.get()
    l2 = txt2.get()

    if l1 == '' or l2 == '':
        err_screen()
        return

    try:
        cam = cv2.VideoCapture(rtsp_url)
        detector = cv2.CascadeClassifier(
            r'D:/Project/Face-Recognition-Attendance-System-main/haarcascade_frontalface_default.xml')
        Enrollment = l1
        Name = l2
        sampleNum = 0

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite("TrainingImage/" + Name + "." + Enrollment + '.' + str(sampleNum) + ".jpg", gray)

                print("Images Saved for Enrollment :")
                cv2.imshow('Frame', img)

            if cv2.waitKey(100) & 0xFF == ord('q'): break

            if sampleNum > 70: break

        cam.release()
        cv2.destroyAllWindows()

        row = [Enrollment, Name, Date, Time]

        with open('D:/Project/Face-Recognition-Attendance-System-main/StudentDetails/StudentDetails.csv',
                  'a+') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow(row)

        res = "Images Saved for Enrollment : " + Enrollment + " Name : " + Name
        Notification.configure(text=res, bg="SpringGreen3", width=(50), font=('times', 18, 'bold'))
        Notification.place(x=(250), y=(400))

except FileExistsError as F:
f = 'Student Data already exists'
Notification.configure(text=f, bg="Red", width=(21))
Notification.place(x=(450), y=(400))


# Function to choose subject and fill attendance automatically
def subjectchoose():
    def Fillattendances():
        sub = tx.get()

        now = time.time()

        future = no + 20

        if time.time() < future:
            if sub == '':
                err_screen1()
            else:
                recognizer = cv2.createLBPHFaceRecognizer()

                try:
                    recognizer.read("TrainingImageLabel/Trainner.yml")
                except:
                    e = 'Model not found Please train model'
                    Notifica.configure(text=e, bg="red", fg="black", width=(33), font=('times', 15, 'bold'))
                    Notifica.place(x=-20, y=-250)

                harcascadePath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(harcascadePath)
                df = pd.read_csv("StudentDetails/StudentDetails.csv")

                cam = cv2.VideoCapture(rtsp_url)
                font = cv2.FONT_HERSHEY_SIMPLEX

                col_names = ['Enrollment', 'Name', 'Date', 'Time']
                attendance = pd.DataFrame(columns=(col_names))

                while True:
                    ret, img(cam.read())
                    gray = cv2.cvtColor(im(cv.COLOR_BGR2GRAY))
                    faces(faceCascade.detectMultiScale(gray(1.3)))

                    for (x, y, w, h)(faces):
                        global Id
                        Id(conf) = recognizer.predict(gray[y:y + h, x:x + w])

                        if conf < 70:
                            print(conf)

                            global Subject
                            global aa
                            global date
                            global timeStamp

                            Subject(tx.get())
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                            aa = df.loc[df['Enrollment'] == Id]['Name'].values[0]

                            attendance.loc[len(attendance)] = [Id(aa, date, timeStamp)]

                            cv.rectangle(im, (x, y), (x + w, y + h), (0, (260), (0)), 7)
                            cv.putText(im, str(tt), (x + h, y), font, (255), (255), (0,), 4)

                        else:
                            Id('Unknown')
                            tt = str(Id)

                        cv.rectangle(im, (x, y), (x + w, h), (0, (25), (255)), 7)
                        cv.putText(im, str(tt), (x + h, y), font, (0, (25), (255)), 4)

                    if time.time() > future: break

                    attendance.drop_duplicates(['Enrollment'], keep='first')

                    cv.imshow('Filling attendance...', im)

                    key = cv.waitKey(30) & 0xff
                    if key == 27: break

                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                Hour.Minute.Second = timeStamp.split(":")

                fileName = f"Attendance/{Subject}_{date}_{Hour}-{Minute}-{Second}.csv"

                attendance.drop_duplicates(['Enrollment'], keep='first')

                print(attendance.to_csv(fileName, index=False))

                # Create table for Attendance date_for_DB

    date_for_DB = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
    DB_Table_name = str(Subject + "_" + date_for_DB + "_Time_" + Hour + "_" + Minute + "_" + Second)

    import pymysql.connections

    # Connect to the database
    try:
        global cursor
        connection = pymysql.connect(host='localhost', user='root', password='', db='Face_reco_fill')
        cursor(connection.cursor())

    except Exception as e:
        print(e)

    sql = f"""CREATE TABLE {DB_Table_name} (
      ID INT NOT NULL AUTO_INCREMENT,
      ENROLLMENT varchar(100) NOT NULL,
      NAME VARCHAR(50) NOT NULL,
      DATE VARCHAR(20) NOT NULL,
      TIME VARCHAR(20) NOT NULL,
      PRIMARY KEY (ID));"""

    # Now enter attendance in Database insert_data
    insert_data = f"""INSERT INTO {DB_Table_name} (ID,
   ENROLLMENT,
   NAME,
   DATE,
   TIME ) VALUES (0,%s,%s,%s,%s);"""

    VALUES = (str(Id), str(aa), str(date), str(timeStamp))

    try:
        cursor.execute(sql)  # for create a table #For insert data into table cursor.execute(insert_data,
        VALUES)

        except Exception as ex:
        print(ex)

    M = 'Attendance filled Successfully'
    Notifica.configure(text=M, bg="Green", fg="white",
                       width=(33),
                       font=('times',
                             15, 'bold'))
    Notifica.place(x=(20), y=(250))

    cam.release()
    cv.destroyAllWindows()


# GUI to display attendance records

import tkinter as tkinter

root = tkinter.Tk()

root.title(f"Attendance of {Subject}")

root.configure(background='grey80')

cs = f'C:/Users/Pragya Singh/PycharmProjects/Attendace_management_system/{fileName}'

with open(cs, newline='') as file:
    reader(csv.reader(file))
    r, c = (0), (0)

    for col in reader:
        c = (0)
        for row in col:
            label = tkinter.Label(root, width=(10),
                                  height=(1),
                                  fg=("black"),
                                  font=("times",
                                        15, " bold "),
                                  bg=("white"),
                                  text=row,
                                  relief=tkinter.RIDGE)
            label.grid(row=r, column=c)
            c += 1

        r += 1

root.mainloop()

# Window is frame for subject chooser windo

windo = tk.Tk()

windo.title("Enter subject name...")

windo.geometry('580x320')

windo.configure(background='grey80')

Notifica = tk.Label(windo, text=("Attendance filled Successfully"),
                    bg=("Green"), fg=("white"),
                    width=(33),
                    height=(2),
                    font=("times",
                          15, "bold"))


def Attf():
    import subprocess
    subprocess.Popen(
        r'explorer /select,"C:\Users\Pragya Singh\PycharmProjects\Attendace_management_system\Attendance\-------Check atttendance-------"')


attf = tk.Button(windo, text=("Check Sheets"), command=(Attf),
                 fg=("white"), bg=("black"),
                 width=(12),
                 height=(1),
                 activebackground=("white"),
                 font=("times",
                       14, " bold "))
attf.place(x=(430), y=(255))

sub = tk.Label(windo, text=("Enter Subject : "), width=(15),
               height=(2), fg=("black"),
               bg=("grey"),
               font=("times",
                     15, " bold "))
sub.place(x=(30), y=(100))

tx = tk.Entry(windo, width=(20),
              bg=("white"), fg=("black"),
              font=("times",
                    23))
tx.place(x=(250), y=(105))

fill_a = tk.Button(windo, text=("Fill Attendance"),
                   fg=("white"), command=(Fillattendances),
                   bg=("SkyBlue1"),
                   width=(20),
                   height=(2),
                   activebackground=("white"),
                   font=("times",
                         15, " bold "))
fill_a.place(x=(250), y=(160))

windo.mainloop()

