#-*- coding: utf-8 -*-
#视频识别人脸
import face_recognition
import cv2
import numpy as np
import pymysql


video_capture = cv2.VideoCapture('1.mp4')



# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

db = pymysql.connect("localhost","root","","baiduyun" )
cursor = db.cursor()
sql = "select * from face where tzm !=0 order by id asc limit 0,3"
cursor.execute(sql)
results = cursor.fetchall()
for row in results:
	id = row[0]
	imgnames = row[2]
	tzm_list = row[3]
	# 将字符串转为numpy ndarray类型，即矩阵
	# 转换成一个list
	dlist = tzm_list.strip(' ').split(',')
	# 将list中str转换为float
	dfloat = list(map(float, dlist))
	arr = np.array(dfloat) #arr 数据库里出来的矩阵
	known_face_encodings.append(arr)
	known_face_names.append(imgnames)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face 戴
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) 
		
    # Display the resulting image
    cv2.imshow('video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()