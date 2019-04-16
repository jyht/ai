import time
import numpy as np
import sklearn
import sklearn.metrics.pairwise as pw
import cv2
import dlib

prototxt = 'datas/models/caffe/vgg-face/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt'
caffemodel = 'datas/models/caffe/vgg-face/vgg_face_caffe/vgg_face_caffe/VGG_FACE.caffemodel'
dlib_model = 'datas/models/dlib/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model)
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

faces1 = get_faces('datas/images/face-tests/fanbb/f16.jpg')
faces2 = get_faces('datas/images/faces/fanbb.jpg')
for i,face in enumerate(faces1):
    cv2.imshow('face1_%d' % i,face)

for i,face in enumerate(faces2):
    cv2.imshow('face2_%d' % i,face)

face_1 = faces1[0]
face_2 = faces2[0]

result = compare_faces(face_1,face_2)
print('prob of similarity:',result)
cv2.waitKey()
cv2.destroyAllWindows()
