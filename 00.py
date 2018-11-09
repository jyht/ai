#-*- coding: utf-8 -*-

import time
import numpy as np
import sklearn
import sklearn.metrics.pairwise as pw
import cv2
import dlib
import face_recognition


facerec = dlib.face_recognition_model_v1("c:/dlib_face_recognition_resnet_model_v1.dat")

# 计算两个向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("same?: ", dist)
'''
    if dist > 0.4:
        return "diff"
    else:
        return "same"
'''
# 返回单张图像的 128D 特征
def return_128d_features(path_img):
    img = cv2.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)	

    print("检测的人脸图像：", path_img, "\n")

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    # print(face_descriptor)
    return face_descriptor

prototxt = 'C:/vgg_face_caffe/VGG_FACE_deploy.prototxt'
caffemodel = 'C:/vgg_face_caffe/VGG_FACE.caffemodel'
dlib_model = 'C:/dlib_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model)
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)


faces1 = return_128d_features('c:/11.jpg')
faces2 = return_128d_features('c:/22.jpg')

		

face_1 = faces1[0]
face_2 = faces2[0]

return_euclidean_distance(face_1,face_2)
cv2.waitKey(1)
#cv2.destroyAllWindows()