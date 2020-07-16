# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:05:28 2019

@author: acm
"""

from keras.models import load_model
import cv2

model = load_model('cat_and_dog.h5') #引入模型，上面的代码运行后，模型就在你的保存代码的目录下，可以自己修改地址

model.summary()  #显示一下模型形状

image = cv2.imread(r'3.jpg')  #读取测试图片
image = cv2.resize(image, (224, 224))   #将测试图片缩小
image = image.reshape(1, 224, 224, 3)   #把图片转换成模型输入的维度
print('识别为:')
predict = model.predict_classes(image)
if (predict[0] == 0):
    print('猫')
    print(predict[0])
else:
    print('狗')
    print(predict[0])

#展示图片
#images = image 
#images = images.reshape(224, 224, 3)
#cv2.imshow('Image1', images)
#cv2.waitKey(0)
