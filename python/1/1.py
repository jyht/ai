# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:51:41 2019

@author: acm
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras import optimizers

train_dir = r'train'  #训练集地址，根据自己下载位置进行更改
test_dir = r'test'   #测试集地址

#构建训练集数据的generator
train_pic_gen = ImageDataGenerator(rescale=1./255,  #像素缩到0-1之间
                                   rotation_range=20, #数据随机转动角度
                                   width_shift_range=0.2,  #图片水平偏移的角度
                                   height_shift_range=0.2,  #图片数值偏移的角度
                                   shear_range=0.2,  #剪切强度 
                                   zoom_range=0.2,   #随机缩放的幅度
                                   horizontal_flip=True,   #时候进行随机水平翻转
                                   fill_mode='nearest')   #进行变换时，超出边界的点的处理方式

#构建测试集数据的generator
test_pic_gen = ImageDataGenerator(rescale=1./255)  

#生成训练集数据
train_flow = train_pic_gen.flow_from_directory(train_dir,
                                  target_size=(224, 224),
                                  batch_size=64,
                                  class_mode='binary')

#生成测试集数据
test_flow = test_pic_gen.flow_from_directory(test_dir,
                                 target_size=(224, 224),
                                 batch_size=64,
                                 class_mode='binary')

#输出一下图片的分类，由于数据中分为了两个文件夹，他会自动把一个文件夹看作一个类
print(train_flow.class_indices)

#构造模型
model = Sequential()
#加卷积层，activation在卷积神经网络中基本上都用relu
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
#加池化层
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(2, 2))
#扁平层，把多维数据压成一维，是卷积和全连接网络的过度层
model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

#配置训练模型，分别为损失函数，优化程序使用keras中自带的optimizers，步长设置好，评价函数使用acc
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

#训练数据，保存模型
history = model.fit_generator(
        train_flow,
        steps_per_epoch=100,
        epochs=30,
        validation_data=test_flow,
        validation_steps=50)

model.save('cat_and_dog.h5')

