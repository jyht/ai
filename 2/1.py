import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
#https://blog.csdn.net/m0_38056893/article/details/105207945
 
 
base_dir = './dataset/'
train_dir = os.path.join(base_dir, 'train/')
validation_dir = os.path.join(base_dir, 'validation/')
 
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
 
 
# 设置预处理数据集和训练网络时要使用的变量。
 
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
 
num_cats_tr = len(os.listdir(train_cats_dir))  # total training cat images: 1000
num_dogs_tr = len(os.listdir(train_dogs_dir))  # total training dog images: 1000
 
num_cats_val = len(os.listdir(validation_cats_dir))  # total validation cat images: 500
num_dogs_val = len(os.listdir(validation_dogs_dir))  # total validation dog images: 500
 
total_train = num_cats_tr + num_dogs_tr  # Total training images: 2000
total_val = num_cats_val + num_dogs_val  # Total validation images: 1000
 
"""
数据准备
    将图像格式化成经过适当预处理的浮点张量，然后输入网络:
    - 从磁盘读取图像。
    - 解码这些图像的内容，并根据它们的RGB内容将其转换成适当的网格格式。
    - 把它们转换成浮点张量。
    - 将张量从0到255之间的值重新缩放到0到1之间的值，因为神经网络更喜欢处理小的输入值。
    幸运的是，所有这些任务都可以用tf.keras提供的ImageDataGenerator类来完成。
    它可以从磁盘读取图像，并将它们预处理成适当的张量。它还将设置发生器，将这些图像转换成一批张量——这对训练网络很有帮助。
"""
 
# 生成训练数据集和验证数据集
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
 
# 在为训练和验证图像定义生成器之后，flow_from_directory方法从磁盘加载图像，应用重新缩放，并将图像调整到所需的尺寸。
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
# output:Found 2000 images belonging to 2 classes.
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
# output:Found 1000 images belonging to 2 classes.
 
 
# 可视化训练图像：通过从训练生成器中提取一批图像(在本例中为32幅图像)来可视化训练图像，然后用matplotlib绘制其中五幅图像。
 
sample_training_images, _ = next(train_data_gen)
# next函数：从数据集中返回一个批处理。
# 返回值：(x_train，y_train)的形式，其中x_train是训练特征，y_train是其标签。丢弃标签，只显示训练图像。
 
 
# 该函数将图像绘制成1行5列的网格形式，图像放置在每一列中。
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
 
 
plotImages(sample_training_images[:5])
 
# 创建模型：该模型由三个卷积块组成，每个卷积块中有一个最大池层。有一个完全连接的层，上面有512个单元，由relu激活功能激活。
 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])
 
# 编译模型：这边选择ADAM优化器和二进制交叉熵损失函数。传递metrics参数查看每个训练时期的训练和验证准确性。
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
 
# 训练模型：使用ImageDataGenerator类的fit_generator方法来训练网络。
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
 
# 可视化训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
 
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs_range = range(epochs)
 
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
 
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

