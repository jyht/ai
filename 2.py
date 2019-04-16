#-*- coding: utf-8 -*-

import face_recognition as fr
# 读取图像
img_a = fr.load_image_file("tmp/1.jpg")
img_b = fr.load_image_file('tmp/2.jpg')
img_c = fr.load_image_file('tmp/3.jpg')
img_d = fr.load_image_file('tmp/4.jpg')
img_x = fr.load_image_file('tmp/0.jpg')
# 进行特征编码
encoding_a = fr.face_encodings(img_a)[0]
encoding_b = fr.face_encodings(img_b)[0]
encoding_c = fr.face_encodings(img_c)[0]
encoding_d = fr.face_encodings(img_d)[0]
encoding_x = fr.face_encodings(img_x)[0]
# 将示例图片与目标人脸逐一对比，返回 list, 包含True/False, 表示是否匹配
fr.compare_faces([encoding_a, encoding_b, encoding_c, encoding_d], encoding_x)
# 返回 [True, True, True, False]
# 我擦，不对啊，怎么第2，3张也是True。

# 差值默认为0.5，越小对比越严格
fr.compare_faces([encoding_a, encoding_b, encoding_c, encoding_d], encoding_x, tolerance=0.2)
# 返回值[True, False, False, False]，得知目标人脸属于范爷

ad = fr.face_distance([encoding_a, encoding_b, encoding_c, encoding_d], encoding_x)
print(ad)
