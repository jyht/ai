#-*- coding: utf-8 -*-
#矩阵入库
import face_recognition as fr
import numpy as np
import pymysql


db = pymysql.connect("localhost","root","","baiduyun" )
cursor = db.cursor()
sql = "select * from face where tzm=0 order by id asc"
cursor.execute(sql)
results = cursor.fetchall()
for row in results:
	id = row[0]
	name = row[1]
	img = row[2]
	img_a = fr.load_image_file(img)
	#特征码转换矩阵
	try:
		encoding_a = fr.face_encodings(img_a,known_face_locations=None, num_jitters=0)[0] #0池化操作。数值越高，精度越高，但耗时越长
		
		encoding__array_list = encoding_a.tolist()
		# 将列表里的元素转化为字符串
		encoding_str_list = [str(i) for i in encoding__array_list]
		# 拼接列表里的字符串
		encoding_str = ','.join(encoding_str_list)
	except IndexError:
		print ("Error: 没有识别人脸")
		encoding_str = str(0)
	else:
		print ('ok')
	#为了入库 numpy.darray转换list
	ins = "update face set tzm='"+encoding_str+"' where id=" + str(id)
	cursor.execute(ins)
	db.commit()
	
db.close()		 



