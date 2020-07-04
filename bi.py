#-*- coding: utf-8 -*-
#list 出库并且矩阵对比
import face_recognition as fr
import numpy as np
import pymysql


db = pymysql.connect("localhost","root","","baiduyun" )
cursor = db.cursor()
sql = "select * from face order by id asc"
cursor.execute(sql)
results = cursor.fetchall()

img_a = fr.load_image_file("1.jpeg")
encoding_a = fr.face_encodings(img_a)[0]

mao = []
for row in results:
	id = row[0]
	names = row[1]
	tzm_list = row[3]
	# 将字符串转为numpy ndarray类型，即矩阵
	# 转换成一个list
	dlist = tzm_list.strip(' ').split(',')
	# 将list中str转换为float
	dfloat = list(map(float, dlist))
	arr = np.array(dfloat) #arr 数据库里出来的矩阵
	ad = fr.face_distance([arr], encoding_a)
	#print("aaa %f" % (ad))
	mao.append([ad[0],names])
	
db.close()
#list = [(1,93),(2,71),(3,89),(4,93),(5,85),(6,77)] 
mao.sort(key=lambda x:x[0]) 
#print (mao)
print(mao[0])
print(mao[1])
print(mao[2])
print(mao[3])
print(mao[4])
