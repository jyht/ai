# -*- coding: utf-8 -*-
"""
@author: wangyao
"""
#改变默认工作路径
import os 


##爬取百度百科剧情
import urllib.request
from bs4 import BeautifulSoup
import re
import pandas as pd
url = "https://baike.baidu.com/item/%E4%BA%BA%E6%B0%91%E7%9A%84%E5%90%8D%E4%B9%89/17545218"

import sys
import importlib
importlib.reload(sys)

response = urllib.request.urlopen(url)
con = response.read()
#使用beautifulsoup中的html解析器
cont = BeautifulSoup(con,"html.parser")
content = cont.find_all('ul',{'id':'dramaSerialList'})
content = str(content)
##去掉HTML标签
content1 = re.sub(r'<[^>]+>','',content) 
f = open('rmdmy.txt','w',encoding= 'utf-8') #直接用open打开会报错，需要指定编码方式
f.write(content1)
f.close()

#爬取名字
f = open('rmdmy_name.txt','a',encoding= 'utf-8')
name_content = cont.find_all("dl",attrs={"class": "info"})
for i in name_content:
    name_d = i.get_text().strip().split(u'\n')[0]
    name = name_d.split(u'\xa0')[2]
    #加decode()byte和str才能相加
    f.write(name.encode('utf-8').decode()+'\n')

f.close()