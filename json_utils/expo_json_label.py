import os
import json
from config import *
'''
    短袖：short sleeve top
    短裤：shorts
    长裤：trousers
    长袖：long sleeve top

'''
sst,s,t,lst = 0,0,0,0
json_list = os.listdir(json_dir)
for file in json_list:
    f = open(json_dir+file, 'r')
    content = f.read()
    data = json.loads(content)
    for item in items:
        if item in data.keys():
            if data[item]['category_name'] == 'short sleeve top':
                sst += 1
            elif data[item]['category_name'] == 'shorts':
                s += 1
            elif data[item]['category_name'] == 'trousers':
                t += 1
            elif data[item]['category_name'] == 'long sleeve top':
                lst += 1
    f.close()
print('short sleeve top :',sst)
print('shorts :',s)
print('trousers :',t)
print('long sleeve top :',lst)

'''
output1:
short sleeve top : 70962
shorts : 36128
trousers : 53725
long sleeve top : 35430

'''