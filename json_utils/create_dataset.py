import os
import json
from config import *
'''
clean label
'''

def run():
    json_list = os.listdir(json_dir)
    for file in json_list:
        f = open(json_dir+file, 'r')
        content = f.read()
        data = json.loads(content)
        for item in items:
            if item in data.keys():
                if data[item]['category_name'] not in labels.values():
                    data[item]['category_name'] = 'unsure'
        dic = data
        f.close()
        with open(dataset_save_dir+file , 'w') as r:
            # 定义为写模式，名称定义为r
            json.dump(dic, r)
            # 将dict写入名称为r的文件中
        r.close()

