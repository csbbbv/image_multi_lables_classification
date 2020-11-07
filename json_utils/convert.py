import json
import os
from config import *
json_dir = 'D:\\pycharm\\data\\test\\anno'
json_files = os.listdir(json_dir)

with open('test.json', 'w') as fp:
    fp.write('{"samples":[')
    for file in json_files:
        fp.write('{"image_name":'+'"'+file[0:file.index('.')]+'.jpg","image_labels":[')
        j_file = os.path.join(json_dir,file)
        write = 0
        with open(j_file, 'r') as load_f:
            obj = json.load(load_f)
            if 'item1' in obj and 'item2' in obj:
                if obj['item1']['category_name'] == 'unsure' and obj['item2']['category_name'] == 'unsure':
                    fp.write('"' + obj['item1']['category_name'] + '"')
                elif obj['item1']['category_name'] != 'unsure' and obj['item2']['category_name'] != 'unsure':
                    fp.write('"' + obj['item1']['category_name'] + '"')
                    fp.write(',"' + obj['item2']['category_name'] + '"')
                else:
                    if obj['item1']['category_name'] != 'unsure':
                        fp.write('"' + obj['item1']['category_name'] + '"')
                    if obj['item2']['category_name'] != 'unsure':
                        fp.write('"' + obj['item2']['category_name'] + '"')
            else:
                if 'item1' in obj:
                    fp.write('"' + obj['item1']['category_name'] + '"')
                elif 'item2' in obj:
                    fp.write('"' + obj['item2']['category_name'] + '"')
        # if 'item1' in obj:
        #     if not write:
        #         write=1
        #         fp.write('"'+obj['item1']['category_name']+'"')
        #     else:
        #         fp.write(',"'+obj['item1']['category_name']+'"')
        # if 'item2' in obj:
        #     if not write:
        #         write=1
        #         fp.write('"'+obj['item2']['category_name']+'"')
        #     else:
        #         fp.write(',"'+obj['item2']['category_name']+'"')
        fp.write(']},')
    fp.write(']}')
            

