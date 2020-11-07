import json
import os
json_dir = 'D:\\pycharm\\data\\train\\anno'
json_files = os.listdir(json_dir)

with open('train.json', 'w') as fp:
    fp.write('{"samples":[')
    for file in json_files:
        fp.write('{"image_name":'+'"'+file[0:file.index('.')]+'.jpg","image_labels":[')
        j_file = os.path.join(json_dir,file)
        write = 0
        with open(j_file, 'r') as load_f:
            obj = json.load(load_f)
        if 'item1' in obj:
            if not write:
                write=1
                fp.write('"'+obj['item1']['category_name']+'"')
            else:
                fp.write(',"'+obj['item1']['category_name']+'"')
        if 'item2' in obj:
            if not write:
                write=1
                fp.write('"'+obj['item2']['category_name']+'"')
            else:
                fp.write(',"'+obj['item2']['category_name']+'"')
        fp.write(']},')
    fp.write(']}')
            

