import shutil
import glob,os
def move_file(file_list,old_dir,new_dir,format='.jpg'):
    for i in file_list:
        file_name = i.split('.')[0]
        shutil.move(old_dir+file_name+format, new_dir)
if __name__=='__main__':
    test = '/home/forest/workspace/image_multi_lables_classification/clothes/test/image/'
    test_list =os.listdir(test)
    for file in test_list:
        shutil.move(test+file, '/home/forest/workspace/image_multi_lables_classification/clothes/train/image/')
    print('ok')