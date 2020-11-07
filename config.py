items =['item1','item2']
json_dir = 'D:\\pycharm\\data\\clothes\\annos\\'
image_dir = 'D:\\pycharm\\data\\clothes\\image\\'
dataset_save_dir = 'D:\\pycharm\\data\\new_clothes\\'
new_label = 'D:\\pycharm\\data\\new_label\\'

label = ['short sleeve top','shorts','trousers','long sleeve top','unsure']
labels = {  0:'short sleeve top',
            1:'shorts',
            2:'trousers',
            3:'long sleeve top'}
save_path = 'chekpoints/'
# Save path for logs
logdir = 'logs/'
img_folder = ''

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 初始化训练参数
num_workers = 8  # Number of CPU processes for data preprocessing
lr = 1e-4  # Learning rate
batch_size = 32
save_freq = 1  # Save checkpoint frequency (epochs)
test_freq = 200  # Test model frequency (iterations)
max_epoch_number = 35  # Number of epochs for training
# Note: on the small subset of data overfitting happens after 30-35 epochs

if __name__ == '__main__':
    print('shorts' in labels.values())