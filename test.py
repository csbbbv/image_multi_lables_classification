# Run inference on the test data
from torchvision import models
from PIL import Image
from matplotlib import pyplot as plt
import  numpy as np
model =

model.eval()
for sample_id in [1,2,3,4,6]:
    test_img, test_labels = test_dataset[sample_id]
    test_img_path = os.path.join(img_folder, test_dataset.imgs[sample_id])
    with torch.no_grad():
        raw_pred = model(test_img.unsqueeze(0)).cpu().numpy()[0]
        raw_pred = np.array(raw_pred > 0.5, dtype=float)

    predicted_labels = np.array(dataset_val.classes)[np.argwhere(raw_pred > 0)[:, 0]]
    if not len(predicted_labels):
        predicted_labels = ['no predictions']
    img_labels = np.array(dataset_val.classes)[np.argwhere(test_labels > 0)[:, 0]]
    plt.imshow(Image.open(test_img_path))
    plt.title("Predicted labels: {} \nGT labels: {}".format(', '.join(predicted_labels), ', '.join(img_labels)))
    plt.axis('off')
    plt.show()