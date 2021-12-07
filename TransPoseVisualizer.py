import torchvision.transforms as T
import matplotlib.pyplot as plt
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
import torch
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
DATASET_PATH = './dataset/'
positions = os.listdir(DATASET_PATH)
images = list()

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

class YogaPoseDataset(Dataset):

    def __init__(self, dataset_path):
        self.data_path = dataset_path
        # call to init the data
        self.size = 256,192
        self._init_data()

    def _init_data(self):
        positions = os.listdir(self.data_path)
        images = list()
        labels = list()
        for idx, position in enumerate(positions):
            for file in os.listdir(DATASET_PATH + position):
                if not file.endswith('.gif'):
                    f = cv2.imread(DATASET_PATH + position + '/' + file,cv2.IMREAD_COLOR)

                    #print(position + '/' + file)
                    #print(f.shape)
                    height,width, channels = f.shape
                    if height > width:
                        scale_percent = 256/height
                        f = cv2.resize(f,(int(width*scale_percent),256))
                        if f.shape[1] > 192:
                            scale_percent = 192/width
                            f = cv2.resize(f,(192,int(height*scale_percent)))
                    else:
                        scale_percent = 192/width
                        f = cv2.resize(f,(192,int(height*scale_percent)))
                        if f.shape[0] > 256:
                            scale_percent = 256/height
                            f = cv2.resize(f,(int(width*scale_percent),256))

                    height,width, channels = f.shape
                    #print(f.shape,"after resize")
                    f = cv2.copyMakeBorder(f, 256-height, 0, 192-width, 0, cv2.BORDER_CONSTANT,value=0)
                    #print(f.shape,"after padding")
                    #if torch.FloatTensor(f.getdata()).size()[1] != 3: f.show()
                    data = torch.reshape(torch.FloatTensor(f).to(device),(3,256,192))
                    #print(data.shape)
                    #print(data.size())
                    images.append(data)
                    labels.append(torch.tensor(idx))


        #print(images)
        images = torch.stack(images)
        labels = torch.stack(labels)
        #print(labels)
        print(images.size())
        mask = np.arange(labels.size()[0])
        np.random.shuffle(mask)
        #print(type(images),type(images[0]))
        self.images = images[mask]
        self.labels = labels[mask]

    def __len__(self):
        # returns the number of samples in our dataset
        return len(self.images)


    def getX(self):
        return self.images

    def getY(self):
        return self.labels

    def __getitem__(self, idx):
        # returns the idx-th sample
        return self.images[idx],self.labels[idx]

# Custom implementation based on TransPose, not yet implemented nor utilized
class MyTranspose(nn.Module):
    def __init__(self):
        super(MyTranspose, self).__init__()

        self.tph = torch.hub.load('yangsenius/TransPose:main', 'tph_a4_256x192', pretrained=True, device=device)

    def forward(self, x):
        x = x.to(device)
        out = self.tph(x)
        return out

# get model from torch hub
model_name = "tpr_a4_256x192"
assert model_name in ["tpr_a4_256x192", "tph_a4_256x192"]
modelyaml = {"tph_a4_256x192": "models_yaml/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc4_mh1.yaml",
             "tpr_a4_256x192": "models_yaml/TP_R_256x192_d256_h1024_enc4_mh8.yaml"}

model = torch.hub.load('yangsenius/TransPose:main', model_name, pretrained=True, force_reload=True, verbose=2)
model.to(device)
# print("model params:{:.3f}M".format(sum([p.numel() for p in model.parameters()])/1000**2))
dataset = YogaPoseDataset(DATASET_PATH)
split_position = int((len(dataset)/10)*7)
trainset = dataset[:split_position]
testset = dataset[split_position:]
print(trainset)

batch = dataset.getX()[:32].to(device)
print(batch.size(),dataset.getX().size())

model(batch)