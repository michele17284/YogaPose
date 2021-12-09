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

    def __init__(self, dataset_path, size=(256,192),transform=None):
        self.data_path = dataset_path
        # call to init the data
        self.size = size
        self.transform = transform


        
        self._init_data()


    def _init_data(self):
        positions = os.listdir(self.data_path)
        images = list()
        labels = list()
        for idx, directory_class in enumerate(os.listdir(DATASET_PATH)):
            class_path = os.path.join(DATASET_PATH,directory_class) 
            for file_name in os.listdir(class_path):
                f = cv2.imread(os.path.join(class_path, file_name),cv2.IMREAD_COLOR)
                if (self.transform != None): 
                    f = self.transform(f)
                data = torch.reshape(torch.FloatTensor(f).to(device),(3,self.size[0],self.size[1]))
                images.append((idx,data))
        #np.random.shuffle(images)
        self.images = images


    def __len__(self):
        # returns the number of samples in our dataset
        return len(self.images)

    def getData(self):
        return self.images

    def __getitem__(self, idx):
        # returns the idx-th sample
        return self.images[idx]

    def getOriginalImage(self, idx):
        return torch.reshape(self.images[idx][1],(self.size[0],self.size[1], 3))
    
    def collate_fn(self,data):
        Xs = torch.stack([x[1] for x in data])
        y = torch.stack([torch.tensor(x[0]) for x in data])
        return Xs,y


# get model from torch hub
model_name = "tpr_a4_256x192"
assert model_name in ["tpr_a4_256x192", "tph_a4_256x192"]
modelyaml = {"tph_a4_256x192": "models_yaml/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc4_mh1.yaml",
             "tpr_a4_256x192": "models_yaml/TP_R_256x192_d256_h1024_enc4_mh8.yaml"}

model = torch.hub.load('yangsenius/TransPose:main', model_name, pretrained=True, force_reload=True, verbose=2)
model.to(device)


norm_transform = T.Compose([T.ToTensor(),
                                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ])


dataset = YogaPoseDataset(DATASET_PATH, transform=norm_transform)
split_position = int((len(dataset)//10)*7)
trainset = dataset[:split_position]
testset = dataset[split_position:]

image1 = dataset[0][1]
image2 = torch.reshape(dataset.getOriginalImage(0),(3,256, 192))

print(image1 - image2)

#cv2.imwrite("./original_img.jpg", orimage)


def test(image):
    from TransPose.lib.config import cfg
    from TransPose.lib.core.inference import get_final_preds
    from TransPose.lib.utils import transforms, vis

    with torch.no_grad():
        model.eval()
        tmp = []
        tmp2 = []
        img = image

        inputs = torch.cat([img.to(device)]).unsqueeze(0)
        outputs = model(inputs)
        
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs
            
        if cfg.TEST.FLIP_TEST: 
            input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = transforms.flip_back(output_flipped.cpu().numpy(),
                                    dataset.flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            output = (output + output_flipped) * 0.5
            

        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), None, None, transform_back=False)

    # from heatmap_coord to original_image_coord
    query_locations = np.array([p * 4 + 0.5 for p in preds[0]])
    print(query_locations)

    from TransPose.visualize import inspect_atten_map_by_locations

    inspect_atten_map_by_locations(img, model, query_locations, model_name="transposer", mode='dependency', save_img=True,
                                threshold=0.0)


#test(dataset[0][1])