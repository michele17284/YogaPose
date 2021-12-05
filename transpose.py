import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
device = torch.device('cuda')
#tpr = torch.hub.load('yangsenius/TransPose:main', 'tpr_a4_256x192', pretrained=True)
tph = torch.hub.load('yangsenius/TransPose:main', 'tph_a4_256x192', pretrained=True, device=device)


#print(tph)
DATASET_PATH = './dataset/'
positions = os.listdir(DATASET_PATH)
#print(os.listdir(DATASET_PATH+entries[0]))
images = list()



class YogaPoseDataset(Dataset):

    def __init__(self, dataset_path):
        self.data_path = dataset_path
        # call to init the data

        self._init_data()

    def _init_data(self):
        positions = os.listdir(self.data_path)
        images = list()
        for idx, position in enumerate(positions):
            for file in os.listdir(DATASET_PATH + position):
                with Image.open(DATASET_PATH + position + '/' + file, "r") as f:
                    # f.show()
                    data = np.asarray(f)
                    images.append((idx, data))
                    # print(data)
        self.images = np.array(images)
        np.random.shuffle(self.images)

    def __len__(self):
        # returns the number of samples in our dataset
        return len(self.images)

    def __getitem__(self, idx):
        # returns the idx-th sample
        return self.images[idx]

dataset = YogaPoseDataset(DATASET_PATH)
split_position = int((len(dataset)/10)*7)
trainset = dataset[:split_position]
testset = dataset[split_position:]




class PoseClassifier(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(PoseClassifier, self).__init__()

        self.tph = torch.hub.load('yangsenius/TransPose:main', 'tph_a4_256x192', pretrained=True, device=device)
        self.fc1 = nn.Linear(1000,1000).to(device)
        self.fc2 = nn.Linear(1000,n_class).to(device)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.to(device)
        out = self.tph(x)
        out = self.fc1(out)
        #m = nn.BatchNorm1d(1000,device=device)
        #out = m(out)
        out = self.relu(out)
        out = self.fc2(out)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out

X = np.array(dataset[:,1],dtype=float)
#X = X.astype(float)
out = tph(torch.tensor(X))
