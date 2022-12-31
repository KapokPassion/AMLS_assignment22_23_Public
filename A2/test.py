# coding:utf8
import os
from torchvision import transforms as T
from PIL import Image
from torch.utils import data
import pandas as pd

import visdom
from torchvision.utils import make_grid
import torch

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from BNAlexNet import BNAlexNet

#global var
results = []
confusion_matrix = np.zeros([2, 2], dtype=int)

class SmileDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, dataframe, transform=None):
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.img = dataframe["img_name"].tolist()
        self.label = dataframe['smiling'].map(lambda x: (x + 1) // 2).tolist() # 1 for smiling, 0 for not
        index = len(self.img) - 1
        for img_name in self.img[::-1]:
            if not os.path.exists(os.path.join(self.imgs_dir, img_name)):
                self.img.pop(index)
                self.label.pop(index)
            index -= 1
        
    def __getitem__(self, idx):
        img_name = self.img[idx]
        img_path = os.path.join(self.imgs_dir, img_name)
        img = Image.open(img_path)
        img = img.convert("RGB")
#         img = np.array(img)
        label = self.label[idx]
        
        if self.transform:
            img = self.transform(img) # transform needs a PIL img
        return img, label
    
    def __len__(self):
        return len(self.img)


# CPU
def visualize(data, label, pred):
    viz = visdom.Visdom(env='main')
    # print(data.size()) # torch.Size([4, 3, 224, 224])
    out = make_grid(data) # 4D to 3D
    # print(out.size()) # torch.Size([3, 228, 906])
    inp = torch.transpose(out, 0, 2)
    # print(inp.size()) # torch.Size([906, 228, 3])
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = torch.transpose(inp, 0, 2)
    # print(inp.size()) # torch.Size([3, 228, 906])
    viz.images(inp, opts=dict(title='label:{} pred:{}'.format(label.item(), pred.item())))


def self_dataset():
    device = torch.device('cpu')
    global confusion_matrix
    model_test = torch.load(r'a2model.pkl', map_location='cpu')
    model_test.eval()

    trans = T.Compose([
                T.Resize([224, 224]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                ])
    
    dataframe = pd.read_csv(r'Datasets\celeba_test\labels.csv', sep='	')
    test_dataset = SmileDataset(imgs_dir=r'Datasets\celeba_test\img_line', dataframe=dataframe, transform=trans)
    dataloaders = data.DataLoader(test_dataset, batch_size=8 ,shuffle=True, num_workers=0)

    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        outputs = model_test(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(labels)):
            results.append([labels[i], preds[i]])
            confusion_matrix[labels[i].item()][preds[i].item()] += 1
#             if labels[i].item() != preds[i].item():
#                 visualize(inputs[i], labels[i], preds[i]) # "python -m visdom.server" to start visdom server

def run():
    self_dataset()
    correct = 0
    fail = 0
    for result in results:
        if(result[0] == result[1]):
            correct += 1
        else:
            fail += 1

    print("Acc: {:.4f}".format(correct / (correct + fail)))
    
    draw()
    

def draw():
    global confusion_matrix
    config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
    "axes.unicode_minus": False,
    "xtick.direction":'out',
    "ytick.direction":'out',
    }
    rcParams.update(config)

    classes = ['0','1']

    proportion = []
    for i in confusion_matrix:
        for j in i:
            temp=j/(np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt="%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])
    pshow = np.array(pshow).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])

    plt.figure(figsize=(5,3))
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar().ax.tick_params(labelsize=10)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    ax = plt.gca()

    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)  

    iters = np.reshape([[[i,j] for j in range(2)] for i in range(2)],(confusion_matrix.size,2))
    for i, j in iters:
        if(i==j):

            plt.text(j, i-0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white', weight=5)
            plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i-0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)
            plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('Labels', fontsize=12)
    plt.xlabel('pred', fontsize=12)
    ax = plt.gca()

    ax.xaxis.set_label_position('top')   
    plt.tight_layout()

    plt.show()
    