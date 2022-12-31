# coding:utf8
from torchvision import datasets, models
from torch import nn, optim
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt

import os
import copy
import time
import torch

from PIL import Image
import pandas as pd

from BAlexNet import BAlexNet


#global var
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_acc_arr = []
train_loss_arr = []
val_acc_arr = []
val_loss_arr = []

class EyeDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, dataframe, transform=None):
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.img = dataframe["file_name"].tolist()
        self.label = dataframe['eye_color'].tolist()
        
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


# Training
# Param:
# model: model to train
# criterion：criterion funcion
# optimizer：optimizer
# scheduler：scheduler for learning rate
# num_epochs：number of epochs

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    global device
    #Dataset
    # image format parse
    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    train_trans = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    val_trans = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])

    # Load to customized Dataset
    dataframe = pd.read_csv(r'Datasets\cartoon_set\labels.csv', sep='	')
    train_partial_rate = 0.8
    total_dataset = EyeDataset(imgs_dir=r'Datasets\cartoon_set\img', dataframe=dataframe, transform=train_trans)
    train_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [int(len(total_dataset) * train_partial_rate), len(total_dataset) - int(len(total_dataset) * train_partial_rate)])
    dataset_sizes = {'train' : len(train_dataset), 'val' : len(val_dataset)}
    print(dataset_sizes)

    dataloaders = {'train' : torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=0),
                  'val' : torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True,num_workers=0)}

    since = time.time()
    epoch_start = time.time()
    epoch_end = time.time()
    # Save the optimal weight and the corresponding data
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_iteration = 0
    
    # Starting epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{} '.format(epoch, num_epochs - 1))
        epoch_start = time.time()
        # temp train acc before val acc
        temp = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
#                 scheduler.step()
                model.train()
            else:
            # Turn off some neurons
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # autograd
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                    # backpropagation
                        loss.backward()
                        # optimize
                        optimizer.step()
                        
                # loss
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_arr.append(epoch_loss)
                train_acc_arr.append(epoch_acc)
            if phase =='val':
                val_loss_arr.append(epoch_loss)
                val_acc_arr.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model      
            if phase == 'train' and epoch_acc > best_train_acc:
                temp = epoch_acc
            if phase =='val' and epoch_acc > 0 and epoch_acc < temp:
                best_train_acc = temp
                best_val_acc = epoch_acc
                best_iteration = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
        epoch_end = time.time()
        print('Epoch cost {}'.format(epoch_end - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {}'.format(best_iteration))  
    print('Best train Acc: {:4f}'.format(best_train_acc)) 
    print('Best val Acc: {:4f}'.format(best_val_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def draw():
    plt.figure(1, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.xlabel("num of epoch")
    plt.ylabel("value of loss")
    plt.plot(epoch_arr, train_loss_arr, color='red', label='train')
    plt.plot(epoch_arr, val_loss_arr, color='blue', label='val')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("accruacy")
    train_acc_arr_float = []
    val_acc_arr_float = []

    for dtype in train_acc_arr:
        train_acc_arr_float.append(dtype.item())
    for dtype in val_acc_arr:
        val_acc_arr_float.append(dtype.item())
    plt.plot(epoch_arr, train_acc_arr_float, color='red', label='train')
    plt.plot(epoch_arr, val_acc_arr_float, color='blue', label='val')
    plt.legend()
    plt.show()
    
    
def run():
    # Use BAlexnet for training
    model_conv = BAlexNet(num_classes=5)
    # Check net
    print(model_conv)
    global device
    model_conv.to(device)
    # Define the loss function 
    criterion = nn.CrossEntropyLoss()
    #Define the optimizer (Adam for 2 classes)
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.0001, betas=(0.9, 0.99))
    #Define scheduler
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)
    
    model_train = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=30)
    
    torch.save(model_train, 'b2model.pkl')
    
    draw()
