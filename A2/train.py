# coding:utf8

import dlib
import cv2
import numpy

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

from BNAlexNet import BNAlexNet

#global var
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_acc_arr = []
train_loss_arr = []
val_acc_arr = []
val_loss_arr = []

def image_to_line_train():
    print("A2 train dataset process")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
    path = r'Datasets/celeba/img'
    files = os.listdir(path)
    if not os.path.exists(r'Datasets/celeba/img_line'):
        os.makedirs(r'Datasets/celeba/img_line')
    subprocess = 0
    for file in files:
        if subprocess % 100 == 0:
            print('{:.2%}'.format(subprocess / len(files)))
        subprocess += 1
        image = cv2.imread(path + "/" + file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) == 1:
            face = faces[0]
            empty_image = numpy.zeros((224, 224, 1), numpy.uint8)
            shape = predictor(gray, face)
            parts = shape.parts()
            i = 0
            j = 1
            for pt in parts[:16]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[17:21]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[22:26]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[27:30]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[31:35]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[36:41]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            cv2.line(empty_image, (parts[36].x, parts[36].y), (parts[41].x, parts[41].y), 255)
            for pt in parts[42:47]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            cv2.line(empty_image, (parts[42].x, parts[42].y), (parts[47].x, parts[47].y), 255)
            for pt in parts[48:59]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            cv2.line(empty_image, (parts[48].x, parts[48].y), (parts[59].x, parts[59].y), 255)
            for pt in parts[60:67]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            cv2.line(empty_image, (parts[60].x, parts[60].y), (parts[67].x, parts[67].y), 255)
            
            cv2.imwrite(r'Datasets/celeba/img_line/' + file, empty_image)

    print('done')

def image_to_line_test():
    print("A2 test dataset process")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

    path = r'Datasets/celeba_test/img'
    files = os.listdir(path)
    if not os.path.exists(r'Datasets/celeba_test/img_line'):
        os.makedirs(r'/Datasets/celeba_test/img_line')
    subprocess = 0
    for file in files:
        if subprocess % 100 == 0:
            print('{:.2%}'.format(subprocess / len(files)))
        subprocess += 1
        image = cv2.imread(path + "/" + file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) == 1:
            face = faces[0]
            empty_image = numpy.zeros((224, 224, 1), numpy.uint8)
            shape = predictor(gray, face)
            parts = shape.parts()

            i = 0
            j = 1
            for pt in parts[:16]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[17:21]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[22:26]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[27:30]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[31:35]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            for pt in parts[36:41]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            cv2.line(empty_image, (parts[36].x, parts[36].y), (parts[41].x, parts[41].y), 255)
            for pt in parts[42:47]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            cv2.line(empty_image, (parts[42].x, parts[42].y), (parts[47].x, parts[47].y), 255)
            for pt in parts[48:59]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            i += 1
            j += 1
            cv2.line(empty_image, (parts[48].x, parts[48].y), (parts[59].x, parts[59].y), 255)
            for pt in parts[60:67]:
                cv2.line(empty_image, (parts[i].x, parts[i].y), (parts[j].x, parts[j].y), 255)
                i += 1
                j += 1
            cv2.line(empty_image, (parts[60].x, parts[60].y), (parts[67].x, parts[67].y), 255)
            
            cv2.imwrite(r'../Datasets/celeba_test/img_line/' + file, empty_image)
            
    print('done')


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
    dataframe = pd.read_csv(r'Datasets\celeba\labels.csv', sep='	')
    train_partial_rate = 0.8
    total_dataset = SmileDataset(imgs_dir=r'Datasets\celeba\img_line', dataframe=dataframe, transform=train_trans)

    train_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [int(len(total_dataset) * train_partial_rate), len(total_dataset) - int(len(total_dataset) * train_partial_rate)])
    dataset_sizes = {'train' : len(train_dataset), 'val' : len(val_dataset)}
    print(dataset_sizes)

    dataloaders = {'train' : torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=0),
                  'val' : torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True,num_workers=0)}

    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    # stop flag
    stop = False
    since = time.time()
    epoch_start = time.time()
    epoch_end = time.time()
    
    # Starting epochs
    for epoch in range(num_epochs):
        epoch_arr.append(epoch)
        print('Epoch {}/{} '.format(epoch, num_epochs - 1))
        epoch_start = time.time()

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
                if len(val_loss_arr) > 0 and epoch_loss < min(val_loss_arr):
                    best_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                if len(val_loss_arr) > 10 and epoch_loss > (val_loss_arr[-1] + val_loss_arr[-2] + val_loss_arr[-3]) / 3: # called Mean stop in report
                    stop = True
                val_loss_arr.append(epoch_loss)
                val_acc_arr.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
           
        scheduler.step()
        epoch_end = time.time()
        print('Epoch cost {}'.format(epoch_end - epoch_start))
        if stop == True:
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {}'.format(best_epoch))
    print('Best train Acc: {:4f}'.format(train_acc_arr[best_epoch] if stop else train_acc_arr[-1]))
    print('Best val Acc: {:4f}'.format(val_acc_arr[best_epoch] if stop else val_acc_arr[-1]))

    # load best model weights
    model.load_state_dict(best_weights)
    return model


def draw():
    epoch_arr = range(30)
    plt.figure(1, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.xlabel("num of epoch")
    plt.ylabel("value of loss")
    plt.plot(epoch_arr, train_loss_arr, color='red', label='loss')
    plt.plot(epoch_arr, val_loss_arr, color='blue', label='loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("accruacy")
    train_acc_arr_float = []
    val_acc_arr_float = []

    for dtype in train_acc_arr:
        train_acc_arr_float.append(dtype.item())
    for dtype in val_acc_arr:
        val_acc_arr_float.append(dtype.item())
    plt.plot(epoch_arr, train_acc_arr_float, color='red', label='accruacy')
    plt.plot(epoch_arr, val_acc_arr_float, color='blue', label='accruacy')
    plt.legend()
    plt.show()

    
def run():
    image_to_line_train()
    image_to_line_test()
    # Use BNAlexnet for training
    model_conv = BNAlexNet(num_classes=2)
    # Check net
    print(model_conv)
    global device
    model_conv.to(device)
    # Define the loss function 
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    #Define the optimizer (Adam for 2 classes)
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.00002, betas=(0.9, 0.99))
    #Define scheduler
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)
    
    model_train = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=30)
    
    torch.save(model_train, 'a2model.pkl')
    
    draw()




