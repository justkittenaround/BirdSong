#https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d

import copy
import random
import os, sys
import numpy as np
from PIL import Image
from skimage import data, color
import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


DATA = 'Cropped_birdsong_2_images_per_song/'
EPOCHS = 5000






trill = []
whistle = []
for f in os.listdir(DATA):
    if 'segment1' in f:
        im = Image.open(DATA+f)
        trill.append(im)
    if 'segment2' in f:
        im = Image.open(DATA+f)
        whistle.append(im)

shapesr = []
shapesc = []
for i, s in enumerate(trill):
    shapesr.append(s.size[0])
    shapesc.append(s.size[1])
    shapesr.append(whistle[i].size[0])
    shapesc.append(whistle[i].size[1])

print(min(shapesr), max(shapesr), np.average(shapesr))
print(min(shapesc), max(shapesc), np.average(shapesc))

width = max(shapesr)
height = max(shapesc)



c = np.ones([(47*2), 1, height, width])
clight = np.ones([(47*2), 1, height, width])
cdark = np.ones([(47*2), 1, height, width])
for idx in range(len(cdark)):
    cdark[idx,...] = 0.3

for idx in range(len(clight)):
    clight[idx,...] = 0.7

c = torch.FloatTensor(c)
cdark = torch.FloatTensor(cdark)
clight = torch.FloatTensor(clight)






trill = []
whistle = []
for f in os.listdir(DATA):
    if 'segment1' in f:
        im = Image.open(DATA+f)
        trill.append(im.resize((width, height), Image.ANTIALIAS))
    if 'segment2' in f:
        im = Image.open(DATA+f)
        whistle.append(im.resize((width, height), Image.ANTIALIAS))

print('trill samples:', len(trill), 'whistle samples:', len(whistle))


data = []
for idx in range(len(trill)):
    # for i in np.arange(20,95,5):
    p = (99*np.asarray(trill[idx]).shape[1])//100
    a = np.asarray(trill[idx])[:, :p, :]
    p = (99*np.asarray(whistle[idx]).shape[1])//100
    b = np.asarray(whistle[idx])[:, -p:, :]
    data.append((a,b))

mrow = []
mcols = []
for k, v in data:
    mrow.append(k.shape[0])
    mcols.append(k.shape[1])
    mrow.append(v.shape[0])
    mcols.append(v.shape[1])

r = max(mrow)
c = max(mcols)

d = []
for samp in data:
    k,v = samp
    k = color.rgb2gray(k)
    k = resize(k, (r,c))
    v = color.rgb2gray(v)
    v = resize(v, (r,c))
    d.append((k,v))


# tens = transforms.ToTensor()
# d = {}
# for k, v in data:
#     krad = np.zeros([r-k.shape[0], k.shape[1], 3])
#     kcad = np.zeros([r, c-k.shape[1], 3])
#     plt.imshow(k)
#     plt.show()
#     k = np.append(k, krad, axis=0)
#     k = np.append(k, kcad, axis=1)
#     plt.imshow(k)
#     plt.show()
#     vrad = np.zeros([r-v.shape[0], v.shape[1], 3])
#     vcad = np.zeros([r, c-v.shape[1], 3])
#     v = np.append(v, vrad, axis=0)
#     v = np.append(v, vcad, axis=1)
#     d.update({tens(k): tens(v)})



random.shuffle(d)


data = 0
trill = 0
whistle = 0

validx = np.random.choice( len(d), int( len(d)*.2 ) )
train = [d[i] for i in range(len(d)) if i not in validx]
val = [d[i] for i in validx]
data = {'train':train, 'val':val}
print('Train Size:', len(train), 'Val Size:', len(val))

# validx = np.random.choice( len(d.keys()), int( len(d.keys())*.2 ) )
# train = [list(d.items())[i] for i in range(len(d.keys())) if i not in validx]
# val = [list(d.items())[i] for i in validx]
# data = {'train':train, 'val':val}
# print('Train Size:', len(train), 'Val Size:', len(val))

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc = nn.Linear(31744, 4096)
        self.pool = nn.MaxPool2d(3)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.linear = nn.Linear(4096, 1)
        self.flat = nn.Flatten(start_dim=1)
    def forward(self, x):
        out = self.pool(self.relu(self.conv(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = self.pool(self.relu(self.conv3(out)))
        out = self.relu(self.conv4(out))
        out = self.flat(out)
        out = self.fc(out)
        out = self.sig(out)
        return out
    def predict(self, L):
        pred = self.sig(self.linear(L))
        return pred

model = Siamese()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00006)
criterion = nn.BCELoss()

best_acc = 0
for epoch in range(EPOCHS):
    print('Epoch', epoch)
    for phase in ['train','val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        n_correct = 0
        # for i, samp in enumerate(data[phase]):
        for i in range(len(c)):
            # k, v = samp
            # #if i < len(data[phase])-20:
            #  #   idx = [idx for idx in np.random.choice(np.arange(i+20, len(data[phase])), 1, replace=False) if idx != i]
            # #elif i >= len(data[phase])-20:
            #  #   idx = [idx for idx in np.random.choice(i-20, 1, replace=False) if idx != i]
            # a = np.arange(0, len(data[phase]))
            # a = a[a!=i]
            # idx = [idx for idx in np.random.choice(a, 1, replace=False) if idx != i]
            # randomwhistle = torch.from_numpy(data[phase][idx[0]][1]).to(torch.FloatTensor()).unsqueeze(0).unsqueeze(0)
            # xtrill = torch.from_numpy(k).to(torch.FloatTensor()).unsqueeze(0).unsqueeze(0).to(device)
            # intrill = torch.cat((xtrill.to(device), xtrill.to(device)), axis=0)
            inc = (torch.cat((c[:1], c[:1]), axis=0)).to(device)
            # xwhistle = torch.from_numpy(v).to(torch.FloatTensor()).unsqueeze(0).unsqueeze(0)
            # inwhistle = torch.cat((randomwhistle.to(device), xwhistle.to(device)), axis=0)
            inw = (torch.cat((cdark[:1], clight[:1]), axis=0)).to(device)
            # target = torch.cat((torch.FloatTensor([[0]]).to(device), torch.FloatTensor([[1]]).to(device)), axis=0)
            target = torch.cat((torch.FloatTensor([[0]]).to(device), torch.FloatTensor([[1]]).to(device)), axis=0)
            with torch.set_grad_enabled(phase == 'train'):
                # outtrill = model(intrill)
                # outwhistle = model(inwhistle)
                outtrill = model(inc)
                outwhistle = model(inw)
                L1 = torch.abs(outtrill- outwhistle)
                prediction = model.predict(L1)
                preds = torch.round(prediction)
                loss = criterion(prediction, target)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            if preds[0,] == target[0,]:
                n_correct += 1
            if preds[1,] == target[1,]:
                n_correct += 1
        if i > 0:
            print(phase, (n_correct/(i*2)) )
    if phase == 'val' and n_correct > best_acc:
        best_acc = n_correct/(i)
    best_model_wts = copy.deepcopy(model.state_dict())


# model.load_state_dict(best_model_wts)
torch.save(model, ('siamese_sliding'+str(best_acc)))
