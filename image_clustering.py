# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
from tqdm import tqdm
       

"""
https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
"""
            
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

data_transform  = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])



class ResNet18_Backbone(nn.Module):
    ''' n = no of output classes'''

    def __init__(self, n=3):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.dims = model.fc.in_features
        # Remove the softmax layer and the pooling layer
        self.bottom = nn.Sequential(*list(model.children())[:-2])
        # Initialize a new pooling layer and a new softmax layer
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=self.dims, out_features=n)

    def extract_feats(self, imgs):
        out = self.bottom(imgs)
        out = self.pool(out)
        out = out.view(-1, self.dims)
        return out

    def forward(self, imgs):
        out = self.extract_feats(imgs)
        return self.fc(out)


def extract_features(file, model):
    model.eval()
    with torch.no_grad():
        img = Image.open(file).convert("RGB")
        image = data_transform(img).unsqueeze(0)
        features = model.extract_feats(image)
        return features


model = ResNet18_Backbone()

#%%
from pathlib import Path
fpath = Path('flower_dataset')

data = []
filenames = []
# lop through each image in the dataset
for flower in tqdm(fpath.joinpath('flower_images').iterdir()):
    if str(flower).endswith(".png"):
        fname = flower.parts[-1] #last part
        feat = extract_features(flower,model)
        data.append(feat.detach().numpy())
        filenames.append(str(flower))
# reshape so that there are 210 samples of d vectors
feat =  np.concatenate(data)

#%%
# get the unique labels (from the flower_labels.csv)
df = pd.read_csv('flower_dataset/flower_images/flower_labels.csv')
label = df['label'].tolist()
unique_labels = list(set(label))

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=len(unique_labels),n_jobs=-1, random_state=22)
kmeans.fit(x)

from collections import defaultdict
# holds the cluster id and the images { id: [images] }
groups = defaultdict(list)
for file, cluster in zip(filenames,kmeans.labels_):
        groups[cluster].append(file)

# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:30]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = Image.open(file).convert("RGB")
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
   
# this is just incase you want to see which value for k might be the best 
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22, n_jobs=-1)
    km.fit(x)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')

#%%
for i in groups:
    print ("group ", i,len(groups[i]))
    
    view_cluster(i)
        