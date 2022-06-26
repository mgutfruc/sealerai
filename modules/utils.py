from PIL import Image,ImageStat
import multiprocessing as mp
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

import torch

import torch.nn as nn
from  torchvision.io import read_image
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset

import io

transform_tensor = T.ToPILImage()
transform_pil = T.ToTensor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pandas_output_format():
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 400)
    pd.set_option('display.max_colwidth',100)

# calculations should work on small arrays and not dataframes to make code reusable when training and running
# to process a datafram chuck the call to array processing routine.

def load_model(yolo_location:str,weights:str):
    model = torch.hub.load(yolo_location, 'custom', path=weights , source='local') # load yolo models with 2 classes
    #model = model.cuda()
    return model

# 3. Store results in df:
def store_results(img_names,yolo_results):
    res = [x.cpu().numpy() for x in yolo_results]
    data_res = {'img': img_names,'yolo': res}
    df = pd.DataFrame(data_res)
    return df

# 2. Add yolo labels:
def batched_detect(img_names,model,pbar):
    "run inference in batches of 32 images"
    all_results=[] # leere Liste

    # Step 1: Init multiprocessing.Pool() to read jpegs

    #transform = T.ToPILImage()

    pool = mp.Pool(mp.cpu_count())

    for i in range(0, len(img_names), 32): # 32er steps
        name_sub_set=img_names[i:i+32] # nehme 32 imgs in sub_set
        #images=[ Image.open(f) for f in name_sub_set] # öffne imgs aus sub_set

        # Step 2: `pool.apply` the `howmany_within_range()`
        images = pool.map(Image.open,name_sub_set)

        #images = pool.map(read_image,name_sub_set)
        #images = [transform(x) for x in images]
    
        #results = self.model(images,size=384)
        results = model(images) # speicher Ergebnisse der Bilder aus sub_set
        all_results.extend(results.xywh) # Füge sub_sets in die Liste hinzu
        #all_images.extend(images)
        pbar.update(1)
    # Step 3: Don't forget to close
    pool.close()   
    return all_results # gib alle Ergebnisse aus

# 4. Filter images with corners:
def point_diff(a,b):
    if (a is None) or (b is None):
        return (None,None)
    a=a[0:2]
    b=b[0:2]
    return tuple(a-b)

def search_corner_circle_new(yolo):
    if len(yolo) == 0:
        return (None,None)
    corner_found=None
    corner_found_pj=0
    circle_found=None
    circle_found_pj=0
    for j in range(len(yolo)):
        _,yj,_,_,pj,cj=yolo[j]
        if cj == 1 and pj > 0.5 and pj > corner_found_pj:            
            corner_found=yolo[j]
            corner_found_pj=pj
        if cj == 0 and pj > 0.5 and pj > circle_found_pj:            
            circle_found=yolo[j]
            circle_found_pj=pj
    return (corner_found,circle_found)

def orientation_old(a,b):
    if (a is None) or (b is None):
        return None
    return a[0]>b[0]

def orientation(a):
    if a == 'caml1c1':
        return True
    if a == 'caml1c2':
        return False
    if a == 'caml2c2':
        return True
    if a == 'picamera':
        return False

def show_image(img,model):
    if isinstance(img, str):
         img =[img]
    images = [read_image(x) for x in img]
    results = model(images)
    return results

def euclidean_distance(a,b):
    if (a is None) or (b is None):
        return None
    a=a[0:2]
    b=b[0:2]
    return np.sqrt(np.sum(np.square(a-b)))

def section_line(p1, p2, m, n):
 
    (x1,y1) = p1
    (x2,y2) = p2 
    # Applying section formula
    x = (float)((n * x1)+(m * x2))/(m + n)
    y = (float)((n * y1)+(m * y2))/(m + n)
 
    # Printing result
    return (x, y)

def cut_insteresting_section(img_name,point,width,height,return_tensor=False):

    image = read_image(img_name)
    image_cropped = T.functional.crop(image,round(point[1]),round(point[0]),height,width) 
    if return_tensor:
        return transform_tensor(image_cropped)
    img_byte_arr = io.BytesIO()
    transform_tensor(image_cropped).save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def define_corner(point,orientation,euclidean_distance):
    if math.isnan(euclidean_distance):
        euclidean_distance=250
    h=23*euclidean_distance/250
    if orientation:
        return (round(point[0]-(55*euclidean_distance/250)-94),round(point[1]+h))
    else:
        return (round(point[0]+(55*euclidean_distance/250)),round(point[1]+h))

def define_corner_223(point,orientation,euclidean_distance):
    if orientation:
        return (round(point[0]-224),round(point[1]+10))
    else:
        return (round(point[0]),round(point[1]+10))

def convert_byte_to_image(img):
    f = io.BytesIO(img)
    im1=Image.open(f)
    return im1

def extract_mean_223(img):
    im1=convert_byte_to_image(img)
    stat = ImageStat.Stat(im1)
    r,g,b = stat.mean
    return (r,g,b)
    #
    #return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def extract_mean(point,orientation,image_name):
    try:
        y = round(point[1]+23)

        if orientation:
            x = round(point[0]+100)
        else:
            x = round(point[0]-100)
        h = 50
        w = 50 
        image = read_image(image_name)
        image_cropped = T.functional.crop(image,y,x,h,w)
        image_mean = torch.mean(image_cropped.type(torch.DoubleTensor)).item()
        return image_mean
    except:
        return None

def name_color(avg):
    if avg < 160:
        return "anthracite"
    if avg < 220:
        return "mid-gray"
    return "light-gray"

def image_brightness(avg):
    if avg < 120:
        return 0
    if avg < 140:
        return 1
    if avg < 160:
        return 2
    if avg < 180:
        return 3
    if avg < 180:
        return 4
    if avg < 200:
        return 5
    if avg < 220:
        return 6
    return 7

def split_df(df,ratio):
    mask = np.random.rand(len(df)) <= ratio
    training_data = df[mask]
    testing_data = df[~mask]
    return training_data, testing_data

class ConvAutoencoder(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        # N, 3, 100, 50
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, (4,2), stride=(4,2), ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(8, 16, 3, stride=2, ), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, ), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, ), 
        )
        
        # N 128 , 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, ), # N, 64, 10, 6
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, ), # N, 32, 23, 12
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, ), # N, 16, 49, 25
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, (4,2), stride=(4,2) ), # N, 3, 100, 50
            nn.Sigmoid()
        )
    
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConvAutoencoder_223(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        # N, 3, 100, 50
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, (3,3), stride=(2), ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(4, 8, 3, stride=2, ), 
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, ), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, ), 
        )
        
        # N 128 , 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, ), # N, 64, 10, 6
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, ), # N, 32, 23, 12
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, ), # N, 16, 49, 25
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, (3,3), stride=(2) ), # N, 3, 100, 50
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConvAutoencoder_old(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        # N, 3, 100, 50
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (4,2), stride=(4,2), ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, 3, stride=2, ), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, ), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, ), 
        )
        
        # N 128 , 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, ), # N, 64, 10, 6
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, ), # N, 32, 23, 12
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, ), # N, 16, 49, 25
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (4,2), stride=(4,2) ), # N, 3, 100, 50
            nn.Sigmoid()
        )
    
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ResnetModel(nn.Module):
    def __init__(self, weights, pretrained, no_classes):
        super(ResnetModel, self).__init__()

        self.model_ft = models.resnet18(pretrained=pretrained)

        num_features = self.model_ft.fc.in_features

        self.model_ft.fc = nn.Sequential(
            # how big?
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, no_classes),
            )

        if not pretrained:
            if weights:
                print('weights loaded')
                self.model_ft.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

    def forward(self, x):
        x = self.model_ft(x)
        return x

    def get_vis_hook_layer(self):
        return self.model_ft.layer4[1]

class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(16 * 13 * 13, 40)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        #print("xxx ",x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ClassifierNet_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(12 * 13 * 13, 20)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        #print("xxx ",x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DatasetFromPandas(Dataset):

    def __init__(self, df, variables):
        
        self.df = df
        self.variables = variables

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        rr = [ self.df.iloc[idx][x] for x in self.variables ]

        rr[0] 

        return rr

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            x[0] = self.transform(x[0])
        return x

    def __len__(self):
        return len(self.subset)

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """

    #if ymap is not None:
    #    y_pred = ymap.inverse_transform(y_pred)
    #    y_true = ymap.inverse_transform(y_true)
    #    labels = ymap.inverse_transform(labels)
    if isinstance(ymap, list):
        y_pred=[ymap[x] for x in y_pred]
        y_true=[ymap[x] for x in y_true]
        labels=[ymap[x] for x in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="YlGnBu")
    plt.show()