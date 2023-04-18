#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np

from datetime import datetime, timedelta
import time

import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential
from keras import metrics
from keras.models import model_from_json

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import cv2

import pandas as pd


# In[2]:


#VGG-Face for face recognition: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    return vgg_face_descriptor


# In[3]:


model = loadVggFaceModel()


# In[4]:


from keras.models import model_from_json
model.load_weights('C:/Users/Computer/Documents/FacialRecNN/vgg_face_weights/vgg_face_weights.h5')


# In[5]:


#open-cv's face detection module
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[6]:


mat = scipy.io.loadmat('C:/Users/Computer/Documents/FacialRecNN/imdb_files/imdb')


# In[7]:


columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score", "celeb_names", "celeb_id"]


# In[8]:


instances = mat['imdb'][0][0][0].shape[1]


# In[9]:


df = pd.DataFrame(index = range(0,instances), columns = columns)


# In[10]:


for i in mat:
    if i == "imdb":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):           
            df[columns[j]] = pd.DataFrame(current_array[j][0])


# In[11]:


#remove pictures does not include face
df = df[df['face_score'] != -np.inf]

#some pictures include more than one face, remove them
df = df[df['second_face_score'].isna()]

#check threshold
df = df[df['face_score'] >= 4]

#remove old photos
df = df[df['photo_taken']> 2012]


# In[12]:


def extractNames(name):
    return name[0]


# In[13]:


df['celebrity_name'] = df['name'].apply(extractNames)


# In[14]:


df.shape


# In[15]:


#Load Data set Images

def getImagePixels(image_path):
    return cv2.imread('C:/Users/Computer/Documents/FacialRecNN/imdb_crop/imdb_crop/' + image_path[0]) #pixel values in scale of 0-255


# In[46]:


dftest=df.head(n=5)


# In[47]:


tic = time.time()
dftest['pixels'] = dftest['full_path'].apply(getImagePixels)
toc = time.time()

print("this block completed in ",toc-tic," seconds...") #562.80 seconds


# In[48]:


dftest


# In[49]:


from deepface import DeepFace
 
def findFaceRepresentation(img):
    try:
        representation = DeepFace.represent(img_path = img, model_name = "VGG-Face", detector_backend ="skip")
        outvec = representation[0]["embedding"]
    except:
        outvec = None
 
    return outvec


# In[50]:


tic = time.time()
dftest['face_vector_raw'] = dftest['pixels'].apply(findFaceRepresentation) #vector for raw image
toc = time.time()

print("this block completed in ",toc-tic," seconds...")


# In[51]:


dftest


# In[55]:


len(dftest)


# In[56]:


dftest.index


# In[33]:


facevec=dftest['face_vector_raw']


# In[57]:


#facevec[588]


# In[43]:


initial_representation = DeepFace.represent(img_path = 'C:/Users/Computer/Documents/FacialRecNN/face1.jpg',model_name="VGG-Face", detector_backend="opencv")
yourself_representation = initial_representation[0]["embedding"]


# In[25]:


from deepface.commons import distance


# In[26]:


def findCosineSimilarity(source_representation, test_representation=yourself_representation):
   try:
      return distance.findCosineDistance(source_representation, yourself_representation)
   except:
       return 10 #assign a large value in exception. similar faces will have small value.
 


# In[59]:


#dftest['face_vector_raw'][588]


# In[36]:


len(dftest['face_vector_raw'][588])


# In[38]:


#dftest['face_vector_raw'].values


# In[37]:


dftest['distance'] = dftest['face_vector_raw'].apply(findCosineSimilarity)


# In[68]:


dftest


# In[39]:


#dftest['face_vector_raw'].values


# In[62]:


dftest.index


# In[69]:


distance1 = []
for index in dftest.index:
    distance1.append(distance.findCosineDistance(dftest['face_vector_raw'][index],yourself_representation))


# In[71]:


dftest['distance']=distance1


# In[72]:


dftest


# In[60]:


distance.findCosineDistance(dftest['face_vector_raw'][596],yourself_representation)


# In[40]:


type(facevec)


# In[41]:


type(testface)

