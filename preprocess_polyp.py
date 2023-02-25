import cv2 as cv
import numpy as np
import os

image_path = './Kvasir-SEG/images/'
mask_path = './Kvasir-SEG/masks/'
files = os.listdir(image_path)
images = np.zeros((len(files),256,256,3),dtype = np.float32)
masks = np.zeros((len(files),256,256,1),dtype = np.float32)
count = 0
for i in files:
    img = cv.imread(os.path.join(image_path,i))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    images[count] = cv.resize(img,(256,256))/255
    mask = cv.imread(os.path.join(mask_path,i))
    masks[count] = np.asarray(np.expand_dims(cv.resize(mask[:,:,0],(256,256)),axis = -1)/255>0.5, dtype = np.float32)
    count = count+1

np.savez('./dataset/Kvasir/samples.npz',images = images, masks = masks)