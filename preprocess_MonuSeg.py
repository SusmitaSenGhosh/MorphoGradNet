import skimage
import cv2 as cv
import numpy as np
import os
from PIL import Image
import random

def extractPatches(im, window_shape=(256,256, 3),stride=128):
    nr, nc, ncolor = im.shape
    patches = skimage.util.view_as_windows(im, window_shape, stride)
    nR, nC, t, H, W, C = patches.shape
    nWindow = nR * nC
    patches =  np.reshape(patches, (nWindow, H, W, C))
    patches = patches.astype(np.int32, copy=False)
    #pathces = np.asarray(patches, dtype=np.float16)
    print(patches.shape)
    return patches



image_path = './Nucli/Train/images'
mask_path = './Nucli/Train/masks'
files = os.listdir(image_path)
random.seed(0)
train_files = files
#train_files = random.sample(files,28)
# validation_files = list(set(files)-set(train_files))

count = 0
patch_size = 256
N = 1024

for i in train_files:
    img = cv.resize(cv.cvtColor(cv.imread(os.path.join(image_path,i)), cv.COLOR_BGR2RGB),(N,N))
    img_patches = extractPatches(img, window_shape=(patch_size,patch_size, 3))
    img_patches = np.asarray(img_patches/255, dtype = np.float32)
    mask = cv.resize(np.array(cv.imread(os.path.join(mask_path,i))),(N,N))
    print(mask.shape)
    mask_patches = extractPatches(mask,window_shape=(patch_size,patch_size, 1))
    mask_patches = np.asarray(np.expand_dims(mask_patches[:,:,:,0],axis = -1)/255>0.5, dtype = np.float32)

    if count == 0:
        images =img_patches
        masks =mask_patches
    else:
        images = np.concatenate((images,img_patches),axis = 0)
        masks = np.concatenate((masks,mask_patches),axis = 0)

    count = count+1

print(images.shape,masks.shape)
np.savez('./dataset/MonuSeg/samples_train.npz',images = images, masks = masks)



image_path = './Nucli/Test/images'
mask_path = './Nucli/Test/masks'
files = os.listdir(image_path)


count = 0
for i in files:
    img = cv.resize(cv.cvtColor(cv.imread(os.path.join(image_path,i)), cv.COLOR_BGR2RGB),(N,N))
    img_patches = extractPatches(img, window_shape=(patch_size,patch_size, 3),stride=patch_size)
    img_patches = np.asarray(img_patches/255, dtype = np.float32)
    mask = cv.resize(np.array(cv.imread(os.path.join(mask_path,i))),(N,N))
    mask_patches = extractPatches(mask,window_shape=(patch_size,patch_size, 1),stride=patch_size)
    mask_patches = np.asarray(np.expand_dims(mask_patches[:,:,:,0],axis = -1)/255>0.5, dtype = np.float32)


    if count == 0:
        images =img_patches
        masks =mask_patches
    else:
        images = np.concatenate((images,img_patches),axis = 0)
        masks = np.concatenate((masks,mask_patches),axis = 0)

    count = count+1

print(images.shape,masks.shape)
np.savez('./dataset/MonuSeg/samples_test.npz',images = images, masks = masks)
