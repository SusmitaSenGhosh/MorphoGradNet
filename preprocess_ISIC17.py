import cv2 as cv
import numpy as np
import os

image_path = './ISIC2017/ISIC-2017_Training_Data/'
mask_path = './ISIC2017/ISIC-2017_Training_Part1_GroundTruth'
files = os.listdir(image_path)
new_files = []
for i in files:
    if i[-4::] == '.jpg':
        new_files.append(i)

files = new_files
images = np.zeros((len(files),256,256,3),dtype = np.float32)
masks = np.zeros((len(files),256,256,1),dtype = np.float32)
count = 0
for i in files:
    print(count)
    img = cv.imread(os.path.join(image_path,i))
    print(img.shape)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    images[count] = cv.resize(img,(256,256))/255
    # print(os.path.join(mask_path,i[0:-4]+'_segmentation.png'))
    mask = cv.imread(os.path.join(mask_path,i[0:-4]+'_segmentation.png'))
    masks[count] = np.asarray(np.expand_dims(cv.resize(mask[:,:,0],(256,256)),axis = -1)/255>0.5, dtype = np.float32)
    count = count+1
np.savez('./dataset/ISIC2017/samples_train.npz',images = images, masks = masks)



image_path = './ISIC2017/ISIC-2017_Validation_Data/'
mask_path = './ISIC2017/ISIC-2017_Validation_Part1_GroundTruth'
files = os.listdir(image_path)
new_files = []
for i in files:
    if i[-4::] == '.jpg':
        new_files.append(i)

files = new_files
images = np.zeros((len(files),256,256,3),dtype = np.float32)
masks = np.zeros((len(files),256,256,1),dtype = np.float32)
count = 0
for i in files:
    print(count)
    img = cv.imread(os.path.join(image_path,i))
    print(img.shape)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    images[count] = cv.resize(img,(256,256))/255
    # print(os.path.join(mask_path,i[0:-4]+'_segmentation.png'))
    mask = cv.imread(os.path.join(mask_path,i[0:-4]+'_segmentation.png'))
    masks[count] = np.asarray(np.expand_dims(cv.resize(mask[:,:,0],(256,256)),axis = -1)/255>0.5, dtype = np.float32)
    count = count+1
np.savez('./datasets/ISIC2017/samples_val.npz',images = images, masks = masks)


image_path = './ISIC2017/ISIC-2017_Test_v2_Data/'
mask_path = './ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth'
files = os.listdir(image_path)
new_files = []
for i in files:
    if i[-4::] == '.jpg':
        new_files.append(i)

files = new_files
images = np.zeros((len(files),256,256,3),dtype = np.float32)
masks = np.zeros((len(files),256,256,1),dtype = np.float32)
count = 0
for i in files:
    print(count)
    img = cv.imread(os.path.join(image_path,i))
    print(img.shape)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    images[count] = cv.resize(img,(256,256))/255
    # print(os.path.join(mask_path,i[0:-4]+'_segmentation.png'))
    mask = cv.imread(os.path.join(mask_path,i[0:-4]+'_segmentation.png'))
    masks[count] = np.asarray(np.expand_dims(cv.resize(mask[:,:,0],(256,256)),axis = -1)/255>0.5, dtype = np.float32)
    count = count+1
np.savez('./dataset/ISIC2017/samples_test.npz',images = images, masks = masks)

