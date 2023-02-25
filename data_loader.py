import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import random
BATCH_SIZE = 16

def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
       
    
def load_data(path,dataset):
    if dataset =='Kvasir':
        temp = np.load(path+'/'+dataset+'/samples.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)


        data = torch.moveaxis(data,-1,1)
        mask = torch.moveaxis(mask,-1,1)


        kf = KFold(n_splits=5,shuffle= True, random_state=0)
        kf.get_n_splits(data)
        count = 0
        for train_index, test_index in kf.split(data):
            count = count+1
            if count == 1:
                print("Dataset: ", 'Polyp', "K:", 5)
                train_data, test_data = data[train_index], data[test_index]
                train_mask, test_mask = mask[train_index], mask[test_index]
                break
            
        train_data, val_data, train_mask, val_mask = train_test_split(train_data, train_mask, test_size=0.2, random_state=0)
        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=BATCH_SIZE)
        valid_loader = DataLoader(TensorDataset(val_data, val_mask), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)



    if dataset =='test_CVC-300':
        temp = np.load(path+'/'+dataset+'/samples.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)

        return test_loader

    if dataset =='test_CVC-ClinicDB':
        temp = np.load(path+'/'+dataset+'/samples.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)

        return test_loader

    if dataset =='test_ETIS-LaribPolypDB':
        temp = np.load(path+'/'+dataset+'/samples.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)

        return test_loader


    if dataset =='ISIC17':
        temp = np.load(path+'/'+dataset+'/samples_train.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        train_data = torch.moveaxis(data,-1,1)
        train_mask = torch.moveaxis(mask,-1,1)

        temp = np.load(path+'/'+dataset+'/samples_val.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        val_data = torch.moveaxis(data,-1,1)
        val_mask = torch.moveaxis(mask,-1,1)

        temp = np.load(path+'/'+dataset+'/samples_test.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
            
        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=BATCH_SIZE,shuffle = True)
        valid_loader = DataLoader(TensorDataset(val_data, val_mask), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)


    if dataset == 'ISIC18':
        base_add = path+dataset
        train_data    = torch.from_numpy(dataset_normalized(np.load(base_add+'/data_train.npy')))
        val_data   = torch.from_numpy(dataset_normalized(np.load(base_add+'/data_val.npy')))
        test_data    = torch.from_numpy(dataset_normalized(np.load(base_add+'/data_test.npy')))

        train_mask    = torch.from_numpy(np.load(base_add+'/mask_train.npy'))
        val_mask   = torch.from_numpy(np.load(base_add+'/mask_val.npy'))
        test_mask    = torch.from_numpy(np.load(base_add+'/mask_test.npy'))

        train_mask    = torch.from_numpy(np.expand_dims(train_mask, axis=3))
        val_mask   = torch.from_numpy(np.expand_dims(val_mask, axis=3))
        test_mask   = torch.from_numpy(np.expand_dims(test_mask, axis=3))


        train_mask   = train_mask /255.
        val_mask  = val_mask /255.
        test_mask  = test_mask /255.


        train_data = torch.moveaxis(train_data,-1,1)
        train_mask = torch.moveaxis(train_mask,-1,1)

        val_data = torch.moveaxis(val_data,-1,1)
        val_mask = torch.moveaxis(val_mask,-1,1)
                
        test_data = torch.moveaxis(test_data,-1,1)
        test_mask = torch.moveaxis(test_mask,-1,1)

        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=BATCH_SIZE)
        valid_loader = DataLoader(TensorDataset(val_data, val_mask), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)
        
        print(train_mask.shape,val_mask.shape,test_mask.shape)
        print('ISIC18 Dataset loaded')
 
    if dataset =='MonuSeg':
        temp = np.load(path+'/'+dataset+'/samples_train.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        data = torch.moveaxis(data,-1,1)
        mask = torch.moveaxis(mask,-1,1)
        random.seed(0)
        print(data.shape[0])
        train_index = random.sample(list(np.arange(0,data.shape[0])),int(data.shape[0]*.8))
        test_index = list(set(list(np.arange(0,data.shape[0])))-set(train_index))
        train_data = data[train_index,:,:,:]
        train_mask = mask[train_index,:,:,:]
        val_data = data[test_index,:,:,:]
        val_mask = mask[test_index,:,:,:]
        print(train_data.shape,val_data.shape)
        del data, mask


        temp = np.load(path+'/'+dataset+'/samples_test.npz')
        data = temp['images']
        mask = temp['masks']
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
            
        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=BATCH_SIZE,shuffle = True)
        valid_loader = DataLoader(TensorDataset(val_data, val_mask), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)

    if dataset =='pannuke1':
        data = np.load(path+'/pannuke/images_1.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_1.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        train_data = torch.moveaxis(data,-1,1)
        train_mask = torch.moveaxis(mask,-1,1)
        print('fold1 loaded')
        del data,mask
        data = np.load(path+'/pannuke/images_2.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_2.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        val_data = torch.moveaxis(data,-1,1)
        val_mask = torch.moveaxis(mask,-1,1)
        print('fold2 loaded')
        del data,mask

        data = np.load(path+'/pannuke/images_3.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_3.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
        print('fold3 loaded')
        del data,mask

        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=BATCH_SIZE,shuffle = True)
        valid_loader = DataLoader(TensorDataset(val_data, val_mask), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)


    if dataset =='pannuke2':
        data = np.load(path+'/pannuke/images_2.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_2.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        train_data = torch.moveaxis(data,-1,1)
        train_mask = torch.moveaxis(mask,-1,1)
        print('fold1 loaded')
        del data,mask

        data = np.load(path+'/pannuke/images_1.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_1.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        val_data = torch.moveaxis(data,-1,1)
        val_mask = torch.moveaxis(mask,-1,1)
        print('fold2 loaded')
        del data,mask

        data = np.load(path+'/pannuke/images_3.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_3.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
        print('fold3 loaded')
        del data,mask

        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=BATCH_SIZE,shuffle = True)
        valid_loader = DataLoader(TensorDataset(val_data, val_mask), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)



    if dataset =='pannuke3':
        data = np.load(path+'/pannuke/images_3.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_3.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        train_data = torch.moveaxis(data,-1,1)
        train_mask = torch.moveaxis(mask,-1,1)
        print('fold1 loaded')
        del data,mask

        data = np.load(path+'/pannuke/images_2.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_2.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        val_data = torch.moveaxis(data,-1,1)
        val_mask = torch.moveaxis(mask,-1,1)
        print('fold2 loaded')
        del data,mask

        data = np.load(path+'/pannuke/images_1.npy')
        mask_index = np.ravel(np.load(path+'/pannuke/masks_1.npy'))
        mask = np.zeros((mask_index.size, mask_index.max() + 1),dtype = np.uint8)
        mask[np.arange(mask_index.size), mask_index] = 1
        mask = np.reshape(mask,(data.shape[0],data.shape[1],data.shape[2],mask.shape[-1]))
        data = torch.from_numpy(data)
        mask    = torch.from_numpy(mask)
        test_data = torch.moveaxis(data,-1,1)
        test_mask = torch.moveaxis(mask,-1,1)
        print('fold3 loaded')
        del data,mask

        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=BATCH_SIZE,shuffle = True)
        valid_loader = DataLoader(TensorDataset(val_data, val_mask), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_mask), batch_size=BATCH_SIZE)


    return train_loader,valid_loader,test_loader
