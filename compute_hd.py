import seg_metrics.seg_metrics as sg
import cv2
import os
import numpy as np
import pandas as pd

labels = [0,1]
seeds = [0,1,2]
seed_result = np.zeros((len(seeds),6))
m = 0
for seed in seeds:
    gdth_path = './outputs/Kvasir/gt_pred/seed'+str(seed)+'/gt/'  # this folder saves a batch of ground truth images
    pred_path= ['.outputs/Kvasir/gt_pred/seed'+str(seed)+'/MorphoDeepLabV3PlusSv3_ResNet50/',
                './outputs/Kvasir/gt_pred/seed'+str(seed)+'/DeepLabV3PlusSv2_ResNet50/' , 
                './outputs/Kvasir/gt_pred/seed'+str(seed)+'/Unet_resnet34/' ,#, 
                './outputs/Kvasir/gt_pred/seed'+str(seed)+'/Unet_resnet50/',
                 './outputs/Kvasir/gt_pred/seed'+str(seed)+'/UnetPlusPlus_resnet34/',  
                 './outputs/Kvasir/gt_pred/seed'+str(seed)+'/UnetPlusPlus_resnet50/' ]

    n = 0
    for path in pred_path:
        result =[]
        for i in os.listdir(gdth_path): 
            # print(gdth_path,i)
            gdth_img =cv2.imread(os.path.join(gdth_path,i))
            pred_img  =cv2.imread(os.path.join(path,i))
            gdth_img = np.array(gdth_img[:,:,0]/255,np.int32)
            pred_img = np.array(pred_img[:,:,0]/255,np.int32)
            csv_file = path+'/result.csv'  # results will be saved to this file and prented on terminal as well. If not set, results 
            # print(np.unique(gdth_img),np.unique(gdth_img))
            if(len(np.unique(gdth_img)) != 1):
                metrics = sg.write_metrics(labels=labels,  # exclude background
                                gdth_img=gdth_img,
                                pred_img=pred_img,
                                # csv_file=csv_file,
                                verbose = False,
                                metrics=['msd'],)
                if (metrics[0]['msd'][1])<1000:
                    result.append(metrics[0]['msd'][1])
        print(m,n)
        print(max(result),len(result))
        seed_result[m,n] = np.mean(result)
        n = n+1
    m = m+1
print(seed_result, np.mean(seed_result),np.std(100*seed_result))
DF = pd.DataFrame(seed_result)
DF.to_csv('.outputs/kvasir/gt_pred/ahd.csv')
