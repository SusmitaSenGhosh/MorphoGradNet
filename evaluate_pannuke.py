import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from models import *
import os
import json
import segmentation_models_pytorch as smp
from utils import set_seed
import matplotlib.pyplot as plt
from torchmetrics import Dice
from torchsummary import summary
import pandas as pd
from data_loader import load_data


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str, default='pannuke1',
    help='dataset name')
parser.add_argument('--datasetpath', type=str, default= './dataset/',
    help='dataset path')
parser.add_argument('--model', type=str, default='MorphoGradNet',
    help='model name')
parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train our network for')
parser.add_argument('--basepath', type=str, default='./outputs/',
    help='path for saving output')
parser.add_argument('--noclass', type=int, default=1,
    help='number of nodes at output layer')
parser.add_argument('--backbone', type=str, default='resnet50',
    help='backbone')
parser.add_argument('--pretrain', type=str, default=True,
    help='pretrain backbone')
parser.add_argument('--lr', type=float, default= 1e-3,
    help='learning rate')
parser.add_argument('--loss', type=str, default='diceloss',
    help='loss function')
parser.add_argument('--seed', nargs = "+",type=int, default=0,
    help='seed')
args = vars(parser.parse_args())
parser.add_argument('--postprocessing', type=str, default= False,
    help='post processing')
args = vars(parser.parse_args())



seeds= args['seed']
best_result = np.zeros((len(args['seed']),7))
final_result = np.zeros((len(args['seed']),7))

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# define model
pretrain = args['pretrain']
no_class = args['noclass']
# build the model
if args['model'] == 'DeepLabV3plus':
    model = DeepLabV3plus(pretrained=args['pretrain'], num_classes=no_class).to(device)
elif args['model'] == 'MorphoGradNet':
    model = MorphoGradNet(pretrained=args['pretrain'], num_classes=no_class).to(device)
elif args['model'] == 'Unet':
    model = smp.Unet(encoder_name=args['backbone'], encoder_depth=4, encoder_weights='imagenet', decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32), decoder_attention_type=None, in_channels=3, classes=no_class, activation='sigmoid', aux_params=None).to(device)
elif args['model'] == 'UnetPlusPlus':
    model = smp.UnetPlusPlus(encoder_name=args['backbone'], encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3, classes=no_class, activation='sigmoid', aux_params=None).to(device)
elif args['model'] == 'myunet':
    model = myunet(n_classes=no_class,n_channels = 3).to(device)
elif args['model'] == 'myunet_mspp':
    model = myunet_mspp(n_classes=no_class,n_channels = 3).to(device)


def evaluate(model, testloader, savePath1,savePath2):
    model.eval()
    print('Validation')
    running_loss = 0.0
    running_correct = 0
    counter = 0
    count = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            images, labels = data
            # print(labels.dtype)
            images = images/255 
            images = images.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            # forward pass
            outputs = model(images)
            # calculate the loss
            # loss = criterion(outputs, labels)
            # running_loss += loss.item()
            # calculate the accuracy
            preds = torch.argmax(outputs,dim= 1)
            preds = preds.int()
            labels= torch.argmax(labels,dim= 1)
            labels = labels.int()
            preds_pp = torch.zeros_like(preds)
            running_correct += ((preds == labels).sum().item())
            if args['postprocessing']== True:
                for j in range(0,preds.shape[0]):
                    a = ((preds[j].cpu().detach().numpy())/5)*255
                    b = a[:, :, None] * np.ones(3, dtype=int)[None, None, :]
                    # print(np.uint8(b))
                    c = postproc_pannuke(np.uint8(b))
                    preds_pp[j] = torch.from_numpy(np.asarray(c/255*5, dtype=np.int32))
                preds = preds_pp.to(device,dtype=torch.int32)
            if counter == 1:
                output = preds
                label = labels.to(device,dtype=torch.int32)
            else:
                output = torch.cat([output,preds],axis =  0)
                label = torch.cat([label, labels.to(device,dtype=torch.int32)],axis = 0)

    
            for j in range(0,images.shape[0]):

                cv2.imwrite(savePath1+str(count)+'.png',np.array(255*torch.squeeze(labels[j]).to('cpu'),dtype = np.uint8))
                cv2.imwrite(savePath2+str(count)+'.png',np.array(255*torch.squeeze(preds[j]).to('cpu'),dtype = np.uint8))
                count = count+1

    return label,output



def get_metrics(pred,target):
 
    pred[torch.where(pred<=4)] = 1
    pred[torch.where(pred==5)] = 0
    target[torch.where(target<=4)] = 1
    target[torch.where(target==5)] = 0

    tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode='binary',num_classes = 6)
    accuracy = smp.metrics.functional.accuracy(tp, fp, fn, tn, reduction='micro', zero_division='warn')
    iou_score = smp.metrics.functional.iou_score(tp, fp, fn, tn, reduction="micro",zero_division='warn')
    f1_score = smp.metrics.functional.f1_score(tp, fp, fn, tn, reduction="micro",zero_division='warn')
    recall = smp.metrics.functional.recall(tp, fp, fn, tn, reduction="micro",zero_division='warn')
    precision = smp.metrics.functional.precision(tp, fp, fn, tn, reduction='micro',zero_division='warn')
    specificity = smp.metrics.functional.specificity(tp, fp, fn, tn, reduction="micro", class_weights=None, zero_division='warn')
    sensitivity = smp.metrics.functional.sensitivity(tp, fp, fn, tn, reduction="micro", class_weights=None, zero_division='warn')


    return [accuracy.item(),iou_score.item(),f1_score.item(),recall.item(),precision.item(),specificity.item(),sensitivity.item()]




#print(model)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

seed = 0
savePath = args['basepath']+args['dataset']+'/'+args['model']+'_'+args['backbone']+'_pretrain_'+str(args['pretrain'])+'_'+args['loss']+'_aug_'+str(args['aug'])+'_seed_'+str(seed)+'_epoch_'+str(args['epochs'])


# get the test loader
_, _, test_loader = load_data(args['datasetpath'],args['dataset'])
print(len(test_loader))
best_model_cp = torch.load(savePath +'/best_model.pth')
best_model_epoch = best_model_cp['epoch']
print(f"Best model was saved at {best_model_epoch} epochs\n") 
model.load_state_dict(best_model_cp['model_state_dict'])


savePath1 = args['basepath']+args['dataset']+'/gt_pred/seed'+str(seed)+'/'+'gt'+'/'
savePath2 = args['basepath']+args['dataset']+'/gt_pred/seed'+str(seed)+'/'+args['model']+'_'+args['backbone']+'/'
if not  os.path.exists(savePath1):
    os.makedirs(savePath1)
if not  os.path.exists(savePath2):
        os.makedirs(savePath2)
target,pred= evaluate(model, test_loader, savePath1,savePath2)
best_result[0,:] = get_metrics(pred,target)
print(best_result)
print(np.around(100*np.mean(best_result,axis = 0),2))
print(np.around(100*np.std(best_result,axis = 0),2))
DF = pd.DataFrame(best_result)
DF.to_csv(args['basepath']+args['dataset']+'/'+args['model']+'_'+args['backbone']+'_result.csv')
