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
parser.add_argument('--dataset', type=str, default='ISIC18',
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

# get the test  loader
_, _, test_loader = load_data(args['datasetpath'],args['dataset'])

# print(model)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

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
            # print(labels.shape)
            images = images.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            # forward pass
            outputs = model(images)
            # calculate the loss
            # loss = criterion(outputs, labels)
            # running_loss += loss.item()
            # calculate the accuracy
            preds = (outputs>=0.5).float()
            running_correct += ((preds == labels).sum().item())
            if counter == 1:
                output = preds
                label = labels.to(device,dtype=torch.int32)
            else:
                output = torch.cat([output,preds],axis =  0)
                label = torch.cat([label, labels.to(device,dtype=torch.int32)],axis = 0)

            for j in range(0,images.shape[0]):
                # print(np.array(255*torch.squeeze(preds[j]).to('cpu'),dtype = np.uint8))

                if not  os.path.exists(savePath1):
                    os.makedirs(savePath1)
                if not  os.path.exists(savePath2):
                    os.makedirs(savePath2)
                cv2.imwrite(savePath1+str(count)+'.png',np.array(255*torch.squeeze(labels[j]).to('cpu'),dtype = np.uint8))
                cv2.imwrite(savePath2+str(count)+'.png',np.array(255*torch.squeeze(preds[j]).to('cpu'),dtype = np.uint8))
                count = count+1

    # loss and accuracy for the complete epoch
    # epoch_loss = running_loss / counter
    epoch_acc = 100. * (running_correct / (len(testloader.dataset)*preds.shape[-2]*preds.shape[-1]))
    print(output.dtype, label.dtype)
    # label = torch.unsqueeze(label,dim = 1)
    tp, fp, fn, tn = smp.metrics.get_stats(output, label, mode='binary', threshold=0.5)
    accuracy = smp.metrics.functional.accuracy(tp, fp, fn, tn, reduction='micro-imagewise')
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise",)
    # sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro", class_weights=None, zero_division=1.0)
    label = label.to(device,dtype = torch.float32)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sm = ssim(output, label)
    return [accuracy.item(),iou_score.item(),f1_score.item(),recall.item(),precision.item(),specificity.item(),sm.item()]


i = 0
for seed in seeds:
    print(seed)
    set_seed(seed)
    savePath = args['basepath']+args['dataset']+'/'+args['model']+'_'+args['backbone']+'_pretrain_'+str(args['pretrain'])+'_'+args['loss']+'_seed_'+str(seed)+'_epoch_'+str(args['epochs'])
    print(savePath)

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
    best_result[i,:] = evaluate(model, test_loader, savePath1,savePath2)

    i = i+1

    # if args['visulaize'] == True:
print(best_result)
print(np.around(100*np.mean(best_result,axis = 0),2))
print(np.around(100*np.std(best_result,axis = 0),2))
DF = pd.DataFrame(best_result)
DF.to_csv(args['basepath']+args['dataset']+'/'+args['model']+'_'+args['backbone']+'_result.csv')
        