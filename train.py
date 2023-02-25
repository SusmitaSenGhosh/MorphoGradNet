import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from models import *
# from data_loader_aug import load_data
from utils import save_model, save_plots, SaveBestModel, set_seed
import os
import json
import segmentation_models_pytorch as smp
from losses import *
from torch.optim.lr_scheduler import StepLR
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


def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0 
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device,dtype=torch.float)
        labels = labels.to(device,dtype=torch.float)
        # print(image.shape, labels.shape)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        # print(outputs.shape,labels.shape)
        loss = criterion(outputs, labels)
        if args['noclass']==1:
            preds = (outputs>0.5).float()
        else:
            preds, _ = torch.max(outputs,dim= 1)
            preds = preds.float()

        train_running_loss += loss.item()
        # calculate the accuracy
        train_running_correct +=((preds == labels).sum().item())
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / (len(trainloader.dataset)*preds.shape[-2]*preds.shape[-1]))
    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            if args['noclass']==1:
                preds = (outputs>0.5).float()
            else:
                preds, _ = torch.max(outputs,dim= 1)
                preds = preds.float()

            valid_running_correct += ((preds == labels).sum().item())
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / (len(testloader.dataset)*preds.shape[-2]*preds.shape[-1]))
    return epoch_loss, epoch_acc

seeds= args['seed']
for seed in seeds:
    print(seed)
    set_seed(seed)
    savePath = args['basepath']+args['dataset']+'/'+args['model']+'_'+args['backbone']+'_pretrain_'+str(args['pretrain'])+'_'+args['loss']+'_seed_'+str(seed)+'_epoch_'+str(args['epochs'])
    if not  os.path.exists(savePath):
        os.makedirs(savePath)

    with open(savePath+'/commandline_args.txt', 'w') as f:
        json.dump(args, f, indent=2)

    # get the train validation loader
    train_loader, valid_loader, test_loader = load_data(args['datasetpath'],args['dataset'])

    # learning_parameters 
    lr = args['lr']
    epochs = args['epochs']
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

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss function
    if args['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif args['loss'] == 'diceloss':
        criterion = smp.losses.DiceLoss(mode= 'binary', classes=None, log_loss=False, from_logits=True, smooth=0.0, eps=1e-07)


    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    # lists to keep track of losses and accuracies
    train_loss, valid_loss,test_loss = [], [],[]
    train_acc, valid_acc,test_acc = [], [], []

    # start the training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion)
        test_epoch_loss, test_epoch_acc = validate(model, test_loader,  
                                                    criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        test_loss.append(test_epoch_loss)

        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        test_acc.append(test_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print(f"Test loss: {test_epoch_loss:.3f}, test acc: {test_epoch_acc:.3f}")
        # save the best model till now if we have the least loss in the current epoch
        save_best_model(
            valid_epoch_loss, epoch, model, optimizer, criterion,savePath)
        print('-'*50)
        # scheduler.step()

    # save the trained model weights for a final time
    save_model(epochs, model, optimizer, criterion,savePath)
    # save the loss and accuracy plots
    save_plots(train_acc, valid_acc, test_acc,train_loss, valid_loss, test_loss,  savePath)
    print('TRAINING COMPLETE')

