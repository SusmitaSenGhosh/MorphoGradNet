import torch
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion,path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path+'/final_model.pth')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss,epoch, model, optimizer, criterion,path
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path+'/best_model.pth')

def save_plots(train_acc, valid_acc, test_acc,train_loss, valid_loss,test_loss,path):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.plot(
        test_acc, color='red', linestyle='-', 
        label='test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path+'/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='green', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='blue', linestyle='-', 
        label='validataion loss'
    )
    plt.plot(
        test_loss, color='red', linestyle='-', 
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss.png')


def set_seed(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def draw_contour(image,label,mask1,mask2):
  imgray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
#   image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#   image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  #add this line
  image = np.ascontiguousarray(image, dtype=np.uint8)

#   print(type(image),type(imgray))
  ret,thresh = cv2.threshold(imgray,127,255,0)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  boundary = cv2.drawContours(image, contours, 0, (0,0,255), 4)

  imgray = cv2.cvtColor(mask1,cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(imgray,127,255,0)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  boundary = cv2.drawContours(boundary, contours, -1, (0,255,0), 4)

  imgray = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(imgray,127,255,0)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  boundary = cv2.drawContours(boundary, contours, -1, (255,0,0), 4)

  return boundary




def postproc_pannuke(img):
    img = 255 - img
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        cimg = np.zeros_like(img)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
        pts = np.where(cimg == 255)
        lst_intensities = list(img[pts[0], pts[1]])
        if len(lst_intensities)<=50:
            img[pts[0], pts[1]] = 0
        if (len(np.unique(img[pts[0], pts[1]]))!=1):
            max_intensity = max(lst_intensities,key = lst_intensities.count)
            img[pts[0], pts[1]] = max_intensity
    img = 255-img
    return img


