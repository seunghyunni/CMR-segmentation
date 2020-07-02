import torch
import numpy as np

def dice_coef(y_pred, y_true):

    # Convert to Binary
    zeros = torch.zeros(y_pred.size())
    ones = torch.ones(y_pred.size())

    y_pred = y_pred.cpu()
    y_pred = torch.where(y_pred > 0.5, ones, zeros)

    if torch.cuda.is_available():
        y_pred = y_pred.cuda()

    y_true = y_true.cpu()
    y_true = torch.where(y_true > 0, ones, zeros)
    
    if torch.cuda.is_available():
        y_true = y_true.cuda()


    # Calculate Dice Coefficient Score
    smooth = 1.
    
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    #print(y_pred.size())

    intersection = (y_pred * y_true).sum()

    return (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
