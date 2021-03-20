import torch
import torch.nn as nn
import torch.nn.functional as F

class Diceloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=1e-12):
        assert prediction.shape==target.shape,'the dimensions of prediction and target do not match/diceloss'
        batch_size=prediction.shape[0]
        intersection=(prediction.view(batch_size,-1) * target.view(batch_size,-1)).sum(dim=1)
        union=prediction.view(batch_size,-1).sum(dim=1)+target.view(batch_size,-1).sum(dim=1)
        dice=(2*intersection+smooth)/(union+smooth)
        dice_loss = 1 - dice
        dice_loss=torch.mean(dice_loss)
        return dice_loss

class Dicecoeff(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=1e-12):
        assert prediction.shape == target.shape, 'the dimensions of prediction and target do not match/dice'
        batch_size = prediction.shape[0]
        intersection = (prediction.view(batch_size, -1) * target.view(batch_size, -1)).sum(dim=1)
        union = prediction.view(batch_size, -1).sum(dim=1) + target.view(batch_size, -1).sum(dim=1)
        dice = (2 * intersection + smooth) / (union + smooth)
        dice = torch.mean(dice)
        return dice

class Iou(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=0.00001):
        assert prediction.shape == target.shape, 'the dimensions of prediction and target do not match/iou'
        mask=prediction<0.5
        prediction[mask]=0#小于0.5的预测为False,在prediction——copy里小于0.5的被置为0
        mask=prediction>=0.5
        prediction[mask]=1
        batch_size = prediction.shape[0]
        intersection=(prediction.view(batch_size,-1) * target.view(batch_size,-1)).sum(dim=1)
        union=target.view(batch_size, -1).sum(dim=1)+prediction.view(batch_size, -1).sum(dim=1)-intersection
        iou=intersection/(union+smooth)
        iou=torch.mean(iou)
        return iou

def CE(input,target):
    assert input.shape==target.shape,'the dimensions of prediction and target do not match/BCE'
    return F.binary_cross_entropy(input,target)

class L2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target,smooth=0.00001):
        assert prediction.shape == target.shape, 'the dimensions of prediction and target do not match/iou'
        batch_size = prediction.shape[0]
        pred=prediction.view(batch_size,-1)
        target=target.view(batch_size,-1)
        l2=torch.mean((pred-target).pow(2),dim=1)
        l2=torch.mean(l2)
        return l2
# target=torch.eye(3)
# prediction=target
#
# bceloss=CE(prediction,target)
# print(bceloss)

# import matplotlib.pyplot as plt
# for i in range(prediction.shape[0]):
#     pred=transforms.ToPILImage()(prediction[i])
#     msk=transforms.ToPILImage()(mask[i])
#     plt.subplot(1,2,1)
#     plt.imshow(pred)
#     plt.subplot(1,2,2)
#     plt.imshow(msk)
#     plt.show()
