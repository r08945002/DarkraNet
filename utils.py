import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def structure_loss(pred, mask):

    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    wdice = 1 - (2*inter + 1)/(union + 1)
    wdice_log = torch.log((torch.exp(wdice) + torch.exp(-wdice)) / 2)

    # return (wbce + wiou).mean()
    return (wbce + wdice_log + wiou).mean()

def log_cosh_dice_loss(pred, mask):

    smooth = 1.
    mask_f = torch.flatten(mask)
    pred_f = torch.flatten(pred)
    intersection = torch.sum(mask_f * pred_f)
    score = (2. * intersection + smooth) / (
                    torch.sum(mask_f) + torch.sum(pred_f) + smooth)
    dice_loss = 1 - score

    return torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss)) / 2.0)

'''
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -infty and +infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))

    loss_bce = binary_xloss(logits, labels)
    return (loss+loss_bce).mean()



def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -infty and +infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -infty and +infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
'''

def metrics(pred, label):
    '''
    Define metrics
    '''
    SMOOTH = 1e-6
    pred = pred.data.cpu().numpy().squeeze()
    label = label.data.cpu().numpy().squeeze()

    tp = np.sum(pred * label)
    tn = np.sum(pred * label) - tp
    fp = np.sum((1 - label) * pred)
    fn = np.sum((1 - pred) * label)

    mean_iou = tp/(tp+fp+fn+SMOOTH)
    mean_dice = (2*tp)/(2*tp+fp+fn+SMOOTH)
    precision = tp/(tp+fp+SMOOTH)
    recall = tp/(tp+fn+SMOOTH)
    f2 = (5*precision*recall)/(4*precision+recall+SMOOTH)

    return mean_iou, mean_dice, precision, recall, f2



def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def learning_curve(t_mIoU, t_loss, v_mIoU, v_loss):

    plt.figure(figsize=(20,8))
    plt.plot(t_mIoU, lw=3, label = 'Train')
    plt.plot(v_mIoU, lw=3, label = 'Val')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('mIoU', fontsize=20)
    plt.title('mIoU Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('mIoU.png')

    plt.figure(figsize=(20,8))
    plt.plot(t_loss, lw=3, label = 'Train')
    plt.plot(v_loss, lw=3, label = 'Val')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('Loss.png')


