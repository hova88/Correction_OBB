import numpy as np 
import torch

from ops import boxes_iou3d_gpu

def calculate_iou(pred_boxes , fake_boxes , gt_boxes):
    """
    Args:
        pred_boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        fake_boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        gt_boxes: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        pred_iou: (N, 1)
        fake_iou: (N, 1)
        diff_iou: (N, 1)
    """
    pred_boxes = pred_boxes.float().cuda()
    fake_boxes = fake_boxes.float().cuda()
    gt_boxes   = gt_boxes.float().cuda()
    

    pred_iou = boxes_iou3d_gpu(pred_boxes[:, 0:7], gt_boxes[:, 0:7]) 
    fake_iou = boxes_iou3d_gpu(fake_boxes[:, 0:7], gt_boxes[:, 0:7]) 

    pred_iou = torch.diagonal(pred_iou)
    fake_iou = torch.diagonal(fake_iou)
    
    return pred_iou.cpu().numpy(), fake_iou.cpu().numpy(), (pred_iou - fake_iou).cpu().numpy()


def statistics_diff_iou(fake_iou , diff_iou , step=0.01):
    """
    Args:
        diff_iou: (N, 1) 
        fake_iou: (N, 1)
    Returns:
        interval:(M)
        count_diff_iou: (M, 2) [#in_interval , [fake_iou<0.5 , fake_iou>=0.5]]
    """
    start , end = diff_iou.min() - step ,diff_iou.max() + step 
    interval = np.arange(start , end , step)
    length =  interval.shape[0] -1
    quantity = np.zeros(( length, 2))
    for idx in range(length):
        count = [0,0] 
        for iou_idx in range(diff_iou.shape[0]):
            flg = int(fake_iou[iou_idx] > 0.5)
            if (interval[idx] < diff_iou[iou_idx]) & (diff_iou[iou_idx] < interval[idx+1]):
                count[flg] += 1
        
        quantity[idx] = count

    return interval , quantity


def plot_histogram(interval , quantity , width = 0.01):
    """
    Args:
        interval:(M)
        quantity: (M, 2) [x, y, z, dx, dy, dz, heading]
        
    Returns:
        diff_iou: (N, 1)
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # calculate x and y
    x = (interval[1:] +interval[:-1]) * 0.5  
    x *= (width / (x[1] - x[0]))
    ax.bar(x, quantity[:,1], width, bottom = quantity[:,0] , color = 'tomato', label = r'$IoU_{main}\geq0.5$')
    ax.bar(x, quantity[:,0], width, color = 'royalblue', label = r'$IoU_{main}<0.5$', hatch = '')
    
    # init the figture parameters
    ax.legend(prop={'size': 10})
    ax.set_xlabel(r'$\Delta$IoU')
    ax.set_ylabel('Statistical Quantity')
    ax.set_title(r'Evaluate of PostDetector')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
