import numpy as np
import matplotlib.pyplot as plt
import cv2


def vis_img(img, maxh=512, maxw=512, debuglevel=0, brightness=0.4, bgr=True):
    '''
    Creates a colored image representing the C channels across 
    an H x W x C heatmap of prediction or GT probabilities for
    each of the C classes.
    '''
    if debuglevel > 2:
        print("vis_img")

    # debug
    vis = img * brightness
    if bgr:
        v = vis[:,:,0].copy()
        vis[:,:,0] = vis[:,:,2]
        vis[:,:,2] = v
    if vis.shape[2] > 5:
        if bgr:
            vis[:,:,0] += vis[:,:,5]
        else:
            vis[:,:,2] += vis[:,:,5]
        vis[:,:,1] += vis[:,:,5]
    if vis.shape[2] > 4:
        vis[:,:,2] += vis[:,:,4]
        vis[:,:,0] += vis[:,:,4]
    if vis.shape[2] > 3:
        if bgr:
            vis[:,:,0] += vis[:,:,3]
        else:
            vis[:,:,2] += vis[:,:,3]
        vis[:,:,1] += vis[:,:,3]
    if debuglevel > 1:
        print("vis_img Min/Max, dtype", np.min(vis), np.max(vis), vis.dtype)
    return vis[:,:,:3]

def get_pred_gt_diff(pred, gt, img=None):
    return np.abs(pred-gt)
