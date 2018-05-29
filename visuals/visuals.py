from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2


def vis_img(img, maxh=512, maxw=512, debuglevel=0, brightness=0.65, bgr=True):
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
    vis = np.clip(vis, 0, 1)
    if debuglevel > 1:
        print("vis_img Min/Max, dtype", np.min(vis), np.max(vis), vis.dtype)
    return vis[:,:,:3]

def get_pred_gt_diff(pred, gt, img=None):
    return np.abs(pred-gt)

# TODO: Consolidate and delete!
from data_loaders.utils import enum_to_onehot as make_onehot
from evaluation.evaluations import score
def show(im, gt, pred, maxh, maxw, impath=None, batch_num=0, label="", do_show=False):
    #print("SHOW gt shape:", gt.shape, pred.shape, im.shape)
    fx = 1.0
    if im.shape[0] > 3000:
        fx = 1.0 #1.0/8
    image=im
    show_image = True
    if fx < 1:
        if show_image:
            im = cv2.resize(im, (0,0), fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)
        pred = cv2.resize(pred, (0,0), fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (0,0), fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)
        maxh = int(maxh * fx)
        maxw = int(maxw * fx)
        print("Resized ", im.shape, maxh, maxw)

    # Shrink the visuals by a factor if too large.
    if do_show:
         cv2.imshow(label+"Input", image)
    ps = vis_img(pred, maxh, maxw)
    if do_show:
        cv2.imshow(label+'Pred image', ps)
    if impath is not None:
        print("Writing result to ", impath)
        cv2.imwrite(impath + "_pred"+str(batch_num)+".jpg", ps*255.0)

    #print("Argmaxing predictions...")

    pred_argmax = make_onehot(np.argmax(pred, axis=-1), pred.shape[-1])
    #print("Min,Max,Maxpred", np.min(pred_argmax), np.max(pred_argmax), np.max(pred))
    if False:
        ps_argmax = vis_img(pred_argmax, maxh, maxw)
        cv2.imshow(label+'Argmaxed predictions', ps_argmax)
        if impath is not None:
            cv2.imwrite(impath + "_predargmax"+str(batch_num)+".jpg", ps_argmax*255.0)

    gt_mask = None
    if gt is not None and len(gt.shape) == 2:
        #print("Using supplied GT mask")
        gt_mask = gt
        print(gt_mask.shape)
        gt_mask = np.clip(gt_mask, 0, 1)
        gt_mask = np.expand_dims(gt_mask, -1)
        gt_mask = np.tile(gt_mask, (1, 1, pred.shape[-1])) # [0,1]


    if gt_mask is None and gt is not None:
        #print("Generating GT mask...")
        gt_mask = np.sum(gt, axis=-1) * 255.0
        gt_mask = np.clip(gt_mask, 0, 1)
        gt_mask = np.reshape(gt_mask, (gt_mask.shape[0], gt_mask.shape[1], 1))
        gt_mask = np.tile(gt_mask, (1, 1, gt.shape[-1])) # [0,1]

    if gt_mask is not None:
        #print("Masking predictions...")
        if False:
            pred_masked = np.multiply(gt_mask, pred)
            ps_masked = vis_img(pred_masked, maxh, maxw)
            cv2.imshow('Pred masked by GT', ps_masked)
            if impath is not None:
                cv2.imwrite(impath + "_predmasked"+str(batch_num)+".jpg", ps_masked*255.0)

        #print("Masking argmaxed predictions...")
        pred_argmax_masked = np.multiply(pred_argmax, gt_mask)
        ps_argmax_masked = vis_img(pred_argmax_masked, maxh, maxw)
        if do_show:
            cv2.imshow('Pred argmax masked by GT', ps_argmax_masked)
        if impath is not None:
            cv2.imwrite(impath + "_predargmaxmasked"+str(batch_num)+".jpg", ps_argmax_masked*255.0)


    if gt is not None and len(gt.shape) > 2:
        #print("Generating Pred and GT diff image...")
        #pred_gt_diff = np.clip(gt*255.0,0,1) - pred_argmax_masked #np.less()
        pred_gt_diff = np.abs(np.clip(gt*255.0,0,1) - pred) #_argmax_masked #np.less()
        #pred_argmax = np.clip(gt*255.0,0,1) - pred_argmax #np.less()
        # Squish down to one layer
        # TODO: Bad to flatten classes... (WIP)

        # TODO: THis is counting totals. Is it not taking into account the (just_right)
        # pixels in the WRONG locations?????
        #pred_gt_diff = np.mean(pred_gt_diff, axis=-1) #Conflates TP of one with FP of another... Hmmm...
        pred_gt_abs_diff = np.abs(pred_gt_diff)
        #pred_argmax = np.abs(pred_argmax)
        #print("Mean Absolute Difference between GT and Preds:", np.mean(pred_gt_abs_diff))
        #print("Mean Squared Difference between GT and Preds:", np.mean(np.square(pred_gt_abs_diff)))

        #ps_gt_abs_diff = vis_img(pred_gt_abs_diff, maxh, maxw)
        ps_gt_abs_diff = vis_img(pred_gt_abs_diff, maxh, maxw)
        if do_show:
            cv2.imshow("GT Pred diff", ps_gt_abs_diff)
        if impath is not None:
            cv2.imwrite(impath + "_predgtdiff"+str(batch_num)+".jpg", ps_gt_abs_diff*255.0)

        gtso = vis_img(gt, maxh, maxw)
        if do_show:
            cv2.imshow('GT image', gtso)
        if impath is not None:
            cv2.imwrite(impath + "_gt.jpg", gtso*255.0)
    cv2.waitKey(50)
