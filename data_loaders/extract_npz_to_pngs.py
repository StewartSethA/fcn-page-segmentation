from __future__ import print_function
import numpy as np
import cv2
import sys, os

if __name__ == "__main__":
    image_path = sys.argv[1]
    if not ".npz" in image_path:
        image_path = image_path+"_gt.npz"
    gt = np.load(image_path)['arr_0']
    for channel in range(gt.shape[-1]):
        gt_layer = gt[:,:,channel]
        cv2.imwrite(image_path.replace("_gt.npz", "_"+str(channel)+".png"), gt_layer*255)
