import numpy as np
import cv2
from matplotlib import pyplot as plt

import sys, os

def graphcut_refine(image, gtlayer, lamb=0.5):
    bg_model = np.zeros((1,65),np.float64)
    fg_model = np.zeros((1,65),np.float64)
    gtlayer[gtlayer >= 128] = cv2.GC_FGD
    gtlayer[gtlayer < 128] = cv2.GC_BGD
    
    mask, bg_sample, fg_sample = cv2.grabCut(image, gtlayer, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    refined = image*mask[:,:,np.newaxis]
    plt.imshow(refined),plt.colorbar(),plt.show()
    return mask, refined
    
if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])[:256,:256,:]
    gtlayer = cv2.imread(sys.argv[2], 0)[:256,:256]
    print image.dtype, gtlayer.dtype, image.shape, np.sum(image), gtlayer.shape, np.sum(gtlayer)
    gcr,imgref = graphcut_refine(image, gtlayer, 0.5)
