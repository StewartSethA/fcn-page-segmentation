from __future__ import print_function
import cv2
import numpy as np
import sys, os
import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def dilate(img, iterations=1):
    kernel = np.ones((3,3), np.uint8)
    #kernel[0:0] = kernel[2:0] = kernel[0:2] = kernel[2:2] = 0
    dilation = cv2.dilate(img, kernel, iterations)
    return dilation

if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("Usage: python dilate_gts.py <filesdir> <outdir> dilate_amt")
        exit()
    outdir = sys.argv[2]
    files = os.listdir(sys.argv[1])
    pngs = [f for f in files if ".png" in f[-4:]]
    dilate_amt = int(sys.argv[3])
    mkdir_p(outdir)
    for png in pngs:
        img = cv2.imread(os.path.join(sys.argv[1], png), 0)
        dilated = img
        for i in range(dilate_amt):
            dilated = dilate(dilated, dilate_amt)
        cv2.imwrite(os.path.join(outdir, png), dilated)
    
