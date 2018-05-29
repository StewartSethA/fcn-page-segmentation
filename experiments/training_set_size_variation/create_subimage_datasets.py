from __future__ import print_function
import numpy as np
import cv2

import sys
import os
try:
    from data_loaders.gt_loaders import *
    from visuals.visuals import *
except ImportError:
    sys.path.append('../../')
    from data_loaders.gt_loaders import *
    from visuals.visuals import *

import shutil
import errno

from utils import mkdir_p

if __name__ == "__main__":
    crop_policy = "equal_length"
    subdivision_amounts = [2, 3, 4, 5, 8, 16]
    data_dir = sys.argv[1]
    for jpg in [os.path.join(data_dir, f) for f in os.listdir(data_dir) if ".jpg" == f[-4:]]:
        print("Processing", jpg)
        img = cv2.imread(jpg)
        base = os.path.basename(jpg).replace(".jpg", "")
        shape = img.shape

        gt = load_gt(jpg)

        for subdiv_amt in subdivision_amounts:
            output_folder = os.path.join(data_dir, "subdivs_"+str(subdiv_amt))
            h = shape[0] / subdiv_amt
            w = shape[1] / subdiv_amt
            mkdir_p(output_folder)
            quadrant_with_most_hw_plus_mp_pixels = (0,0)
            most_hw_plus_mp_pixels = 0
            for x in range(subdiv_amt):
                for y in range(subdiv_amt):
                    l,u = x*w, y*h
                    r,b = (x+1)*w, (y+1)*h
                    output_file = os.path.join(output_folder, base + "_" + str(x) + "_" + str(y))
                    out_jpg = output_file + ".jpg"
                    # Create and save each of the subcrops
                    out_crop = img[u:b, l:r, :]
                    train = False
                    mp_plus_hw = np.sum(gt[u:b, l:r, 1]) + np.sum(gt[u:b, l:r, 2])
                    if mp_plus_hw > most_hw_plus_mp_pixels:
                        most_hw_plus_mp_pixels = mp_plus_hw
                        quadrant_with_most_hw_plus_mp_pixels = (x,y)
                    print("Writing", out_crop.shape, "to", out_jpg)
                    cv2.imwrite(out_jpg, out_crop)
                    for c in range(gt.shape[-1]):
                        gt_crop = gt[u:b, l:r, c]
                        layersum = np.sum(gt_crop)
                        if layersum == 0:
                            continue
                        out_gtlayer = output_file + "_" + str(c) + ".png"
                        cv2.imwrite(out_gtlayer, gt_crop*255)
                        print("Writing", gt_crop.shape, "to", out_gtlayer)

            x,y = quadrant_with_most_hw_plus_mp_pixels
            best_quadrant_base = base + "_" + str(x) + "_" + str(y)
            # Now move best quadrant out for training
            train_dir = output_folder + "_train"
            mkdir_p(train_dir)

            src_jpg = os.path.join(output_folder, best_quadrant_base + ".jpg")
            dst_jpg = os.path.join(train_dir, best_quadrant_base + ".jpg")
            shutil.move(src_jpg, dst_jpg)
            for c in range(gt.shape[-1]):
                src_png = os.path.join(output_folder, best_quadrant_base + "_" + str(c) + ".png")
                dst_png = os.path.join(train_dir, best_quadrant_base + "_" + str(c) + ".png")
                if os.path.exists(src_png):
                    shutil.move(src_png, dst_png)
