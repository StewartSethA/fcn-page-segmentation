from __future__ import print_function
import numpy as np
import cv2
import image_warper
import time
from scipy import ndimage
import random

def preproc(imgs,gts,num_classes,batch_size=64,width=28,height=28):
    blur = True
    blur = int(random.random() * 3 + 0.5)
    for n in range(0,batch_size):
        nim = np.zeros((height,width))
        max_noise_pct = 0.85
        mx = random.randint(10,int(max_noise_pct*255))
        mn = random.randint(0,min(mx-1,255))
        nim = np.random.uniform(mn, mx, (height,width))/255.0
        img = imgs[n]
        gt = gts[n]
        # Randomly rotate each sample
        angle_range=15
        if random.random() > 0.5:
            angle = random.randint(-angle_range,angle_range)
            p = (img.shape[1]/2, img.shape[0]/2)
            M = cv2.getRotationMatrix2D(p, angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            gt = cv2.warpAffine(gt, M, (gt.shape[1],gt.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        amt_signal=(255.0-mx)/255.0
        img = (img*amt_signal+nim*(1.0-amt_signal))
        if blur > 0:
            img = cv2.blur(img, (blur,blur))
            contrastdiff = random.uniform(0.2,1)
            img *= contrastdiff
            offset = random.uniform(0.0, 1.0-contrastdiff)
            img += offset
        imgs[n] = img
        gts[n] = gt

    return imgs,gts

class DataAugmenter(object):
    def __init__(self, noise_probability=0.25, max_noiselevel=0.56, warp_probability=0.0, dilate_gt_amount=0, zoom_probability=0):
        self.noise_probability = noise_probability
        self.max_noiselevel = max_noiselevel
        self.warp_probability = warp_probability
        self.zoom_probability = zoom_probability
        self.dilate_gt_amount = dilate_gt_amount
        self.train = True

    def augment_batch(self, images, gts):
        pass

    def augment(self, image, gt):
        #print(("AUGMENTING!!!"))
        zoom_variability = 0 #self.zoom_probability #0 #0.13 #0.13 #33
        warp_probability = self.warp_probability #0.25
        noise_probability = self.noise_probability
        #print(("AUGMENTING!!! 1"))
        downsampling_rate = 1.0
        if self.train and zoom_variability > 0:
            downsampling_rate *= (1.0 + zoom_variability * (2 * np.random.random() - 1))
        #print(("AUGMENTING!!! 2"))
        augmentation_time = time.time()
        if self.train:
            for c in range(3):
                contrastdiff = 0.3
                #if np.random.random() > 0.8:
                #    blur = 3
                #    image[:,:,c] = cv2.blur(image[:,:,c], (blur,blur))
                cd = np.random.uniform(contrastdiff,1)
                image[:,:,c] *= cd
                offset = np.random.uniform(0.0, 1.0-cd)
                image[:,:,c] += offset
        #print(("AUGMENTING!!! 3"))
        if self.train and np.random.random() < noise_probability:
            #print("Messing up input image for data augmentation!")
            #max_noise_pct = max_noise_level
            #mx = random.randint(5,int(max_noise_pct*255))
            #amt_signal=(255.0-mx)/255.0
            #image = (image*amt_signal)
            # Add some random vinegar noise to the image.
            #print("Adding noise")
            noiselevel = np.random.random() * self.max_noiselevel #0.36
            image += noiselevel * np.random.random(size=(image.shape))*(2*np.random.random()-1)**3
            for s in [2,4,8,16]:
                #print(s)
                if len(image.shape) > 2:
                    size = [image.shape[0]/s,image.shape[1]/s,image.shape[2]]
                    res = [s,s,1]
                else:
                    size = [image.shape[0]/s,image.shape[1]/s]
                    res = [s,s]
                #print("Adding", image)
                image[0:size[0]*s,0:size[1]*s,:] += noiselevel * ndimage.zoom(np.random.random(size=size)*(2*np.random.random()-1)**3, res, order=2)
        image = np.clip(image, 0, 1)
        #print(("AUGMENTING!!! 5"))

        for gtl in range(gt.shape[-1]):
            if self.dilate_gt_amount > 1:
                #print("NONZERO pixels of class in field of view:", np.count_nonzero(gt[:,:,gtl]))
                gt[:,:,gtl] = cv2.dilate(gt[:,:,gtl], kernel=np.ones((self.dilate_gt_amount,self.dilate_gt_amount)))

        #print(("AUGMENTING!!! 5"))

        #if self.train and np.random.random() < warp_probability:
        #    imb, gtb = image_warper.warp_images(image, gt, borderValue=(1,1,1))
        #    imb = np.clip(imb, 0, 1)
        #    gtb = np.clip(gtb, 0, 1)

        augmentation_time = time.time() - augmentation_time
        #print("Augmentation time:", augmentation_time)

        return image, gt
