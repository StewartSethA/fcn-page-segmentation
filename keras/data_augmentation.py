import numpy as np
import cv2
import image_warper
import time
from scipy import ndimage

class DataAugmenter(object):
    def __init__(self, noise_probability=0.25, max_noiselevel=0.56, warp_probability=0.0, dilate_gt_amount=0, zoom_probability=0):
        self.noise_probability = noise_probability
        self.max_noiselevel = max_noiselevel
        self.warp_probability = warp_probability
        self.zoom_probability = zoom_probability
        self.dilate_gt_amount = dilate_gt_amount
        self.train = True

    def augment(self, image, gt):
        #print ("AUGMENTING!!!")
        zoom_variability = 0 #self.zoom_probability #0 #0.13 #0.13 #33
        warp_probability = self.warp_probability #0.25
        noise_probability = self.noise_probability
        #print ("AUGMENTING!!! 1")
        downsampling_rate = 1.0
        if self.train and zoom_variability > 0:
            downsampling_rate *= (1.0 + zoom_variability * (2 * np.random.random() - 1))
        #print ("AUGMENTING!!! 2")
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
        #print ("AUGMENTING!!! 3")
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
        #print ("AUGMENTING!!! 5")

        for gtl in range(gt.shape[-1]):
            if self.dilate_gt_amount > 1:
                #print("NONZERO pixels of class in field of view:", np.count_nonzero(gt[:,:,gtl]))
                gt[:,:,gtl] = cv2.dilate(gt[:,:,gtl], kernel=np.ones((self.dilate_gt_amount,self.dilate_gt_amount)))

        #print ("AUGMENTING!!! 5")

        #if self.train and np.random.random() < warp_probability:
        #    imb, gtb = image_warper.warp_images(image, gt, borderValue=(1,1,1))
        #    imb = np.clip(imb, 0, 1)
        #    gtb = np.clip(gtb, 0, 1)

        augmentation_time = time.time() - augmentation_time
        #print("Augmentation time:", augmentation_time)

        return image, gt
