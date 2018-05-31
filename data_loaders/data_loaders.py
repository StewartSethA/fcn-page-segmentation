from __future__ import print_function
"""
Data loaders (inputs and ground truth) for semantic segmentation (pixel-labeling) tasks.

This is PERFORMANCE-CRITICAL code, as it may be invoked numerous times per batch
throughout model training and inference.
A reasonable latency per call will be < 50 ms.

Author: Seth Stewart
stewart.seth.a@gmail.com
Brigham Young University
February 2018
"""
import math
import numpy as np
import cv2
from collections import defaultdict
import multiprocessing
import time
import os
from utils import get_power_of_two_padding
from gt_loaders import load_gt, index_training_set_by_class
from preprocessing.data_augmentation import DataAugmenter
from legacy_training_sample_generators import *

# An image batcher that loads image batches from a folder containing .jpg images.
class WholeImageOnlyBatcher(object):
    def __init__(self, folder, batch_size=1, exts=['.jpg'], max_downsampling=8):
        """
        Initialize the TestImageLoader.

        Parameters
        ----------
        folder : str
            The folder to load images from, at depth=1
        exts : list of str
            The file extensions to load images from.
        max_downsampling : int
            The maximum downsampling that will be applied to the image.
            It will be padded to a multiple of this power of two.
        """
        self.folder = folder
        self.current_index = 0
        self.cycles = 0
        self.image_list = [os.path.join(folder, filepath) for filepath in sorted(os.listdir(folder)) if (os.path.splitext(filepath.lower())[-1] in exts)]
        self.max_downsampling=max_downsampling
        self.has_next = True
        self.batch_size = 1 # TODO: Not implemented.

    # Generator that retrieves the next image in the folder listing, allowing looping.
    # The image will be normalized to the range [0,1.0] and zero-padded to the
    # nearest simple multiple of a power of two in each dimension.
    # Currently only a batch size of 1 is supported.
    def generate(self):
        while True:
            image_path = self.image_list[self.current_index]
            self.current_index += 1
            if self.current_index >= len(self.image_list):
                self.has_next = False
                self.current_index = 0
                self.cycles += 1
            image = cv2.imread(image_path).astype('float32')/255.0
            pad_x, pad_y = get_power_of_two_padding(image, self.max_downsampling)
            image = np.pad(image, [(0,pad_y),(0,pad_x),(0,0)], mode='constant')
            # Reshape as a batch
            batch = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
            yield batch

# An ImageSampler takes a subcrop of an image (or the entire image) using some
# sampling strategy. The vanilla sampler uses uniform random crop coordinates.
class WholeImageSampler(object):
    def __init__(self):
        pass

    # Gets a crop of the image, or pads it to be a multiple of a power of two.
    def sample_image_and_gt(self, image, gt, c=-1, subcrop_dim=-1, image_path="", max_downsampling=8, debuglevel=0):
        pad_x, pad_y = get_power_of_two_padding(image, max_downsampling)
        # Pad image to nearest multiple of appropriately large power of two.
        if pad_x > 0 or pad_y > 0:
            image = np.pad(image, [(0,pad_y),(0,pad_x),(0,0)], mode='constant')
            gt = np.pad(gt, [(0,pad_y),(0,pad_x),(0,0)], mode='constant')
        return image, gt

# Samples image crops of the specified size using a uniform random distribution.
class UniformImageCropSampler(object):
    def __init__(self):
        pass
    # Gets a crop of the image, or pads it to be a multiple of a power of two.
    def sample_image_and_gt(self, image, gt, c=-1, subcrop_dim=512, image_path="", max_downsampling=8, debuglevel=0):
        starty = np.random.randint(0, image.shape[0]-subcrop_dim)
        startx = np.random.randint(0, image.shape[1]-subcrop_dim)
        image = image[starty:starty+subcrop_dim, startx:startx+subcrop_dim,:]
        gt = gt[starty:starty+subcrop_dim, startx:startx+subcrop_dim,:]
        return image, gt

# An ImageSampler takes a subcrop of an image (or the entire image) using some
# sampling strategy. The vanilla sampler uses uniform random crop coordinates.
class ClassImageSampler(object):
    def __init__(self):
        pass

    # Gets the upper left coordinates of a window centered on at least one
    # pixel of the desired class.
    def get_start_indices(self, classnum, gt, subcrop_dim=512, image_path=None, cached_images={}, debuglevel=0):
        if debuglevel > 0:
            t = time.time()
        if (image_path is not None) and image_path + str(classnum)+"where" in cached_images:
            x,y = cached_images[image_path + str(classnum)+"where"]
            if debuglevel > 1:
                t = time.time() - t
                print("Cache where retrieval time:", t, image_path, classnum)
        else:
            gt_layer = gt[:,:,classnum]
            y,x = np.where(gt_layer > 0.0)
            if image_path is not None:
                cached_images[image_path + str(classnum)+"where"] = (x,y)
                if debuglevel > 0:
                    t = time.time() - t
                    print("NP Where generation time:", t, image_path, classnum)

        # Select a random pixel of the target class in the current GT image.
        # Build a window that contains it.
        i = np.random.randint(len(x))
        x,y = x[i],y[i]
        x -= subcrop_dim/2
        y -= subcrop_dim/2
        if x + subcrop_dim >= gt.shape[1]:
            x = gt.shape[1]-subcrop_dim-1
        if y + subcrop_dim >= gt.shape[0]:
            y = gt.shape[0]-subcrop_dim-1
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        x = int(x)
        y = int(y)
        if debuglevel > 1:
            print("FOUND sampled pixel for class ", classnum, ":", x, y, x+subcrop_dim, y+subcrop_dim, gt.shape, subcrop_dim)
            #print(x,y,subcrop_dim,classnum)
        if debuglevel > 1:
            mx = np.max(gt[y:y+subcrop_dim,x:x+subcrop_dim,classnum])
            mn = np.mean(gt[y:y+subcrop_dim,x:x+subcrop_dim,classnum])
            ct_nonzero = np.count_nonzero(gt[y:y+subcrop_dim,x:x+subcrop_dim,classnum])
            print("Nonzero pixels in field:", ct_nonzero, mx, mn, gt.dtype)
            print("")
        return x,y

    # Gets a crop of the image, or pads it to be a multiple of a power of two.
    def sample_image_and_gt(self, image, gt, c, image_path, subcrop_dim=512, max_downsampling=8, debuglevel=0):
        startx = 0
        starty = 0
        do_subcrop = subcrop_dim < image.shape[0] and subcrop_dim < image.shape[1]

        # GET SUBCROP INFO FROM THE GT LAYER FOR THE CURRENT CLASS
        subcropinfo_time = time.time()
        if do_subcrop:
            if debuglevel > 0:
                print("Acquiring subcrop information...")
            #print("Sampling in class-balanced fashion!")
            # Sample the patch from an area containing GT pixels of the specified class.
            classnum = c
            startx, starty = self.get_start_indices(classnum, gt, subcrop_dim, image_path)

            # TODO: Move to a logger in the using class!
            #if hasattr(self.dataset_sampler, resampled_classbalance):
            #    for cd in range(gt.shape[-1]):
            #        self.dataset_sampler.resampled_classbalance[cd] += np.count_nonzero(gt[:,:,cd])
            image = image[starty:starty+subcrop_dim, startx:startx+subcrop_dim,:]
            gt = gt[starty:starty+subcrop_dim, startx:startx+subcrop_dim,:]
        # Pad image to fit crop window if necessary.
        pad_x, pad_y = image.shape[1]-subcrop_dim, image.shape[0]-subcrop_dim
        if pad_x > 0 or pad_y > 0:
            image = np.pad(image, [(0,pad_y),(0,pad_x),(0,0)], mode='constant')
            gt = np.pad(gt, [(0,pad_y),(0,pad_x),(0,0)], mode='constant')
        return image, gt


# A DatasetSampler generates an image path to be used in a training batch,
# based on some sampling procedure, such as uniform random over samples or class-balanced.
# The vanilla sampler obtains training samples uniformly by instance.
# Interestingly, a very smart sampler could choose Support Vectors in some feature
# space, eliminating or downweighting relatively uninformative samples.
# This is a vanilla dataset sampler, sampling uniform randomly with replacement.
class DatasetSampler(object):
    def __init__(self, data_folder, exts = [], num_classes=4, train=True, index=None):
        if index is None:
            class_to_samples, image_list, pixel_counts_byclass = index_training_set_by_class(data_folder, num_classes=num_classes)
        else:
            class_to_samples, image_list, pixel_counts_byclass = index
        self.image_list = image_list
        #self.image_list = [os.path.join(data_folder, filepath) for filepath in os.listdir(data_folder) if (os.path.splitext(filepath.lower())[-1] in ['.jpg'] and not '.jpg.' in filepath and not '.jpg_' in filepath)]
        self.train = train
        self.sample_classbalanced_by_pixel = False

    def get_image_path_and_class(self):
        # Simply pick a random training image, without regard to class balancing.
        randind = 0 if len(self.image_list) == 1 else np.random.randint(0, len(self.image_list))
        image_path = self.image_list[randind]
        # TODO: This sampling method is currently operating with replacement.
        # Implement another sampler without replacement (traditional training set sampler)
        return image_path,-1

# Put this in wherever it is needed...
def preproc(imgs,gts,num_classes,batch_size=64,width=28,height=28, do_preproc=True,rotate_freq=0.15,angle_range=5,blur_freq=0.5,blur_amt=3, noise_freq=0.5, max_noise_pct = 0.65,contrastdiff = 0.3):
    if not do_preproc:
        return imgs, gts
    blur = 0
    if random.random() < blur_freq:
        blur = int(random.random() * blur_amt + 0.5)
     #0.3
    for n in range(0,batch_size):
        img = imgs[n]
        gt = gts[n]
        # Randomly rotate each sample
        if random.random() > rotate_freq:
            h,w = img.shape[0], img.shape[1]
            angle = random.randint(-angle_range,angle_range)
            p = (img.shape[1]/2, img.shape[0]/2)
            M = cv2.getRotationMatrix2D(p, angle, 1)
            # New improved rotate
            import imutils
            img = imutils.rotate_bound(img, angle)
            gt = imutils.rotate_bound(gt, angle)
            oy = (img.shape[0] - h)/2
            ox = (img.shape[1] - w)/2
            img = img[oy:oy+h,ox:ox+w]
            gt = gt[oy:oy+h,ox:ox+w,:]
            #img = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            #gt = cv2.warpAffine(gt, M, (gt.shape[1],gt.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        if random.random() < noise_freq:
            mx = random.randint(5,int(max_noise_pct*255))
            mn = random.randint(0,min(mx-1,255))
            nim = np.random.uniform(mn, mx, imgs.shape[1:])/255.0
            amt_signal=(255.0-mx)/255.0
            img = (img*amt_signal+nim*(1.0-amt_signal))
        if blur > 0:
            img = cv2.blur(img, (blur,blur))
            cd = random.uniform(contrastdiff,1)
            img *= cd
            offset = random.uniform(0.0, 1.0-cd)
            img += offset
        imgs[n] = img
        gts[n] = gt
    return imgs,gts

class ClassPixelQuotaSampler(DatasetSampler):
    def __init__(self, data_folder, index=None, class_weights=None, instance_weights=None, num_classes=4, train=True):
        self.train = train
        if index is None:
            class_to_samples, image_list, pixel_counts_byclass = index_training_set_by_class(training_folder, num_classes=num_classes)
        else:
            class_to_samples, image_list, pixel_counts_byclass = index
        self.image_list = image_list
        self.class_to_samples = class_to_samples
        self.pixel_counts_byclass = pixel_counts_byclass
        self.sum_of_classes = 0.0
        for c in pixel_counts_byclass.keys():
            self.sum_of_classes += pixel_counts_byclass[c]
        self.class_weights = {c:1.0+math.log(self.sum_of_classes/pixel_counts_byclass[c]) for c in pixel_counts_byclass.keys()}
        sorted_classweight_indices = sorted([(self.class_weights[c], c) for c in self.class_weights.keys()]) #range(len(self.class_weights))])
        self.sorted_classweight_indices = [si[1] for si in sorted_classweight_indices]
        self.resampled_classbalance = defaultdict(lambda:0)
        self.resampled_imagebalance = defaultdict(lambda:defaultdict(lambda:0)) # Class to image path to frequency
        self.class_weights = class_weights
        if class_weights is None:
            self.class_weights = [1.0/num_classes]*num_classes #defaultdict(lambda:1.0/num_classes) # TODO: We need a callback from the score function to modify this!
        self.sum_class_weights = 0
        self.sample_classbalanced_by_pixel = False # Whether to go out of our way to find pixels to match a class of interest in our training samples

        # TODO: instance_weights is not currently used, but in the future,
        # it may help to bias some proportion of each training batch
        # in favor of difficult training samples, as is being done for classes.
        self.instance_weights = instance_weights
        self.num_classes = num_classes
        self.class_indices = range(self.num_classes)
        self.pixel_gt_counts = defaultdict(lambda:0)

    def get_image_path_and_class(self, c=None, debuglevel=0):
        minclass = 0
        mincount = 100000000000000
        for c in range(self.num_classes):
            if self.pixel_gt_counts[c] < mincount:
                mincount = self.pixel_gt_counts[c]
                minclass = c
        c = minclass
        print("Sampling greedily from class with fewest pixels:", c, mincount, [int(self.pixel_gt_counts[p]) for p in sorted(self.pixel_gt_counts.keys())])

        while len(self.class_to_samples[c]) == 0:
            c = np.random.choice(self.class_to_samples.keys())
        if len(self.class_to_samples[c]) == 1:
            randind = 0
        else:
            randind = np.random.randint(0, len(self.class_to_samples[c]))
        #print(self.class_to_samples[c])
        image_path = self.class_to_samples[c][randind]
        self.resampled_imagebalance[c][image_path] += 1

        return image_path, c

# Obtain the next sample based on sampling rules, caching, and weights.
# This Sampler samples uniformly by class until the class weights are updated
# (e.g. by computing performance metrics on a validation set), at which point
# it weights the class constitution of each batch to favor classes with poorer
# validation performance metrics.
class PerformanceFeedbackDatasetSampler(DatasetSampler):
    def __init__(self, data_folder, index=None, class_weights=None, instance_weights=None, num_classes=4, train=True):
        self.train = train
        if index is None:
            class_to_samples, image_list, pixel_counts_byclass = index_training_set_by_class(training_folder, num_classes=num_classes)
        else:
            class_to_samples, image_list, pixel_counts_byclass = index
        self.image_list = image_list
        self.class_to_samples = class_to_samples
        self.pixel_counts_byclass = pixel_counts_byclass
        self.sum_of_classes = 0.0
        for c in pixel_counts_byclass.keys():
            self.sum_of_classes += pixel_counts_byclass[c]
        self.class_weights = {c:1.0+math.log(self.sum_of_classes/pixel_counts_byclass[c]) for c in pixel_counts_byclass.keys()}
        sorted_classweight_indices = sorted([(self.class_weights[c], c) for c in self.class_weights.keys()]) #range(len(self.class_weights))])
        self.sorted_classweight_indices = [si[1] for si in sorted_classweight_indices]
        self.resampled_classbalance = defaultdict(lambda:0)
        self.resampled_imagebalance = defaultdict(lambda:defaultdict(lambda:0)) # Class to image path to frequency
        self.class_weights = class_weights
        if class_weights is None:
            self.class_weights = [1.0/num_classes]*num_classes #defaultdict(lambda:1.0/num_classes) # TODO: We need a callback from the score function to modify this!
        self.sum_class_weights = 0
        self.sample_classbalanced_by_pixel = False # Whether to go out of our way to find pixels to match a class of interest in our training samples

        # TODO: instance_weights is not currently used, but in the future,
        # it may help to bias some proportion of each training batch
        # in favor of difficult training samples, as is being done for classes.
        self.instance_weights = instance_weights
        self.num_classes = num_classes
        self.class_indices = range(self.num_classes)

    # Allow biased sampling, for example, to train more heavily on classes with
    # a low validation f-score.
    def set_class_sampling_weights(self, weights, invert=True, debuglevel=0):
        #return # SethS: Experimental; don't allow adjusting of the sampling weights based on validation metrics.
        self.class_weights = weights
        # Sample primary classes for batch based on which classes are performing most poorly.
        self.sum_class_weights = sum([self.class_weights[tc] for tc in self.class_to_samples.keys()])
        inds = [tc for tc in self.class_to_samples.keys()]
        probs = [1.0/len(inds)] * len(inds)
        if invert and self.sum_class_weights > 0:
            probs = []
            inds = []
            for tc in self.class_to_samples.keys():
                prob = (1.0 - self.class_weights[tc]) # Make probability proportional to 1-fscore, to make lower f-score samples more likely.
                probs.append(prob)
                inds.append(tc)
            sumprobs = sum(probs)
            if sumprobs > 0:
                i = 0
                for tc in inds:
                    probs[i] /= sumprobs
                    i += 1
        self.class_indices = inds
        self.class_weights = probs

        # Display the updated class balance.
        if debuglevel > 0:
            resbal = defaultdict(lambda:0)
            resbalsum = 0
            for c in self.resampled_classbalance.keys():
                resbalsum += self.resampled_classbalance[c]
            for c in self.resampled_classbalance.keys():
                resbal[c] = float(self.resampled_classbalance[c]) / resbalsum if resbalsum > 0 else float(self.resampled_classbalance[c])
            print("   Resampled class balance:")
            for cn in resbal.keys():
                print(cn, resbal[cn])


    def get_image_path_and_class(self, c=None, debuglevel=0):
        #print("Probabilitiy indices:", inds, probs, "Sum fscores", self.sum_class_fscores, "Sumprobs", sumprobs)
        # DONE updating the distribution of difficult/easy samples.
        #print(self.sample_classbalanced, self.num_classes, self.sum_class_fscores, sumprobs, self.class_to_samples)
        # PICK which training sample comes next. =========================
        #
        # Draw from a uniform distribution by chance
        if (c is not None) and (self.class_weights is None or self.sum_class_weights == 0):
            while not c in self.class_to_samples.keys():
                c += 1
                if c >= self.num_classes:
                    c = 0

            if debuglevel > 0:
                print("Sampling from random uniform by class", c, self.sum_weights)
        else:
            # Select according to performance-weighted probabilities.
            if debuglevel > 2:
                print(self.class_indices, self.class_weights)
            c = np.random.choice(self.class_indices, p=self.class_weights)
            if debuglevel > 0:
                print("Randomly sampling class by fscore. Got class ", c, "with weight", self.class_weights[c], " and index ", self.class_indices.index(c))
        while len(self.class_to_samples[c]) == 0:
            c = np.random.choice(self.class_to_samples.keys())
        if len(self.class_to_samples[c]) == 1:
            randind = 0
        else:
            randind = np.random.randint(0, len(self.class_to_samples[c]))
        #print(self.class_to_samples[c])
        image_path = self.class_to_samples[c][randind]
        self.resampled_imagebalance[c][image_path] += 1

        return image_path, c

# Load whole or cropped images and ground truth.
class ImageAndGTBatcher(object):
    def __init__(self, training_folder, num_classes=6, batch_size=1, downsampling_rate=1, dataset_sampler=None, image_sampler=None, data_augmenter=DataAugmenter(), image_channels=3, train=True, max_downsampling=8, cache_images=True, crop_size=256, index=None, suffix_to_class_map={"DL":0, "HW":1, "MP":2, "LN":3, "ST":4}):
        data_augmenter = None #TODO Fix augmenter
        if dataset_sampler is None:
            if train:
                print("Using ClassPixelQuotaSampler dataset sampler")
                dataset_sampler = ClassPixelQuotaSampler(training_folder, index=index, num_classes=num_classes)
                print("Using Performance Feedback dataset sampler")
                dataset_sampler = PerformanceFeedbackDatasetSampler(training_folder, index=index, num_classes=num_classes)
            else:
                print("Using simple dataset sampler")
                dataset_sampler = DatasetSampler(training_folder, num_classes=num_classes, index=index)
        self.dataset_sampler = dataset_sampler
        if image_sampler is None:
            if train:
                print("Using class-wise image sampler")
                image_sampler = ClassImageSampler()
            else:
                print("Using whole image sampler")
                image_sampler = UniformImageCropSampler() # For memory reasons. Used to be WholeImageSampler.
        self.image_sampler = image_sampler
        self.data_augmenter = data_augmenter
        self.num_classes = num_classes
        print("Using data batcher with num_classes", self.num_classes)
        self.batch_size = batch_size
        self.downsampling_rate = downsampling_rate
        self.image_channels = image_channels
        self.train = train
        self.batch_num = 0
        self.max_downsampling = max_downsampling
        self.cached_images = {}
        self.image_frequency = defaultdict(lambda:0)
        self.cache_images = cache_images
        self.crop_size = crop_size
        self.suffix_to_class_map = suffix_to_class_map
        self.gt = None
        self.image = None
        self.imb = None
        self.gtb = None

    def load_image(self, image_path, debuglevel=0):
        if image_path + ".image" in self.cached_images:
            if debuglevel > 0:
                print("Loading cached image", image_path)
            image = self.cached_images[image_path+'.image']
        else:
            if debuglevel > 0:
                print("Reading image", image_path, "from disk.")
            image = cv2.imread(image_path) # Three channel!
            #print("Finished reading image", image_path)
            if len(image.shape) < 3:
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            if self.downsampling_rate > 1.0:
                image = cv2.resize(image, (0,0), fx=1.0/self.downsampling_rate, fy=1.0/self.downsampling_rate)
            image = image.astype('float32') / 255.0
            self.cached_images[image_path+'.image'] = image
        #image = cv2.blur(image, (3,3))
        return image

    def mixed_get_batch(self, batch_generators, batch_generator_probabilities=None):
        if batch_generator_probabilities is None:
            batch_generator_probabilities = [1.0/len(batch_generators)]*len(batch_generators)
        while True:
            batcher = np.random.choice(batch_generators, p=batch_generator_probabilities)
            yield batcher(subcrop_dim=None, batch_size=None, compress_in_ram=False, cache_in_ram=True, debuglevel=0)

    def legacy_get_batch(self, subcrop_dim=None, batch_size=None, compress_in_ram=False, cache_in_ram=True, debuglevel=0, max_height_scale=4, min_height_scale=0.25, warp_probability=0.25):
        cache_in_ram = self.train # There is a bug when validation samples are being read from the memory cach
        if batch_size is None:
            batch_size = self.batch_size
        if debuglevel > 0:
            print("GENERATING a LEGACY training sample!")
        if subcrop_dim is None:
            subcrop_dim = self.crop_size
        input_height=input_width=subcrop_dim
        num_channels = 3

        # This will not allow changing the batch size over time.
        if num_channels == 3:
            np_batch_imgs = np.zeros((batch_size, input_height, input_width, num_channels), dtype=np.float32) #np.uint8)
        else:
            np_batch_imgs = np.zeros((batch_size, input_height, input_width), dtype=np.float32) #np.uint8)
        np_batch_gts = np.zeros((batch_size, input_height, input_width, self.num_classes))

        ims = []
        gts = []

        while True:
            for b in range(batch_size):
                ms = int(subcrop_dim*max_height_scale)

                # Pick a training sample to use with its GT.
                image_path,c = self.dataset_sampler.get_image_path_and_class()
                self.last_path = image_path

                # Load the GT and image.
                gt = load_gt(image_path, self.num_classes, memory_cache=self.cached_images, compress_in_ram=compress_in_ram, downsampling_rate=self.downsampling_rate, debuglevel=debuglevel, suffix_to_class_map=self.suffix_to_class_map)
                img = self.load_image(image_path)

                # TODO: GT images are still off by a bit, and sometimes the GT image is scaled up in one dimension vs the original.
                #if self.textsempred:
                #    #print(clasnum, clas, self.class_training_exemplar_generators, self.class_training_exemplar_generators[0])
                #    imb, gtb = self.class_training_exemplar_generators[clas](img, mask, self.charmasks, maskval=0, numclasses=self.num_classes, label=clasnum, minsize=int(self.height*self.min_height_scale), maxsize=ms, height=self.height, width=self.width, maskchannel=clasnum, page=self.page, char_to_idx=self.char_to_idx)
                #elif self.textpred:
                #    imb, gtb, cph = self.class_training_exemplar_generators[clas](img, mask, maskval=0, numclasses=self.num_classes, label=clasnum, minsize=int(self.height*self.min_height_scale), maxsize=ms, height=self.height, width=self.width, maskchannel=clasnum, page=self.page, char_to_idx=self.char_to_idx)
                #else:
                #    imb, gtb = self.class_training_exemplar_generators[clas](img, gt, maskval=0, numclasses=self.num_classes, label=clasnum, minsize=int(self.height*self.min_height_scale), maxsize=ms, height=self.height, width=self.width, maskchannel=clasnum)
                imb, gtb = get_masked_regionsampler_semanticseg_multichannel(img, gt, maskval=0, numclasses=self.num_classes, label=c, minsize=int(input_height*min_height_scale), maxsize=ms, height=input_height, width=input_width, maskchannel=c)

                #if random.random() < warp_probability:
                #    imb, gtb = image_warper.warp_images(imb, gtb, borderValue=(1.0,1.0,1.0), warp_amount=0.1) # TODO: Warp both images by the same amount!
                #print("===", imb.shape, gtb.shape)
                if len(ims) == b:
                    ims.append(imb)
                    gts.append(gtb)
                    np_batch_imgs[b] = ims[b]
                    np_batch_gts[b]  = gts[b]
                else:
                    ims[b] = imb
                    gts[b] = gtb
                    np_batch_imgs[b] = imb
                    np_batch_gts[b] = gtb
                #if self.textpred:
                #    np_batch_chs[start_b] = cph
            np_batch_imgs, np_batch_gts = preproc(np_batch_imgs, np_batch_gts, self.num_classes, np_batch_imgs.shape[0], np_batch_imgs.shape[1], np_batch_imgs.shape[2])
            self.image = np_batch_imgs[0]
            self.gt = np_batch_gts[0]
            #if self.textpred:
            #    return np_batch_imgs, np_batch_gts, np_batch_chs
            #print("Woohoo!!!!!", np_batch_imgs.shape, np_batch_gts.shape)
            yield(np_batch_imgs, np_batch_gts)

    ##### GENERATE a training sample.
    def generate(self, subcrop_dim=None, batch_size=None, compress_in_ram=False, cache_in_ram=True, debuglevel=0):
        cache_in_ram = self.train # There is a bug when validation samples are being read from the memory cach
        if batch_size is None:
            batch_size = self.batch_size
        if debuglevel > 0:
            print("GENERATING a training sample!")
        if subcrop_dim is None:
            subcrop_dim = self.crop_size
        self.samples_obtained = 0
        while True:
            if not self.train:
                self.ims = []
                self.gts = []
            elif (self.imb is None) or (batch_size != self.imb.shape[0]):
                self.ims = []
                self.gts = []
                self.imb = np.zeros((batch_size, subcrop_dim, subcrop_dim, 3), dtype=np.float32)
                self.gtb = np.zeros((batch_size, subcrop_dim, subcrop_dim, self.num_classes), dtype=np.float32)
            c = 0
            # PYGO TODO: EVEN BETTER class sampling algorithm: Go for a quota that EQUALIZES the pixel distributions
            # for each class within a factor of 2. This won't always be possible, but we can try...
            # Just greedily select instances from the most under-represented class.
            if debuglevel > 1:
                start_time = time.time()
                #all_gt_time = time.time()
            pixel_gt_counts = defaultdict(lambda:0)
            for b in range(batch_size):
                if debuglevel > 1:
                    gt_elem_time = time.time()
                # Pick a training sample.
                image_path,c = self.dataset_sampler.get_image_path_and_class()
                self.last_path = image_path

                # Load the GT and image.
                debuglevel = 0
                #if not self.train:
                #    debuglevel = 5
                gt = load_gt(image_path, self.num_classes, memory_cache=self.cached_images, compress_in_ram=compress_in_ram, downsampling_rate=self.downsampling_rate, debuglevel=debuglevel, suffix_to_class_map=self.suffix_to_class_map)
                image = self.load_image(image_path)

                # Pad or crop the image.
                image, gt = self.image_sampler.sample_image_and_gt(image, gt, c=c, image_path=image_path, subcrop_dim=subcrop_dim, max_downsampling=self.max_downsampling)

                # DO Data augmentation!
                if self.data_augmenter:
                    image, gt = self.data_augmenter.augment(image, gt)

                for c in range(gt.shape[-1]):
                    pixel_gt_counts[c] += np.sum(gt[:,:,c])
                self.dataset_sampler.pixel_gt_counts = pixel_gt_counts

                # Append to the batch in a "smart" way.
                if not self.train:
                    self.ims.append(image)
                    self.gts.append(gt)
                elif b == len(self.ims):
                    self.ims.append(image)
                    self.imb[b] = self.ims[b]
                    self.gts.append(gt)
                    self.gtb[b] = self.gts[b]
                else:
                    # THIS gives a HUGE (~4x) speedup! No need to rebuild the
                    # batch numpy array each time.
                    self.ims[b] = image
                    self.gts[b] = gt
                self.samples_obtained += 1
                self.image = image
                self.gt = gt

            # Format the batch: This is the single most expensive operation here!
            # If you did not need to format the data elements in a Numpy array,
            # This method would be ~5x faster (4s per 1000 elems instead of 20s)
            # I have achieved a ~4x improvement by caching the batch numpy array
            # and only assigning to elements of it through a list.
            if self.train:
                imb = self.imb
                gtb = self.gtb
            else:
                imb = np.stack(self.ims, axis=0)
                gtb = np.stack(self.gts, axis=0)

            self.batch_num += 1
            yield (imb, gtb)
