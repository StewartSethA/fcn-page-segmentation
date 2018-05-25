"""
Ground truth loaders for semantic segmentation (pixel-labeling) tasks.

This is PERFORMANCE-CRITICAL code, as it may be invoked numerous times per batch
throughout model training and inference.
A reasonable latency per call will be < 50 ms.

Author: Seth Stewart
stewart.seth.a@gmail.com
Brigham Young University
February 2018
"""
import numpy as np
import cv2
import os
from collections import defaultdict

# TODO: Import code from txt_to_pngs here?
def load_gt_from_txt(image_path, num_classes=6, dontcare_idx=-1, image_size=(0,0)):
    pass

def load_gt_from_suffices(image_path, num_classes=6, dontcare_idx=-1, suffix_to_class_map={"DL":0, "HW":1, "MP":2, "LN":3, "ST":4}, gtext="jpg"):
    gtdim = num_classes = len(suffix_to_class_map)
    classnums = range(0, num_classes)
    if dontcare_idx > 0:
        gtdim = num_classes + 1
        classnums.append(dontcare_idx)

    gt = None
    for class_suffix, classnum in suffix_to_class_map.items():
        gt_layer_path = image_path + "_" + class_suffix + "." + gtext
        if os.path.exists(gt_layer_path):
            print "reading", gt_layer_path
            gt_layer = cv2.imread(gt_layer_path, 0).astype('float32') / 255.0
            if gt is None:
                gt = np.zeros((gt_layer.shape[0], gt_layer.shape[1], gtdim))
            gt[:,:,classnum] = gt_layer
    # Catch-all: No GT channels loaded? Still return the appropriate sized block of zeros.
    if gt is None:
        img = cv2.imread(image_path)
        gt = np.zeros((img.shape[0], img.shape[1], gtdim))
    return gt

# Load GT from a set of greyscale PNGs.
def load_gt_pnglayers(image_path, num_classes=6, dontcare_idx=-1):
    """
    Loads a set of ground truth (GT) channels for an image from correspondingly named .PNG files.

    If the image is named "dir/my_image.jpg", then the ground truth channels are expected to be named
    "dir/my_image_0.png", "dir/my_image_1.png", etc., numbered for 0-based channels up to num_classes-1.
    If a given channel image .PNG file is absent, the corresponding channel will be all zeros.
    All of the GT channel images must be of the same spatial dimensions (HxW) as the original and must
    be greyscale.
    Allows OVERLAP of multiple classes per pixel.

    Parameters
    ----------
    image_path : str
        The path to the original image next to which to find the correspondingly named GT .PNG files.
    num_classes : int
        The maximum number of channels to load.
    dontcare_idx: int
        The index of a "don't care" class superlabel; set to -1 to ignore (default)
    Returns
    -------
    A [H x W x num_channels] numpy array containing all of the loaded GT channels, normalized to the range [0,1.0]. If dontcare_idx >= 0, an additional channel is concatenated, representing the "don't care" label.
    """
    gtdim = num_classes
    classnums = range(0, num_classes)
    if dontcare_idx > 0:
        gtdim = num_classes + 1
        classnums.append(dontcare_idx)
    #print("Gtdim:", gtdim)
    #print("Loading from layers FOR REAL!")

    gt = None
    base_path = os.path.splitext(image_path)[0]
    for classnum in classnums:
        gt_layer_path = base_path + "_" + str(classnum) + ".png"
        if not os.path.exists(gt_layer_path):
            gt_layer_path = base_path + "_" + str(classnum) + ".jpg"
        if os.path.exists(gt_layer_path):
            gt_layer = cv2.imread(gt_layer_path, 0).astype('float32') / 255.0
            if gt is None:
                gt = np.zeros((gt_layer.shape[0], gt_layer.shape[1], gtdim))
            gt[:,:,classnum] = gt_layer
    # Catch-all: No GT channels loaded? Still return the appropriate zeros.
    if gt is None:
        print "Trying to load blank GT having shape of original:", image_path
        img = cv2.imread(image_path)
        gt = np.zeros((img.shape[0], img.shape[1], gtdim))
    return gt

# Load GT from a single multi-hot bit-indexed color PNG.
def load_gt_multihot_bit_indexed_png(image_path, num_classes=6, dontcare_idx=-1):
    """
    Loads ground truth (GT) channels from a single correspondingly named color PNG file.
    If the image is named "dir/my_image.jpg", then the ground truth file is named "dir/my_image.png".
    Allows OVERLAP of multiple classes per pixel (multi-hot).

    This loader is used by the ICDAR2017 Competition on Layout Analysis for Challenging Medieval Manuscripts
    http://diuf.unifr.ch/main/hisdoc/icdar2017-hisdoc-layout-comp

    In that competition, the classes are:
    main text body, comment, decoration, background.
    Importantly, they can overlap.
    0x8000... signifies a boundary pixel. Boundary pixels may be classified either the same
    as the foreground pixels they surround, or as background with no additional penalty.
    0x01 is background,
    0x02 is comment,
    0x04 is decoration,
    0x08 is main text body.

    Parameters
    ----------
    image_path : str
        The path to the original image next to which to find the correspondingly named GT .PNG files.
    num_classes : int
        The maximum number of channels to load.
    dontcare_idx: int
        The index of a "don't care" class superlabel; set to -1 to ignore (default)
    Returns
    -------
    A [H x W x num_channels] numpy array containing all of the loaded GT channels, in the range [0,1.0]. If dontcare_idx >= 0, an additional channel is concatenated, representing the "don't care" label.
    """
    print("Loading from RGB!")

    gt_path = image_path.replace(".jpg", ".png")
    #print("Loading GT for multi-indexed PNG file:", gt_path)
    gt_rgb = cv2.imread(gt_path)
    gtdim = num_classes
    if dontcare_idx > 0:
        gtdim = num_classes + 1
    gt = np.zeros((gt_rgb.shape[0], gt_rgb.shape[1], gtdim))
    channels = []
    for c in range(gt_rgb.shape[-1]):
        channels.append(gt_rgb[:,:,c])

    mask = 0x01
    channel = 0
    for classnum in range(0, num_classes):
        gt_layer = (np.bitwise_and(mask, channels[channel])).astype('bool').astype('float32')
        gt[:,:,classnum] = gt_layer
        mask = mask << 1
        if mask > 255:
            mask = 0x01
            channel += 1
    if dontcare_idx > 0: # This expects dontcare_idx is greater than the highest class index.
        channel = dontcare_idx / 8 # bits per channel
        mask = 0x01 << (dontcare_idx % 8)
        gt[:,:,num_classes] = (np.bitwise_and(mask, channels[channel])).astype('bool').astype('float32')

    print("RGB GT shape:", gt.shape)
    return gt

# Disk-Caching wrapper for loading ground truth. Currently uses .npz format.
def load_gt_diskcached(image_path, num_classes=6, gt_loader=load_gt_multihot_bit_indexed_png, dontcare_idx=-1, use_disk_cache=True, memory_cache=None, compress_in_ram=False, downsampling_rate=1.0, debuglevel=-1):
    # First check the memory cache.
    if memory_cache is not None:
        if debuglevel > 1:
            print("Using memory cache")
        if image_path + "_gt" in memory_cache:
            if debuglevel > 1:
                print("Loading from memory cache")
            arr = memory_cache[image_path + "_gt"]
            if debuglevel > 2:
                print("arr.shape:", arr.shape)
            return arr
        elif image_path + "_gt.npz" in memory_cache:
            if debuglevel > 2:
                print("Loading .npz from memory cache")
            f = self.cached_images[image_path+"_gt.npz"]
            f.seek(0)
            arr = np.load(f)['arr_0']
            if debuglevel > 2:
                print("arr.shape:", arr.shape)
            return arr
        if debuglevel > 2:
            print("Not found in memory cache")
    gt_path = image_path.replace(".jpg", ".png")
    #if not os.path.exists(gt_path):
    #    gt_loader = load_gt_pnglayers
    #    #print("Loading from layers")
    #else:
    #    #print("Loading from", gt_loader)
    #    pass
    # Load the disk-cached ground truth as compressed .npz
    if use_disk_cache:
        if debuglevel > 1:
            print("using disk cache")
        if os.path.exists(image_path+"_gt.npz"):
            try:
                if debuglevel > 0:
                    print("Loading npz from disk!", image_path+"_gt.npz")
                gt = np.load(image_path+"_gt.npz")['arr_0']
            except Exception:
                if debuglevel > 0:
                    print("Loading npz failed! Using primary loader...")
                gt = gt_loader(image_path, num_classes, dontcare_idx=dontcare_idx)
                if debuglevel > 1:
                    print("Saving GT to disk, shape:", gt.shape)
                with open(image_path+"_gt.npz", 'wb') as f:
                    np.savez_compressed(f, gt)
        else:
            if debuglevel > 2:
                print("Not found in disk cache. Using primary loader...")
            gt = gt_loader(image_path, num_classes, dontcare_idx=dontcare_idx)
            # Save the compressed GT into the disk cache.
            if debuglevel > 1:
                print("Saving GT to disk, shape:", gt.shape)
            with open(image_path+"_gt.npz", 'wb') as f:
                np.savez_compressed(f, gt)
    else:
        if debuglevel > 2:
            print("Not using disk cache. Using primary loader...")
        gt = gt_loader(image_path, num_classes, dontcare_idx=dontcare_idx)

    if debuglevel > 1:
        print("load_gt GT shape:", gt.shape)


    # Downsample if a rate has been specified. NOTE that this is very slow
    # compared to statically resizing all images in a dataset prior to training,
    # but can be reasonably fast if the memory cache is used.
    if downsampling_rate > 1.0:
        gt = cv2.resize(gt, (0,0), fx=1.0/downsampling_rate, fy=1.0/downsampling_rate)
    if debuglevel > 1:
        print("Resized GT shape", gt.shape)

    # TODO: Only add a sample to the cache if there is more than enough free memory?
    # (Check using guppy)
    # TODO: If needed, pop the least frequently and least recently used elements off the cache first.
    # Imbalanced training is amenable to a cache speedup through exemplar dropping.
    # Currently there is no check for out-of-memory errors, so this will fail
    # for large datasets.
    if memory_cache is not None:
        if compress_in_ram:
            f = io.BytesIO()
            np.savez_compressed(f, gt)
            memory_cache[image_path+"_gt.npz"] = f
        else:
            memory_cache[image_path + "_gt"] = gt

    return gt


# Automatically determines the GT loader to use.
def load_gt_automatic(image_path, num_classes=6, dontcare_idx=-1, use_disk_cache=False, memory_cache=None, compress_in_ram=False, downsampling_rate=1.0, debuglevel=-1):
    if debuglevel > 3:
        print "Loading GT for base file", image_path
    gt_loader = load_gt_from_suffices
    if os.path.exists(image_path.replace("jpg", "png")):
        gt_loader = load_gt_multihot_bit_indexed_png
        if debuglevel > 2:
            print "Using multihot layer loader on", image_path
    else:
        for classnum in range(num_classes):
            gt_layer_path = image_path.replace(".jpg", "_" + str(classnum) + ".jpg")
            if os.path.exists(gt_layer_path):
                if debuglevel > 2:
                    print "Using png layer loader on", gt_layer_path
                gt_loader = load_gt_pnglayers
                break
            gt_layer_path = image_path.replace(".jpg", "_" + str(classnum) + ".png")
            if os.path.exists(gt_layer_path):
                if debuglevel > 2:
                    print "Using png layer loader on", gt_layer_path
                gt_loader = load_gt_pnglayers
                break
    if gt_layer_path is not None and not os.path.exists(gt_layer_path):
        suffix_to_class_map={"DL":0, "HW":1, "MP":2, "LN":3, "ST":4}
        gtext="jpg"
        for class_suffix, classnum in suffix_to_class_map.items():
            gt_layer_path = image_path + "_" + class_suffix + "." + gtext
            if os.path.exists(gt_layer_path):
                if debuglevel > 2:
                    print "Using suffix GT loader on ", image_path
                gt_loader = load_gt_from_suffices
                break

    return load_gt_diskcached(image_path, num_classes=num_classes, gt_loader=gt_loader, dontcare_idx=dontcare_idx, use_disk_cache=use_disk_cache, memory_cache=memory_cache, compress_in_ram=compress_in_ram, downsampling_rate=downsampling_rate, debuglevel=debuglevel)

load_gt = load_gt_automatic


# Index a training set for efficient per-class exemplar access.
def index_training_set_by_class(training_folder, num_classes=4, debuglevel=-1):
    """
    Creates an index of a training set, with GT presence per class, per image.
    This index can be used to create class-balanced batches efficiently.

    Parameters
    ----------
    training_folder : str
        A folder of training image and ground truth samples.
    num_classes : int
        The maximum number of classes to search for.

    Returns
    -------
    a dictionary mapping classes to a list of samples that contain that class,
    a list of all training images,
    and a dictionary containing the total number of pixels in each class.
    """
    image_list = [os.path.join(training_folder, filepath) for filepath in os.listdir(training_folder) if (os.path.splitext(filepath.lower())[-1] in ['.jpg'] and not '.jpg.' in filepath and not '.jpg_' in filepath)]
    if debuglevel >= 0:
        print("Indexing training set; image list:", image_list)
    class_to_samples = defaultdict(list)
    pixel_counts_byclass = defaultdict(lambda:0)
    total_pixel_count = 0.0
    for image_path in image_list:
        gt = load_gt(image_path, num_classes)
        for classnum in range(0, num_classes):
            gt_layer = gt[:,:,classnum]
            pixel_count = np.count_nonzero(gt_layer)
            total_pixel_count += pixel_count
            if pixel_count > 0:
                class_to_samples[classnum].append(image_path)
                pixel_counts_byclass[classnum] += pixel_count
            elif debuglevel > 1:
                print("WARNING! CLASS PIXEL COUNT " + str(classnum) + " FOR IMAGE " + image_path + " IS ZERO; DROPPING FROM TRAINING SET!")

    if debuglevel >= 0:
        print("Total number of classes:", str(len(class_to_samples)))
        print("Total number of images:", str(len(image_list)))
        print("Total number of images per class:")
        for c in class_to_samples.keys():
            print(c, str(len(class_to_samples[c])))
        for c in class_to_samples.keys():
            print("Class", c, "Samples:")
            for im in class_to_samples[c]:
                print(im)
        print("Total number of pixels per class:")
        for c in pixel_counts_byclass.keys():
            print(c, str(pixel_counts_byclass[c]))
        print("Proportion of pixels by class:")
        for c in pixel_counts_byclass.keys():
            print(c, str(float(pixel_counts_byclass[c])/total_pixel_count))
    for c in range(num_classes):
        if len(class_to_samples[c]) == 0:
            print "WARNING!!! Dataset has zero instances representing class ", c, ". Training and validation performance will be affected adversely."
    return class_to_samples, image_list, pixel_counts_byclass


def load_image(image_path, debuglevel=0, downsampling_rate=1.0, channels=3):
    if debuglevel > 0:
        print("Reading image", image_path, "from disk.")
    image = cv2.imread(image_path) # Three channel!
    if len(image.shape) < channels:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    if downsampling_rate > 1.0:
        image = cv2.resize(image, (0,0), fx=1.0/downsampling_rate, fy=1.0/downsampling_rate)
    image = image.astype('float32') / 255.0
    return image
