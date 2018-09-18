from __future__ import print_function
import scipy
import numpy as np
import cv2
from collections import defaultdict

def find_connected_components(preds):
    pass

def filter_connected_components(preds, minsizes=defaultdict(lambda:0)):
    pass

def get_dominant_label(cc, preds):
    pass

def relabel_preds(preds, cc, label):
    pass

def relabel_connected_components(preds):
    flat_predictions = np.sum(preds, axis=-1)
    # Now find connected components
    ccs = find_connected_components(preds)
    for cc in ccs:
        dominant_label = get_dominant_label(cc, preds)
        relabel_preds(preds, cc, dominant_label)

def multihot_to_multiindexed_rgb(preds):
    # Each array element is 1.0 or 0.0 signaling the predicted presence or absence of each class.
    # Convert these to RGB format where RGB[y,x] = Sum_{c = 0...N-1} [ (0x1 << c) & preds[y,x,c] ]
    predsrgb = np.zeros(preds.shape[:2], dtype='uint32')
    shift = 0
    predsint = preds.astype('uint32')
    print("multihot_to_multiindexed_rgb preds max, min, avg:", np.max(predsint), np.min(predsint), np.mean(predsint))
    for c in range(preds.shape[-1]):
        classgroup = np.left_shift(predsint[:,:,c], shift)
        print("classgroup min, max, avg:", np.min(classgroup), np.max(classgroup), np.mean(classgroup))
        predsrgb[:,:] += classgroup
        shift += 1
    predsrgbchannels = np.zeros((preds.shape[0], preds.shape[1], 3), dtype='uint8')
    mask = 0xFF
    for c in range(3):
        predsrgbchannels[:,:,c] += np.bitwise_and(predsrgb[:,:], mask)
        mask = mask << 8
    print("predsrgb max, min, avg:", np.max(predsrgbchannels), np.min(predsrgbchannels), np.mean(predsrgbchannels))
    return predsrgbchannels

def postprocess_preds(batch_x, preds, gt=None, gt_mask=None, pixel_counts_byclass=None, thresh=defaultdict(lambda:0.5)):
    if thresh is None:
        thresh = defaultdict(lambda:0.5)
    #return preds

    preds_orig = preds.copy()
    # HisDoc modification: (deleted lines) NO filtering with foreground mask!

    if pixel_counts_byclass is not None:
        #print("Using pixel counts by class.", preds.shape)
        for c in range(preds.shape[-1]): #pixel_counts_byclass.keys():
            if (c not in pixel_counts_byclass) or pixel_counts_byclass[c] == 0:
                preds[:,:,:,0] = preds[:,:,:,0] + preds[:,:,:,c] # Zero out the unseen and hence unpredictable class.
                preds[:,:,:,c] = 0 # Zero out the unseen and hence unpredictable class.
                print("Zeroing out class", c, "because it has no representation in the training data.")


    # NOW do some fancy postprocessing based on the number of images containing each class and the amount of each class in each training image.

    # HisDoc modification: DON'T argmax!!!
    #argmax_preds = np.argmax(preds, axis=-1)
    preds = preds_orig
    #thresh = defaultdict(lambda:0.9)
    if len(thresh) > 0:
        print("Postprocessing using thresholds:", thresh)
    # Rather than do arg-maxing, threshold the array.
    if type(thresh) == float:
        preds[preds < thresh] = 0.0
        preds[preds >= thresh] = 1.0
    else:
        for c in range(preds.shape[-1]):
            pct = preds[:,:,:,c]
            pct[pct < thresh.get(c, 0.5)] = 0.0
            pct[pct >= thresh.get(c, 0.5)] = 1.0
            preds[:,:,:,c] = pct

    ## Turning off this filtering for now.
    ##
    #for c in range(preds.shape[-1]):
    #    count_on = np.count_nonzero(np.equal(preds, c).astype('float32'))
    #    #print("Channel and nonzero pixels:", c, count_on)
    #    if count_on < 200:
    #        print("ZEROING out sparse class without GT mask", c)
    #        preds[:,:,:,0] = preds[:,:,:,0] + preds[:,:,:,c] # Zero out the unseen and hence unpredictable class.
    #        preds[:,:,:,c] = 0 # Zero out the unseen and hence unpredictable class.

    #preds = scipy.ndimage.filters.convolve(preds, np.ones((1,25,25,1))/(25.0*25.0))
    preds = np.clip(preds, 0, 1)

    return preds

    # TODO: Needs OpenCV3!!
    # Hole filling: If the value at the center of the convolution kernel is not
    # consistent with surrounding predictions, coerce the current value to be
    # consistent.
    print("Performing Domain transform on all channels of predicted image...")
    for b in range(batch_x.shape[0]):
        edgemap = cv2.Laplacian(batch_x[b],cv2.CV_64F)
        guided_filter = cv2.createGuidedFilter(edgemap, radius=50, eps=10)
        for c in range(preds.shape[-1]):
            filtered_channel = cv2.dtFilter(guided_filter, preds[b,:,:,c], sigmaSpatial=10, sigmaColor=5, mode="DTF_RF", numIters=3)
            preds[b,:,:,c] = filtered_channel

    return preds
