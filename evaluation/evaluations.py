from __future__ import print_function
"""
evaluations.py

Performs evaluations of pixel-wise predictions against ground-truth.
Includes both visual and numeric results for detailing accuracy and errors.

Author: Seth Stewart
stewart.seth.a@gmail.com
Brigham Young University
February 2018
"""

import math
import numpy as np
from collections import defaultdict

# Import ground truth loaders.
import sys
sys.path.append('./')
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
# "As a caveat: This works so long as the importing script is run from its containing directory. Otherwise the parent directory of whatever other directory the script is run from will be appended to the path and the import will fail."
try:
    from data_loaders.gt_loaders import *
    from visuals.visuals import *
except ImportError:
    sys.path.append('../')
    from data_loaders.gt_loaders import *
    from visuals.visuals import *

# Ignore divide by zero. For many metrics, these have special meaning.
np.seterr(divide='ignore', invalid='ignore')
def fix_div_by_zero(arr, numerators):
    num_mask = np.zeros_like(arr)
    num_mask = np.multiply(np.isnan(arr).astype(arr.dtype), np.equal(numerators, 0.0).astype(arr.dtype))
    arr[np.where(np.isnan(arr))] = 0.0
    arr += num_mask
    return arr

# Divide with the special meaning that 0/0 = 1.0, and K/0 = 0.0 for K != 0.
# This is used in evaluation criteria in instances where no ground truth labels are present,
# so that a metric that predicts no labels on it is not penalized.
def smart_div(num, denom):
    arr = num / denom
    arr = fix_div_by_zero(arr, num)
    return arr

def weight_mask(channels, mask):
    if channels.shape == mask.shape:
        return np.multiply(channels, mask)
    ch = np.transpose(channels, (2, 0, 1))
    masked = np.multiply(ch, mask).transpose(1,2,0)
    return masked

# Saves visualizations of the GT, predictions, and errors to disk.
def visualize_errors(outpath, gt, pred, gtthreshold=0.5, predthreshold=0.5):
    cv2.imwrite(outpath+"_gt.png", vis_img(gt, bgr=False)*255)
    cv2.imwrite(outpath+"_pred.jpg", vis_img(pred, bgr=False)*255)
    predthresh = np.greater(pred, predthreshold).astype('float32')
    cv2.imwrite(outpath+"_predthresh"+str(predthreshold)+".jpg", vis_img(predthresh, bgr=False)*255)
    pred_gt_diff=np.abs(pred-gt)
    cv2.imwrite(outpath+"_predgtdiff.jpg", vis_img(pred_gt_diff, bgr=False)*255)
    predthresh_gt_diff=np.abs(predthresh-np.greater(gt, 0.5).astype('float32'))
    cv2.imwrite(outpath+"_predthreshgtdiff"+str(predthreshold)+".jpg", vis_img(predthresh_gt_diff, bgr=False)*255)

class CachedMetrics(dict):
    '''
    CachedMetrics defines a lazy dictionary of functions operating on a single ground truth, prediction pair.
    Any of the metrics can be accessed or reused by a simple dictionary access to the CachedMetrics instance:

    cm = CachedMetrics(gt, pred)
    print(cm["recall"])
    print(cm["confusion"])

    For per-class metrics, set byclass=True.
    For global metrics weighted by total number of pixels irrespective of class, set byclass=False.

    weights is a dictionary used to weight pixels in some computations.
    It takes the form of a dictionary from strings to 2D numpy arrays,
    where the string key is the name of the method the mask is to be applied in.

    example:
    weights={"recall":recall_weights, "precision":precision_weights}
    '''
    def __init__(self, gt, pred, byclass=True, weights=None, print_warnings=True):
        '''
        Assumes that gt and pred are already binarized in {0,1}.
        If not, it will attempt to compute a differentiable form of each metric.
        '''
        self.metrics = {}
        self.metric_names = self.__dict__.keys()
        self.gt = gt
        self.pred = pred
        self.eps = 0.00000000000001
        self.axes=(0,1)
        self.converse_axes = 2
        self.print_warnings = print_warnings
        if not byclass:
            self.axes=None
            self.converse_axes = 2
            # We're going to use a trick to get global metrics:
            # We're simply going to flatten all of the channels into a single channel
            # by concatenating them along a spatial dimension.
            self.gt = np.reshape(gt, (gt.shape[0], gt.shape[1]*gt.shape[-1], 1))
            self.pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[-1], 1))
        self.gt = np.clip(self.gt, 0, 1)
        self.pred = np.clip(self.pred, 0, 1)
        if weights is None:
            self.weights = defaultdict(lambda:None)
        else:
            self.weights = weights

    # Awesome! The following works for a caching function memory:
    # (see) https://stackoverflow.com/questions/815110/is-there-a-decorator-to-simply-cache-function-return-values
    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = getattr(self, key)()
        return result

    # Define evaluation functions below.
    def mean_squared_error(self):
        return np.mean(np.square(self.gt - self.pred), axis=self.axes)

    def sum_abs_diff(self):
        return np.sum(np.abs(self.gt - self.pred), axis=self.axes)

    def true_positive_mask(self):
        return np.multiply(self.gt, self.pred)

    def true_positives(self):
        return np.sum(self["true_positive_mask"], axis=self.axes)

    def true_negative_mask(self):
        return np.multiply(1.0-self.gt, 1.0-self.pred)

    def true_negatives(self):
        return np.sum(self["true_negative_mask"], axis=(0,1))

    def false_positive_mask(self):
        return np.multiply(1.0-self.gt, self.pred)

    def false_negative_mask(self):
        return np.multiply(self.gt, 1.0-self.pred)

    def gt_mass(self):
        return np.sum(self.gt, axis=self.axes)

    def pred_mass(self):
        return np.sum(self.pred, axis=self.axes)

    def pred_mask(self):
        return np.sum(self.pred, axis=-1)

    def pred_mask_binary(self):
        return np.greater(self["pred_mask"], 0.0).astype(self.pred.dtype)

    def pred_bg(self):
        pred_b = np.ones(self.pred.shape[:2], dtype=self.pred.dtype)
        pred_b -= self["pred_mask"]
        pred_b = np.maximum(pred_b, 0) # In case some of the values across multiple classes sum to > 1
        return pred_b

    def gt_mask(self):
        return np.sum(self.gt, axis=-1)

    def gt_mask_binary(self):
        return np.greater(self["gt_mask"], 0.0).astype(self.gt.dtype)

    def gt_bg(self):
        gt_b = np.ones(self.gt.shape[:2], dtype=self.gt.dtype)
        gt_b -= self["gt_mask"]
        gt_b = np.maximum(gt_b, 0) # In case some of the values across multiple classes sum to > 1
        return gt_b

    def recall(self):
        return smart_div(self["true_positives"], self["gt_mass"])

    def precision(self):
        return smart_div(self["true_positives"], self["pred_mass"])

    def f1_score(self):
        return 2 * self["precision"] * self["recall"] / np.maximum(self["precision"] + self["recall"], self.eps)

    def accuracy(self):
        return (self["true_positives"] + self["true_negatives"]) / (np.product(self.gt.shape[:-1]))

    def global_true_positives(self):
        return np.sum(self["true_positives"])

    def global_true_negatives(self):
        return np.sum(self["true_negatives"])

    def global_accuracy(self):
        return (self["global_true_positives"] + self["global_true_negatives"]) / np.product(self.gt.shape)

    def foreground_mass(self):
        return np.sum(self["foreground_mask"], axis=self.axes)

    # https://arxiv.org/pdf/1711.07695.pdf
    # FGA and global recall are the EXACT same thing. It is the true positive rate of the foreground pixels.
    # This suggests global precision, global recall, and global f1-score can be computed as well.
    # These global measures are the EXACT same things as frequency-weighted precision, recall, and f-measure.
    def foreground_accuracy_global_recall(self):
        return np.sum(np.multiply(self["true_positive_mask"], self.gt), axis=(0,1,2)) / np.maximum(np.sum(self["gt_mass"]), self.eps)

    # TODO: THIS version assumes no pixel overlap, the same as in the paper.
    # https://arxiv.org/pdf/1711.07695.pdf.
    # However, it is less meaningful when applied to datasets that allow pixel overlap.
    # It simply picks the arg max for each channel, which can be arbitrary.
    def foreground_accuracy(self):
        return np.sum(np.multiply(self["true_positive_mask"], self.gt), axis=(0,1,2)) / np.maximum(np.sum(self["gt_mass"]), self.eps)


    # TODO: Weighted F-measure is the harmonic mean of class-agnostic precision
    # and recall across all classes.
    # Average F-measure, on the other hand, is the average of all per-class
    # f-measures across all classes, uniformly weighted.

    # TODO add foreground accuracy!

    def intersection(self):
        return self["true_positives"]

    def union_mask(self):
        return self["true_positive_mask"] + self["false_positive_mask"] + self["false_negative_mask"]

    def union(self):
        return np.sum(self["union_mask"], axis=self.axes)

    def intersection_over_union(self):
        return smart_div(self["intersection"], self["union"])

    def foreground_mask(self):
        return np.array([np.sum(self.gt, axis=self.converse_axes)]*self.gt.shape[-1]).transpose((1,2,0))

    ############################################################################################
    # Overlapped Pixel metrics
    ############################################################################################

    # How to compute the performance metrics for a given class, counting only overlapped regions?
    # Obviously, if classes overlap in the GT, then we'll focus on whether it got those regions right.
    # But what if it overlaps in the prediction as well?
    # We could compute an overlap f-measure score.
    # Of overlapped pixels, how many did we get??
    # Of things that we said are overlapped, how many are overlapped?
    # So we CAN compute an "overlap prediction" accuracy and f-measure.
    # Should that be separate from rating the quality of the overlap predictions themselves?
    def overlapped_gt_pixels(self):
        return self["gt_mask"]

    def count_overlapped_gt_pixels(self):
        return np.sum(np.greater(self["overlapped_gt_pixels"], 1.0).astype('float32'))

    def proportion_overlapped_pixels(self):
        return self["count_overlapped_gt_pixels"] / np.maximum(self["gt_mass"], 0.01)

    # Returns a histogram of the number of overlapped pixels per location.
    # How many have 1, how many have 2, etc.
    def counts_overlapped_gt_classes(self):
        return np.unique(self["overlapped_gt_pixels"], return_counts=True)
        # TODO: Reconcile with the following earlier code:
        print("Class:", c, "GT Mass:", gt_mass, "true_positive_mass:", true_positives[c], "pred mass:", pred_mass, "union mass:", union_mass)
        true_positive_overlap_mass = np.count_nonzero(np.multiply(true_positive_mask, gt_overlap_mask))
        print("True positive overlap mass:", true_positive_overlap_mass)
        overlap_gt_mass = np.count_nonzero(np.multiply(gt_c, gt_overlap_mask))
        print("overlap_gt_mass:", overlap_gt_mass)
        overlap_pred_mass = np.count_nonzero(np.multiply(pred_c, gt_overlap_mask))
        print("overlap_pred_mass:", overlap_pred_mass)

        overlap_precisions[c] += float(true_positive_overlap_mass) / max(1, float(overlap_pred_mass))
        overlap_recalls[c] += float(true_positive_overlap_mass) / max(1, float(overlap_gt_mass))
        overlap_f_measures[c] += 2*overlap_precisions[c]*overlap_recalls[c] / (overlap_precisions[c] + overlap_recalls[c]) if (overlap_precisions[c] + overlap_recalls[c]) > 0 else 0
        overlap_accuracies[c] += float(true_positive_overlap_mass) / max(1, float(true_positive_overlap_mass+np.count_nonzero(np.multiply(true_negative_mask, gt_overlap_mask))))

    ############################################################################################
    # Per-pixel weighted metrics
    ############################################################################################
    # In order to incorporate per-pixel, per-metric weighting,
    # we just introduce new methods that reference a mask parameter, if it has been provided,
    # for each computation that optionally depends on such a mask.
    def weighted_precision(self):
        if self.weights["precision"] is None:
            if self.print_warnings:
                print("Warning: No weights supplied for weighted precision computation. Returning ordinary precision...")
            return self["precision"]
        return smart_div(np.sum(weight_mask(self["true_positive_mask"], self.weights["precision"]), axis=self.axes), np.sum(weight_mask(self.pred, self.weights["precision"]), axis=self.axes))

    def weighted_recall(self):
        if self.weights["recall"] is None:
            if self.print_warnings:
                print("Warning: No weights supplied for weighted recall computation. Returning ordinary recall...")
            return self["recall"]
        return smart_div(np.sum(weight_mask(self["true_positive_mask"], self.weights["recall"]), axis=self.axes), np.sum(weight_mask(self.gt, self.weights["recall"]), axis=self.axes))

    def weighted_f1_score(self):
        return 2 * self["weighted_precision"] * self["weighted_recall"] / np.maximum(self["weighted_precision"] + self["weighted_recall"], self.eps)

    # We don't want to have to compute 2^(2C) entries for confusion, since there are
    # this many pairs of unique class presence/absence bitstring pairs to be compared for overlapping classes
    # (which even for small C, is large: for C=5, 2^(2C) = 32x32 = 1024 entries in the confusion matrix.)
    # Therefore, let's only consider a simpler type of confusion, in which presence or absence of each class
    # is only contrasted with its predicted presence or absence, and tallies are made based only on false positives
    # and false negatives per-class, where there is further a background class to which all votes are compared.
    def confusion(self):
        # An (N+1)x(N+1) Confusion matrix can be computed as follows:
        # for each class:
        # Count the diagonal as correctly classified pixels of that class.
        # Of the remainder:
        # count pixels wrongly attributed from other classes.
        # Count pixels wrongly classified as background, and background wrongly classified as the class in question.
        # You can perform this continuously as well.
        n = self.gt.shape[-1] + 1
        confusion_matrix = np.zeros((n,n))
        for c1 in range(n):
            if c1 < n - 1:
                gt_mask = self.gt[:,:,c1]
            else:
                gt_mask = self["gt_bg"]

            for c2 in range(n):
                if c2 < n - 1:
                    pred_mask = self.pred[:,:,c2]
                else:
                    # Background: Everything not "on" in the prediction
                    pred_mask = self["pred_bg"]
                # Now we have a GT mask for the current class and a predicted mask for another, possibly distinct class.
                # Compute the intersection between the two:
                c1c2 = np.multiply(gt_mask, pred_mask)
                sum_c1c2 = np.sum(c1c2)
                # This is the amount of real estate that should be C1 but was predicted as C2.
                # In the special case that C1 == C2, this is the per-class True Positive Rate.
                confusion_matrix[c1,c2] = sum_c1c2
        return confusion_matrix

import unittest
class TestCachedMetrics(unittest.TestCase):
    def setUp(self):
        self.proportions_by10s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        np.random.seed(1234)

    def test_weighted_measures(self):
        gt = np.array([[[0., 1, 0, 1], [0, 1, 1, 0]]])
        pr = np.array([[[1., 0, 0, 1], [0, 1, 0, 1]]])
        recall_weights = np.array([[2., 0.5]])
        precision_weights = np.array([[1., 0.2]])
        weights={"precision":precision_weights, "recall":recall_weights}
        expected_weighted_recall = ewr = np.array([0., 0.5/2.5, 0., 1.0])
        expected_weighted_precision = ewp = np.array([0., 1.0, 1.0, 1.0/1.2])
        expected_weighted_f1_score = np.array([2*ewr[i]*ewp[i]/(ewr[i]+ewp[i]) for i in range(gt.shape[-1])])
        cm = CachedMetrics(gt, pr, weights=weights)
        np.testing.assert_array_almost_equal(cm["weighted_precision"], expected_weighted_precision)
        np.testing.assert_array_almost_equal(cm["weighted_recall"], expected_weighted_recall)
        np.testing.assert_array_almost_equal(cm["weighted_f1_score"], expected_weighted_f1_score)

    def test_confusion_matrix(self):
        gt1 = [[0., 1.], [0., 0.]]
        gt2 = [[1., 0.], [0., 0.]]
        gt3 = [[0., 0.], [0., 0.]]
        gt4 = [[1., 1.], [1., 0.]]
        gt = np.zeros((2,2,4))
        gt[:,:,0] = gt1
        gt[:,:,1] = gt2
        gt[:,:,2] = gt3
        gt[:,:,3] = gt4
        pr1 = [[0., 0.], [1., 1.]]
        pr2 = [[1., 1.], [0., 0.]]
        pr3 = [[0., 0.], [0., 1.]]
        pr4 = [[1., 1.], [1., 0.]]
        pr = np.zeros((2,2,4))
        pr[:,:,0] = pr1
        pr[:,:,1] = pr2
        pr[:,:,2] = pr3
        pr[:,:,3] = pr4
        confusion_matrix = [[0., 1., 0., 1., 0.], [0., 1., 0., 1., 0.], [0., 0., 0., 0., 0.], [1., 2., 0., 3., 0.], [1., 0., 1., 0., 0.]]
        cm = CachedMetrics(gt, pr)
        np.testing.assert_array_almost_equal(cm["confusion"], np.array(confusion_matrix))

    # Caveat: Traditional foreground accuracy
    # presumes only one label per pixel.
    # FGA is the % of foreground pixels that are correct.
    # Generalized to multiple labels per pixel,
    # it can reduce to recall.
    # https://arxiv.org/pdf/1711.07695.pdf
    # TODO: Fix this test case!
    '''
    def test_foreground_accuracy(self):
        gt1 = [[0., 1.], [0., 0.]]
        gt2 = [[1., 0.], [0., 0.]]
        gt3 = [[0., 0.], [0., 0.]]
        gt4 = [[1., 1.], [1., 0.]]
        gt = np.zeros((2,2,4))
        gt[:,:,0] = gt1
        gt[:,:,1] = gt2
        gt[:,:,2] = gt2
        gt[:,:,3] = gt4
        pr1 = [[0., 0.], [1., 1.]]
        pr2 = [[1., 1.], [0., 0.]]
        pr3 = [[0., 0.], [0., 1.]]
        pr4 = [[1., 1.], [1., 0.]]
        pr = np.zeros((2,2,4))
        pr[:,:,0] = pr1
        pr[:,:,1] = pr2
        pr[:,:,2] = pr3
        pr[:,:,3] = pr4
        fgas = [4.0/5.0]
        cm = CachedMetrics(gt, pr)
        np.testing.assert_array_almost_equal(cm["foreground_accuracy"], np.array(fgas))
    '''

    def test_f1_score(self):
        gt1 = [[0., 1.], [0., 1.]]
        gt2 = [[1., 0.], [0., 0.]]
        gt3 = [[0., 0.], [0., 0.]]
        gt4 = [[1., 1.], [1., 0.]]
        gt = np.zeros((2,2,4))
        gt[:,:,0] = gt1
        gt[:,:,1] = gt2
        gt[:,:,2] = gt3
        gt[:,:,3] = gt4
        pr1 = [[0., 0.], [1., 1.]]
        pr2 = [[1., 1.], [0., 0.]]
        pr3 = [[0., 0.], [0., 1.]]
        pr4 = [[1., 1.], [1., 0.]]
        pr = np.zeros((2,2,4))
        pr[:,:,0] = pr1
        pr[:,:,1] = pr2
        pr[:,:,2] = pr3
        pr[:,:,3] = pr4
        f1s = [2*0.5*0.5/(0.5+0.5), 2*1.0*0.5/(1.0+0.5), 0.0, 1.0]
        cm = CachedMetrics(gt, pr)
        np.testing.assert_array_almost_equal(cm["f1_score"], np.array(f1s))

    def test_intersection_over_union(self):
        gt1 = [[0., 1.], [0., 1.]]
        gt2 = [[1., 0.], [0., 0.]]
        gt3 = [[0., 0.], [0., 0.]]
        gt4 = [[1., 1.], [1., 0.]]
        gt = np.zeros((2,2,4))
        gt[:,:,0] = gt1
        gt[:,:,1] = gt2
        gt[:,:,2] = gt3
        gt[:,:,3] = gt4
        pr1 = [[0., 0.], [1., 1.]]
        pr2 = [[1., 1.], [0., 0.]]
        pr3 = [[0., 0.], [0., 1.]]
        pr4 = [[1., 1.], [1., 0.]]
        pr = np.zeros((2,2,4))
        pr[:,:,0] = pr1
        pr[:,:,1] = pr2
        pr[:,:,2] = pr3
        pr[:,:,3] = pr4
        ious = [1.0/3.0, 1.0/2.0, 0.0, 1.0]
        cm = CachedMetrics(gt, pr)
        np.testing.assert_array_almost_equal(cm["intersection_over_union"], np.array(ious))

    # TODO: Unit test stubs.
    def test_weighted_recall(self):
        pass

    def test_global_recall(self):
        pass

    def test_per_class_performance_measures(self):
        shape=(10,10,6)
        gt = np.ones(shape)
        pred = gt.copy()
        numvals = (gt.shape[0]*gt.shape[1])
        amts_on = []
        amts_pred = []
        amts_true_positive = []
        amts_false_positive = []
        amts_true_negative = []
        amts_false_negative = []
        supposed_precisions = []
        supposed_recalls = []
        supposed_f1_scores = []
        supposed_accuracies = []
        supposed_ious = []
        supposed_fg_accuracies = []
        for c in range(gt.shape[-1]):
            # Set each class to the target proportion in the GT.
            amt_on = np.random.randint(0, numvals+1)
            amt_off = numvals-amt_on
            amts_on.append(amt_on)
            on_vals = np.zeros(numvals)
            on_vals[:amt_on] = 1.0 # Set the appropriate number on for this class.
            np.random.shuffle(on_vals) # Shuffle their locations.
            gt[:,:,c] = on_vals.reshape(shape[:2]) # Set the GT mask layer to this map.

            # Now form which will be guessed right by the simulated predictor (true positives)
            amt_fp = np.random.randint(0, amt_off+1) # 0-100% correct
            amts_false_positive.append(amt_fp)
            amt_tn = amt_off - amt_fp
            amts_true_negative.append(amt_tn)
            pred_vals = on_vals.copy()
            offs = np.where(on_vals == 0.0) # Which locations are "off" in the original
            # Now we are going to set a subset of the original off pixels to "on", reducing precision.
            if amt_fp > 0:
                fp_ons = np.random.choice(offs[0], amt_fp, replace=False)
                pred_vals[fp_ons] = 1.0

            # Now likewise, penalize recall:
            amt_fn = np.random.randint(0, amt_on+1) # 0-100% correct
            amts_false_negative.append(amt_fn)
            amt_tp = amt_on - amt_fn
            amts_true_positive.append(amt_tp)

            ons = np.where(on_vals == 1.0) # Which locations are "on" in the original
            # Now we are going to set a subset of the original on pixels to "off", reducing recall.
            if amt_fn > 0:
                fn_ons = np.random.choice(ons[0], amt_fn, replace=False)
                pred_vals[fn_ons] = 0.0

            pred[:,:,c] = pred_vals.reshape(shape[:2])
            amt_pred = amt_on + amt_fp - amt_fn
            if amt_pred > 0:
                supposed_precision = float(amt_tp) / amt_pred
            else:
                supposed_precision = 1.0
            supposed_precisions.append(supposed_precision)
            if amt_on > 0:
                supposed_recall = float(amt_tp) / amt_on
            else:
                assert(amt_tp == 0)
                supposed_recall = 1.0
            supposed_recalls.append(supposed_recall)
            f1_score = 2*supposed_precision*supposed_recall/max(0.000000000001, supposed_precision+supposed_recall)
            supposed_f1_scores.append(f1_score)
            acc = float(amt_tp+amt_tn)/max(0.01, amt_tp+amt_tn+amt_fp+amt_fn)
            supposed_accuracies.append(acc)
            assert(amt_tp+amt_tn+amt_fp+amt_fn == numvals)
            iou = float(amt_tp) / max(0.01, amt_fp + amt_tp + amt_fn)
            supposed_ious.append(iou)
            if amt_tp + amt_fn == 0:
                fg_acc = 1.0
            else:
                fg_acc = float(amt_tp) / max(0.01, amt_tp + amt_fn) # Same thing as recall!
            supposed_fg_accuracies.append(fg_acc)

        cm = CachedMetrics(gt, pred, print_warnings=False)
        np.testing.assert_array_almost_equal(cm["precision"], np.array(supposed_precisions))
        np.testing.assert_array_almost_equal(cm["weighted_precision"], np.array(supposed_precisions))
        np.testing.assert_array_almost_equal(cm["recall"], np.array(supposed_recalls))
        np.testing.assert_array_almost_equal(cm["weighted_recall"], np.array(supposed_recalls))
        np.testing.assert_array_almost_equal(cm["f1_score"], np.array(supposed_f1_scores))
        np.testing.assert_array_almost_equal(cm["weighted_f1_score"], np.array(supposed_f1_scores))
        np.testing.assert_array_almost_equal(cm["intersection_over_union"], np.array(supposed_ious))
        np.testing.assert_array_almost_equal(cm["accuracy"], np.array(supposed_accuracies))
        #np.testing.assert_array_almost_equal(cm["recall"], np.array(supposed_fg_accuracies))

    def test_per_class_precision(self):
        shape=(10,10,6)
        gt = np.ones(shape)
        pred = gt.copy()
        numvals = (gt.shape[0]*gt.shape[1])
        amts_on = []
        amts_correct = []
        supposed_precisions = []
        for c in range(gt.shape[-1]):
            # Set each class to the target proportion.
            amt_on = 0 #np.random.randint(0, numvals+1)
            amt_off = numvals-amt_on
            amts_on.append(amt_on)
            on_vals = np.zeros(numvals)
            on_vals[:amt_on] = 1.0 # Set the appropriate number on for this class.
            np.random.shuffle(on_vals) # Shuffle their locations.
            gt[:,:,c] = on_vals.reshape(shape[:2]) # Set the GT mask layer to this map.

            # Now form which will be guessed right by the simulated predictor
            amt_incorrect = np.random.randint(0, amt_off+1) # 0-100% correct
            amts_correct.append(amt_incorrect)
            pred_vals = on_vals.copy()
            offs = np.where(on_vals == 0.0) # Which locations are "on" in the original
            # Now we are going to set a random subset of the originals to "off", reducing recall.
            if amt_incorrect > 0:
                fp_ons = np.random.choice(offs[0], amt_incorrect, replace=False)
                pred_vals[fp_ons] = 1.0
            pred[:,:,c] = pred_vals.reshape(shape[:2])
            supposed_precision = 1.0-float(amt_incorrect) / max(0.1, amt_on + amt_incorrect)
            supposed_precisions.append(supposed_precision)
        cm = CachedMetrics(gt, pred, print_warnings=False)
        np.testing.assert_array_almost_equal(cm["precision"], np.array(supposed_precisions))
        np.testing.assert_array_almost_equal(cm["weighted_precision"], np.array(supposed_precisions))

    def test_per_class_recall(self):
        shape=(10,10,6)
        gt = np.ones(shape)
        pred = gt.copy()
        numvals = (gt.shape[0]*gt.shape[1])
        amts_on = []
        amts_correct = []
        supposed_recalls = []
        for c in range(gt.shape[-1]):
            # Set each class to the target proportion.
            amt_on = 0 #np.random.randint(0, numvals+1)
            amts_on.append(amt_on)
            on_vals = np.zeros(numvals)
            on_vals[:amt_on] = 1.0 # Set the appropriate number on for this class.
            np.random.shuffle(on_vals) # Shuffle their locations.
            gt[:,:,c] = on_vals.reshape(shape[:2]) # Set the GT mask layer to this map.

            # Now form which will be guessed right by the simulated predictor
            amt_correct = np.random.randint(0, amt_on+1) # 0-100% correct
            amts_correct.append(amt_correct)
            pred_vals = on_vals.copy()
            ons = np.where(on_vals == 1.0) # Which locations are "on" in the original
            # Now we are going to set a random subset of the originals to "off", reducing recall.
            amt_wrong = amt_on - amt_correct
            if amt_wrong > 0:
                offs = np.random.choice(ons[0], amt_wrong, replace=False)
                pred_vals[offs] = 0.0
            pred[:,:,c] = pred_vals.reshape(shape[:2])
            if amt_on > 0:
                supposed_recall = float(amt_correct) / amt_on
            else:
                assert(amt_correct == 0)
                supposed_recall = 1.0
            supposed_recalls.append(supposed_recall)
        cm = CachedMetrics(gt, pred, print_warnings=False)
        np.testing.assert_array_almost_equal(cm["recall"], np.array(supposed_recalls))
        np.testing.assert_array_almost_equal(cm["weighted_recall"], np.array(supposed_recalls))

# If you want weighted precision or weighted recall, put in
# weights={"precision":precisionweights, "recall"=recallweights}, etc.
def score(gt, pred, predthreshold=0.5, gtthreshold=0.5, predweights=None, gtweights=None, weights={}):
    if gtthreshold is not None and gtthreshold > 0:
        gt = np.greater(gt, gtthreshold).astype('float32')
    if predthreshold is not None and predthreshold > 0:
        pred = np.greater(pred, predthreshold).astype('float32')
    if predweights is not None:
        pred = np.multiply(pred, predweights)
    if gtweights is not None:
        gt = np.multiply(gt, gtweights)
    cm = CachedMetrics(gt, pred, weights=weights)
    return cm

standard_metrics = ["f1_score", "recall", "precision", "intersection_over_union", "accuracy", "confusion"]

def trim_to_common_size(arr1, arr2):
    maxdims = md = [min(arr1.shape[i], arr2.shape[i]) for i in range(len(arr1.shape))]
    arr1 = arr1[:md[0],:md[1],:md[2]]
    arr2 = arr2[:md[0],:md[1],:md[2]]
    return arr1, arr2

def score_and_visualize_errors(gtfile, predfile, gtthreshold=0.5, predthreshold=0.5, suffix_to_class_map=None):
    gt = load_gt(gtfile, suffix_to_class_map=suffix_to_class_map)
    if np.sum(gt) == 0:
        print("WARNING: GT sum for file", gtfile, "is zero!")
    pred = load_gt(predfile, suffix_to_class_map=suffix_to_class_map)
    gt,pred = trim_to_common_size(gt,pred)
    s = score(gt, pred, gtthreshold, predthreshold)
    visualize_errors(predfile, gt, pred, gtthreshold, predthreshold)
    for metric in standard_metrics:
        print(metric, s[metric])
    for metric in standard_metrics:
        print("Average", metric, np.mean(s[metric]))
    return s

def visualize_confusion(outpath, confusion):
    sqrt_conf = np.sqrt(confusion.astype('float32'))
    scaled_conf = sqrt_conf / np.max(sqrt_conf, axis=0)
    cv2.imwrite(outpath+"_confusion.png", confusion*255)

def score_and_visualize_folders(gt_dir, test_dir, gtthreshold=0.5, predthreshold=0.5, suffix_to_class_map=None):
    test_images = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if '.jpg' == f[-4:]]
    pred_files = [f.replace(gt_dir, test_dir) for f in test_images]
    print("Test images", test_images)
    print("")
    print("Pred files", pred_files)
    print("")
    scores = defaultdict(list)
    for f,f2 in zip(test_images, pred_files):
        print("Evaluating", f, f2)
        s = score_and_visualize_errors(f, f2, gtthreshold, predthreshold, suffix_to_class_map=suffix_to_class_map)
        visualize_confusion(f2, s["confusion"])
        for metric in standard_metrics:
            scores[metric].append(s[metric])
        print("")

    print("Averages for entire directory: (", len(test_images) ,"items )")
    for metric in standard_metrics:
        scores[metric] = np.array(scores[metric])
        print(metric, np.mean(scores[metric], axis=0))
    visualize_confusion(f2, np.mean(scores["confusion"]))
    for metric in standard_metrics:
        print("Average", metric, np.mean(scores[metric]))
    print("Pixel accuracy", None)

from data_loaders.gt_loaders import autodiscover_suffix_to_class_map
if __name__ == "__main__":
    import sys
    test_folder = sys.argv[1]
    if not os.path.isdir(test_folder):
        test_folder = os.path.dirname(test_folder)
    print("Test folder", test_folder)
    suffix_to_class_map = autodiscover_suffix_to_class_map(test_folder, ["jpg", "png", "tif"])
    print("Inferred suffix to class map:", suffix_to_class_map)
    if len(sys.argv) > 2:
        gtthreshold = 0.5
        predthreshold = 0.5
        if len(sys.argv) > 3:
            predthreshold = float(sys.argv[3])
        if len(sys.argv) > 4:
            gtthreshold = float(sys.argv[4])
        if os.path.isfile(sys.argv[1]):
            f,f2 = sys.argv[1],sys.argv[2]
            print("Evaluating", f, f2)
            score_and_visualize_errors(f, f2, gtthreshold, predthreshold, suffix_to_class_map=suffix_to_class_map)
            print("")
        else:
            score_and_visualize_folders(sys.argv[1], sys.argv[2], gtthreshold, predthreshold, suffix_to_class_map=suffix_to_class_map)
    else:
        print("Usage: python evaluations.py gt_folder pred_folder [predthreshold] [gtthreshold]")
        print("With no arguments, runs unit tests.")
        if len(sys.argv) == 1:
            print("Running unit tests...")
            unittest.main(verbosity=2)
            cm = CachedMetrics(np.random.random((10,10,6)), np.ones((10,10,6)))
            cm = CachedMetrics(np.random.random((10,10,6)), np.random.random((10,10,6)))
            print(cm["recall"])
            print(cm["precision"])
            print(cm["f1_score"])
            print(cm["accuracy"])
            print(cm["intersection_over_union"])
            print(cm["foreground_accuracy"])
            # TO compute overlap versions of the metric, recompute with masked versions, only including in preds and gts
            # those areas that contain overlapped pixels.
        # Perform evaluation on the contents of the target folder.

# TODO: Implement DRD, MPM, PSNR, PPB.
def distance_reciprocal_distortion_metric(gt, pred):
    pass

def mask_image(img, layer_mask):
    pass

# Computes a reconstruction error metric for each class.
def masked_image_reconstruction_accuracy(gt, pred, img, error='mse'):
    return
