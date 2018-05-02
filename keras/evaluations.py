import numpy as np
from collections import defaultdict

# TODO: Move all of the different performance measures into their own function definitions?

def score(gt, pred, predthreshold=0.5, gtthreshold=0.5, additional_measures=[]):
    # We're going to threshold the Ground Truth (GT) to make it binary, so let's
    # save the original (potentially fuzzy for float-valued) preds and GT here:
    fuzzy_gt = gt.copy()
    fuzzy_pred = pred.copy()

    # Most evaluation measures assume that the ground truth and predictions are
    # binary. So we are putting them into that format:
    gt = np.greater(gt, gtthreshold).astype('float32')
    pred = np.greater(pred, predthreshold).astype('float32')

    # A few simple statistics:
    ct_gt_on_pixels = np.sum(np.greater(bin_gt, gtthreshold).astype('float32'))
    ct_gt_overlapped_pixels = np.sum(np.greater(np.sum(bin_gt, axis=-1), 1.0).astype('float32'))
    proportion_overlapping_gt_pixels = ct_gt_overlapped_pixels / max(1, ct_gt_on_pixels)


    get_num_overlapped_pixels = np.unique(np.sum(gt, axis=-1), 1.0).astype('float32')

    print("Scoring...")
    # Store each of the metrics by class.
    true_positives = defaultdict(lambda:0)
    false_positives = defaultdict(lambda:0)
    true_negatives = defaultdict(lambda:0)
    false_negatives = defaultdict(lambda:0)

    precisions = defaultdict(lambda:0)
    recalls = defaultdict(lambda:0)
    f_measure = defaultdict(lambda:0)
    accuracies = defaultdict(lambda:0)
    tot_gt_mass = defaultdict(lambda:0)
    fg_accuracies = defaultdict(lambda:0)
    # We don't want to have to compute 2^(2C) entries for confusion, since there are
    # this many pairs of unique class presence/absence bitstring pairs to be compared
    # (which even for small C, is large: for C=5, 2^(2C) = 1024 entries in the confusion matrix.)
    # Therefore, let's only consider a simpler type of confusion:
    # (1) We assume that most pixels do not overlap.
    # We can assume that at most, two types of content are present per image.
    # pseudo_confusion =

    overall_correct = []
    # TODO: Weighted F-measure is the harmonic mean of class-agnostic precision
    # and recall across all classes.
    # Average F-measure, on the other hand, is the average of all per-class
    # f-measures across all classes, uniformly weighted.

    # TODO add foreground accuracy!

    total_pixel_count = gt.shape[0] * gt.shape[1]
    tot_gt_mass = 0
    overall_correct = 0
    for c in range(pred.shape[-1]):
        gt_c = gt[:,:,c]
        pred_c = pred[:,:,c]
        true_positive_mass = true_positives[c]  = np.count_nonzero(np.multiply(gt_c, pred_c))
        false_positives[c] = np.count_nonzero(np.multiply(1.0-gt_c, pred_c))
        true_negatives[c]  = np.count_nonzero(np.multiply(1.0-gt_c, 1.0-pred_c))
        false_negatives[c] = np.count_nonzero(np.multiply(gt_c, 1.0-pred_c))
        gt_mass = np.count_nonzero(gt_c)
        tot_gt_mass += gt_mass
        pred_mass = np.count_nonzero(pred_c)
        union_mass = gt_mass + false_positives[c]
        overall_correct += true_positive_mass + true_negatives[c]

        print("Class:", c, "GT Mass:", gt_mass, "true_positive_mass:", true_positives[c], "pred mass:", pred_mass, "union mass:", union_mass)
        recall = float(true_positive_mass) / float(gt_mass) if gt_mass != 0 else 1
        precision = float(true_positive_mass) / float(pred_mass) if pred_mass != 0 else 1
        f_score = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = float(true_positive_mass) / float(union_mass) if union_mass != 0 else 1 if precision == recall == 1 else 0 # Accuracy in this case is equivalent to intersection over union.
        #print("Accuracy:", accuracy, true_positive_mass, union_mass)
        precisions[c] += precision
        recalls[c] += recall
        f_scores[c] += f_score
        accuracies[c] += accuracy
    #print("DONE scoring.")
    return precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct

# score_with_argmax allows only one prediction per pixel.
def score_batch_with_argmax(batch_gts, batch_preds, predthreshold=127, gtthreshold=127, measures=[]):
    #print("Scoring...")
    precisions = defaultdict(lambda:0)
    recalls = defaultdict(lambda:0)
    f_scores = defaultdict(lambda:0)
    accuracies = defaultdict(lambda:0)
    tot_gt_mass = defaultdict(lambda:0)
    overall_correct = []
    if len(batch_preds.shape) == 3:
        batch_preds = np.reshape(batch_preds, (1, batch_preds.shape[0], batch_preds.shape[1], batch_preds.shape[2]))#.astype('float32'
    if len(batch_gts.shape) == 3:
        batch_gts = np.reshape(batch_gts, (1, batch_gts.shape[0], batch_gts.shape[1], batch_gts.shape[2]))#.astype('float32')
    for b in range(preds.shape[0]):
        #global_pred = np.mean(np.mean(preds[b],   axis=0), axis=0)
        #global_gt   = np.mean(np.mean(batch_y[b], axis=0), axis=0)
        #digit_prediction = global_pred
        #print("Argmaxes:", np.argmax(digit_prediction), np.argmax(global_gt))
        predsheet = np.argmax(batch_preds[b], axis=2)
        gtsheet = np.argmax(batch_gts[b], axis=2)
        #print("Predsheet, GT sheet shape:", predsheet.shape, gtsheet.shape, np.max(predsheet), np.max(gtsheet))
        correct = np.mean(np.equal(predsheet, gtsheet).astype('float32'))
        overall_correct.append(correct)
        for c in range(preds.shape[-1]):
            predc = np.equal(c, predsheet).astype('float32')
            gtc = np.equal(c, gtsheet).astype('float32')
            pred_by_gt = predc * gtc
            print("pred_by_gt:", np.max(pred_by_gt), np.min(pred_by_gt), np.mean(pred_by_gt))
            #print(c, gtc.shape)
            true_positive = np.equal(1.0, pred_by_gt)#.astype('float32')
            #print(c, true_positive.shape)
            true_positive_mass = np.count_nonzero(true_positive)
            gt_mass = np.count_nonzero(gtc) #np.count_nonzero((0.0 < batch_y[b:,:,c]).astype('float32')) #np.count_nonzero(gtc)
            tot_gt_mass[c] += gt_mass
            pred_mass = np.count_nonzero(predc)
            union_mass = np.count_nonzero(np.equal(predsheet, c).astype('float32') + np.equal(gtsheet, c).astype('float32'))
            print("Class:", c, "GT Mass:", gt_mass, "true_positive_mass:", true_positive_mass, "pred mass:", pred_mass, "union mass:", union_mass)
            recall = float(true_positive_mass) / float(gt_mass) if gt_mass != 0 else 1
            precision = float(true_positive_mass) / float(pred_mass) if pred_mass != 0 else 1
            f_score = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = float(true_positive_mass) / float(union_mass) if union_mass != 0 else 1 if precision == recall == 1 else 0 # Accuracy in this case is equivalent to intersection over union.
            #print("Accuracy:", accuracy, true_positive_mass, union_mass)
            precisions[c] += precision
            recalls[c] += recall
            f_scores[c] += f_score
            accuracies[c] += accuracy
    #print("DONE scoring.")
    return precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct
