import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt

import Levenshtein
import editdistance
def cer(r, h):
    if len(r) == 0 and len(h) != 0:
        return 1.0
    #Remove any double or trailing
    r = u' '.join(r.split())
    h = u' '.join(h.split())

    return err(r, h)


def err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    return float(dis) / float(len(r))

def wer(r, h):
    r = r.split()
    h = h.split()

    return err(r,h)

def bleu_score_transcription(prediction, gt):
    references = [[list(gt)]]
    candidates = [list(prediction)]
    scores = []
    for max_ngram in range(1,50):
        score = corpus_bleu(references, candidates, weights=[1.0/max_ngram]*max_ngram)
        print("Max N-gram length:", max_ngram, "Cumulative BLEU Score:", score)
        scores.append(score)
    plt.clf()
    plt.plot(range(1,50), scores, label="Cumulative BLEU Score")
    #ax = plt.figure().add_subplot(111)
    plt.xlabel = "Maximum N-gram length"
    plt.ylabel = "Cumulative BLEU Score"
    plt.show()
    plt.savefig(sys.argv[1]+"_bleuscores.png")

def box_iou(b1, b2, generosity=100):
    #print("Being GENEROUS!")
    g=generosity
    box1 = (b1[0]-g, b1[1]-g, b1[2]+g, b1[3]+g)
    box2 = (b2[0]-g, b2[1]-g, b2[2]+g, b2[3]+g)
    left_intersection = max(box1[0], box2[0])
    right_intersection = min(box1[2], box2[2])
    top_intersection = max(box1[1], box2[1])
    bottom_intersection = min(box1[3], box2[3])
    #print left_intersection, right_intersection
    #print top_intersection, bottom_intersection
    h_intersection = bottom_intersection - top_intersection
    w_intersection = right_intersection - left_intersection
    #print h_intersection, w_intersection
    if h_intersection < 0 or w_intersection < 0:
        return 0.0
    else:
        a_intersection = h_intersection * w_intersection
        h1 = box1[3]-box1[1]
        w1 = box1[2]-box1[0]
        h2 = box2[3]-box2[1]
        w2 = box2[2]-box2[0]
        a1 = h1*w1
        a2 = h2*w2
        a_union = a1+a2-a_intersection
        #print a1, a2, a_union, a_intersection
        iou = float(a_intersection) / a_union
        return iou

# TODO: Build a textual precision (based on 1.0-insertions/N) and textual recall (based on 1.0-deletions/N) measure for
# text recognition. A third metric, also multiplied, could be textual confusion (based on 1.0-substitutions/N).
# Taking a 3-way harmonic mean of these might get a good estimate of how biased the model is towards precision, recall,
# and substitution.

import numpy as np
def align_boxes_cer(gt_boxes, gt_texts, pred_boxes, pred_texts, case_sense=True, include_punctuation=True):
    '''
    Returns a Character Error Rate (CER) in the range 0.0-1.0 based on the alignment of bounding
    boxes in the prediction against bounding boxes in the ground truth.

    Args:
        gt_boxes : a list or tuple of boxes specified by [left, top, right, bottom] integer pixel coordinates.
        gt_texts : a list of strings containing ground truth box transcriptions.
        pred_boxes : a list or tuple of boxes specified by [left, top, right, bottom] integer pixel coordinates.
        pred_texts : a list of strings containing predicted box transcriptions.

    Returns:
        Average Character Error Rate in range 0.0-1.0,
        Average Word-length-normalized Character Error Rate in range 0.0-1.0,
        List of per-word-pair Character Error Rates,
        List of matched bounding box indices as (gt, pred) index pairs,
        The prediction box indices that aligned to no ground truth box,
        The ground truth box indices that had no predictions aligning to them,
        A collection of all prediction box indices, sorted by decreasing error rate,
        A collection of all ground truth box indices, sorted by decreasing error rate.
    '''
    # Eliminate punctuation-only boxes.
    gt_boxdict = {i:gt_boxes[i] for i in range(len(gt_boxes))}
    pred_boxdict = {i:pred_boxes[i] for i in range(len(pred_boxes))}
    if not include_punctuation:
        import re
        pattern = re.compile('[\W]+')
        ngbd = {}
        for g,gt_box in gt_boxdict.iteritems():
            gt_text = gt_texts[g]
            print g, gt_text
            gt_text = pattern.sub('', gt_text)
            if len(gt_text) > 0:
                gt_texts[g] = gt_text
                ngbd[g] = gt_box
        print "Pruned items from gt boxdict:", len(gt_boxdict), len(ngbd)
        gt_boxdict = ngbd
        npbd = {}
        for p,pred_box in pred_boxdict.iteritems():
            pred_text = pred_texts[p]
            pred_text = pattern.sub('', pred_text)
            if len(pred_text) > 0:
                pred_texts[p] = pred_text
                npbd[p] = pred_box
        pred_boxdict = npbd
        print "Pruned items from pred boxdict:", len(pred_boxdict), len(npbd)

    #overlap_scores = np.zeros((len(gt_texts), len(pred_texts)))
    matched_idxs = {}
    overlap_ratio_to_pairs = []
    # Match all pairs greedily for the highest overlap scores.
    for gti, gtbox in gt_boxdict.iteritems():
        for pi, pbox in pred_boxdict.iteritems():
            overlap_ratio = box_iou(gtbox, pbox)
            #overlap_scores[gti,pi] = overlap_ratio
            overlap_ratio_to_pairs.append((overlap_ratio, (gti,pi)))
    not_taken_gti = set(i for i in gt_boxdict.keys())
    not_taken_pi = set(i for i in pred_boxdict.keys())
    taken_pi = set()
    taken_gti = set()
    paired_boxes = []
    paired_transcription_cers = []
    # Sort by largest overlap ratio, pair the corresponding transcriptions, and score.
    overlap_ratio_to_pairs = reversed(sorted(overlap_ratio_to_pairs, key=lambda x:x[0]))
    print("Matching and scoring box pairs...")

    for score, pair in overlap_ratio_to_pairs:
        if score == 0:
            break
        if pair[0] in taken_gti:
            continue
        elif pair[1] in taken_pi:
            continue
        gt = gt_texts[pair[0]]
        pred = pred_texts[pair[1]]
        if not case_sense:
            gt = gt.lower()
            pred = pred.lower()
        if not include_punctuation:
            gt = pattern.sub('', gt)
            pred = pattern.sub('', pred)
            if len(gt) == 0 and len(pred) == 0:
                continue
            if len(gt) == 0:
                print "WARNING! GT length is zero!", gt_texts[pair[0]], "pred is", pred_texts[pair[1]]
                # Throw it back!
                continue
        paired_boxes.append(pair)
        taken_gti.add(pair[0])
        taken_pi.add(pair[1])
        not_taken_gti.remove(pair[0])
        not_taken_pi.remove(pair[1])
        ptc = cer(gt, pred)
        paired_transcription_cers.append(ptc)
        print score, ptc, gt, pred
    print("Length of paired transcription CERS:", len(paired_transcription_cers))

    for pi in not_taken_pi:
        print "Unaligned pred:", pi, pred_texts[pi], pred_boxes[pi]

    for gti in not_taken_gti:
        print "Unaligned GT:", gti, gt_texts[gti], gt_boxes[gti]

    print "Aligned box pairs", len(paired_transcription_cers), "gt boxes", len(gt_boxdict), "pred boxes", len(pred_boxdict)
    #print "Unaligned GT boxes", len(not_taken_gti)
    #print "Unaligned Pred boxes", len(not_taken_pi)

    all_transcription_errors = list(paired_transcription_cers)
    all_transcription_error_weights = [float(len(gt_texts[p[0]])) for p in paired_boxes]
    weighted_cer = sum([all_transcription_errors[i]*all_transcription_error_weights[i] for i in range(len(all_transcription_errors))]) / sum(all_transcription_error_weights)
    #print "ATEW", len(all_transcription_error_weights)

    unaligned_pred_cers = [1.0 for i in not_taken_pi]
    unaligned_gt_cers = [1.0 for i in not_taken_gti]
    all_transcription_errors.extend(unaligned_pred_cers)
    all_transcription_errors.extend(unaligned_gt_cers)
    pred_weights = [float(len(pred_texts[i])) for i in not_taken_pi]
    gt_weights = [float(len(gt_texts[i])) for i in not_taken_gti]
    all_transcription_error_weights.extend(pred_weights)
    all_transcription_error_weights.extend(gt_weights)
    #print len(all_transcription_error_weights)
    weighted_all = sum([all_transcription_errors[i]*all_transcription_error_weights[i] for i in range(len(all_transcription_errors))]) / (sum(all_transcription_error_weights) - sum(pred_weights))

    total_predicted_chars = sum([len(pred_texts[i]) for i in pred_boxdict.keys()])
    total_gt_chars = sum([len(gt_texts[i]) for i in gt_boxdict.keys()])
    unaligned_pred_chars = sum(pred_weights)
    unaligned_gt_chars = sum(gt_weights)
    # Accumulate the unpaired boxes also as errors.
    return np.mean(paired_transcription_cers), weighted_cer, np.mean(all_transcription_errors), weighted_all, paired_transcription_cers, not_taken_pi, not_taken_gti, total_predicted_chars, total_gt_chars, unaligned_pred_chars, unaligned_gt_chars

if __name__ == "__main__":
    import json
    # Load the GT file as a Google Vision API JSON.
    # Load the Pred file in same format.
    print("Loading GT JSON file:", sys.argv[1])
    with open(sys.argv[1], 'r') as f:
        gt_json = json.load(f)
    print("Loading prediction JSON file:", sys.argv[2])
    with open(sys.argv[2], 'r') as f:
        pred_json = json.load(f)
    conf=0.2
    if len(sys.argv) > 3:
        conf = float(sys.argv[3])
    case_sense=True
    include_punctuation=True

    from OCR_google_vision_api import convert_google_api_json_to_bboxes_and_transcriptions
    gtbb, gtt, _, _ = convert_google_api_json_to_bboxes_and_transcriptions(gt_json, confidence_threshold=conf)
    prbb, prt, _, _ = convert_google_api_json_to_bboxes_and_transcriptions(pred_json, confidence_threshold=conf)
    avg_cer, weighted_cer, avg_all, weighted_all, cers, not_taken_pi, not_taken_gti, total_predicted_chars, total_gt_chars, unaligned_pred_chars, unaligned_gt_chars = align_boxes_cer(gtbb, gtt, prbb, prt, case_sense=case_sense, include_punctuation=include_punctuation)
    print("Unmatched Predictions:", len(not_taken_pi))
    print("Unmatched GT:", len(not_taken_gti))
    print("Total Predicted chars:", total_predicted_chars)
    print("Total GT chars:", total_gt_chars)
    print("Unmatched Predicted chars:", unaligned_pred_chars)
    print("Unmatched GT chars:", unaligned_gt_chars)
    print("Character accuracy:", 1.0-avg_cer)
    print("Matched, Word CER:", avg_cer)
    print("Matched, Document CER:", weighted_cer)
    print("All boxes, Word CER:", avg_all)
    print("All boxes, Document CER:", weighted_all)
