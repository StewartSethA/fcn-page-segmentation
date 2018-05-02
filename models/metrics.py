import numpy as np
import keras.backend as K
from collections import defaultdict

# TODO: Write Confusion Matrix display together with accs.

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def single_class_accuracy(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_preds, 3), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def foreground_accuracy(y_true, y_pred, threshold=0.5):
    # Ignores background pixels in accuracy computation.
    # We care about pixels that may have multiple classes,
    # And we do not penalize any correct classes.
    # All we'd have to do is predict all 1s to get this right, though...
    # So this is equivalent to recall of true pixels.
    return K.sum(K.multiply(y_true, K.cast(K.equal(y_true,y_pred), K.floatx())))/K.sum(y_true)

def background_accuracy(y_true, y_pred):
    # Ignores foreground pixels in accuracy computation.
    # We care about pixels that may have multiple classes,
    # And we do not penalize any correct classes.
    # All we'd have to do is predict all 1s to get this right, though...
    return K.sum(K.multiply(1.0-y_true, K.cast(K.equal(y_true,y_pred), K.floatx())))/K.sum(y_true)

def score(batch_y, preds, predthreshold=127, gtthreshold=127):
    print("Scoring...")
    precisions = defaultdict(lambda:0)
    recalls = defaultdict(lambda:0)
    f_scores = defaultdict(lambda:0)
    accuracies = defaultdict(lambda:0)
    tot_gt_mass = defaultdict(lambda:0)
    overall_correct = []
    if len(preds.shape) == 3:
        preds = np.reshape(preds, (1, preds.shape[0], preds.shape[1], preds.shape[2]))#.astype('float32')
    if len(batch_y.shape) == 3:
        batch_y = np.reshape(batch_y, (1, batch_y.shape[0], batch_y.shape[1], batch_y.shape[2]))#.astype('float32')
    for b in range(preds.shape[0]):
        #global_pred = np.mean(np.mean(preds[b],   axis=0), axis=0)
        #global_gt   = np.mean(np.mean(batch_y[b], axis=0), axis=0)
        #digit_prediction = global_pred
        #print("Argmaxes:", np.argmax(digit_prediction), np.argmax(global_gt))
        predsheet = np.argmax(preds[b], axis=2)
        gtsheet = np.argmax(batch_y[b], axis=2)
        print("Predsheet, GT sheet shape:", predsheet.shape, gtsheet.shape, np.max(predsheet), np.max(gtsheet))
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
    print("DONE scoring.")
    return precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct
