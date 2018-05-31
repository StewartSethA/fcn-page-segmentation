from __future__ import print_function
import numpy as np
import keras.backend as K
import keras
import tensorflow as tf

from collections import defaultdict

# TODO: Alternatively to f-measure, simply minimize the variance of the diff.
# This will help balance positives and negatives.
# False positives are where pred_argmax_masked > gt
# Softmax each pixel location, THEN do the continuous f-measure loss!!
# WITH margin!!!
# Precision + Recall + F_measure + MSE

# False negatives are where gt > pred_argmax_masked
# TP = 1 -
# Pseudo-f-measure penalty is the product of
# 1.0 - 2 * (1-continuous FN) * (1-continuous FP) /
# [(1-continuous FN) + (1-continuous FP)]

floattype='float32'#floattype
nc=4

# https://github.com/keras-team/keras/issues/3720
def gaussian(x, mu, sigma):
    return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

# https://github.com/keras-team/keras/issues/3720
def make_gaussian_kernel(sigma, num_input_channels=5, num_output_channels=5):
    global floattype
    # Kernel radius = 2*sigma, at least 3x3
    kernel_size = max(3, int(2*2*sigma+1))
    mean = np.floor(0.5*kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    # make 2D kernel using outer product of 1D kernel, leveraging dimensional symmetry
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
    # normalize the kernel
    kernel = np_kernel / np.sum(np_kernel)
    # Make into Identity channel->channel map for 4D convolution
    kernel = np.stack([kernel])
    kernel_4d = np.zeros((kernel_size, kernel_size, num_input_channels, num_output_channels), dtype=floattype)
    for i in range(num_input_channels):
        kernel_4d[:,:,i,i] = kernel
    return kernel_4d
    # TODO: Potentially could do kernels of varying widths....

# https://github.com/keras-team/keras/issues/3720
from keras.utils.generic_utils import get_custom_objects
def class_weighted_loss(internal_loss, weights=defaultdict(lambda:1.)):
    # Cheap way to heighten the loss values: Scale up both predictions and gt by sqrt of weight?
    # No, we actually need a per-class loss. (Why? Think what MSE does when logits > 1.)
    ## Then weight the outputs of these.
    # PYGO TODO Interface: Per-class loss functions. Then combining & weighting is trivial.
    # The mathematics for just about every loss function I know of are amenable to this.
    return

def blurred_loss(internal_loss, blur_sigma_pred=5.0, blur_sigma_gt=5.0, num_channels=5, weights=defaultdict(lambda:1.0)):
    print("Using Seth's Blurred loss preprocessor with weights", weights, "and internal loss function", internal_loss)
    p_kernel = make_gaussian_kernel(blur_sigma_pred, num_channels, num_channels)
    g_kernel = make_gaussian_kernel(blur_sigma_gt, num_channels, num_channels)
    blur_gt_op = lambda gt: K.conv2d(gt, g_kernel, padding='same')
    blur_pred_op = lambda gt: K.conv2d(gt, p_kernel, padding='same')
    return lambda gt, pred: apply_preop_loss(gt, pred, internal_loss, gt_ops=[blur_gt_op,], pred_ops=[blur_pred_op,])

def apply_preop_loss(gt, pred, internal_loss, gt_ops=[], pred_ops=[], weights=defaultdict(lambda:1.0)):
    for op in gt_ops:
        gt = op(gt)
    for op in pred_ops:
        pred = op(pred)
    return internal_loss(gt, pred, weights=weights)

def continuous_recall(gt, pred):
    # Recall: Only counts where GT is true. How much of it did we get?
    return keras.layers.multiply([gt, K.clip(pred, 0., 1.)])

def continuous_precision(gt, pred):
    # Precision: Only counts against us where GT is false. How much of false positive is there?
    #return K.max(gt)-keras.layers.multiply([K.max(gt)-gt, 1.0-K.clip(pred, 0., 1.)])
    return 1.0-keras.layers.multiply([1.0-gt, 1.0-K.clip(pred, 0., 1.)])

def continuous_f_measure(weights=defaultdict(lambda:1.)):
    print("Using Seth's Continuous F-Measure loss with weights", weights)
    return lambda gt,pred:continuous_f_measure_loss(gt, pred, weights)

def continuous_f_measure_loss(gt, pred, weights=defaultdict(lambda:1.)):
    global floattype
    gt = K.cast(gt, dtype=floattype)
    # Simply multiply predicted pseudo-probabilities against GT to get pseudo-recall and pseudo-precision.
    total_loss = K.variable(0.0, dtype=floattype)
    if len(weights) == 0:
        cr = continuous_recall(gt, pred)
        cp = continuous_precision(gt, pred)
        f1_score_approx = keras.layers.multiply([cr, cp])
        return 1.0-f1_score_approx
    global floattype
    for c in range(len(weights)):
        gtc = K.cast(gt[:,:,:,c], floattype)
        pdc = pred[:,:,:,c]
        cr = continuous_recall(gtc, pdc)
        cp = continuous_precision(gtc, pdc)
        f1_score_approx = keras.layers.multiply([cr, cp])
        class_loss = weights[c] * (1.0 - f1_score_approx)
        total_loss = total_loss + K.mean(class_loss)
        sumweights += weights[c]

    #h,w = K.shape(gt)[1], K.shape(gt)[2]
    #total_loss = total_loss / K.cast((h*w), dtype=floattype)
    return total_loss / sumweights

def blurred_continuous_f_measure(weights=defaultdict(lambda:1.), sigma=5.0, num_classes=5):
    return blurred_loss(continuous_f_measure_loss, weights=weights, blur_sigma_gt=sigma, blur_sigma_pred=sigma, num_channels=num_classes)

get_custom_objects().update({"blurred_continuous_f_measure": blurred_continuous_f_measure})
get_custom_objects().update({"continuous_f_measure": continuous_f_measure})

def get_per_class_margin(weights=defaultdict(lambda:1.)):
    print("Using Get Per-class Margin loss")
    return lambda gt,pred: per_class_margin(gt, pred, weights=weights)

def per_class_margin(gt, pred, num_classes=nc, weights=defaultdict(lambda:1.)):
    if len(weights) == 1:
        weights = weights * num_classes
    global floattype
    margin_plus = 0.9
    margin_neg  = 0.1
    lmbda = 0.5
    loss = K.variable(0.0, dtype=floattype)
    for c in range(num_classes):
        # Added mean reduction.
        loss_piece = K.mean(keras.layers.multiply([K.cast(gt[:,:,:,c], floattype), K.square(keras.layers.maximum([0.1*gt[:,:,:,c], margin_plus - pred[:,:,:,c]]))]) + lmbda * keras.layers.multiply([K.cast(0.9 - gt[:,:,:,c], floattype), K.square(keras.layers.maximum([0.1*gt[:,:,:,c], pred[:,:,:,c]-margin_neg]))]))
        loss_piece = loss_piece * weights[c]
        loss = loss + loss_piece

    return loss
get_custom_objects().update({"per_class_margin": per_class_margin})

def masked_per_class_margin(gt, pred, num_classes=nc):
    global floattype
    margin_plus = 0.9
    margin_neg  = 0.1
    lmbda = 0.5
    loss = K.variable(0.0)

    gt_mask = K.sum(gt, axis=-1) * 255.0
    gt_mask = K.clip(gt_mask, 0, 1)
    gt_mask = K.expand_dims(gt_mask, axis=-1)
    gt_mask = K.tile(gt_mask, (1,1,1,num_classes))

    gt_mask_by_pred = keras.layers.multiply([gt_mask, pred])

    for c in range(num_classes):
        loss_piece = keras.layers.multiply([K.cast(gt[:,:,:,c], floattype), K.square(keras.layers.maximum([0.0*gt[:,:,:,c], margin_plus - gt_mask_by_pred[:,:,:,c]]))]) + lmbda * keras.layers.multiply([K.cast(1.0 - gt[:,:,:,c], floattype), K.square(keras.layers.maximum([0.0*gt[:,:,:,c], gt_mask_by_pred[:,:,:,c]-margin_neg]))])
        loss = loss + loss_piece
    return loss

def masked_mse(gt, pred, num_classes=nc):
    gt_mask = K.sum(gt, axis=-1) * 255.0
    gt_mask = K.clip(gt_mask, 0, 1)
    gt_mask = K.expand_dims(gt_mask, axis=-1)
    gt_mask = K.tile(gt_mask, (1,1,1,num_classes))

    gt_mask_by_pred = keras.layers.multiply([gt_mask, pred])
    gt_pred_diff = gt - gt_mask_by_pred
    mse = K.mean(K.square(gt_pred_diff))
    return mse

def pseudo_f_measure_loss_more(gt, pred, num_classes=nc):
    global floattype
    gt_mask = K.sum(gt, axis=-1) * 255.0
    gt_mask = K.clip(gt_mask, 0, 1)
    gt_mask = K.expand_dims(gt_mask, axis=-1)
    gt_mask = K.tile(gt_mask, (1,1,1,num_classes)) #K.get_shape(gt.shape)[-1]))
    #gt_mask = K.reshape(gt_mask, (gt.shape[0], gt_mask.shape[1], num_classes))

    margin = 0.5 # TODO: Figure out what we can do with a margin.
    gt_mask_by_class = K.clip(gt * 255.0, 0, 1)
    pred_clipped = K.clip(pred, 0, 1)
    gt_true = gt_mask_by_class
    gt_false = 1.0 - gt_true
    #pseudo_recall = gt_true * pred_clipped # This will be mean=1 iff recall=1. It is continuous and increasingly monotonic w.r.t. recall.
    #pseudo_precision = 1.0-(gt_false * pred_clipped) # This will be mean=1 iff precision=1. It is continuous and increasingly monotonic w.r.t. precision.

    '''
    MARGIN stuff.
    '''
    margin_plus = 0.9
    margin_neg  = 0.1
    lmbda = 0.5
    loss = K.variable(0.0)
    for c in range(num_classes):
        loss_piece = keras.layers.multiply([K.cast(gt[:,:,:,c], floattype), K.square(keras.layers.maximum([0.0*gt[:,:,:,c], margin_plus - pred[:,:,:,c]]))]) + lmbda * keras.layers.multiply([K.cast(1.0 - gt[:,:,:,c], floattype), K.square(keras.layers.maximum([0.0*gt[:,:,:,c], pred[:,:,:,c]-margin_neg]))])
        loss = loss + loss_piece

    '''
    END margin stuff.
    '''

    pseudo_tp = pred - margin

    pseudo_correct = gt_true * pred_clipped

    pred_gt_diff = gt_mask_by_class - pred #np.less()

    deviations_from_1 = pred_gt_diff * gt_mask_by_class
    deviations_from_zero = pred_gt_diff * (1.0-gt_mask_by_class)
    # Squish down to one layer
    # TODO: Bad to flatten classes... (WIP)

    # TODO: THis is counting totals. Is it not taking into account the (just_right)
    # pixels in the WRONG locations?????
    #pred_gt_diff = np.mean(pred_gt_diff, axis=-1) #Conflates TP of one with FP of another... Hmmm...
    pred_too_low_fn_mask = K.cast(K.greater(pred_gt_diff, 0.0), floattype)
    pred_too_low_fn = keras.layers.multiply([pred_too_low_fn_mask, pred_gt_diff]) # Make it not just a 0-1 mask, but a float representing the degree of deviation. BUT it's going to be misclassified either way... SO add a margin!!!
    pred_too_high_fp_mask = K.cast(K.less(pred_gt_diff, 0.0),floattype)
    pred_too_high_fp = keras.layers.multiply([pred_too_high_fp_mask, pred_gt_diff])
    # TRY 2 Margin version:
    pred_just_right_tp = keras.layers.multiply([1.0 - K.clip(K.abs(pred_gt_diff)*gt_mask_by_class, 0, 1), gt_mask])

    # TRY 3 Margin and counts version:
    #pred_just_right_tp = keras.layers.multiply([gt_true, K.cast(K.greater(pred - margin - gt_mask_by_class, 0.0), floattype)])
    # True positives: They are true, and we called them true.
    pred_just_right_tp = keras.layers.multiply([gt_true, pred]) # The reward is proportional to our confidence that it was true when it was true.
     # - margin - gt_mask_by_class])
    # False negatives: They are true, but we called them false.
    pred_too_low_fn = keras.layers.multiply([gt_true, 1.0-pred_clipped]) # The penalty is proportional to how much we underestimated it.
    # False positives: They are false, but we called them true. # The penalty is proportional to how confident we were it was true when it was false.
    pred_too_high_fp = keras.layers.multiply([1.0-gt_true, pred, gt_mask])

    #pred_just_right_tp = keras.layers.multiply([1.0 - K.clip(K.abs(pred_gt_diff), 0, 1), gt_mask]) # TODO: Needs to be masked by "TRUE" pixels (GT)

    # Compute class-wise statistics
    macro_pseudo_precision = K.variable(0.0)
    macro_pseudo_recall = K.variable(0.0)
    macro_pseudo_fmeasure = K.variable(0.0)
    total_weight = 0.0
    product_of_fmeasures = K.variable(0.0)
    eps = 0.00000001
    print("PSEUDO-STATISTICS")
    for classnum in range(num_classes):
        ptl_fn_c = pred_too_low_fn[:,:,:,classnum]
        pth_fp_c = pred_too_high_fp[:,:,:,classnum]
        pjr_tp_c = pred_just_right_tp[:,:,:,classnum]



        surrogate_tp = pjr_tp_c
        surrogate_fp = pth_fp_c
        surrogate_fn = ptl_fn_c
        surrogate_tp_quantity = K.sum(K.abs(surrogate_tp))
        surrogate_fp_quantity = K.sum(K.abs(surrogate_fp))
        surrogate_fn_quantity = K.sum(K.abs(surrogate_fn))
        surrogate_pred_mass = (surrogate_tp_quantity + surrogate_fp_quantity)
        surrogate_true_mass = (surrogate_tp_quantity + surrogate_fn_quantity)
        surrogate_precision = surrogate_tp_quantity / (surrogate_pred_mass + eps)
        surrogate_recall = surrogate_tp_quantity / (surrogate_true_mass + eps)

        # LASTDAY
        #surrogate_precision = K.mean(pseudo_precision[:,:,:,classnum])
        #surrogate_recall = K.mean(pseudo_recall[:,:,:,classnum])

        surrogate_f_measure = 2 * surrogate_precision * surrogate_recall / (surrogate_precision + surrogate_recall + eps)

        # Compute the product of all f measures to force the penalty high for missing a single class
        product_of_fmeasures *= (0.5 + surrogate_f_measure)

        # Square each term to make the penalty nonlinear.
        surrogate_f_measure = surrogate_f_measure * surrogate_f_measure

        # Experimental class re-weighting for precision and recall
        weight = 1.0
        if classnum >= 2:
            weight = 8.0
            surrogate_f_measure *= weight
        total_weight += weight

        macro_pseudo_precision += surrogate_precision
        macro_pseudo_recall += surrogate_recall
        macro_pseudo_fmeasure += surrogate_f_measure

    macro_pseudo_precision /= total_weight
    macro_pseudo_recall /= total_weight
    macro_pseudo_fmeasure /= total_weight

    # Now compute overall
    surrogate_tp = pred_just_right_tp
    surrogate_fp = pred_too_high_fp
    surrogate_fn = pred_too_low_fn
    surrogate_tp_quantity = K.sum(K.abs(surrogate_tp))
    surrogate_fp_quantity = K.sum(K.abs(surrogate_fp))
    surrogate_fn_quantity = K.sum(K.abs(surrogate_fn))
    surrogate_pred_mass = (surrogate_tp_quantity + surrogate_fp_quantity)
    surrogate_true_mass = (surrogate_tp_quantity + surrogate_fn_quantity)
    surrogate_precision = surrogate_tp_quantity / (surrogate_pred_mass + eps)
    surrogate_recall = surrogate_tp_quantity / (surrogate_true_mass + eps)
    surrogate_f_measure = 2 * surrogate_precision * surrogate_recall / (surrogate_precision + surrogate_recall + eps)
    micro_pseudo_fmeasure = surrogate_f_measure

    #print("Macro Pseudo-Precision:", macro_pseudo_precision)
    #print("Macro Pseudo-Recall:", macro_pseudo_recall)
    #print("Macro Pseudo-F-Measure:", macro_pseudo_fmeasure)
    return 2.0 - macro_pseudo_fmeasure - micro_pseudo_fmeasure, macro_pseudo_fmeasure, micro_pseudo_fmeasure, macro_pseudo_recall, macro_pseudo_precision, product_of_fmeasures, surrogate_recall, surrogate_precision, surrogate_true_mass, surrogate_pred_mass, surrogate_fn_quantity, surrogate_tp_quantity, surrogate_fp_quantity, K.sum(gt_mask), K.mean(K.abs(pred_gt_diff)), K.mean(K.square(pred_gt_diff)), K.sum(pred_too_low_fn_mask), K.sum(pred_too_high_fp_mask), K.sum(pred_just_right_tp), K.shape(gt_mask)

def pseudo_f_measure_loss(gt, pred, num_classes=nc):
    # Confusion weight matrix.'

    # THese weights should be learned too, according to the loss function.
    weights = np.ones((6,6))
    '''
    weights[0,:] = 30.0
    weights[:,0] = 30.0
    weights[0,0] = 1.0
    weights[1,:] = 4.0
    weights[:,1] = 4.0
    weights[1,1] = 1.0
    weights[2,:] = 5.0
    weights[:,2] = 5.0
    weights[2,2] = 1.0
    weights[3,:] = 6.0
    weights[:,3] = 6.0
    weights[3,3] = 1.0
    weights[4,:] = 3.0
    weights[:,4] = 3.0
    weights[4,4] = 1.0
    weights[5,:] = 3.0
    weights[:,5] = 3.0
    weights[5,5] = 1.0
    '''
    weights = np.ones((6,))
    weights[0] = 1.0
    weights[2] = 2.0
    weights[3] = 3.0
    #w_ce = weighted_pixelwise_crossentropy(gt, pred, weights=weights)


    w_ce = weighted_pixelwise_crossentropy(weights)(gt, pred)

    basic_loss, macro_pseudo_fmeasure, micro_pseudo_fmeasure, macro_pseudo_recall, macro_pseudo_precision, product_of_fmeasures, surrogate_recall, surrogate_precision, surrogate_true_mass, surrogate_pred_mass, surrogate_fn_quantity, surrogate_tp_quantity, surrogate_fp_quantity, gt_mask_sum, sum_abs_diff, sum_squared_diff, sum_too_low, sum_too_high, sum_just_right, gt_mask_shape = pseudo_f_measure_loss_more(gt, pred, num_classes=nc)
    return 0.1*(1.0-macro_pseudo_precision * macro_pseudo_recall) + 0.1*w_ce + per_class_margin(gt, pred) #(gt, pred) #- micro_pseudo_fmeasure +  # - micro_pseudo_fmeasure #+ pixelwise_crossentropy(gt, pred) #3.0-macro_pseudo_recall-macro_pseudo_precision-macro_pseudo_precision*macro_pseudo_recall + pixelwise_crossentropy(gt, pred) # pixelwise_crossentropy(gt, pred) - 1.0*K.sqrt(macro_pseudo_precision) - macro_pseudo_precision*macro_pseudo_recall #0.01*((2.0 - macro_pseudo_fmeasure - micro_pseudo_fmeasure +0.5 - product_of_fmeasures)/5.0 - macro_pseudo_precision/2.0 + macro_pseudo_recall / 4.0) + 0.01*pixelwise_crossentropy(gt, pred) - macro_pseudo_precision * macro_pseudo_recall - 0.2*macro_pseudo_precision - K.sqrt(K.sqrt(0.01 + product_of_fmeasures))
    #(10.0 - 7.0*macro_pseudo_fmeasure - 3.0*micro_pseudo_fmeasure + 5.0*pixelwise_crossentropy(gt, pred))/2.0# - 3.0*product_of_fmeasures)/1.0 #+ sum_squared_diff + sum_abs_diff # - K.sum(gt*pred) / (K.sum(K.clip(gt+pred,1,0))) # Differentiable IoU score.

import tensorflow as tf
def pixelwise_crossentropy(y_true, y_pred):
    global floattype
    epsilon = 0.000000001

    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    weights = K.constant([1.0, 2.0, 3.0, 4.0, 8.0, 10.0], dtype=floattype)

    y_pred /= tf.reduce_sum(y_pred, axis=-1, keep_dims=True) # Softmax it now.
    #pixel_losses = -tf.reduce_mean(K.dot(weights, y_true) * tf.log(y_pred))
    pixel_losses = -tf.reduce_mean(y_true * tf.log(y_pred))
    return pixel_losses

# https://github.com/fchollet/keras/issues/5916 Thanks to dluvzion!
keras.losses.pseudo_f_measure_loss = pseudo_f_measure_loss
keras.losses.pixelwise_crossentropy = pixelwise_crossentropy

# TODO: Force zero-mean filters for text rec
def materialize_loss(model, np_input, np_gt):
    tensor_losses = pseudo_f_measure_loss_more(K.variable(np_gt), K.variable(model.predict(np_input)))
    return [K.eval(tensor_loss) for tensor_loss in tensor_losses]

    #input_tensors = [model.inputs[0], model.sample_weights[0], model.targets[0], K.learning_phase()]
    #keras_function_get_loss = K.function(inputs=input_tensors, outputs=pseudo_f_measure_loss(outputs)
    #inputs = [np_input, np.ones((last_input.shape[0])), np_gt, 0]
    #np_gradients = keras_function_get_loss(inputs)



# Custom loss function with costs
def weighted_categorical_crossentropy(y_true, y_pred, weights):
    global floattype
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max),dtype=floattype)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


# https://github.com/fchollet/keras/issues/6261
def weighted_pixelwise_crossentropy(class_weights):

    def loss(y_true, y_pred):
        _EPSILON = 0.00000001
        epsilon = K.constant(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_mean(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss


#import tensorflow.Tensor as T
def f_measure_loss(y_true, y_pred):
    #y_true = tf.cast(y_true, tf.bool)
    print("Using F-measure loss!")
    print(y_true, y_pred)
    #auc = tf.contrib.metrics.streaming_auc(y_true, y_pred)

    #return 1-auc[0]
    argmax_prediction = tf.argmax(y_pred, -1)
    argmax_y = tf.argmax(y_true, -1)

    TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
    TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
    FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
    FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    #f1_score = tf.multiply(2,  tf.multiply(precision, recall / (precision + recall)
    return 2 - precision - recall
    #true_positive = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    #true_positive =
    #y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    #y_pred /= y_pred.sum(axis=-1, keepdims=True)
    #cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    #return cce


#def pixel_cross_entropy(y_true, y_conv):
#    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv)
