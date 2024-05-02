from __future__ import print_function
import matplotlib.pyplot as plt
#plt.ion()
import numpy as np
import cv2
import json
from keras.callbacks import Callback
import keras
from collections import defaultdict
import time
from postprocessing.postprocessing import postprocess_preds, multihot_to_multiindexed_rgb
from visuals.visuals import vis_img as show
from evaluation.evaluations import score
from evaluation.evaluations import CachedMetrics
import psutil
import shutil
import os
#from keras_visualize_activations.read_activations import *

# Pixels per second logging too!!!

def visualize_weights(model, screen_height, screen_width, batch_num=0):
    weights = model.get_weights()
    x,y = 0,0
    for l, layer_weights in enumerate(weights):
        lws = layer_weights.shape
        if len(lws) == 4:
            # H, W, Cin, Cout
            # CNN filter visualization
            display_image = np.zeros((lws[0]*lws[2], lws[1]*lws[3]))
            formatted_display_image = np.zeros((lws[0]*lws[2], lws[1]*lws[3], 3))
            h,w=lws[:2]
            for fin in range(lws[2]):
                for fout in range(lws[3]):
                    lw = layer_weights[:,:,fin,fout]
                    formatted_lw = np.zeros((lw.shape[0], lw.shape[1], 3))
                    mx = np.max(lw)
                    mn = np.min(lw)
                    std = np.std(display_image)
                    #print("  Filter ", fin, "x", fout, " std:", std, "maxmin:", mx, mn, "mean:", np.mean(lw))
                    # Normalize and display each filter, colorized by negative, positive weights and magnitude
                    mag = (lw - mn) / (mx-mn)
                    posmag = lw / mx
                    negmag = lw / mn
                    formatted_lw[:,:,0] = std
                    formatted_lw[:,:,1] = np.multiply(posmag, np.greater(lw, 0.0).astype('float32'))
                    formatted_lw[:,:,2] = np.multiply(negmag, np.less(lw, 0.0).astype('float32'))
                    formatted_display_image[h*fin:h*(fin+1),w*fout:w*(fout+1),:] = formatted_lw
                    display_image[h*fin:h*(fin+1),w*fout:w*(fout+1)] = lw
            if display_image.shape[0] < 600 and display_image.shape[1] < 800:
                # Scale up 4x
                display_image = cv2.resize(display_image, (0,0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            # Display positive weights as green, negative as red, and overall magnitude as blue channel.
            di = np.zeros((display_image.shape[0], display_image.shape[1], 3))
            display_image = np.sqrt(np.abs(display_image)) * np.sign(display_image)
            std = np.std(display_image)

            #normalized_image = display_image / (max(0.0000000000000001, std / 2))
            mx = np.max(display_image)
            mn = np.min(display_image)
            if batch_num % 100 == 0:
                print("  Layer std:", std, "maxmin:", mx, mn, "mean:", np.mean(display_image))
            mag = (display_image - mn) / (mx-mn)
            posmag = display_image / mx
            negmag = display_image / mn
            di[:,:,0] = std
            di[:,:,1] = np.multiply(posmag, np.greater(display_image, 0.0).astype('float32'))
            di[:,:,2] = np.multiply(negmag, np.less(display_image, 0.0).astype('float32'))

            cv2.imshow('Layer '+str(l), di)
            '''
            cv2.resizeWindow('Layer '+str(l), display_image.shape[0], display_image.shape[1])
            y += display_image.shape[0]
            if y >= screen_height:
                y = 0
                x += display_image.shape[1]
            cv2.moveWindow('Layer' +str(l), x, y)
            '''
            cv2.imwrite('filters/Layer'+str(l)+"_batchnum"+str(batch_num)+".png", di*255)
            pass
        if batch_num < 100:
            print("Layer weights shape:", l, lws)
    cv2.waitKey(10)

class SaveEveryEpochCallback(Callback):
    def __init__(self, model, model_save_path, interval=1):
        self.interval = interval
        #self.model = model
        self.set_model(model)
        if model_save_path[-3:] != ".h5":
            model_save_path = model_save_path + ".h5"
        self.model_save_path = model_save_path
        self.epoch = 0
    
    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.interval == 0:
            print("Saving model at epoch", self.epoch+1, epoch)
            filepath = self.model_save_path.replace(".h5", "_" + str(self.epoch) + ".h5")
            self.model.save(filepath)
        self.epoch = self.epoch + 1

import visuals
class DisplayTrainingSamplesCallback(Callback):
    def __init__(self, training_generator_class, model=None, interval=10, log_dir=None, dac=None):
        self.training_generator = training_generator_class
        self.interval = interval
        self.batch_num = -2
        #self.model = model
        self.set_model(model)
        self.log_dir = log_dir
        self.pixels_per_class_per_sample = defaultdict(list)
        self.historic_weighted_thresholds = defaultdict(list)
        self.dac = dac

    def on_batch_end(self, batch, logs={}):
        self.batch_num += 1
        if self.batch_num % self.interval == 0:
            print("DisplayTrainingSamplesCallback")
            if self.training_generator.gt is not None:
                #plt.figure("Training_GT")
                #plt.clf()
                v = show(self.training_generator.gt, bgr=False)
                #plt.imshow((v), interpolation='none')
                if self.log_dir is not None:
                    cv2.imwrite(os.path.join(self.log_dir, "TrainGT.png"), v*255)
                    #plt.savefig(os.path.join(self.log_dir, "TrainGT.png"))
            if self.training_generator.image is not None:
                #plt.figure("Training_Image")
                #plt.clf()
                #plt.imshow((self.training_generator.image), interpolation='none')
                if self.log_dir is not None:
                    pass
                    #plt.savefig(os.path.join(self.log_dir, "TrainImage.png"))
                if self.model is not None:
                    im = self.training_generator.image
                    gt = self.training_generator.gt
                    im = np.pad(im, ((0,gt.shape[0]-im.shape[0]),(0,gt.shape[1]-im.shape[1]),(0,0)), mode='constant', constant_values=0)
                    #plt.figure("Training_Prediction")
                    #plt.clf()
                    if self.log_dir is not None:
                        cv2.imwrite(os.path.join(self.log_dir, "Train.jpg"), show(im, bgr=False)*255)
                    print("DisplayTrainingSamplesCallback input shape", im.shape)
                    preds = self.model.predict(np.reshape(im, [1,im.shape[0],im.shape[1],im.shape[2]]))
                    self.pred = preds[0]
                    print("Pred shape:", self.pred.shape)
                    print("Pred min/max:", np.min(self.pred), np.max(self.pred))
                    #plt.imshow((show(self.pred, bgr=True)), interpolation='none')
                    if self.log_dir is not None:
                        cv2.imwrite(os.path.join(self.log_dir, "TrainPred.jpg"), show(self.pred, bgr=False)*255)
                        #plt.savefig(os.path.join(self.log_dir, "TrainPred.png"))
                    #plt.pause(0.001)

                    # Find best per-class thresholds as well.
                    best_fscores = defaultdict(lambda:0)
                    best_thresholds = {}
                    step = 0.05
                    gt = gt.astype('float32')
                    gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1], gt.shape[2]))
                    preds = preds.astype('float32')
                    print("GT shape:", gt.shape, "Preds shape:", preds.shape)
                    print("Computing TRAINING F1-scores, precisions, recalls, etc.!")
                    for thresh in np.arange(step,1.0,step):
                        predsthresh = postprocess_preds(im, preds, gt, None, None, thresh=defaultdict(lambda:thresh))
                        cm = CachedMetrics(gt[0], predsthresh[0])
                        precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct = cm["precision"], cm["recall"], cm["accuracy"], cm["f1_score"], cm["gt_mass"], cm["true_positives"]
                        for classnum in range(gt.shape[-1]):
                            if f_scores[classnum] > best_fscores.get(classnum, 0):
                                best_fscores[classnum] = f_scores[classnum]
                                best_thresholds[classnum] = thresh
                    # Weight by number of pixels of each...
                    # In the last sample of 100 or so?
                    self.best_thresholds = best_thresholds
                    pix_counts_per_class = np.sum(gt, axis=(0,1,2))
                    self.weighted_best_thresholds = {c:pix_counts_per_class[c]*best_thresholds[c] for c in best_thresholds.keys()}
                    for c in best_thresholds.keys():
                        if pix_counts_per_class[c] > 0:
                            self.pixels_per_class_per_sample[c].append(pix_counts_per_class[c])
                            self.historic_weighted_thresholds[c].append(self.weighted_best_thresholds[c])
                            if len(self.pixels_per_class_per_sample[c]) > 30:
                                del self.pixels_per_class_per_sample[c][0]
                                del self.historic_weighted_thresholds[c][0]
                    self.historic_averaged_best_thresholds = {c:(np.mean(self.historic_weighted_thresholds.get(c, [0.5]))/np.sum(self.pixels_per_class_per_sample.get(c, [1.0,]))) for c in range(preds.shape[-1])}
                    print("Best historic averaged thresholds:", self.historic_averaged_best_thresholds)
                    if self.dac is not None:
                        print("Updating Validation to use best thresholds computed on training set...")
                        self.dac.thresholds = self.historic_averaged_best_thresholds



#if (self.iteration+1)%self.display_interval == 0:
#ls,train_accuracy,logits = self.sess.run([self.loss, self.accuracy, self.y_conv], feed_dict={self.x:1.0-batch[0], self.y_:batch[1], self.keep_prob: 1.0})
#print(ls, train_accuracy)
#print(("Step %d, loss %g, training accuracy %g"%(self.iteration, ls, train_accuracy))#, "logits:", logits))

class DisplayActivationsCallback(Callback):
    def __init__(self, model=None, input_generator=None):
        #self.model = model
        self.set_model(model)
        self.input_generator = input_generator
        self.display_interval = 50
        self.batch_num = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_num += 1
        if self.batch_num % self.display_interval != 0:
            return
        img = self.input_generator.next()[0]
        activations = get_activations(self.model, img)
        display_activations(activations)

import subprocess
def get_screen_resolution():
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return {'width': resolution[0], 'height': resolution[1]}

class VisualizeWeightsCallback(Callback):
    def __init__(self, model=None, display_interval=10):
        self.set_model(model)
        self.display_interval = display_interval
        self.batch_num = 0
        res = get_screen_resolution()
        self.screen_width = res['width']
        self.screen_height = res['height']

    def on_batch_end(self, batch, logs={}):
        self.batch_num += 1
        if self.batch_num % self.display_interval != 0:
            return
        visualize_weights(self.model, self.screen_height, self.screen_width, self.batch_num)

def normalize_weights(model, batch_num):
    new_weights = []
    sumabses = []
    for l,layer_weights in enumerate(model.get_weights()):
        new_weights.append(layer_weights)
        if len(layer_weights.shape) == 1:
            layer_weights = np.minimum(np.maximum(layer_weights, -1.0), 1.0)
            new_weights[-1] = layer_weights
            continue # Skip bias weights.
        # Normalize all convolutional kernels!
        if len(layer_weights.shape) == 4:
            for feature in range(layer_weights.shape[-1]):
                sumabs = np.sum(np.abs(layer_weights[:,:,:,feature]))
                #print("Sumabs weights for layer", l, "feature", feature, ":", sumabs)
                sumabses.append(sumabs)
                layer_weights = np.multiply(layer_weights, 1.0/max(0.00000000001, sumabs))
                new_weights[-1] = layer_weights
    print("Average sumabs, min, max, std:", np.mean(sumabs), np.min(sumabs), np.max(sumabs), np.std(sumabs))
    print("Setting normalized weights...")
    model.set_weights(new_weights)
    print("Done.")

class WeightNormalizationCallback(Callback):
    def __init__(self, model=None, normalization_interval=100, max_normalization_batchnum=200):
        self.set_model(model)
        self.normalization_interval = normalization_interval
        self.batch_num = 0
        self.max_normalization_batchnum = max_normalization_batchnum

    def on_batch_end(self, batch, logs={}):
        self.batch_num += 1
        if self.batch_num >= self.max_normalization_batchnum:
            return
        if self.batch_num % self.normalization_interval != 0:
            return
        normalize_weights(self.model, self.batch_num)

class ShowTestPredsCallback(Callback):
    def __init__(self, generator_class=None):
        self.generator_class = generator_class
        self.generator = generator_class.generate()
        self.batch_num = 0

    def on_epoch_end(self, batch, logs={}):
        print("Showing test preds!")
        batch_x = next(self.generator)
        preds = self.model.predict(batch_x, batch_size=batch_x.shape[0])
        do_show = True
        #if do_show:
        #    show(batch_x[0], None, preds[0], 1024, 768, "", self.batch_num, "Test")


class DisplayAccuracyCallback(Callback):
    def __init__(self, model, generator=None, generator_class=None, training_generator_class=None, pixel_counts_by_class=defaultdict(lambda:0), eval_interval=50, log_dir="./"):
        self.generator = generator
        self.generator_class = generator_class
        self.eval_interval = eval_interval
        self.accs = []
        self.smoothedlosseswindow = []
        self.recs = defaultdict(list)
        self.precs = defaultdict(list)
        self.fscores = defaultdict(list)
        self.pixaccs = defaultdict(list)
        self.ctaccs = 10
        self.gt_masses = defaultdict(list)

        self.losses = []
        self.smoothedlosses = []
        self.acc = []
        self.prec = []
        self.reca = []
        self.fscore = []
        self.pixacc = []
        self.h = 6656 # 1024
        self.w = 4608 # 768
        self.batch_num = 0
        self.val_json_iters = []
        self.pixel_counts_by_class = pixel_counts_by_class
        print("DisplayAccuracyCallback Pixel counts by class:", self.pixel_counts_by_class)
        self.training_generator_class = training_generator_class
        self.batch_num = 0
        self.best_macro_fscore = 0.0
        self.log_dir = log_dir
        self.set_model(model)
        self.thresholds = None

    # on_batch_end
    def on_batch_end(self, batch, logs={}):
        do_show = True
        self.batch_num += 1
        if self.batch_num % self.eval_interval != 0:
            return

        print("")
        print("========= Performing Validation... ===========")
        batch_x, batch_y = next(self.generator)
        cropsize=256
        try:
            x,y = np.random.randint(0,batch_x.shape[2]-cropsize),np.random.randint(0,batch_x.shape[1]-cropsize)
        except Exception as ex:
            x,y = 0,0
        batch_x, batch_y = batch_x[:,y:y+cropsize,x:x+cropsize,:],batch_y[:,y:y+cropsize,x:x+cropsize,:]
        print("Batch x, y shapes:", batch_x.shape, batch_y.shape)
        print("Making predictions...")
        preds = self.model.predict(batch_x, batch_size=batch_x.shape[0])
        print("Preds shape:", preds.shape)
        print("postprocessing preds!", preds.shape)
        print("")
        print("")
        print("")
        print("")
        print("")
        print("Classwise predicted means:")
        print(np.mean(preds, axis=(0,1,2)))
        print("Classwise ground truth means:")
        print(np.mean(batch_y, axis=(0,1,2)))
        # return 1
        #preds = postprocess_preds(batch_x, preds, batch_y, None, self.pixel_counts_by_class, thresh=self.thresholds)
        print("done!")
        #scores = self.model.evaluate(batch_x, batch_y, batch_size=batch_x.shape[0])
        #print("ymax", np.max(batch_y), np.min(batch_y))

        #grads = get_gradients(self.model)
        #print("GRADIENTS:", grads)

        batch_y = batch_y.astype('float32')
        preds = preds.astype('float32')

        # Display image, ground truth, and predictions.
        #print(self.generator_class)
        #print("Materializing loss...")
        #loss = materialize_loss(self.model, batch_x, batch_y)
        #print("LOSS:", loss)
        print("Showing predictions and batch...")
        impath = self.generator_class.last_path
        #if do_show:
        #    show(batch_x[0], batch_y[0], preds[0], self.h, self.w, impath, self.batch_num, do_show=False)

        #if self.training_generator.gt is not None:
        #plt.figure("Validation_GT")
        #plt.clf()
        v = show(batch_y[0], bgr=True)
        #plt.imshow((v), interpolation='none')
        #if self.training_generator.image is not None:
        #plt.figure("Validation_Image")
        #plt.clf()
        im = batch_x[0]
        #plt.imshow((im), interpolation='none')
        #if self.model is not None:
        #plt.figure("Validation_Prediction")
        #plt.clf()
        print("DisplayValidationSamplesCallback input shape", im.shape)
        #self.pred = self.model.predict(np.reshape(im, [1,im.shape[0],im.shape[1],im.shape[2]]))[0]
        self.pred = preds[0]
        print("Pred shape:", self.pred.shape)
        print("Pred min/max:", np.min(self.pred), np.max(self.pred))
        #plt.imshow((show(self.pred, bgr=True)), interpolation='none')
        # TODO plt.savefig(
        #plt.pause(0.001)

        print("Computing precisions, recalls, etc.!")
        # Remember to call plt.clf() to avoid a memory leak!

        #predsrgbchannels = multihot_to_multiindexed_rgb(preds[0])
        #print("Performing HisDoc Evaluation...")

        #print("Scores", scores)
        # print(per-digit accuracy rate!!!)

        # COMPARE prediction to "MOST COMMON"

        # BLANK image:
        # precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct = score(batch_y, np.zeros(preds.shape))
        '''
        Average activation: 0.00833214 Max activation: 0.999519 Min activation: 0.0
        Average validation accuracy: 0.892559296486
        Average validation precision: 0.982093216081
        Average validation recall: 0.533333333333
        Average validation F-score: 0.523820697386
        Average validation IoU Pixel accuracy: 0.515426549414
        '''
        # Using most common label instead of predictions:
        '''
        most_common_gt = np.argmax(batch_y[0], axis=-1)
        from scipy import stats
        most_common_gt = stats.mode(most_common_gt, axis=None)[0]
        print("Most Common GT:", most_common_gt)
        baseline_pred = np.zeros(preds.shape)
        print(baseline_pred.shape)
        baseline_pred[:,:,:,most_common_gt] = 1
        precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct = score(batch_y, baseline_pred)
        '''

        '''
        ========== Overall stats: =============
        Average activation: 0.0459793 Max activation: 0.999983 Min activation: 5.16211e-31
        Average validation accuracy: 0.86616964416
        Average validation precision: 0.977694940693
        Average validation recall: 0.555555555556
        Average validation F-score: 0.543486426286
        Average validation IoU Pixel accuracy: 0.533250496249
        '''

        cm = CachedMetrics(batch_y[0], preds[0])
        precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct = cm["precision"], cm["recall"], cm["accuracy"], cm["f1_score"], cm["gt_mass"], cm["true_positives"]
        import sys
        print("Size of CM:", sys.getsizeof(cm))
        print("Size of self:", sys.getsizeof(self))
        cm = None

        #precisions, recalls, accuracies, f_scores, tot_gt_mass, overall_correct = score(batch_y, preds)

        print(precisions.shape, recalls.shape, accuracies.shape, f_scores.shape, tot_gt_mass.shape, overall_correct.shape)
        if self.training_generator_class is not None:
            self.training_generator_class.class_fscores = f_scores

        self.accs.extend(overall_correct)
        self.smoothedlosseswindow.append(float(logs['loss']))
        while len(self.accs) > self.ctaccs:
            del self.accs[0]
        while len(self.smoothedlosseswindow) > self.ctaccs:
            del self.smoothedlosseswindow[0]
        av_prec = 0.0
        av_rec = 0.0
        av_fsc = 0.0
        av_acc = 0.0

        prec_cs = {}
        rec_cs = {}
        fsc_cs = {}
        acc_cs = {}

        # These are used for moving average plotting. TODO: Abstract this into the variable plotting class!
        for c in range(preds.shape[-1]):
            precisions[c] /= preds.shape[0]
            recalls[c] /= preds.shape[0]
            f_scores[c] /= preds.shape[0]
            accuracies[c] /= preds.shape[0]
            self.precs[c].append(precisions[c])
            self.recs[c].append(recalls[c])
            self.fscores[c].append(f_scores[c])
            self.pixaccs[c].append(accuracies[c])
            self.gt_masses[c].append(tot_gt_mass[c])

            if len(self.precs[c]) > self.ctaccs:
                del self.precs[c][0]
                del self.recs[c][0]
                del self.fscores[c][0]
                del self.pixaccs[c][0]
                del self.gt_masses[c][0]
            prec_c = sum(self.precs[c]) / len(self.precs[c])
            rec_c = sum(self.recs[c]) / len(self.recs[c])
            fsc_c = sum(self.fscores[c]) / len(self.fscores[c])
            acc_c = sum(self.pixaccs[c]) / len(self.pixaccs[c])
            print("")
            print("Average precision for class " + str(c+1) + ": " + str(prec_c))
            print("Average recall for class " + str(c+1) + ": " + str(rec_c))
            print("Average f-score for class " + str(c+1) + ": " + str(fsc_c))
            print("Average accuracy for class " + str(c+1) + ": " + str(acc_c))
            print("Average GT mass for class " + str(c+1) + ": " + str(np.mean(self.pixaccs[c])))
            # TODO: We can and should log the net amount of gradient updates incurred by each class.
            prec_cs[c] = prec_c
            rec_cs[c] = rec_c
            fsc_cs[c] = fsc_c
            acc_cs[c] = acc_cs
            av_prec += prec_c
            av_rec += rec_c
            av_fsc += fsc_c
            av_acc += acc_c

        if self.training_generator_class is not None:
            print("Setting F-scores!", fsc_cs)
            if hasattr(self.training_generator_class.dataset_sampler, 'set_class_sampling_weights'):
                self.training_generator_class.dataset_sampler.set_class_sampling_weights(fsc_cs, invert=True)

        #if self.training_generator_class is not None:
        #    self.training_generator_class.class_fscores = f_scores

        import os
        inst_fscore = 0.0
        for c in range(preds.shape[-1]):
            inst_fscore += self.fscores[c][-1]
        inst_fscore /= preds.shape[-1]
        if inst_fscore > self.best_macro_fscore:
            self.best_macro_fscore = inst_fscore
            print("NEW BEST INSTANT MACRO-FSCORE:", self.best_macro_fscore)
            if self.training_generator_class is not None:
                modeldest = os.path.join(os.path.dirname(self.training_generator_class.dataset_sampler.image_list[0]), "model_checkpoint_BEST_"+str(inst_fscore)+".h5")
                if os.path.exists('model_checkpoint.h5'):
                    print("Copying best model to ", modeldest)
                    shutil.copy('model_checkpoint.h5', modeldest)
                    shutil.copy('model_checkpoint.h5', modeldest)
                    modeldest = os.path.join(os.path.dirname(self.training_generator_class.dataset_sampler.image_list[0]), "model_checkpoint_BEST.h5")

        #plt.figure("Mini-Validation Metrics By Class")
        #plt.clf()
        #fig, ax = plt.subplots()
        index = np.arange(preds.shape[-1]+1)
        bar_width = 0.15
        opacity = 0.90
        error_config = {'ecolor': '0.3'}
        #for classnum in range(preds.shape[-1]):
        prec_means = [np.mean(np.array(self.precs[c])) for c in range(preds.shape[-1])]
        prec_means.append(av_prec/preds.shape[-1])
        prec_stds = [np.std(np.array(self.precs[c])) for c in range(preds.shape[-1])]
        prec_stds.append(np.std(np.array([self.precs[c] for c in range(preds.shape[-1])])))
        rec_means = [np.mean(np.array(self.recs[c])) for c in range(preds.shape[-1])]
        rec_means.append(av_rec/preds.shape[-1])
        rec_stds = [np.std(np.array(self.recs[c])) for c in range(preds.shape[-1])]
        rec_stds.append(np.std(np.array([self.recs[c] for c in range(preds.shape[-1])])))
        fsc_means = [np.mean(np.array(self.fscores[c])) for c in range(preds.shape[-1])]
        fsc_means.append(av_fsc/preds.shape[-1])
        fsc_stds = [np.std(np.array(self.fscores[c])) for c in range(preds.shape[-1])]
        fsc_stds.append(np.std(np.array([self.fscores[c] for c in range(preds.shape[-1])])))
        acc_means = [np.mean(np.array(self.pixaccs[c])) for c in range(preds.shape[-1])]
        acc_means.append(av_acc/preds.shape[-1])
        acc_stds = [np.std(np.array(self.pixaccs[c])) for c in range(preds.shape[-1])]
        acc_stds.append(np.std(np.array([self.pixaccs[c] for c in range(preds.shape[-1])])))
        #print(prec_means, prec_stds, len(prec_means), len(prec_stds))
        #prec_rects = plt.bar(index, prec_means, bar_width, alpha=opacity, color='r', yerr=prec_stds, error_kw=error_config, label="Precision")
        #rec_rects = plt.bar(index+bar_width, rec_means, bar_width, alpha=opacity,color='b',yerr=rec_stds, error_kw=error_config, label="Recall")
        #fsc_rects = plt.bar(index+2*bar_width, fsc_means, bar_width, alpha=opacity,color='xkcd:purple',yerr=fsc_stds, error_kw=error_config, label="F-Measure")
        #pixacc_rects = plt.bar(index+3*bar_width, acc_means, bar_width, alpha=opacity, color='g', yerr=acc_stds, error_kw=error_config, label="Pixel Accuracy")
        #plt.xlabel('Class Number')
        #plt.ylabel('Performance Measure')
        # TODO: Genericize!
        #plt.xticks(index + bar_width / 2, (1, 2, 3, 4, 5, "All"))
        #plt.xticks(index + bar_width / 2, (1, 2, 3, 4, 5, "All"))
        #plt.legend()
        #plt.title('Metric Chart')
        #plt.tight_layout()
        #plt.pause(0.0001)
        #plt.savefig(os.path.join(self.log_dir, 'LossMetrics.png'))

        print("========== Overall stats: =============")
        print("Average activation:", np.mean(preds), "Max activation:", np.max(preds), "Min activation:", np.min(preds))
        print("Average validation accuracy:", sum(self.accs)/len(self.accs))
        print("Average validation precision:", av_prec/preds.shape[-1])
        print("Average validation recall:", av_rec/preds.shape[-1])
        print("Average validation F-score:", av_fsc/preds.shape[-1])
        print("Average validation IoU Pixel accuracy:", av_acc/preds.shape[-1])

        self.smoothedlosses.append(np.mean(np.array(self.smoothedlosseswindow)))
        self.losses.append(float(logs['loss']))
        self.prec.append(av_prec/preds.shape[-1])
        self.reca.append(av_rec/preds.shape[-1])
        self.fscore.append(av_fsc/preds.shape[-1])
        # Scale the pixacc for visualization
        import math
        self.pixacc.append(math.log(av_acc/preds.shape[-1]+0.0000000000001)+1.0)
        #plt.figure("Training Loss and Mini-Validation Metric History")
        #plt.clf()
        #plt.plot(self.losses, color='xkcd:orange')
        #plt.plot(self.smoothedlosses, color='xkcd:teal')
        #plt.plot(self.fscore, color='xkcd:purple')
        #plt.plot(self.reca, color='b')
        #plt.plot(self.prec, color='r')
        #plt.plot(self.pixacc, color='g')
        #plt.tight_layout()
        #plt.title('Loss and Metric History')
        #plt.pause(0.001)
        #plt.savefig(os.path.join(self.log_dir, 'training.png'))
        # Compute Precision, Recall, F-Score, and Pixel accuracy per class.

        #mem = psutil.virtual_memory()
        #print("Memory:", mem)
        #from guppy import hpy
        #h = hpy()
        #print(h.heap())

        val_json = {"Loss": self.losses, "Precision":self.prec, "Recall": self.reca, "F-Score": self.fscore, "Accuracy": self.pixacc}
        #print(val_json)
        self.val_json_iters = val_json #.append(val_json)
        with open(os.path.join(self.log_dir, "loss_history.json"), 'w') as f:
            json.dump(self.val_json_iters, f, indent=2)

        #cv2.waitKey(10)

def display_stats(tensor, name="None", numbins=10):
    # Show mean, median, stddev, and histogram.
    # Really, just show a histogram and save it.
    #fig = plt.figure(name)
    #hist = np.histogram(tensor, bins=numbins)
    #plt.clf()
    #plt.hist(hist, bins=numbins)
    #plt.savefig(name+".png")
    #plt.close(fig)
    #print("Histogram of weights for tensor", name, hist)
    print(name, "Mean:", np.mean(tensor), "StdDev:", np.std(tensor), "Median", np.median(tensor))

class DisplayWeightStatsCallback(Callback):
    def __init__(self, model):
        self.set_model(model)
        self.interval = 100
        self.model.weights_good = True
        self.iteration = 0

    def on_batch_end(self, batch, logs={}):
        if self.iteration % self.interval == 0:
            weights = self.model.get_weights()
            for i,weight in enumerate(weights):
                print("Displaying weights for", i)
                print(weight.shape)
                display_stats(weight, "Weight"+str(i)+"_"+str(weight.shape))
                self.model.weights_good = False
        self.iteration += 1


class DisplayActivationStatsCallback(Callback):
    def __init__(self, model):
        self.set_model(model)

class LogMemoryUsageCallback(Callback):
    def __init__(self):
        self.sess = K.get_session()
        import tensorflow as tf

    def on_epoch_end(self, epoch, logs):
        # maximum across all sessions and .run calls so far
        max_bytes = self.sess.run(tf.contrib.memory_stats.MaxBytesInUse())
        # current usage
        bytes = self.sess.run(tf.contrib.memory_stats.BytesInUse())

        max_gb = float(max_bytes)/(1024*1024*1024)
        gb = float(bytes)/(1024**3)
        print("Max GB in use:", max_gb, max_bytes)
        print("Current GB in use:", gb, bytes)

#callbacks.append(LogMemoryUsageCallback())

class LogTimingCallback(Callback):
    def __init__(self, batch_size=6, width=224, **kwargs):
        self.batch_size = batch_size
        self.width = self.height = 224
        super(LogTimingCallback, self).__init__(**kwargs)
        self.batch_start_time = time.time()
        self.epoch_start_time = time.time()
        self.total_running_time = 0
        self.batches_elapsed = 0
        self.batches_elapsed_total = 0
        self.pixels_processed_total = 0
        self.display_interval = 20
        self.batch_num = 0

    def on_batch_end(self, batch, logs):
        t = time.time()
        batch_duration = t - self.batch_start_time
        self.batch_start_time = t
        self.batches_elapsed += 1
        self.batches_elapsed_total += 1
        self.pixels_processed_total += self.batch_size * self.width * self.height
        self.total_running_time += batch_duration
        self.batch_num += 1
        if self.batch_num % self.display_interval == 0:
            print("Batch duration:", batch_duration, "s")
            batches_per_second = self.batches_elapsed_total / self.total_running_time
            samples_per_second = self.batches_elapsed_total * self.batch_size / self.total_running_time
            pixels_per_second = self.pixels_processed_total / self.total_running_time
            print("Total batches per second", batches_per_second)
            print("Total samples per second", samples_per_second)
            print("Total Mega-Pixels (MP) per second", pixels_per_second/1000000)
            print("Current batches per second", 1.0 / batch_duration)
            print("Current samples per second", self.batch_size / batch_duration)

    def on_epoch_end(self, epoch, logs):
        epoch_duration = time.time() - self.epoch_start_time
        print("Epoch duration:", epoch_duration, "s")
        self.batches_elapsed = 0

def _mkdir(newdir):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        #print("_mkdir %s" % repr(newdir))
        if tail:
            os.mkdir(newdir)


class TFModelSaverCallback(Callback):
    def __init__(self, model, model_save_path, save_interval=5000, start_iteration=0):
        self.set_model(model)
        self.iteration = start_iteration
        self.model_save_path = model_save_path
        self.save_interval = save_interval

    def on_batch_end(self, batch, logs={}):
        self.iteration += 1
        if self.iteration > 0 and self.iteration % self.save_interval == 0:
            print("Saving model...")
            dirname = os.path.dirname(self.model_save_path)
            if not os.path.exists(dirname):
                _mkdir(dirname)
            self.model.save(self.model_save_path)

class TestModelCallback(Callback):
    def __init__(self, model, save_basepath="best_model", testfolder="./test", testscale=1.0, pixel_counts_byclass=None):
        self.set_model(model)
        self.save_basepath = save_basepath
        self.model_version = 0
        self.gradient_history = []
        self.testfolder=testfolder
        self.testscale=testscale
        self.epoch_num = 0
        self.test_interval = 50
        self.loss_history = []
        self.model_saves = []
        self.pixel_counts_byclass = pixel_counts_byclass

    def on_epoch_end(self, batch, logs={}):
        # TODO: Get gradient history working!!!
        '''
        grads = get_gradients(self.model)
        print(grads)
        mean_gradient = np.mean(np.array([np.mean(grads[g] for g in range(len(grads)))]))
        self.gradient_history.append(mean_gradient)
        plt.figure(1)
        plt.clf()
        plt.plot(gradient_history)
        #plt.pause(0.001)
        '''
        self.loss_history.append(logs['loss'])

        l = len(self.loss_history)
        lh = self.loss_history
        ls = lh[l-1]
        if l > 1 and ls < lh[l-2] and ls < lh[l-3] and ls < lh[l-4] and ls < lh[l-5]:
            print("Saving a best snapshot...")
            filepath = self.save_basepath + "_" + str(self.model_version) + ".h5"
            self.model.save(filepath)
            self.model_saves.append(filepath)
            self.model_version += 1

        if self.epoch_num % self.test_interval != 0:
            self.epoch_num += 1
            return
        self.epoch_num += 1

        if len(self.model_saves) > 0:
            # DO Predictions with an ensemble of models, then median-filter the result.
            TestModel(model=self.model, model_basepath=self.save_basepath, testfolder=self.testfolder, testscale=self.testscale, pixel_counts_byclass=self.pixel_counts_byclass)


# https://github.com/fchollet/keras/issues/3358
class TensorBoardWrapper(keras.callbacks.TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0], dtype=np.float32), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)
