import tensorflow as tf
from collections import defaultdict
import os
import cv2
import numpy as np

from data_loaders.legacy_training_sample_generators import *

from data_loaders.gt_loaders.gt_loader import *

from models.tensorflow_models import *

from data_loaders.preprocessing.data_augmentation import preproc as preproc

def accuracy_info(sess, y_conv, y_, params={}):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    class_slices =  []
    gt_slices = []
    cur_ind = 0
    for i in range(len(params['classes'])):
        class_slices.append(tf.slice(y_conv,[0,cur_ind],[-1,params['classes'][i]]))
        gt_slices.append(tf.slice(y_,[0,cur_ind],[-1,params['classes'][i]]))
        cur_ind += params['classes'][i]

    preds_classes = []
    gt_classes = []
    for i in range(len(class_slices)):
        preds_classes.append(tf.argmax(class_slices[i], 1))
        gt_classes.append(tf.argmax(gt_slices[i], 1))

    orientation_accuracy = tf.reduce_mean(tf.cast(tf.equal(gt_classes[1], preds_classes[1]),tf.float32))

    return accuracy, orientation_accuracy

# Variable learning rate
# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.1
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            100000, 0.96, staircase=True)
# # Passing global_step to minimize() will increment it at each step.
# learning_step = (
#     tf.train.GradientDescentOptimizer(learning_rate)
#     .minimize(...my loss..., global_step=global_step)
# )


class TrainerTF:
    def __init__(self, model=model, data_loader=None, class_labels={0: "dotted lines", 1: "handwriting", 2: "machine print", 3: "solid lines", 4: "stamps"}, width=224, height=224, model_factory=model):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.params = {}
        self.num_classes = len(class_labels)
        self.width = width
        self.height = height
        
        self.class_num_to_symbol = {0:'dotted_lines', 1:'hw', 2:'machprint', 3:'lines', 4:'stamp'}
        self.num_classes = len(self.class_num_to_symbol)
        self.class_to_num = {v:k for k,v in self.class_num_to_symbol.items()}
        self.class_training_density = {'machprint':0.3, 'hw':0.3, 'blank':0.0, 'lines':0.1, 'stamp':0.05, 'dotted_lines':0.25}
        self.class_training_exemplar_generators = defaultdict(lambda:get_masked_regionsampler_semanticseg_multichannel)

        self.params['classes'] = [self.num_classes]
        self.params['predictors'] = []
        self.max_iterations = 20001
        self.display_interval = 25
        self.batch_size = 8 if width > 64 else (16 if width >= 32 else 64)
        self.model_path = "BarrettA" #model_path
        loadmodel = self.model_path + "5000"
        print "Attempting to load model from path", loadmodel, os.path.exists(loadmodel + ".meta")
        loadmodel = loadmodel if os.path.exists(loadmodel + ".meta") else None
        self.x, self.y_, self.y_conv, self.train_step, self.accuracy, self.keep_prob, self.loss, self.saver = create_model(self.sess, model_factory=model, loadmodel=loadmodel, params=self.params, trainable=True, batch_size=self.batch_size, height=self.height, width=self.width)
        self.iteration = 0
        self.display_interval = 25
        self.save_interval = 5000

        self.max_height_scale = 4
        self.min_height_scale = 0.5
        self.recognition_scale = 1
        self.extra_offsets = 0
        self.scales_up = 0
        self.scales_down = 0

        self.img = cv2.imread(data_loader, 0)
        self.mask = load_gt(data_loader, num_classes=5) #cv2.imread('multichannel_forms/RGB_form1.png')

        self.read_extra_channels = False
        if self.read_extra_channels:
            print "Data dir:", data_dir
            for jpg in [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d[-3:] == "jpg" and not "jpg." in d and not "jpg_" in d and not "pred" in d]:
                print "read jpg", jpg
                self.img = cv2.imread(jpg, 0)
                print self.img.shape
                self.mask = np.zeros((self.img.shape[0], self.img.shape[1], self.num_classes))
                for clasnum, clas in self.class_num_to_symbol.items():
                    mask_layer = cv2.imread(jpg.replace(".jpg","_"+str(clasnum)+".png"), 0)
                    if mask_layer is not None:
                        print clasnum, clas, mask_layer.shape
                    self.mask[:,:,clasnum] = mask_layer
        print "Done initializing."

    def get_batch(self, batch_size=64, input_height=28, input_width=28):
        start_b = 0
        end_b = batch_size
        tot_fract = 0.0
        np_batch_imgs = np.zeros((batch_size, input_height, input_width), dtype=np.float32) #np.uint8)
        np_batch_gts = np.zeros((batch_size, input_height, input_width, self.num_classes))

        #hw_img = cv2.imread(random.choice(self.hw_imgs), 0)
        #t,bin_hw = cv2.threshold(hw_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #mp_img = cv2.imread(random.choice(self.mp_imgs), 0)
        #t,bin_mp = cv2.threshold(mp_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = self.img
        mask = self.mask
        for clasnum, clas in self.class_num_to_symbol.items():
            prob = self.class_training_density[clas]
            tot_fract += prob
            target_batch = int((tot_fract * batch_size))
            while start_b < target_batch:
                #print start_b, prob, tot_fract, clas, target_batch
                ms = int(self.height*self.max_height_scale)

                # TODO: GT images are still off by a bit, and sometimes the GT image is scaled up in one dimension vs the original.
                imb, gtb = self.class_training_exemplar_generators[clas](img, mask, maskval=0, numclasses=self.num_classes, label=clasnum, minsize=int(self.height*self.min_height_scale), maxsize=ms, height=self.height, width=self.width, maskchannel=clasnum)
                np_batch_imgs[start_b] = imb.astype('float32')/255.0
                np_batch_gts[start_b]  = gtb
                start_b += 1
        np_batch_imgs, np_batch_gts = preproc(np_batch_imgs, np_batch_gts, self.num_classes, np_batch_imgs.shape[0], np_batch_imgs.shape[1], np_batch_imgs.shape[2])
        return np_batch_imgs, np_batch_gts

    def run(self, train=True):
        self.iter = 0
        while self.iter < self.max_iterations:
            self.train_for(50)
            self.iter += 50

    def train_for(self, iters=50):
        for i in range(iters):
            print "Training iteration", i
            batch, targets = self.get_batch(self.batch_size, self.height, self.width)
            batch = [np.reshape(batch, [-1, self.height, self.width, 1]), targets]
            self.sess.run([self.train_step], feed_dict={self.x:1.0-batch[0], self.y_:batch[1], self.keep_prob: 0.9})

            #batch = preproc(batch, params, batch_size)
            if (self.iteration+1)%self.display_interval == 0:
                ls,train_accuracy,logits = self.sess.run([self.loss, self.accuracy, self.y_conv], feed_dict={self.x:1.0-batch[0], self.y_:batch[1], self.keep_prob: 1.0})
                #print ls, train_accuracy
                print ("Step %d, loss %g, training accuracy %g"%(self.iteration, ls, train_accuracy))#, "logits:", logits)
                randInd = random.randint(0,self.batch_size-1)
                sample = batch[0][randInd]
                gt = batch[1][randInd]
                pred = logits[randInd] #np.argmax(logits[randInd], axis=3)
                #cv2.imshow('Train image', np.reshape(sample, [self.height,self.width]))
                #cv2.waitKey(1000)

                predshow = pred
                scalefactor = 10
                if pred.shape[1] < 40:
                    predshow = cv2.resize(pred, (pred.shape[1]*scalefactor, pred.shape[0]*scalefactor), interpolation=cv2.INTER_NEAREST)

                ps = predshow.copy()
                ps[:,:,2] += ps[:,:,4]/2
                ps[:,:,0] += ps[:,:,4]/2
                ps[:,:,2] += ps[:,:,3]/2
                ps[:,:,1] += ps[:,:,3]/2
                cv2.imshow('Pred image', ps[:,:,:3])
                cv2.imwrite('pred.jpg', ps[:,:,:3]*255)
                #cv2.waitKey(10)

                gtshow = gt
                if gt.shape[1] < 40:
                    gtshow = cv2.resize(gt, (gt.shape[1]*scalefactor, gt.shape[0]*scalefactor), interpolation=cv2.INTER_NEAREST)
                gtso = gtshow.copy()
                gtso[:,:,2] += gtso[:,:,4]/2
                gtso[:,:,0] += gtso[:,:,4]/2
                gtso[:,:,2] += gtso[:,:,3]/2
                gtso[:,:,1] += gtso[:,:,3]/2
                cv2.imshow('GT image', gtso[:,:,:3])
                cv2.imwrite('gt.jpg', gtso[:,:,:3]*255)
                #cv2.waitKey(10)

                trainshow = sample
                if sample.shape[1] < 40:
                    trainshow = cv2.resize(sample, (sample.shape[1]*scalefactor, sample.shape[0]*scalefactor), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Train image', trainshow)
                cv2.imwrite('train.jpg', trainshow*255)
                cv2.waitKey(10)

            if self.iteration > 0 and self.iteration % self.save_interval == 0:
                print "Saving model..."
                #B4 = tf.get_default_graph().get_tensor_by_name('b4:0')
                #b4 = sess.run(B4, feed_dict={x:1.0-batch[0], y_:batch[1], keep_prob: 0.5})
                #print b4[0:4]
                save_model(self.sess, "./" + self.model_path + str(self.iteration))
            self.iteration += 1
        return ls, train_accuracy, logits

    def run_batch(self, inputs, targets):
        batch = [np.reshape(inputs, [-1, self.height, self.width, 1]), targets]
        self.iteration += 1
        self.sess.run([self.train_step], feed_dict={self.x:1.0-batch[0], self.y_:batch[1], self.keep_prob: 0.5})

        #batch = preproc(batch, params, batch_size)
        if self.iteration%self.display_interval == 0:
            ls,train_accuracy,logits = self.sess.run([self.loss, self.accuracy, self.y_conv], feed_dict={self.x:1.0-batch[0], self.y_:batch[1], self.keep_prob: 1.0})
            print ("Step %d, loss %g, training accuracy %g"%(self.iteration, ls, train_accuracy))#, "logits:", logits)
            cv2.imshow('Train image', np.reshape(1.0-batch[0][random.randint(0,self.batch_size-1)], [self.height,self.width]))
            cv2.waitKey(1)

        if self.iteration > 0 and self.iteration % self.save_interval == 0:
            print "Saving model..."
            #B4 = tf.get_default_graph().get_tensor_by_name('b4:0')
            #b4 = sess.run(B4, feed_dict={x:1.0-batch[0], y_:batch[1], keep_prob: 0.5})
            #print b4[0:4]
            save_model(self.sess, "mnist_cnn" + str(self.iteration))
        return ls, train_accuracy, logits
