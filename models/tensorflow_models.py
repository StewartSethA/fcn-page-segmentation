import tensorflow as tf
from collections import defaultdict
import os
import cv2
import numpy as np

from models.vanilla_cnn import *
import models.vanilla_cnn
from models.nn_utils import *

def model(x_image, class_splits=[], pred_splits=[], keep_prob=0.5, batch_size=64, size=224):
    if size == 224 or size == 112 and not self.zoning_predictor:
        return cnn224x224_autoencoder_almostoptim(x_image, sum(class_splits) + sum(pred_splits), keep_prob, batch_size=batch_size, width=size, height=size)
    elif size == 224 or size > 100:
        return cnn224x224_autoencoder_regionpred(x_image, sum(class_splits) + sum(pred_splits), keep_prob, batch_size=batch_size, width=size, height=size)

def model28x28_resnet(x_image, class_splits=[], pred_splits=[], keep_prob=0.5):
    #cnn = cnn28x28(x_image, sum(class_splits) + sum(pred_splits), keep_prob)

    num_classes = sum(class_splits) + sum(pred_splits)
    #net = resnet28x28(x_image, 32)
    out_size = (28/7)*16
    net, allwts = resnet(x_image, 28, 28, input_channels=1, start_feats=16, num_layers=7, target_width=7, target_height=7, nonlin=tf.nn.relu)
    net_flat = tf.reshape(net, [-1, 7*7*out_size]) # Batch, EverythingElse
    fc1 = nn_layer(net_flat, 7*7*out_size, 1024, "fc1")
    fc1_drop = tf.nn.dropout(fc1, keep_prob)
    #fc2 = nn_layer(fc1_drop, 1024, 1024, "fc2")
    #fc2_drop = tf.nn.dropout(fc2, keep_prob)
    fc3 = nn_layer(fc1_drop, 1024, num_classes, "fc3", act=lambda x, name="":x)
    cnn = fc3

    split_predictors = multitask_head(cnn, class_splits=class_splits, pred_splits=pred_splits)
    return split_predictors

def create_model(sess, model_factory=model, width=28, height=28, loadmodel=None, params={}, trainable=True, batch_size=64, num_channels=1):
    print "Seth"
    print "Params", params
    if loadmodel is None:
        print "Creating new model..."
        x = tf.placeholder(tf.float32, shape=[None, None, None, num_channels], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, None, None, sum(params['classes'])+sum(params['predictors'])], name='y_')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        y_conv = model_factory(x, class_splits=params['classes'], pred_splits=params['predictors'], keep_prob=keep_prob, batch_size=batch_size, size=width)
        tf.add_to_collection("logits", y_conv)
        tf.add_to_collection("placeholders", x)
        tf.add_to_collection("placeholders", y_)
        loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        #cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))) #, reduction_indices=[1])
        #loss = cross_entropy
        tf.add_to_collection("loss", loss)
        optim = tf.train.AdamOptimizer(1e-3)#4)
        train_step = optim.minimize(loss)
        tf.add_to_collection("train_step", train_step)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
    else:
        x, y_, y_conv, keep_prob, loss, train_step, saver = restore_model(sess, loadmodel)

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #accuracy, orientation_accuracy = accuracy_info(sess, y_conv, y_, params=params)

    if loadmodel is None:
        sess.run(tf.global_variables_initializer())

    return x, y_, y_conv, train_step, accuracy, keep_prob, loss, saver

def restore_model(sess, loadmodel):
    print "Loading model", loadmodel
    saver = tf.train.import_meta_graph(loadmodel+ '.meta')
    dirname = os.path.dirname(loadmodel)
    saver.restore(sess, tf.train.latest_checkpoint(dirname)) # TODO: As of now, this only returns the LATEST checkpoint. We want to be able to restore an ARBITRARY checkpoint!!!
    x = tf.get_default_graph().get_tensor_by_name('x:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
    y_conv = tf.get_collection("logits")[0]
    loss = tf.get_collection("loss")[0]
    train_step = tf.get_collection("train_step")[0]
    print "Success?"
    return x, y_, y_conv, keep_prob, loss, train_step, saver

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



class TFModelKerasStyle:
    def __init__(self, args, profile=False):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.model_path = args.load_model_path
        print ""
        print "TFModelKerasStyle"
        print args
        print self.model_path
        width = height = args.crop_size
        print "Attempting to load model from path", self.model_path, os.path.exists(self.model_path + ".meta")
        if os.path.exists(self.model_path + ".meta"):
            pass
        else:
            self.model_path = self.model_path + "5000" if os.path.exists(self.model_path + "5000" + ".meta") else None
        # TODO: Prune the following. Which are useless?
        self.class_num_to_symbol = {0:'dotted_lines', 1:'hw', 2:'machprint', 3:'lines', 4:'stamp'}
        self.num_classes = len(self.class_num_to_symbol)
        self.class_to_num = {v:k for k,v in self.class_num_to_symbol.items()}
        self.class_training_density = {'machprint':0.3, 'hw':0.3, 'blank':0.0, 'lines':0.1, 'stamp':0.05, 'dotted_lines':0.25}
        self.class_training_exemplar_generators = defaultdict(lambda:get_masked_regionsampler_semanticseg_multichannel)
        self.dropout = args.dropout_rate
        self.params = {}
        self.params['classes'] = [self.num_classes]
        self.params['predictors'] = []
        self.max_iterations = 20001
        self.display_interval = 25
        self.batch_size = args.batch_size
        self.max_height_scale = 1.0#4.0
        self.min_height_scale = 0.125 #0.5
        self.recognition_scale = 1
        self.width = width
        self.height = height
        self.iteration = 0
        self.display_interval = 25
        self.save_interval = 5000
        self.profile = profile
        self.model_type = args.model_type
        self.model_constructor = model
        if getattr(models.vanilla_cnn, self.model_type) is not None:
            print "Using model derived from description:", self.model_type
            self.model_constructor = getattr(models.vanilla_cnn, self.model_type)
        self.x, self.y_, self.y_conv, self.train_step, self.accuracy, self.keep_prob, self.loss, self.saver = create_model(self.sess, model_factory=self.model_constructor, loadmodel=self.model_path, params=self.params, trainable=True, batch_size=self.batch_size, height=self.height, width=self.width)
        if profile:
            for v in tf.trainable_variables():
                tf.summary.histogram(v.name, v)
            self.train_writer = tf.summary.FileWriter('./tf_logs/1/train ', self.sess.graph)


    def get_weights(self):
        return self.sess.run(tf.trainable_variables())

    def get_batch(self, batch_size=64, input_height=28, input_width=28):
        start_b = 0
        end_b = batch_size
        tot_fract = 0.0
        np_batch_imgs = np.zeros((batch_size, input_height, input_width), dtype=np.float32) #np.uint8)
        np_batch_gts = np.zeros((batch_size, input_height, input_width, self.num_classes))

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
                imb, gtb = self.class_training_exemplar_generators[clas](img, mask, maskval=0, numclasses=self.num_classes, label=clasnum, minsize=int(self.height*self.min_height_scale), maxsize=ms, height=input_height, width=input_width, maskchannel=clasnum)
                np_batch_imgs[start_b] = imb.astype('float32')/255.0
                np_batch_gts[start_b]  = gtb
                start_b += 1
        np_batch_imgs, np_batch_gts = preproc(np_batch_imgs, np_batch_gts, self.num_classes, np_batch_imgs.shape[0], np_batch_imgs.shape[1], np_batch_imgs.shape[2])
        return np_batch_imgs, np_batch_gts

    def predict(self, batch, batch_size=1):
        print "TRYING TO DO PREDICTION WITH IMAGE OF SIZE", batch.shape
        # TODO: TF is awful! I can't just make things the shape I want!
        #return np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], self.num_classes))

        #batch = batch[:, :self.height, :self.width, :]
        if batch.shape[-1] == 3:
            print "Predict Batch shape", batch.shape
            batch = np.reshape(np.mean(batch, axis=-1), (batch_size, batch.shape[1], batch.shape[2], 1))
        print "Batch stats:", np.mean(batch), np.std(batch), np.min(batch), np.max(batch)
        logits = self.sess.run(self.y_conv, feed_dict={self.x:batch, self.keep_prob: 1.0})
        print logits.shape
        print "Pred stats:", np.mean(logits), np.std(logits), np.min(logits), np.max(logits)
        return logits

    def fit_generator(self, generator=None, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0):
        if steps_per_epoch is None:
            steps_per_epoch = 5000
        self.epoch = initial_epoch
        self.iteration = initial_epoch * steps_per_epoch

        while self.epoch < epochs:
            print "Epoch:", self.epoch
            for step in range(steps_per_epoch):
                if verbose > 1:
                    print "Training iteration", self.iteration

                if generator is None:
                    batch, targets = self.get_batch(self.batch_size, self.height, self.width)
                    batch = [np.reshape(batch, [-1, self.height, self.width, 1]), targets]
                else:
                    batch, targets = generator.next()
                    if batch.shape[-1] == 3:
                        batch = np.reshape(np.mean(batch, axis=-1), (-1, self.height, self.width, 1))

                if self.profile:
                    merge_reports = tf.summary.merge_all()
                    _, loss, summary = self.sess.run([self.train_step, self.loss, merge_reports], feed_dict={self.x:batch, self.y_:targets, self.keep_prob: self.dropout})
                    self.train_writer.add_summary(summary, self.iteration)
                else:
                    _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.x:batch, self.y_:targets, self.keep_prob: self.dropout})

                # Call all of the callbacks!
                logs = {"loss":loss}
                for callback in callbacks:
                    callback.on_batch_end(batch=self.iteration, logs=logs)
                self.iteration += 1
            # Call all of the callbacks!
            for callback in callbacks:
                callback.on_epoch_end(self.epoch, logs=logs)
            self.epoch += 1
        pass

    def count_params(self):
        return 0

    def summary(self):
        return "Model Summary: Not Implemented"

    def save(self, filepath):
        save_model(self.sess, "./" + filepath + str(self.iteration))

def build_model(args):
    return TFModelKerasStyle(args)
