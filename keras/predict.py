import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
    sys.path.append(dir_path)
import argparse

parser = argparse.ArgumentParser(description='Semantic Segmentation Trainer')
parser.add_argument('input_folder', nargs="?", default="./input")
parser.add_argument('load_model_path', nargs="?", type=str, default="model_checkpoint.h5", metavar='N', help='model load path. If blank or missing, will not load a model.')
parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 4)')
parser.add_argument('--num_classes', type=int, default=4, metavar='N', help='number of classes (default: 4)')
parser.add_argument('--crop_size', type=int, default=128, metavar='M', help='training image crop size (default 128)')
args = parser.parse_args()

#############################################################
# Batch size is small for a low-memory GPU. It can be increased,
batch_size = args.batch_size
num_classes = args.num_classes
in_height = in_width = args.crop_size
input_folder = args.input_folder
load_model_path = args.load_model_path
import numpy as np

#############################################################

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.90
set_session(tf.Session(config=config))
import keras

from keras.models import load_model
from inference import TestModel

# We might need these imports, if a model uses a custom loss function or regularizer.
from regularizers import *
from losses import *
ds_rate = 1.0
import psutil
mem_orig = psutil.virtual_memory().used
print("VIRTUAL MEMORY BEFORE MODEL CREATION:", mem_orig)
model = load_model(load_model_path)
mem_after_load = psutil.virtual_memory().used
print("VIRTUAL MEMORY AFTER MODEL CREATION:", mem_after_load)
print("Estimated model size in memory:", mem_after_load - mem_orig)
print("Total parameters:", model.count_params())

callbacks = []

# TODO: Monitor maximum memory usage during inference.
import time
t = time.time()
TestModel(model=model, model_basepath="best_model", testfolder=input_folder) #, pixel_counts_byclass=training_generator_class.dataset_sampler.pixel_counts_byclass)
time_elapsed = time.time() - t
print("Elapsed time:", time_elapsed, "seconds.")
# TODO: Load cached pixel counts by class.
# TODO: Implement chunked batcher for inference. This would be very good to do.

#testmodelcb = TestModelCallback(model, save_basepath="best_model", testfolder=testfolder, testscale=1.0)
#callbacks.append(testmodelcb)
#callbacks.append(TensorBoardWrapper(validation_generator, nb_steps=1, log_dir='./tensorboard_logs', histogram_freq=1, write_graph=True)) #, write_images=True, embeddings_freq=1, write_grads=True)) #, write_grads=True, write_images=True, embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None))

# TODO: Display overlay of predicted pixels with original image.
