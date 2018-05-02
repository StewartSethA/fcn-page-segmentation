import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
    sys.path.append(dir_path)
import argparse

parser = argparse.ArgumentParser(description='Semantic Segmentation Trainer')
parser.add_argument('training_folder')
parser.add_argument('validation_folder')
parser.add_argument('test_folder', nargs="?", default="")
parser.add_argument('--batch_size', type=int, default=6, metavar='N', help='input batch size for training (default: 4)')
parser.add_argument('--num_classes', type=int, default=5, metavar='N', help='number of classes (default: 4)')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--crop_size', type=int, default=128, metavar='M', help='training image crop size (default 128)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model_type', type=str, default="densenet", metavar='N', help='model type: can be hourglass, densenet, tensmeyer, resnet, dilatednet, cylindricalresnet')
parser.add_argument('--model_save_path', type=str, default="model_checkpoint.h5", metavar='N', help='model save path')
parser.add_argument('--load_model_path', type=str, default="model_checkpoint.h5", metavar='N', help='model load path. If blank or missing, will not load a model.')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--weight_normalization_interval', type=int, default=-1, metavar='N', help='interval, in batches, for weight normalization (UNSTABLE). -1 (default) means never.')
parser.add_argument('--weight_visualization_interval', type=int, default=-1, metavar='N', help='interval, in batches, for weight visualization (SLOW). -1 (default) means never.')
parser.add_argument('--activation_visualization_interval', type=int, default=-1, metavar='N', help='interval, in batches, for activation visualization (VERY SLOW). -1 (default) means never.')

parser.add_argument('--block_layers', type=int, default=4, metavar='N', help='number of block layers with downsampling (default: 5)')
parser.add_argument('--layers_per_block', type=int, default=3, metavar='N', help='number of layers per block (default: 3)')
parser.add_argument('--initial_features_per_block', type=int, default=12, metavar='N', help='initial number of features per block (default: 8)')
parser.add_argument('--feature_growth_rate', type=int, default=5, metavar='N', help='feature growth rate per layer within a block (default: 4)')
parser.add_argument('--upsampling_path_initial_features', type=int, default=24, metavar='N', help='initial number of features per upsampling block (default: 20)')
parser.add_argument('--upsampling_path_growth_rate', type=int, default=2, metavar='N', help='feature growth rate per upsampling layer within a block (default: 4)')
parser.add_argument('--bottleneck_feats', type=int, default=20, metavar='N', help='intial bottleneck features (default: 20)')
parser.add_argument('--bottleneck_growth_rate', type=int, default=4, metavar='N', help='growth rate of bottleneck features (default: 4)')
parser.add_argument('--batcher', type=str, default="simple", metavar='N', help='Batcher: Enables or disables augmentations. Options are simple (default) and legacy (more augmentations)')
args = parser.parse_args()

#############################################################
# Batch size is small for a low-memory GPU. It can be increased,
# but generally smaller batch sizes lead to faster convergence
# (more parameter updates per unit computation), so it may be fine to keep this low.
batch_size = args.batch_size
num_classes = args.num_classes
in_height = in_width = args.crop_size
training_folder = args.training_folder
validation_folder = args.validation_folder
test_folder = args.test_folder
model_save_path = args.model_save_path
model_type = args.model_type
import numpy as np
np.random.seed(args.seed)  # for reproducibility

#############################################################

last_input = None
last_gt = None

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.40
set_session(tf.Session(config=config))
import keras
import keras.losses
from collections import defaultdict
import matplotlib.pyplot as plt
plt.ion()
from data_loaders import index_training_set_by_class
from data_loaders import ImageAndGTBatcher
from data_loaders import WholeImageOnlyBatcher

from models import *
from inference import *

ds_rate = 1.0
index = index_training_set_by_class(training_folder, num_classes=num_classes)
class_to_samples, image_list, pixel_counts_byclass = index
print("Number of samples per class:", {c:len(class_to_samples[c]) for c in class_to_samples.keys()})
training_generator_class = ImageAndGTBatcher(args.training_folder, num_classes, batch_size, index=index, downsampling_rate=ds_rate, crop_size=in_width)
if args.batcher == 'simple':
    training_generator = training_generator_class.generate(in_width) #mixed_get_batch([training_generator_class.legacy_get_batch(in_width), training_generator_class.generate(in_width)], [0.1, 0.9]) # LEGACY batcher. #training_generator_class.generate(in_width)
else:
    training_generator = training_generator_class.legacy_get_batch(in_width)
validation_folder = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
validation_generator_class = ImageAndGTBatcher(args.validation_folder, num_classes, batch_size=1, downsampling_rate=ds_rate, train=False, cache_images=True)
validation_generator = validation_generator_class.generate()

do_samplegenerator_benchmark = False
if do_samplegenerator_benchmark:
    for i in range(500):
        b,g = training_generator.next()
        print i, b.shape, g.shape
    exit()

import psutil
mem_orig = psutil.virtual_memory().used
print("VIRTUAL MEMORY BEFORE MODEL CREATION:", mem_orig)
if model_type == 'densenet':
    model = densenet_for_semantic_segmentation(num_classes=num_classes, dense_block_init_feats=args.initial_features_per_block,
        dense_block_growth_rate=args.feature_growth_rate, updense_init_feats=args.upsampling_path_initial_features, updense_growth_rate=args.upsampling_path_growth_rate,
        bottleneck_feats=args.bottleneck_feats, bottleneck_growth_rate=args.bottleneck_growth_rate,
        block_layers=args.block_layers, layers_per_block=args.layers_per_block, model_save_path=args.load_model_path, use_transpose_conv=False, use_bias=False)
elif model_type == 'hourglass':
    model = build_simple_hourglass(num_classes=num_classes, init_feats=args.initial_features_per_block, feature_growth_rate=args.feature_growth_rate, ds=args.block_layers)
elif model_type == 'tensmeyer':
    print("Not implemented")
    #build_model_functional_old(num_classes=6, num_feats=[[8, 16, 32, 32, 32, 32], [8,]], ks=[[(3,3),(3,3),(3,3),(5,5),(5,5),(5,5)],[(9,9)]], ds=[[2,2,2,-2,-2,-2],[(1,1)]], combine_modes='concat', output_strides=(1,1), input_channels=3, model_save_path='model.h5', use_transpose_conv=False)
elif model_type == 'dilatednet':
    model = build_simple_hourglass(initial_feats=dense_block_init_feats, ds=args.block_layers)
elif model_type == 'unet':
    model = build_simple_hourglass(initial_feats=dense_block_init_feats, ds=args.block_layers)

mem_after_load = psutil.virtual_memory().used
print("VIRTUAL MEMORY AFTER MODEL CREATION:", mem_after_load)
print("Estimated model size in memory:", mem_after_load - mem_orig)
print("Total parameters:", model.count_params())

print("Model Summary:", model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model.png')

callbacks = []

from callbacks import DisplayAccuracyCallback
callbacks.append(DisplayAccuracyCallback(validation_generator, validation_generator_class, training_generator_class=training_generator_class, pixel_counts_by_class=training_generator_class.dataset_sampler.pixel_counts_byclass, eval_interval=500))

from callbacks import DisplayTrainingSamplesCallback
callbacks.append(DisplayTrainingSamplesCallback(training_generator_class, model=model, interval=100))

from keras.callbacks import ModelCheckpoint
save_model_callback = ModelCheckpoint(filepath = model_save_path, verbose=1, save_best_only=True, period=1)
callbacks.append(save_model_callback)

from keras.callbacks import ReduceLROnPlateau
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00000000001))

from callbacks import LogTimingCallback
callbacks.append(LogTimingCallback(batch_size))

#callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto'))

if args.weight_normalization_interval > 0:
    from callbacks import WeightNormalizationCallback
    callbacks.append(WeightNormalizationCallback(model, interval=args.weight_normalization_interval))

if args.weight_visualization_interval > 0:
    from callbacks import VisualizeWeightsCallback
    callbacks.append(VisualizeWeightsCallback(model, display_interval=args.weight_visualization_interval))

if args.activation_visualization_interval > 0:
    from callbacks import DisplayActivationsCallback
    callbacks.append(DisplayActivationsCallback(model, training_generator, display_interval=args.activation_visualization_interval))

if len(args.test_folder) > 0:
    from callbacks import ShowTestPredsCallback
    print("Showing test set predictions during training...")
    callbacks.append(ShowTestPredsCallback(WholeImageOnlyBatcher(test_folder, batch_size=1)))

# Stats of interest:
# Average f-measure on the entire validation set
# Average time in seconds per image of validation inference
# Average time in seconds per pixel of validation inference
# Average change in loss per training sample seen
# Average time in seconds per training sample/batch

print("Training!")
# Profile memory actually used during training:
import tensorflow as tf
from tensorflow.python.client import timeline
with K.get_session() as s:
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()
    #import cProfile
    #import re
    #cProfile.run('re.compile("foo|bar")')
    model.fit_generator(generator=training_generator, epochs=args.epochs, steps_per_epoch=500, validation_data=validation_generator, validation_steps=1, callbacks=callbacks, max_queue_size=4*batch_size, workers=1, use_multiprocessing=False) #, use_multiprocessing=True, workers=10) #validation_data=validation_generator, validation_steps=1, max_queue_size=100) #, show_accuracy=True)

    #to = timeline.Timeline(run_metadata.step_stats)
    #trace = to.generate_chrome_trace_format()
    #with open('full_trace.json', 'w') as out:
    #    out.write(trace)

# Now perform testing!!!
if len(args.test_folder) > 0:
    print("Testing!")
    TestModel(model=model, model_basepath="best_model", testfolder=test_folder, pixel_counts_byclass=training_generator_class.dataset_sampler.pixel_counts_byclass)

#testmodelcb = TestModelCallback(model, save_basepath="best_model", testfolder=testfolder, testscale=1.0)
#callbacks.append(testmodelcb)
#callbacks.append(TensorBoardWrapper(validation_generator, nb_steps=1, log_dir='./tensorboard_logs', histogram_freq=1, write_graph=True)) #, write_images=True, embeddings_freq=1, write_grads=True)) #, write_grads=True, write_images=True, embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None))
