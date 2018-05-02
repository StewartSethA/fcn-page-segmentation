import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
    sys.path.append(dir_path)

#############################################################
# Parse command-line arguments.
# Batch size is small for a low-memory GPU. It can be increased,
# but generally smaller batch sizes lead to faster convergence
# (more parameter updates per unit computation), so it may be fine to keep this low.
from argparser import parse_args

#############################################################
# Import and configure Deep Learning and visualization frameworks.
import numpy as np
import cv2
import random
from collections import defaultdict
import matplotlib.pyplot as plt
plt.ion()
from data_loaders.data_loaders import index_training_set_by_class
from data_loaders.data_loaders import ImageAndGTBatcher
from data_loaders.data_loaders import WholeImageOnlyBatcher
from data_loaders.training_sample_generators import *
from collections import defaultdict
from models.model import build_model
from predict import *
#############################################################

def train(args):
    batch_size = args.batch_size
    num_classes = args.num_classes
    in_height = in_width = args.crop_size
    training_folder = args.training_folder
    validation_folder = args.validation_folder
    test_folder = args.test_folder
    model_save_path = args.model_save_path
    model_type = args.model_type
    ds_rate = 1.0
    
    # Index the training set for efficient caching and batching, including class rebalancing.
    print "Training folder", training_folder
    index = index_training_set_by_class(training_folder, num_classes=num_classes)
    class_to_samples, image_list, pixel_counts_byclass = index
    print("Number of samples per class:", {c:len(class_to_samples[c]) for c in class_to_samples.keys()})
    training_generator_class = ImageAndGTBatcher(args.training_folder, num_classes, batch_size, index=index, downsampling_rate=ds_rate, crop_size=in_width)
    if args.batcher == 'simple':
        training_generator = training_generator_class.generate(in_width)
    elif args.batcher == 'mixed':
        training_generator = mixed_get_batch([training_generator_class.legacy_get_batch(in_width), training_generator_class.generate(in_width)], [0.1, 0.9]) 
    else:
        training_generator = training_generator_class.legacy_get_batch(in_width)
    
    # Optionally include a validation set for display, early stopping, and mini-validation during training.
    # If none is supplied, the training set will be used for real-time display, etc.
    print "Validation folder", validation_folder
    validation_folder = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    validation_generator_class = ImageAndGTBatcher(args.validation_folder, num_classes, batch_size=1, downsampling_rate=ds_rate, train=False, cache_images=True)
    validation_generator = validation_generator_class.generate()

    # For model benchmarking purposes, estimate the memory consumed by model instantiation.
    import psutil
    mem_orig = psutil.virtual_memory().used
    print("VIRTUAL MEMORY BEFORE MODEL CREATION:", mem_orig)
    
    # BUILD the appropriate model, depending on the framework and model construction parameters.
    model = build_model(args)

    # Now print memory stats, including estimated model size.
    mem_after_load = psutil.virtual_memory().used
    print("VIRTUAL MEMORY AFTER MODEL CREATION:", mem_after_load)
    print("Estimated model size in memory:", mem_after_load - mem_orig)
    print("Total parameters:", model.count_params())

    # Print a model summary for debugging.
    print("Model Summary:", model.summary())
    
    # Plot the model as a chart and save to a PNG file
    # TODO Extend TF and PyTorch models to do the same!
    if args.framework.lower() == "keras":
        from keras.utils import plot_model
        plot_model(model, to_file='model.png')

    # CREATE callbacks.
    callbacks = []

    from callbacks import DisplayAccuracyCallback
    callbacks.append(DisplayAccuracyCallback(model, validation_generator, validation_generator_class, training_generator_class=training_generator_class, pixel_counts_by_class=training_generator_class.dataset_sampler.pixel_counts_byclass, eval_interval=500))

    from callbacks import DisplayTrainingSamplesCallback
    callbacks.append(DisplayTrainingSamplesCallback(training_generator_class, model=model, interval=100))

    if args.framework.lower() == "keras":
        from keras.callbacks import ModelCheckpoint
        save_model_callback = ModelCheckpoint(filepath = model_save_path, verbose=1, save_best_only=True, period=1)

        from keras.callbacks import ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00000000001))
    
        #callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto'))
    else:
        from callbacks import TFModelSaverCallback
        save_model_callback = TFModelSaverCallback(model_save_path, model.sess)
    callbacks.append(save_model_callback)
    
    from callbacks import LogTimingCallback
    callbacks.append(LogTimingCallback(batch_size))

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
    # Average time in seconds per training sample/batch/pixel

    # PERFORM TRAINING!!!!
    print("Training!")
    model.fit_generator(generator=training_generator, epochs=args.epochs, steps_per_epoch=5000, validation_data=validation_generator, validation_steps=1, callbacks=callbacks, max_queue_size=4*batch_size, workers=1, use_multiprocessing=False)

    #testmodelcb = TestModelCallback(model, save_basepath="best_model", testfolder=testfolder, testscale=1.0)
    #callbacks.append(testmodelcb)
    #TODO: callbacks.append(TensorBoardWrapper(validation_generator, nb_steps=1, log_dir='./tensorboard_logs', histogram_freq=1, write_graph=True)) #, write_images=True, embeddings_freq=1, write_grads=True)) #, write_grads=True, write_images=True, embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None))

'''
A Trainer consumes a training dataset of the following format:
{
"class_labels": {0: "dotted lines", 1: "handwriting", 2: "machine print", 3: "solid lines", 4: "stamps"}
"abbrev_class_labels": [0: "DL", 1: "HW", 2: "MP", 3: "LN", 4: "ST"]
"training_instance_paths":
    [  
    "path1",
    "path2",
    ...
    ]
"training_gt_paths":
    [
    "gt_path1",
    "gt_path2",
    ...
    ]
"training_instance_dirs":
    [
    "dir1",
    "dir2",
    ...
    ]


}
'''

if __name__ == "__main__":
    args = parse_args()
    import numpy as np
    np.random.seed(args.seed)  # for reproducibility
    train(args)

