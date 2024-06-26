
from __future__ import print_function
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
    sys.path.append(dir_path)

#############################################################
# Parse command-line arguments.
from argparser import parse_args
from models.pytorch_models import build_model

#############################################################
# Import and configure Deep Learning and visualization frameworks.
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
plt.ion()
from data_loaders.gt_loaders import autodiscover_suffix_to_class_map
from data_loaders.data_loaders import index_training_set_by_class
from data_loaders.data_loaders import ImageAndGTBatcher
from data_loaders.data_loaders import WholeImageOnlyBatcher
from data_loaders.training_sample_generators import *
from collections import defaultdict
from models.model import build_model
from infer import *
from utils import _mkdir
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

    print("")
    print("Using framework", args.framework)
    print("Epochs:", args.epochs)
    print("Steps per epoch:", args.steps_per_epoch)
    print("")

    # Index the training set for efficient caching and batching, including class rebalancing.
    print("Training folder", training_folder)
    suffix_to_class_map = autodiscover_suffix_to_class_map(training_folder, ["jpg", "png", "tif"])
    print("Inferred suffix to class map:", suffix_to_class_map)
    if len(suffix_to_class_map) > 0:
        num_classes = len(suffix_to_class_map)
        args.num_classes = num_classes
    index = index_training_set_by_class(training_folder, num_classes=num_classes, suffix_to_class_map=suffix_to_class_map)
    class_to_samples, image_list, pixel_counts_byclass = index
    if type(args.loss_weights) == str:
        sqrt=True
        import math
        pixsum = float(sum([math.sqrt(pixel_counts_byclass[c]) for c in pixel_counts_byclass.keys()]))
        args.loss_weights = {c:pixsum/math.sqrt(pixel_counts_byclass[c]) for c in pixel_counts_byclass.keys()}
        print("Auto-discovered class weights from pixel prevalence in training set:", args.loss_weights)
    if len(image_list) == 0:
        print("No images found in training folder! Cannot train; aborting!")
        exit()
    if sum([len(class_to_samples[c]) for c in class_to_samples.keys()]) == 0:
        print("No class samples indexed! Cannot train; aborting!", class_to_samples)
        exit()
    print("Number of samples per class:", {c:len(class_to_samples[c]) for c in class_to_samples.keys()})
    training_generator_class = ImageAndGTBatcher(args.training_folder, num_classes, batch_size, index=index, downsampling_rate=ds_rate, crop_size=in_width, suffix_to_class_map=suffix_to_class_map)
    if args.batcher == 'simple':
        training_generator = training_generator_class.generate(in_width)
    elif args.batcher == 'mixed':
        training_generator = mixed_get_batch([training_generator_class.legacy_get_batch(in_width), training_generator_class.generate(in_width)], [0.1, 0.9])
    else:
        training_generator = training_generator_class.legacy_get_batch(in_width, min_height_scale=args.min_height_scale, max_height_scale=args.max_height_scale)

    # Optionally include a validation set for display, early stopping, and mini-validation during training.
    # If none is supplied, the training set will be used for real-time display, etc.
    print("Validation folder", validation_folder)
    validation_folder = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    validation_generator_class = ImageAndGTBatcher(args.validation_folder, num_classes, crop_size=args.crop_size, batch_size=1, downsampling_rate=ds_rate, train=False, cache_images=True, suffix_to_class_map=suffix_to_class_map)
    validation_generator = validation_generator_class.generate()

    # For model benchmarking purposes, estimate the memory consumed by model instantiation.
    import psutil
    mem_orig = psutil.virtual_memory().used
    print("VIRTUAL MEMORY BEFORE MODEL CREATION:", mem_orig)

    if not os.path.exists(args.log_dir):
        _mkdir(args.log_dir)

    print(">>>> Log dir", args.log_dir, "Save path", args.model_save_path)
    if args.log_dir != os.path.dirname(args.model_save_path):
        args.model_save_path = os.path.join(args.log_dir, args.model_save_path)
        print("UPDATED", args.model_save_path)
        model_save_path = args.model_save_path

    # BUILD the appropriate model, depending on the framework and model construction parameters.
    if args.framework.lower() == "keras" and not ".h5" in args.load_model_path[-3:]:
        args.load_model_path = args.load_model_path + ".h5"
    model = build_model(args)

    # Now print(memory stats, including estimated model size.)
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
        plot_model(model, to_file=os.path.join(args.log_dir, 'model.png'))

    # CREATE callbacks.
    callbacks = []

    from callbacks import DisplayAccuracyCallback
    dac = DisplayAccuracyCallback(model, validation_generator, validation_generator_class, training_generator_class=training_generator_class, pixel_counts_by_class=training_generator_class.dataset_sampler.pixel_counts_byclass, eval_interval=args.log_interval, log_dir=args.log_dir)
    callbacks.append(dac)

    from callbacks import DisplayTrainingSamplesCallback
    callbacks.append(DisplayTrainingSamplesCallback(training_generator_class, model=model, interval=args.training_sample_visualization_interval, log_dir=args.log_dir, dac=dac))

    if args.framework.lower() == "keras":
        if not ".h5" in model_save_path[-3:]:
            model_save_path = model_save_path + ".h5"
        from keras.callbacks import ModelCheckpoint
        #save_model_callback = ModelCheckpoint(filepath = model_save_path, verbose=1, save_best_only=True, period=1)
        # We don't want to cheat since our validation images are our test images, so we shouldn't allow any flow of information to the model we use for testing/validation.
        save_model_callback = ModelCheckpoint(filepath = model_save_path, verbose=1, save_best_only=False, period=1)
        from callbacks import SaveEveryEpochCallback
        callbacks.append(SaveEveryEpochCallback(model, model_save_path, args.model_save_interval))
        from keras.callbacks import ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00000000001))

        #callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto'))
    else:
        from callbacks import TFModelSaverCallback
        save_model_callback = TFModelSaverCallback(model, model_save_path, args.model_save_interval)
    callbacks.append(save_model_callback)

    #from callbacks import DisplayWeightStatsCallback
    #callbacks.append(DisplayWeightStatsCallback(model))

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

    # PERFORM TRAINING!!!!
    print("Training!")
    #try:
    #model.fit_generator(generator=training_generator, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, validation_data=validation_generator, validation_steps=1, callbacks=callbacks, max_queue_size=4*batch_size, workers=1, use_multiprocessing=False)
    model.fit_generator(generator=training_generator, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, validation_data=None, validation_steps=-1, callbacks=callbacks, max_queue_size=4*batch_size, workers=1, use_multiprocessing=True)
    #except Exception as ex:
    #    print("Exception caught!")
    #    print(ex)
    print("Training is COMPLETE.")
    for i in range(100):
        print("")
    #testmodelcb = TestModelCallback(model, save_basepath="best_model", testfolder=testfolder, testscale=1.0)
    #callbacks.append(testmodelcb)
    #TODO: callbacks.append(TensorBoardWrapper(validation_generator, nb_steps=1, log_dir='./tensorboard_logs', histogram_freq=1, write_graph=True)) #, write_images=True, embeddings_freq=1, write_grads=True)) #, write_grads=True, write_images=True, embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None))

if __name__ == "__main__":
    args = parse_args()
    import numpy as np
    np.random.seed(args.seed)  # for reproducibility
    train(args)
