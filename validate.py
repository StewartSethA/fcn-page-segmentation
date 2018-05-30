from __future__ import print_function
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
    sys.path.append(dir_path)

#############################################################
# Parse command-line arguments.
from argparser import parse_args

#############################################################
# Import and configure Deep Learning and visualization frameworks.
from models.model import build_model
from evaluation.inference import TestModel
from evaluation.evaluations import score_and_visualize_folders
from data_loaders.gt_loaders import autodiscover_suffix_to_class_map

def validate(args):
    test_folder = args.test_folder
    print("Test folder", test_folder)
    suffix_to_class_map = autodiscover_suffix_to_class_map(test_folder, ["jpg", "png", "tif"])
    print("Inferred suffix to class map:", suffix_to_class_map)
    if len(suffix_to_class_map) > 0:
        num_classes = len(suffix_to_class_map)
        args.num_classes = num_classes
    model = build_model(args)
    TestModel(model=model, model_basepath=args.load_model_path, testfolder=test_folder, output_folder=args.output_folder)
    score_and_visualize_folders(test_folder, args.output_folder, suffix_to_class_map=suffix_to_class_map)


# TODO: Monitor maximum memory usage during inference.
# TODO: Load cached pixel counts by class.
# TODO: Implement chunked batcher for inference. This would be very good to do.

# TODO: Display overlay of predicted pixels with original image.

if __name__ == "__main__":
    args = parse_args()
    import numpy as np
    np.random.seed(args.seed)  # for reproducibility

    validate(args)
