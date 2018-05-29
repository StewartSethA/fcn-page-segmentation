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

def validate(args):
    test_folder = args.test_folder
    model = build_model(args)
    TestModel(model=model, model_basepath=args.load_model_path, testfolder=test_folder, output_folder=args.output_folder)
    score_and_visualize_folders(test_folder, args.output_folder)


# TODO: Monitor maximum memory usage during inference.
# TODO: Load cached pixel counts by class.
# TODO: Implement chunked batcher for inference. This would be very good to do.

# TODO: Display overlay of predicted pixels with original image.

if __name__ == "__main__":
    args = parse_args()
    import numpy as np
    np.random.seed(args.seed)  # for reproducibility
    validate(args)
