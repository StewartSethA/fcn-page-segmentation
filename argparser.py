import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Trainer')
    parser.add_argument('--training_folder')
    parser.add_argument('--validation_folder')
    parser.add_argument('--test_folder', nargs="?", default="")
    parser.add_argument('--framework', type=str, default="tensorflow", help="Deep learning framework to use for model and training. Options: keras, tensorflow (Default: tensorflow)")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--loss', type=str, default="categorical_crossentropy", help="Loss function to use (default: categorical_crossentropy)")

    # Training parameters.
    parser.add_argument('--num_classes', type=int, default=5, metavar='N', help='number of classes (default: 4)')

    # Batch size is small for a low-memory GPU. It can be increased,
    # but generally smaller batch sizes lead to faster convergence
    # (more parameter updates per unit computation), so it may be fine to keep this low.
    parser.add_argument('--batch_size', type=int, default=-1, metavar='N', help='input batch size for training (default: -1; uses number of classes)')
    parser.add_argument('--crop_size', type=int, default=224, metavar='M', help='training image crop size (default 128)')
    parser.add_argument('--batcher', type=str, default="legacy", metavar='N', help='Batcher: Enables or disables augmentations. Options are simple (default) and legacy (more augmentations)')

    parser.add_argument('--epochs', type=int, default=40, metavar='N', help='number of epochs to train (default: 40)')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, metavar='N', help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.01)')

    # Model persistence parameters.
    parser.add_argument('--model_save_path', type=str, default="model_checkpoint", metavar='N', help='model save path')
    parser.add_argument('--load_model_path', type=str, default="model_checkpoint", metavar='N', help='model load path. If blank or missing, will not load a model.')
    parser.add_argument('--model_save_interval', type=int, default=1000, help='Number of batches between model saves')

    # Logging and visualizations.
    parser.add_argument('--output_folder', type=str, default="./", help='Path to store output of inference.')
    parser.add_argument('--log_dir', type=str, default="./", help='Path to save logs, visuals, and predictions.')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--weight_visualization_interval', type=int, default=-1, metavar='N', help='interval, in batches, for weight visualization (SLOW). -1 (default) means never.')
    parser.add_argument('--activation_visualization_interval', type=int, default=-1, metavar='N', help='interval, in batches, for activation visualization (VERY SLOW). -1 (default) means never.')
    parser.add_argument('--training_sample_visualization_interval', type=int, default=100)

    # Model parameters.
    parser.add_argument('--model_type', type=str, default="hourglass", metavar='N', help='model type: can be hourglass, densenet, tensmeyer, resnet, dilatednet, cylindricalresnet')
    parser.add_argument('--block_layers', type=int, default=4, metavar='N', help='number of block layers with downsampling (default: 5)')
    parser.add_argument('--layers_per_block', type=int, default=4, metavar='N', help='number of layers per block (default: 3)')
    parser.add_argument('--initial_features_per_block', type=int, default=8, metavar='N', help='initial number of features per block (default: 8)')
    parser.add_argument('--feature_growth_rate', type=int, default=16, metavar='N', help='feature growth rate per layer within a block (default: 4)')
    parser.add_argument('--upsampling_path_initial_features', type=int, default=24, metavar='N', help='initial number of features per upsampling block (default: 20)')
    parser.add_argument('--upsampling_path_growth_rate', type=int, default=2, metavar='N', help='feature growth rate per upsampling layer within a block (default: 4)')
    parser.add_argument('--bottleneck_feats', type=int, default=20, metavar='N', help='intial bottleneck features (default: 20)')
    parser.add_argument('--bottleneck_growth_rate', type=int, default=4, metavar='N', help='growth rate of bottleneck features (default: 4)')
    parser.add_argument('--weight_normalization_interval', type=int, default=-1, metavar='N', help='interval, in batches, for weight normalization (UNSTABLE). -1 (default) means never.')
    parser.add_argument('--dropout_rate', type=float, default=0.9)

    args = parser.parse_args()
    if args.batch_size == -1:
        args.batch_size = args.num_classes
    return args
