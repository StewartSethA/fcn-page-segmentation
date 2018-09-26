# fcn-page-segmentation

This trainable page segmentation system uses fully-convolutional neural networks (FCNs) to label the contents of document images.

To train a Keras U-Net model on a dataset in the folder ./data/, run the following:
```
python train.py --training_folder ./data/training/ --validation_folder ./data/validation/ --framework keras --model_type unet
```

To perform inference on all images in a folder ./images/ using a saved model ./model_snapshot.h5, run the following:
```
python infer.py --test_folder ./images/ --framework keras --load_model_path ./model_snapshot.h5
```

## Input data format
Ground-truth annotations are of the format 
```
.
|--- training_folder
|--- |--- Image1.jpg
|--- |--- Image1_handwriting.png
|--- |--- Image1_machineprint.png
|--- |--- Image1_stamps.png
```
etc., where the .png files are binary or greyscale images for each channel or class, of the same dimensions as the .jpg file sharing the root name before the underscore. Any channels that are omitted for a given image are assumed to be zero everywhere.
When training or inferencing, the names and number of classes will be determined automatically based on the contents of the training folder.
Alternatively, you can encode binary multi-label ground-truth as a .png file with the same base name as the input image ("Image1.jpg", ground truth as "Image1.png") using the RGB bit-masked format introduced 
[here.](https://diuf.unifr.ch/main/hisdoc/icdar2017-hisdoc-layout-comp)

The RGB encoding for such images places classes as single bits in the 24-bit RGB color encoding, with class 1 at the lowest-order bit position, class 2 at the next lowest bit position, and so on.
The highest bit position is reserved for an ambiguity mask, which simply refuses to penalize a classifier for labeling pixels at corresponding locations as either background or the foreground pixels (TODO: This is not yet implemented in scoring functions).
Up to 23 classes can be encoded using this scheme. Note that only binary (no grayscale) values can be encoded using this format.
