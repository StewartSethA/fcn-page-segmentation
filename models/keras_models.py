from __future__ import print_function
import numpy as np
from nn_utils import model_size
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dropout, Activation, LeakyReLU, Layer, SeparableConv2D, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2DTranspose, SeparableConv2D
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras.layers import BatchNormalization, Concatenate, concatenate, add
from keras.layers import Input, merge, maximum, multiply, Lambda, UpSampling2D, GaussianDropout
from keras.models import Model
from keras.models import load_model
import keras.losses
from losses import *
from metrics import *
from keras.backend import spatial_2d_padding as pad2d
from keras.layers import Conv2D as Conv2D_keras
class LeakyRELU(LeakyReLU):
    def __init__(self, *args, **kwargs):
        self.__name__ = "LeakyRELU"
        super(LeakyRELU, self).__init__(*args, **kwargs)

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'LeakyRELU':LeakyRELU})

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from collections import defaultdict

def lookup_loss(loss, args):
    import sys
    current_module = sys.modules[__name__]
    if loss == "get_per_class_margin":
        loss = get_per_class_margin(args.loss_weights)
    elif loss == "blurred_continuous_f_measure":
        loss = blurred_continuous_f_measure(args.loss_weights, args.loss_blur_sigma)
    elif loss == "continuous_f_measure":
        loss = continuous_f_measure(args.loss_weights)
    elif hasattr(current_module, loss):
        loss = getattr(current_module, loss)
    return loss


def reluclip(x, max_value=1.0):
    return K.relu(x, max_value=max_value)

def conv_cylinder(feats, ks=3, nonlinearity=reluclip):
    pass

def unet(args):
    num_classes = args.num_classes
    initial_feats = f = args.initial_features_per_block #4 #8 #16 # Original batch was 32
    inputs = Input((None, None, args.input_channels))
    dropouts = [0, 0.15, 0.15, 0.5, 0.5, 0.5, 0.5, 0.5]
    x = inputs
    downsampled = []
    for l in range(args.block_layers):
        for bl in range(args.layers_per_block):
            x = Conv2D(f, args.kernel_size, activation = LeakyRELU(args.lrelu_alpha), padding = 'same', kernel_initializer = 'he_normal', use_bias=args.use_bias)(x)
            if args.batch_normalization:
                x = BatchNormalization()(x)
        print("x shape:",x.shape)
        x = Dropout(dropouts[l])(x)
        if l != args.block_layers-1:
            downsampled.append(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        if args.batch_normalization:
            x = BatchNormalization()(x)
        print("x shape:",x.shape)
        f *= 2
    # Upsampling layers!
    for l in range(args.block_layers):
        f /= 2
        x = UpSampling2D(size = (2,2))(x)
        x = Conv2D(f, 2, activation = LeakyRELU(args.lrelu_alpha), padding = 'same', kernel_initializer = 'he_normal', use_bias=args.use_bias)(x)
        x = merge([downsampled[-l-1], x], mode = 'concat', concat_axis = 3)
        for bl in range(args.layers_per_block):
            x = Conv2D(f, args.kernel_size, activation = LeakyRELU(args.lrelu_alpha), padding = 'same', kernel_initializer = 'he_normal', use_bias=args.use_bias)(x)
            if args.batch_normalization:
                x = BatchNormalization()(x)
    # Classification layers.
    x = Conv2D(num_classes*2, args.kernel_size, activation = LeakyRELU(args.lrelu_alpha), padding = 'same', kernel_initializer = 'he_normal', use_bias=args.use_bias)(x)
    if args.batch_normalization:
        x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, activation = 'sigmoid', use_bias=args.use_bias)(x)
    #conv10 = Lambda(lambda x:x*10)(conv10)
    #conv10 = Lambda(function=reluclip)(conv10)

    model = Model(input = inputs, output = x)

    model.compile(optimizer = Adam(lr = 1e-4), loss = lookup_loss(args.loss, args), metrics = ['accuracy'])

    return model

# Simplest nets first.

# Simple L-Layer CNN with NO downsampling.
def template_matcher_single_hidden_layer(args):
    input_channels = args.input_channels
    num_classes = args.num_classes
    use_bias = args.use_bias
    kernel_size = ks = args.initial_kernel_size
    feats = args.initial_features_per_block

    # Input layer
    y = model_inputs = Input(shape=(None, None, input_channels))
    y = Conv2D(args.initial_features_per_block, args.initial_kernel_size, padding='same', use_bias=use_bias)(y)
    y = LeakyRELU(args.lrelu_alpha)(y)
    y = Dropout(args.dropout_rate)(y)
    if args.batch_normalization:
        y = BatchNormalization()(y)
    
    ks = args.kernel_size
    # Block layers.
    for layer_num in range(args.block_layers-1):
        y = Conv2D(feats, ks, padding='same', use_bias=use_bias)(y)
        y = LeakyRELU(args.lrelu_alpha)(y)
        y = Dropout(args.dropout_rate)(y)
        if args.batch_normalization:
            y = BatchNormalization()(y)

        if args.feature_growth_type == 'add':
            feats += args.feature_growth_rate
        else:
            feats *= args.feature_growth_rate
    
    # Classification layer
    y = Conv2D(num_classes, (1,1), padding='same', use_bias=use_bias)(y)
    y = Activation('sigmoid')(y)
    #y = Lambda(function=reluclip)(y)

    predictions = y
    model = Model(inputs=model_inputs, outputs=predictions)
    model.compile(loss=lookup_loss(args.loss, args), metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=args.lr, clipvalue=0.5)) #optimizer='nadam') #'adadelta')

    return model



# Replace convolution with normalizing convolution.
#def Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
#

def make_multihead(layer, heads=16, nf=32, atom_nf=2, ks=(3,3), d=1, drop=0.10, rate=1, functions=None, residual=False):
    print("multihead")
    head_outputs = []
    first = layer
    make_functions = False
    if functions is None:
        make_functions = True
        functions = []
    for head_num in range(heads):
        #print("head_num", head_num)
        layer = first
        if make_functions:
            function = Conv2D(atom_nf, (1,1), use_bias=False, activation='linear',padding='same')
            functions.append([])
            functions[head_num].append(function)
        else:
            function = functions[head_num][0]
        depthwise_conv = function(layer)
        layer = depthwise_conv
        start_layer = layer
        for d_e in range(d):
            #print("depth", d_e)
            #layer = GaussianDropout(drop)(layer)
            if make_functions:
                function = Conv2D(atom_nf, ks, use_bias=False, dilation_rate=rate, activation='linear',padding='same')
                functions[head_num].append(function)
            else:
                function = functions[head_num][d_e+1]
            layer = function(layer)
            #layer = BatchNormalization()(layer)
            layer = LeakyRELU(alpha=0.01)(layer)

        if residual:
            layer = keras.layers.Add()([start_layer, layer])

        output = layer
        head_outputs.append(layer)
    multihead_output = concatenate(head_outputs)
    return multihead_output, functions

class SegmentationAwareConvolution2D(Layer):
    def __init__(self):
        pass

# Spatial Warping Layer:
# https://github.com/WeidiXie/New_Layers-Keras-Tensorflow
#

# Lateral inhibition <= zero-mean filters.
# See https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/
# https://github.com/keras-team/keras/issues/4741
# http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

# Input-standardized convolution: Convolve with all 1s/kernel_size of kernel size.
# This is the mean input value over that neighborhood. Subtract that from the input
# for each output channel, and convolve.
# But zero-mean input has no effect, so long as the kernels themselves are zero-mean.

# Zero-mean the kernel, then use a local attention mask.
# A local attention mask will produce a K-dimensional output. Reshape this to match the kernel size.
# Then just do a dot product with the

# The embedding segmentation mask is computed per location pair, mi_j
def compute_embedding_segmentation_masks(segmentation_embeddings, kernel_size, norm=1, dilation_rate=1):
    # TODO: Look at https://arxiv.org/pdf/1605.09673.pdf Dynamic Filter Networks and implement it!

    # Transform the HxWxD segmentation embeddings into an HxW mask.
    # PYGO: We could do a per-weight (input feature channel to output feature channel)
    # mask, OR we could just do a single mask per pixel.
    # WARNING: If it is just one mask, then it is likely to blend away overlapped regions, where multiple classes are present.
    # An improved version could allow information to flow across multiple channels
    # from different locations.
    # OPEN QUESTION: How does this relate to dynamic routing (since they do overlap reconstruction)?

    # Perform im2col to extract the local neighborhoods of each pixel, using kernel_size.
    # This is time efficient but not memory efficient, since embeddings are copied K times
    # prior to distance computation.
    # im2col will be much easier to access in PyTorch: https://github.com/pytorch/pytorch/pull/2580
    # https://github.com/szagoruyko/pyinn/blob/master/pyinn/im2col.py
    # This is like implementing KNN within a neural network. Cool!
    # I have an idea for this. Why don't we compute these symetrically? (cut cost by 1/2)?
    # BUT we could simple have a "merge features" routing mechanism for similar features to join
    # in the network while dissimilar features don't join.
    neigborhood = segmentation_embeddings

    # Let's do this with a for loop if we can.
    # i-1, j-1 comparison: (i+1,j+1 for i-1,j-1 centered kernel)
    pse = pad2d(segmentation_embeddings, padding=((dilation_rate,0), (dilation_rate,0)))
    up_left_distances = K.abs(segmentation_embeddings - pad2d(segmentation_embeddings[:,dilation_rate:,dilation_rate:,:], padding=((0,dilation_rate), (0,dilation_rate))))
    left_distances    = K.abs(segmentation_embeddings - pad2d(segmentation_embeddings[:,0:,dilation_rate:,:], padding=((0,0), (0,dilation_rate))))
    up_distances      = K.abs(segmentation_embeddings - pad2d(segmentation_embeddings[:,dilation_rate:,0:,:], padding=((0,dilation_rate), (0,0))))
    self_distances    = K.zeros(segmentation_embeddings.shape) # Can do without this. Just set the self-weight to 1 (identity).
    down_right_distances   = keras.backend.spatial_2d_padding(left_distances[:,2*dilation_rate:,2*dilation_rate:,:], padding=((0,dilation_rate), (0,dilation_rate)))
    right_distances   = keras.backend.spatial_2d_padding(left_distances[:,:,2*dilation_rate:,:], padding=((0,0), (0,dilation_rate)))
    down_distances    = keras.backend.spatial_2d_padding(left_distances[:,2*dilation_rate:,:,:], padding=((0,dilation_rate), (0,0)))
    up_right_distances = None
    down_left_distances = None


# I get it now! A segmentation-aware network computes a single embedding per pixel at each scale.
# Then these local embedding similarity masks are computed and used at each stage in the computation.
# In other words, you have one network that is trained to optimize pairwise distances in the embedding space
# between pixels based on their class in the embedding space, and another network that learns convolutional
# filters at each layer to transform inputs to outputs. The convolutional kernels themselves can be agnostic
# to the embedding similarity masks, and if I understood the paper right, they are global and are reused at each
# convolution operator at the same scale.
# THEN, the embedding difference maps are used to mask the local inputs to each convolutional kernel.

# Implementation: HxW embedding feature map. Extract K pairwise distances from each pixel's neighborhood.
# Make this into an HxWxK matrix.
# Then take the input feature map. Extract K elements by sampling the local neighborhood features.
# This makes the input feature map HxWxKxFin.
# Then take the KxFout convolutional kernels, and multiply:
# emb(HxWxK) x inp(HxWxKxFin) x conv(KxFout) ->
#    out(HxWxFout).
def embedding_loss(y_true, y_pred, norm=1, similar_radius=0.5, different_margin=2.0, neighborhood_size=(3,3), neighborhood_dilations=[1,2,5]):
    # Compute pairwise normed distances between pixels in neighborhoods.
    return 0

class euclidean_network(Layer):
    pass

# I think I might know enough to be able to implement this now.

def make_multirate_multihead(layer, heads=16, nf=32, atom_nf=2, ks=(3,3), d=1, drop=0.10, rates=[1,2,4,8], residual=False):
    print("multirate_multihead")
    first = layer
    functions = None
    concatted_outputs = []
    for rate in rates:
        print("rate", rate)
        # Tie parameters among different rates. This is gonna be cool.
        output, functions = make_multihead(first, heads=heads, nf=nf, atom_nf=atom_nf, ks=ks, d=d, drop=drop, rate=rate, functions=functions, residual=residual)
        concatted_outputs.append(output)
    multirate_output = concatenate(concatted_outputs)
    return multirate_output

def build_dilated_net_layers(layer, num_classes=6, num_input_channels=3, depth=5, attention_size=None): #(3,3)):
    # No max pooling. Just local convolutional attention over inputs (using a large-ish mask or high rate parameter?)
    heads = 8
    atom_nf=2
    rates=[1,2]
    layer = make_multirate_multihead(layer, heads=heads, atom_nf=atom_nf, ks=(11,11), d=1, drop=0.1, rates=[1,2,4,8])
    layer = BatchNormalization()(layer)
    for layer_num in range(depth):
        print("mainlayer", layer_num)
        layer = make_multirate_multihead(layer, heads=heads, atom_nf=atom_nf, ks=(3,3), d=2, drop=0.1, rates=rates, residual=True)
        if attention_size is not None:
            # TODO: Replace attention itself with multirate multihead conv!!
            attention_mask = Conv2D(atom_nf*heads*len(rates), attention_size, activation='relu', dilation_rate=(11,11), padding='same', use_bias=True, bias_initializer='ones')(layer)
            masked_layer = multiply([attention_mask, layer])
            layer = masked_layer
            layer = LeakyRELU(alpha=0.01)(layer)

        layer = BatchNormalization()(layer)
        layer = GaussianDropout(0.1)(layer)
    return layer

def dense_cnn_block(x, initial_feats=8, growth_rate=4, block_depth=3, ks=(3,3), use_bias=False, dropout_rate=0.5, conv=SeparableConv2D):
    y = x
    y = BatchNormalization()(y)
    y = LeakyRELU(alpha=0.01)(y)
    layers_of_inputs = [y,] #[[y,],]

    num_outputs = initial_feats
    for layer_num in range(block_depth):
        num_outputs += growth_rate
        # Fully connect to previous layer inputs. Sum Conv2Ds instead of concatenating tensors for memory efficiency.
        for prev_layer_num, prev_layer_input in enumerate(layers_of_inputs):
            # Add your features for this layer pair!
            conv_result = conv(num_outputs, ks, padding='same', use_bias=use_bias)(prev_layer_input)
            conv_result = Dropout(dropout_rate)(conv_result)
            if prev_layer_num == 0:
                layer_output = conv_result
            else:
                layer_output = add([layer_output, conv_result]) # In-place add
        # Now the matrix transformation for the layerwise convs has been all added up!
        layers_of_inputs.append(layer_output) # Now your output is input to the next layer!
        y = layer_output

    return layer_output

import densenet.densenet_fc as dc

def densenet_tiramisu(args):
    # TODO Configure this!
    model = dc.DenseNetFCN((None, None, args.input_channels), init_conv_features=args.initial_features_per_block, nb_dense_block=args.block_layers, growth_rate=args.feature_growth_rate, nb_layers_per_block=args.layers_per_block, upsampling_type='upsampling', classes=args.num_classes)
    #model = dc.DenseNetFCN((None, None, 3), nb_dense_block=3, growth_rate=16, nb_layers_per_block=3, upsampling_type='upsampling', classes=args.num_classes)
    #model = dc.DenseNetFCN((None, None, 3), nb_dense_block=4, growth_rate=16, nb_layers_per_block=3, upsampling_type='upsampling', classes=args.num_classes)
    #Following was the original:
    #model = dc.DenseNetFCN((None, None, 3), nb_dense_block=1, growth_rate=8, nb_layers_per_block=2, upsampling_type='upsampling', classes=args.num_classes)
    model.compile(loss=lookup_loss(args.loss, args), metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.001, clipvalue=0.5)) #optimizer='nadam') #'adadelta')
    return model

def densenet_for_semantic_segmentation(args):
    num_classes=args.num_classes
    dense_block_init_feats=8
    dense_block_growth_rate=4
    updense_init_feats=8
    updense_growth_rate=4
    ks=(3,3)
    ds=5
    pooling='average'
    block_layers=5
    layers_per_block=3
    bottleneck_feats=8
    bottleneck_growth_rate=2
    combine_modes='concat'
    output_strides=(1,1)
    input_channels=3
    model_save_path=args.model_save_path
    use_transpose_conv=False
    dropout_rate=0.15
    use_bias=False
    conv=SeparableConv2D

    model_inputs = Input(shape=(None, None, input_channels))
    y = x = model_inputs

    y = Conv2D(32, (9,9), padding='same', use_bias=use_bias)(y)
    #y = Dropout(dropout_rate)(y)
    y = BatchNormalization()(y)
    y = LeakyRELU(alpha=0.01)(y)
    y = conv(8, (1,1), padding='same', use_bias=use_bias)(y)
    y = Dropout(dropout_rate)(y)
    y = LeakyRELU(alpha=0.01)(y)
    fullscale_outputs = []
    scaled_outputs = []
    scale_down = 1
    y = MaxPooling2D(pool_size=(2,2), padding='same')(y)
    for dense_block_num in range(block_layers):
        y = dense_cnn_block(y, initial_feats=dense_block_init_feats, growth_rate=dense_block_growth_rate, block_depth=layers_per_block, ks=ks, use_bias=use_bias, dropout_rate=dropout_rate, conv=conv)
        scaled_outputs.append(y)
        # Downsample and bottleneck between blocks.
        strides = (1,1)
        ks_bottleneck = (1,1)
        if pooling == 'conv': # Conv downsampling is superior since it takes into account relative positions of features.
            strides = (2,2)
            ks_bottleneck = (3,3)
        elif pooling == "average":
            y = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')(y)
        elif pooling == "max":
            y = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(y)
        y = BatchNormalization()(y)
        y = LeakyRELU(alpha=0.01)(y)
        y_bottleneck = conv(bottleneck_feats, ks_bottleneck, padding='same', strides=strides, use_bias=use_bias)(y)
        bottleneck_feats += bottleneck_growth_rate
        y = y_bottleneck
        y = Dropout(dropout_rate)(y)
    scaled_outputs.append(y)

    # Cool. Now we've got our downsampled context stuff. Now what?
    # Let's upsample back through our paths, concatenating features as we go!
    for s, scaled_output in enumerate(reversed(scaled_outputs)):
        if s > 0:
            if use_transpose_conv:
                upsampled = Conv2DTranspose(nf, (5, 5), strides=(2,2), padding='same')(y)
            else:
                upsampled = UpSampling2D(size=(2,2), data_format=None)(y)
                upsampled = conv(bottleneck_feats, (3,3), padding='same', use_bias=True, dilation_rate=(2,2))(upsampled)
            upsampled = BatchNormalization()(upsampled)
            upsampled = LeakyRELU(alpha=0.01)(upsampled)
            upsampled = Dropout(0.05)(upsampled)
        else:
            upsampled = y
        if s > 0:
            # The dense block here should be able to sample from both feature maps.
            y = concatenate([scaled_output, upsampled])

        y = dense_cnn_block(y, initial_feats=updense_init_feats, growth_rate=updense_growth_rate, block_depth=layers_per_block, ks=ks, use_bias=use_bias)
        # Bottleneck
        y = BatchNormalization()(y)
        y = LeakyRELU(alpha=0.01)(y)
        y_bottleneck = conv(bottleneck_feats, ks_bottleneck, padding='same', strides=(1,1), use_bias=use_bias)(y)
        bottleneck_feats -= bottleneck_growth_rate
        y = y_bottleneck

    y = UpSampling2D(size=(2,2), data_format=None)(y)
    y = conv(bottleneck_feats, (3,3), padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyRELU(alpha=0.01)(y)

    predictions = conv(num_classes, (1,1), padding='same', activation='sigmoid', use_bias=False)(y)
    #predictions = keras.layers.Lambda(lambda x: K.cast(x, dtype='float32'))(y)
    model = Model(inputs=model_inputs, outputs=predictions)
    model.compile(loss=per_class_margin, metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.0005, clipvalue=0.5)) #optimizer='nadam') #'adadelta')

    if os.path.exists(model_save_path):
        print("Loading existing model weights...")
        model.load_weights(model_save_path, by_name=True)
    return model

from regularizers import *
def build_model_functional_old(args):
    num_classes=6
    num_feats=[[8, 16, 32, 32, 32, 32], [8,]]
    ks=[[(3,3),(3,3),(3,3),(5,5),(5,5),(5,5)],[(9,9)]]
    ds=[[2,2,2,-2,-2,-2],[(1,1)]]
    combine_modes='concat'
    output_strides=(1,1)
    input_channels=3
    model_save_path='model.h5'
    use_transpose_conv=False
    model_save_path = args.model_save_path
    num_classes = args.num_classes

    print("Building functional model", num_classes, model_save_path)

    model_inputs = Input(shape=(None, None, input_channels))
    current_layer = model_inputs
    tower1 = current_layer
    ud = 3
    previous_layer = []
    nf = 25
    # Branches = 5
    branches = 4
    branch_trunks = []
    branch_outputs = []
    initial_trunk = current_layer
    branch_convs = [(11,11),(5,5),(3,3),(3,3),(7,7)]
    #branch_feats = [10,20,15,15,15]
    # 2015-05-31 eval
    branch_feats = [10,15,20,20,15]
    branch_depths = [2, 1, 1, 1, 3]
    #branch_feats = [20,30,40,40,30]
    #branch_depths = [2, 2, 2, 2, 3]
    previous_trunk = initial_trunk
    #branch_trunks.append(branch_trunk)
    # Create the branch trunks.
    for branch_num in range(0,branches):
        nf = branch_feats[branch_num]
        ks = branch_convs[branch_num]
        branch_trunk = Conv2D(nf, ks, padding='same', use_bias=False)(previous_trunk)
        branch_trunk = BatchNormalization()(branch_trunk)
        branch_trunk = LeakyRELU(alpha=0.01)(branch_trunk)
        branch_trunk = Dropout(0.05)(branch_trunk)
        branch_trunks.append(branch_trunk)
        if branch_num != branches - 1:
            branch_trunk = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(branch_trunk)
        #branch_trunks.append(branch_trunk)
        previous_trunk = branch_trunk

    # Now grow each trunk into its own branches.
    for branch_num in range(branches):
        current_layer = branch_trunks[branch_num]
        ud = branch_depths[branch_num]
        nf = branch_feats[branch_num]

        for l in range(ud):
            ks = (3,3) #branch_convs[branch_num] if l == 0 else (3,3)
            current_layer = Conv2D(nf, ks, padding='same', use_bias=False)(current_layer)
            current_layer = BatchNormalization()(current_layer)
            #current_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(current_layer)
            current_layer = LeakyRELU(alpha=0.01)(current_layer)
            current_layer = Dropout(0.05)(current_layer)
        # Now create (learned?) upsampling layers to improve each branch!
        if branch_num > 0:
            for l in range(branch_num):
                if use_transpose_conv:
                    current_layer = Conv2DTranspose(nf, (5, 5), strides=(2,2), padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(current_layer)
                else:
                    current_layer = UpSampling2D(size=(2,2), data_format=None)(current_layer)
                    #current_layer = Conv2D(nf, (5,5), padding='same', use_bias=True, kernel_regularizer=keras.regularizers.l2(0.01))(current_layer)
                    current_layer = Conv2D(nf, (3,3), padding='same', use_bias=True, dilation_rate=(2,2), kernel_regularizer=keras.regularizers.l2(0.01))(current_layer)
                current_layer = BatchNormalization()(current_layer)
                current_layer = LeakyRELU(alpha=0.0001)(current_layer)
                current_layer = Dropout(0.15)(current_layer)
        # DO dimensionality reduction for each branch before passing on to concatenation.
        current_layer = Conv2D(num_classes, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(current_layer)
        current_layer = BatchNormalization()(current_layer)
        current_layer = LeakyRELU(alpha=0.0001)(current_layer)
        #current_layer = Dropout(0.25)(current_layer)
        branch_outputs.append(current_layer)

    # Now combine the branches in a snazzy way.
    crown_dims = 20
    # Crown 1: Concatenate, followed by 1x1 dimensionality reduction.
    concatted_branches = concatenate(branch_outputs) #merge(branch_outputs, mode='concatenate', concat_axis=1)
    # Crown 2: Multiply with local attention layer
    # multiplied_branches = merge(branch_outputs, mode='multiply', concat_axis=1)
    # Crown 3: Select maximum in each feature channel
    # max_selected_branches = merge(branch_outputs, mode='max', concat_axis=1)
    dimreduct_branches = Conv2D(crown_dims, (1,1), padding='same')(concatted_branches)
    dimreduct_branches = BatchNormalization()(dimreduct_branches)
    dimreduct_branches = LeakyRELU(alpha=0.0001)(dimreduct_branches)
    #dimreduct_branches = Dropout(0.5)(dimreduct_branches)

    #local_attention = Conv2D(crown_dims, (1,1), padding='same', kernel_regularizer=zeromean_regularizer)(dimreduct_branches)
    #multiplied_branches = multiply([local_attention, dimreduct_branches])#merge([local_attention, dimreduct_branches], mode='multiply', concat_axis=1)

    # TODO: Max each branch for the max class, then pick the max branch? Max is transitive.
    #featurewise_max = merge(branch_outputs, mode='max', concat_axis=1)

    # FINAL combination layer: linear
    #top_concatenation = concatenate([featurewise_max, multiplied_branches, dimreduct_branches]) #merge([featurewise_max, multiplied_branches, dimreduct_branches], mode='concatenate', concat_axis=1)
    top_concatenation = dimreduct_branches
    final_conv = Conv2D(crown_dims, (1,1), padding='same')(top_concatenation)
    final_conv = BatchNormalization()(final_conv)
    final_conv = LeakyRELU(alpha=0.1)(final_conv)
    current_layer = final_conv
    #current_layer = Dropout(0.15)(current_layer)

    '''
    tower1 = current_layer
    for l in range(ud):
        # This WORKS so long as the input dimensions are both a multiple of a sufficiently high power of two.
        # 2^l
        first_way = Conv2DTranspose(nf, (5, 5), strides=(2,2), padding='same')(current_layer)
        current_layer = BatchNormalization()(current_layer)
        #current_layer = tf_keras.resize_images(current_layer, 2, 2, "channels_last")
        # Hooray for variable resize!!!!
        #current_layer = Lambda(lambda image: ktf.image.resize_images(image, K.shape(previous_layer[ud-l-1])[1:3]))(current_layer)
        # THe above works in training and validation but throws this error on model save: https://github.com/fchollet/keras/issues/6442

        #second_way = Conv2D(20, (3,3), padding='same')(first_way)
        #multiplied = multiply([first_way, second_way])
        current_layer = first_way #multiplied
    tower1 = current_layer

    #Allow net to choose between multiply, max, and sum at the output branching layer!

    current_layer = model_inputs
    downuptower1 = current_layer
    for l in range(3):
        first_layer = Conv2D(nf, (3,3), padding='same', use_bias=False)(current_layer)
        if l > 2:
            second_layer = Conv2D(nf, (1,1), padding='same', use_bias=False, kernel_regularizer=zeromean_regularizer)(current_layer)
            maxpooled = merge([first_layer, second_layer], mode='max', concat_axis=1)
            current_layer = maxpooled
        elif l > 1000:# %2 == 0 == 0:
            second_layer = SeparableConv2D(nf, (1,1), padding='same', use_bias=False, kernel_regularizer=zeromean_regularizer)(current_layer)
            multiplied = multiply([first_layer, second_layer]) # merge([first_layer, second_layer], mode='sum') #first_layer * second_layer
            current_layer = multiplied
        else:
            current_layer = first_layer
        current_layer = BatchNormalization()(current_layer)
        #current_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(current_layer)
        current_layer = LeakyRELU(alpha=0.1)(current_layer)
        current_layer = Dropout(0.05)(current_layer)
    #a_layer = Conv2D(nf, (1,9), strides=(1,1), padding='same', use_bias=False)(current_layer)
    #a_layer = Conv2D(nf, (9,1), strides=(1,1), padding='same', use_bias=False)(current_layer)
    #a_layer = LeakyRELU(alpha=0.1)(a_layer)
    # TODO: Multiply WEIGHTS, not post-weight activations!!!
    '''

    # LEARN weights (activation arrays) to apply to KERNELS!!
    #b_layer = SeparableConv2D(nf, (1,1), strides=(1,1), padding='same', use_bias=False)(current_layer)
    #current_layer = merge([a_layer, b_layer], mode='max', concat_axis=1) #multiply([a_layer, b_layer])
    #current_layer = LeakyRELU(alpha=0.1)(current_layer)

    #current_layer = merge([current_layer, tower1], mode='max', concat_axis=1) #multiply([a_layer, b_layer])

    # Do upsampling layers now!


    '''
    current_layer = Conv2DTranspose(20, kernel_size=(5, 5),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')(current_layer)
    current_layer = Conv2DTranspose(20, kernel_size=(5, 5),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')(current_layer)
    '''
    #predictions = Conv2D(num_classes, (1,1), padding='same', activation='softmax')(current_layer)
    predictions = Conv2D(num_classes, (1,1), padding='same', activation='relu', use_bias=False)(current_layer)
    model = Model(inputs=model_inputs, outputs=predictions)

    #model.compile(loss=f_measure_loss, metrics=['accuracy'], optimizer='nadam') #'nadam' #'adadelta')
    #model.compile(loss='poisson', metrics=['accuracy'], optimizer='nadam') #'nadam' #'adadelta')
    #flatmodel = Model(inputs=model_inputs, outputs=flatpredictions)
    #model.compile(loss='mse', metrics=['accuracy'], optimizer='nadam') #'adadelta')
    #model.compile(loss=masked_mse, metrics=['accuracy'], optimizer='nadam') #'adadelta')
    #model.compile(loss=pseudo_f_measure_loss, metrics=['accuracy'], optimizer='nadam') #'adadelta')
    model.compile(loss=lookup_loss(args.loss, args), metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.0005, clipvalue=0.5)) #optimizer='nadam') #'adadelta')

    if os.path.exists(model_save_path):
        print("Loading existing model weights...")
        model.load_weights(model_save_path, by_name=True)
    return model #, flatmodel

# Full resolution image processing can accommodate at most 100 channels per original pixel. Stringent memory requirement, but it should be doable with a winning network!
def full_res_net(args):
    num_classes=args.num_classes
    input_channels=3
    model_save_path=args.model_save_path
    model_inputs = Input(shape=(None, None, input_channels))
    layer = model_inputs

    # 3,796,763,136 (4 GB) consumed with a single image in memory, at 100 channels per pixel.
    simple_edges = Conv2D(8, (3,3), padding='same', use_bias=False, kernel_initializer=keras.initializers.Orthogonal())(layer)
    simple_edges = Activation(LeakyRELU(alpha=0.1))(simple_edges)
    simple_colors = Conv2D(8, (1,1), padding='same', use_bias=True)(layer)
    simple_colors = Activation(LeakyRELU(alpha=0.1))(simple_colors)
    first_layer = concatenate([simple_edges, simple_colors])

    second_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(first_layer)
    second_edges = Conv2D(16, (3,3), padding='same', use_bias=False, kernel_initializer=keras.initializers.Orthogonal())(second_layer)
    second_edges = LeakyRELU(alpha=0.05)(second_edges)
    second_colors = Conv2D(8, (1,1), padding='same', use_bias=True)(second_layer)
    second_colors = LeakyRELU(alpha=0.05)(second_colors)
    second_layer = concatenate([second_edges, second_colors])
    second_layer = Dropout(0.05)(second_layer)

    third_layer = MaxPooling2D(pool_size=(2,2), strides = (2,2), padding='same')(second_layer)
    third_edges = Conv2D(32, (3,3), padding='same', use_bias=False)(third_layer)
    third_edges = BatchNormalization()(third_edges)
    third_edges = LeakyRELU(alpha=0.05)(third_edges)
    third_colors = Conv2D(16, (1,1), padding='same', use_bias=False)(third_layer)
    third_colors = BatchNormalization()(third_colors)
    third_colors = LeakyRELU(alpha=0.05)(third_colors)
    third_layer = concatenate([third_edges, third_colors])
    third_layer = Dropout(0.1)(third_layer)

    fourth_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(third_layer)
    fourth_structures = Conv2D(24, (5,5), padding='same', use_bias=False)(fourth_layer)
    fourth_structures = LeakyRELU(alpha=0.1)(fourth_structures)
    fourth_edges = Conv2D(16, (3,3), padding='same', use_bias=False)(fourth_layer)
    fourth_edges = LeakyRELU(alpha=0.05)(fourth_edges)
    fourth_colors = Conv2D(16, (1,1), padding='same', use_bias=False)(fourth_layer)
    fourth_colors = LeakyRELU(alpha=0.05)(fourth_colors)
    fourth_layer = concatenate([fourth_structures, fourth_edges, fourth_colors])
    fourth_layer = Dropout(0.12)(fourth_layer)

    fifth_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(fourth_layer)
    fifth_layer1 = Conv2D(24, (1,1), padding='same', use_bias=True)(fifth_layer)
    fifth_structures = Conv2D(24, (5,5), padding='same', use_bias=False)(fifth_layer1)
    fifth_structures = BatchNormalization()(fifth_structures)
    fifth_structures = LeakyRELU(alpha=0.1)(fifth_structures)
    fifth_layer2 = Conv2D(16, (1,1), padding='same', use_bias=True)(fifth_layer)
    fifth_edges = Conv2D(8, (3,3), padding='same', use_bias=False)(fifth_layer2)
    fifth_edges = BatchNormalization()(fifth_edges)
    fifth_edges = LeakyRELU(alpha=0.05)(fifth_edges)
    fifth_colors = Conv2D(16, (1,1), padding='same', use_bias=False)(fifth_layer2)
    fifth_colors = BatchNormalization()(fifth_colors)
    fifth_colors = LeakyRELU(alpha=0.05)(fifth_colors)
    fifth_layer = concatenate([fifth_structures, fifth_edges, fifth_colors])
    fifth_layer = Dropout(0.15)(fifth_layer)

    sixth_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(fifth_layer)
    sixth_layer1 = Conv2D(24, (1,1), padding='same', use_bias=True)(sixth_layer)
    sixth_structures = Conv2D(24, (5,5), padding='same', use_bias=False)(sixth_layer1)
    sixth_structures = LeakyRELU(alpha=0.1)(sixth_structures)
    sixth_layer2 = Conv2D(16, (1,1), padding='same', use_bias=True)(sixth_layer)
    sixth_edges = Conv2D(8, (3,3), padding='same', use_bias=False)(sixth_layer2)
    sixth_edges = LeakyRELU(alpha=0.05)(sixth_edges)
    sixth_colors = Conv2D(8, (1,1), padding='same', use_bias=False)(sixth_layer2)
    sixth_colors = LeakyRELU(alpha=0.05)(sixth_colors)
    sixth_layer = concatenate([sixth_structures, sixth_edges, sixth_colors])
    sixth_layer = Dropout(0.2)(sixth_layer)

    # It has already undergone 6 downsamplings (1/64th of original.)
    seventh_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(sixth_layer)
    seventh_layer1 = Conv2D(24, (1,1), padding='same', use_bias=True)(seventh_layer)
    seventh_structures = Conv2D(24, (5,5), padding='same', use_bias=False)(seventh_layer1)
    seventh_structures = BatchNormalization()(seventh_structures)
    seventh_structures = LeakyRELU(alpha=0.1)(seventh_structures)
    seventh_layer2 = Conv2D(16, (1,1), padding='same', use_bias=True)(seventh_layer)
    seventh_edges = Conv2D(8, (3,3), padding='same', use_bias=False)(seventh_layer2)
    seventh_edges = BatchNormalization()(seventh_edges)
    seventh_edges = LeakyRELU(alpha=0.05)(seventh_edges)
    seventh_colors = Conv2D(8, (1,1), padding='same', use_bias=False)(seventh_layer2)
    seventh_colors = BatchNormalization()(seventh_colors)
    seventh_colors = LeakyRELU(alpha=0.05)(seventh_colors)
    seventh_layer = concatenate([seventh_structures, seventh_edges, seventh_colors])
    seventh_layer = Dropout(0.25)(seventh_layer)

    hole_filling_layer = SeparableConv2D(32, (5,5), padding='same', use_bias=True)(fifth_layer)
    hole_filling_layer = LeakyRELU(alpha=0.05)(hole_filling_layer)
    hole_filling_attention = Conv2D(32, (1,1), padding='same', activation='sigmoid', use_bias=True, bias_initializer='ones')(fifth_layer)
    hole_filling_product = multiply([hole_filling_layer, hole_filling_attention])
    #seventh_layer = hole_filling_product

    maxout_input = Conv2D(32, (1,1), padding='same', use_bias=True)(fifth_layer)

    maxout_features1 = SeparableConv2D(32, (5,5), dilation_rate=(1,1), padding='same', use_bias=True)(maxout_input)
    maxout_features1 = LeakyRELU(alpha=0.5)(maxout_features1)
    maxout_features2 = SeparableConv2D(32, (5,5), dilation_rate=(1,1), padding='same', use_bias=True)(maxout_input)
    maxout_features2 = LeakyRELU(alpha=0.5)(maxout_features2)
    #maxout_features3 = SeparableConv2D(32, (5,5), dilation_rate=(1,1), padding='same', use_bias=True)(maxout_input)
    #maxout_features3 = LeakyRELU(alpha=0.5)(maxout_features3)
    #maxout_features4 = SeparableConv2D(32, (5,5), dilation_rate=(1,1), padding='same', use_bias=True)(maxout_input)
    #maxout_features4 = LeakyRELU(alpha=0.5)(maxout_features4)

    maxout_features = maximum([maxout_features1, maxout_features2]) #merge(, mode='max', concat_axis=1) #, maxout_features3, maxout_features4], mode='max', concat_axis = 1)

    fifth_layer = concatenate([hole_filling_product, maxout_features])
    fifth_layer = Conv2D(32, (1,1), padding='same', use_bias=True)(fifth_layer)

    # Now, should we perform any kind of multiplicative filtering, e.g. learned morphological operators
    # or Domain Transform?

    upsampled_sixth = UpSampling2D(size=(2,2), data_format=None)(seventh_layer)
    upsampled_sixth = concatenate([upsampled_sixth, sixth_layer])
    upsampled_sixth = Conv2D(256, (3,3), padding='same', use_bias=True)(upsampled_sixth)
    upsampled_sixth = BatchNormalization()(upsampled_sixth)
    upsampled_sixth = LeakyRELU(alpha=0.05)(upsampled_sixth)

    upsampled_fifth = UpSampling2D(size=(2,2), data_format=None)(upsampled_sixth)
    upsampled_fifth = concatenate([upsampled_fifth, fifth_layer])
    upsampled_fifth = Conv2D(128, (5,5), padding='same', use_bias=True)(upsampled_fifth)
    upsampled_fifth = BatchNormalization()(upsampled_fifth)
    upsampled_fifth = LeakyRELU(alpha=0.05)(upsampled_fifth)

    upsampled_fourth = UpSampling2D(size=(2,2), data_format=None)(upsampled_fifth)
    upsampled_fourth = concatenate([upsampled_fourth, fourth_layer])
    upsampled_fourth = Conv2D(64, (3,3), padding='same', use_bias=True)(upsampled_fourth)
    upsampled_fourth = LeakyRELU(alpha=0.05)(upsampled_fourth)

    upsampled_third = UpSampling2D(size=(2,2), data_format=None)(upsampled_fourth)
    upsampled_third = concatenate([upsampled_third, third_layer])
    upsampled_third = Conv2D(32, (3,3), padding='same', use_bias=True)(upsampled_third)
    upsampled_third = BatchNormalization()(upsampled_third)
    upsampled_third = LeakyRELU(alpha=0.05)(upsampled_third)

    upsampled_second = UpSampling2D(size=(2,2), data_format=None)(upsampled_third)
    upsampled_second = concatenate([upsampled_second, second_layer])
    upsampled_second = Conv2D(24, (5,5), padding='same', use_bias=True)(upsampled_second)
    upsampled_second = LeakyRELU(alpha=0.05)(upsampled_second)

    upsampled_first = UpSampling2D(size=(2,2), data_format=None)(upsampled_second)
    upsampled_first = concatenate([upsampled_first, first_layer])
    upsampled_first = Conv2D(16, (5,5), padding='same', use_bias=True)(upsampled_first)
    upsampled_first = BatchNormalization()(upsampled_first)
    upsampled_first = LeakyRELU(alpha=0.05)(upsampled_first)

    current_layer = upsampled_first

    predictions = Conv2D(num_classes, (1,1), padding='same', activation='linear', use_bias=False)(current_layer)
    predictions = Activation(K.softmax)(predictions)
    model = Model(inputs=model_inputs, outputs=predictions)

    print("Model size:", model_size(model))

    model.compile(loss=pseudo_f_measure_loss, metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.0002)) #'nadam' #'adadelta')
    #model.compile(loss='poisson', metrics=['accuracy'], optimizer='nadam') #'nadam' #'adadelta')
    #flatmodel = Model(inputs=model_inputs, outputs=flatpredictions)
    #model.compile(loss=pixelwise_crossentropy, metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.0002)) #'adadelta')

    if os.path.exists(model_save_path):
        print("Loading existing model weights...")
        model.load_weights(model_save_path, by_name=True)
    return model

'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:21:13
 * @modify date 2017-05-25 02:21:13
 * @desc [description]
'''
from keras.models import Model
from keras.layers import Input, merge, concatenate, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.layers import merge, Activation, Dropout
from keras.layers import BatchNormalization
import tensorflow as tf
import keras
import keras.backend as K

def make_multihead(layer, heads=16, nf=32, atom_nf=2, ks=(3,3), d=1, drop=0.10, rate=1, functions=None, residual=False):
    print("multihead", heads, atom_nf, ks)
    head_outputs = []
    first = layer
    make_functions = False
    if functions is None:
        make_functions = True
        functions = []
    for head_num in range(heads):
        #print("head_num", head_num)
        layer = first
        if make_functions:
            function = Conv2D(atom_nf, (1,1), use_bias=False, activation='linear',padding='same')
            functions.append([])
            functions[head_num].append(function)
        else:
            function = functions[head_num][0]
        depthwise_conv = function(layer)
        layer = depthwise_conv
        start_layer = layer
        for d_e in range(d):
            #print("depth", d_e)
            #layer = GaussianDropout(drop)(layer)
            if make_functions:
                function = Conv2D(atom_nf, ks, use_bias=False, dilation_rate=rate, activation='linear',padding='same')
                functions[head_num].append(function)
            else:
                function = functions[head_num][d_e+1]
            layer = function(layer)
            #layer = BatchNormalization()(layer)
            layer = LeakyRELU(alpha=0.01)(layer)

        if residual:
            layer = keras.layers.Add()([start_layer, layer])

        output = layer
        head_outputs.append(layer)
    multihead_output = concatenate(head_outputs)
    return multihead_output, functions

def create_unet_tower(layer, num_class, batchnorm_layers=[True, True, True, True, True, True, True, True, True, True], dropout_rates=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], feat=[8,16,24,32,48,64,96], init_feats=8, num_downsamplings=4, residual=True):
    # MAIN U-Net layers.
    trunks = []
    for depth in range(num_downsamplings):
        print("DS depth:", depth)
        feats = feat[depth]
        #layer = Conv2D(feats, (1, 1), padding='same', use_bias=True)(layer)
        #layer = LeakyRELU(alpha=0.1)(layer)
        layer = Conv2D(feats, (3, 3), padding='same', use_bias=False)(layer)
        layer = LeakyRELU(alpha=0.1)(layer)
        if residual: ### +
            layer_pre_add = layer
        layer = Conv2D(feats, (3, 3), padding='same', use_bias=False)(layer)
        if residual: ### +
            layer = keras.layers.Add()([layer, layer_pre_add])
        if dropout_rates[depth] > 0:
            layer = Dropout(dropout_rates[depth])(layer)
        if batchnorm_layers[depth]:
            layer = BatchNormalization()(layer)
        if num_class > 100:
            l1,_ = make_multihead(layer, heads=feats*2, atom_nf=4, ks=(3,3))
            l2,_ = make_multihead(layer, heads=feats, atom_nf=2, ks=(5,5))
            layer = concatenate([l1, l2])
        #layer = Conv2D(feats, (1,1), padding='same', use_bias=False)(layer)
        #layer = LeakyRELU(alpha=0.1)(layer)
        trunks.append(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)

    for depth in reversed(range(num_downsamplings)):
        print("US depth:", depth)
        base_layer = trunks[depth]
        feats = feat[depth]
        #layer = Conv2D(feats, (1, 1), padding='same', use_bias=True)(layer)
        #layer = LeakyRELU(alpha=0.1)(layer)
        layer = Conv2D(feats, (3, 3), padding='same', use_bias=False)(layer)
        layer = LeakyRELU(alpha=0.1)(layer)
        if residual: ### +
            layer_pre_add = layer
        layer = Conv2D(feats, (3, 3), padding='same', use_bias=False)(layer)
        if residual: ### +
            layer = keras.layers.Add()([layer, layer_pre_add])
        if dropout_rates[depth] > 0:
            layer = Dropout(dropout_rates[depth])(layer)
        if batchnorm_layers[depth]:
            layer = BatchNormalization()(layer)
        layer = UpSampling2D(size=(2, 2))(layer)
        # Layer is now upsampled and prepared to be appended to previous layer.
        layer = concatenate([base_layer, layer])
    if num_class > 100:
        l1,_ = make_multihead(layer, heads=feats*4, atom_nf=8, ks=(1,1))
        l2,_ = make_multihead(layer, heads=feats*2, atom_nf=4, ks=(3,3))
        l3,_ = make_multihead(layer, heads=feats, atom_nf=2, ks=(5,5))
        layer = concatenate([l1, l2, l3])
    layer = Conv2D(num_class, (3, 3), use_bias=False, padding='same')(layer)

    return layer

class AnnealedDropout(Dropout):
    def __init__(self, p, **kwargs):
        self.p = p
        self.t = 0.0
        self.N = 1000000.0
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(Dropout, self).__init__(**kwargs)

    def increment_t_by(self, dt):
        self.t += dt

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.in_train_phase(K.dropout(x, level=self.p * min(1.0, self.t / self.N)), x)
        return x

    def get_config(self):
        config = {'p': self.p, 't': self.t, 'N': self.N}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class UNet():
    def __init__(self):
        print(('build UNet ...'))

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_linear(self, img_shape, num_class, fc_size=32):
        inputs = Input(shape = img_shape)
        outputs = Conv2D(40, (5,5), padding='same')(inputs)
        outputs = Dropout(0.25)(outputs)
        outputs = LeakyRELU(0.01)(outputs)
        outputs = MaxPooling2D(pool_size=(8,8))(outputs)
        outputs = Conv2D(24, (1,1), padding='same')(outputs)
        outputs = Dropout(0.25)(outputs)
        outputs = LeakyRELU(0.01)(outputs)
        outputs = Conv2D(24, (3,3), padding='same')(outputs)
        outputs = Dropout(0.25)(outputs)
        outputs = LeakyRELU(0.01)(outputs)
        outputs = UpSampling2D(size=(8,8))(outputs)
        outputs = Conv2D(24, (3,3), dilation_rate=(8,8), padding='same')(outputs)
        outputs = LeakyRELU(0.01)(outputs)
        outputs = Conv2D(num_class, (3,3), padding='same')(outputs)
        outputs = Activation(K.softmax)(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_unet(self, args):
        num_channels = 3
        img_shape = (None, None, num_channels)
        num_class = args.num_classes
        batchnorm_layers=[True, True, True, True, True, True, True, True, True, True]
        dropout_rates=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        feat=[8,16,24,32,48,64,96]
        init_feats=8
        num_downsamplings=4
        residual=True
        input_noise_amount = 0.25
        print("CREATING U-Net Model!!!")
        concat_axis = 3
        inputs = Input(shape = img_shape)
        init_feats = 8

        layer = inputs
        feats = init_feats
        if input_noise_amount > 0:
            layer = keras.layers.GaussianNoise(input_noise_amount)(layer)
            # TODO: Add the fancy noise layer here. Efficient "Vinegar noise" should be generated in the frequency domain and rendered in the time domain.

        '''
        output1 = create_unet_tower(layer, 1, batchnorm_layers, [dropout_rates[f]/2 for f in range(len(dropout_rates))], [feat[f]/2 for f in range(len(feat))], init_feats, num_downsamplings, residual)
        output2 = create_unet_tower(layer, 1, batchnorm_layers, [dropout_rates[f]/2 for f in range(len(dropout_rates))], [feat[f]/2 for f in range(len(feat))], init_feats, num_downsamplings, residual)
        output3 = create_unet_tower(layer, 1, batchnorm_layers, [dropout_rates[f]/2 for f in range(len(dropout_rates))], [feat[f]/2 for f in range(len(feat))], init_feats, num_downsamplings, residual)
        output4 = create_unet_tower(layer, 1, batchnorm_layers, [dropout_rates[f]/2 for f in range(len(dropout_rates))], [feat[f]/2 for f in range(len(feat))], init_feats, num_downsamplings, residual)
        output5 = create_unet_tower(layer, 1, batchnorm_layers, [dropout_rates[f]/2 for f in range(len(dropout_rates))], [feat[f]/2 for f in range(len(feat))], init_feats, num_downsamplings, residual)
        output6 = create_unet_tower(layer, 1, batchnorm_layers, [dropout_rates[f]/2 for f in range(len(dropout_rates))], [feat[f]/2 for f in range(len(feat))], init_feats, num_downsamplings, residual)
        #
        layer = concatenate([output1, output2, output3, output4, output5, output6])
        '''
        layer = create_unet_tower(layer, num_class, batchnorm_layers, dropout_rates, feat, init_feats, num_downsamplings, residual)
        #layer = keras.layers.Add()([layer, concatenate([output1, output2, output3, output4, output5, output6])])
        def adv_relu(x):
            return K.relu(x, max_value=1.0)
        layer = Activation(adv_relu)(layer)
        #conv10 = K.minimum(conv10, 1.0)
        layer = Activation(K.softmax)(layer)

        model = Model(inputs=inputs, outputs=layer)

        return model

    def create_model(self, img_shape, num_class, batchnorm_layers=[True, True, True, True, True, True, True, True, True, True], init_feats=8, num_downsamplings=4):
        print("CREATING U-Net Model!!!")

        concat_axis = 3
        inputs = Input(shape = img_shape)
        conv1 = keras.layers.GaussianNoise(0.5)(inputs)
        init_feats = 8

        dropout_rate = 0.01250 #50#5 #1#50
        gauss_dropout = 0.010#0.01
        noise_level = 0.005#0.01

        conv1 = Conv2D(init_feats, (3, 3), activation='linear', padding='same', use_bias=False, name='conv1_1')(conv1)
        conv1 = LeakyRELU(alpha=0.1)(conv1)
        conv1_orig = conv1
        conv1 = Conv2D(init_feats, (3, 3), activation='linear', use_bias=False, padding='same')(conv1)
        conv1 = LeakyRELU(alpha=0.1)(conv1)
        conv1 = keras.layers.Add()([conv1, conv1_orig])
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        conv1 = keras.layers.GaussianNoise(noise_level)(conv1)
        conv1 = keras.layers.GaussianDropout(gauss_dropout)(conv1)
        conv2 = Conv2D(init_feats*2, (3, 3), activation='linear', use_bias=False, padding='same')(pool1)
        conv2_orig = conv2
        conv2 = LeakyRELU(alpha=0.1)(conv2)
        conv2 = Conv2D(init_feats*2, (3, 3), activation='linear', use_bias=False, padding='same')(conv2)
        conv2 = LeakyRELU(alpha=0.1)(conv2)
        conv2 = keras.layers.Add()([conv2, conv2_orig])
        conv2 = Dropout(dropout_rate)(conv2)
        if batchnorm_layers[2]:
            conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(init_feats*4, (3, 3), activation='linear', use_bias=False, padding='same')(pool2)
        conv3_orig = conv3
        conv3 = LeakyRELU(alpha=0.1)(conv3)
        conv3 = Conv2D(init_feats*4, (3, 3), activation='linear', use_bias=False, padding='same')(conv3)
        conv3 = LeakyRELU(alpha=0.1)(conv3)
        #conv3 = keras.layers.Add()([conv3, conv3_orig])
        conv3 = Dropout(dropout_rate)(conv3)
        if batchnorm_layers[3]:
            conv3 = BatchNormalization()(conv3)

        conv3 = keras.layers.GaussianDropout(gauss_dropout)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


        conv4 = Conv2D(init_feats*8, (3, 3), activation='linear', use_bias=False, padding='same')(pool3)
        conv4_orig = conv4
        conv4 = LeakyRELU(alpha=0.1)(conv4)
        conv4 = Conv2D(init_feats*8, (3, 3), activation='linear', use_bias=False, padding='same')(conv4)
        conv4 = LeakyRELU(alpha=0.1)(conv4)
        #conv4 = keras.layers.Add()([conv4, conv4_orig])
        #if batchnorm_layers[4]:
        #    conv4 = BatchNormalization()(conv4)
        conv4 = Dropout(dropout_rate)(conv4)
        conv4 = keras.layers.GaussianDropout(gauss_dropout)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(init_feats*16, (3, 3), activation='linear', use_bias=False, padding='same')(pool4)
        conv5_orig = conv5
        conv5 = LeakyRELU(alpha=0.1)(conv5)
        conv5 = Conv2D(init_feats*16, (3, 3), activation='linear', use_bias=False, padding='same')(conv5)
        conv5 = LeakyRELU(alpha=0.1)(conv5)
        conv5 = keras.layers.Add()([conv5, conv5_orig])
        #if batchnorm_layers[5]:
        #    conv5 = BatchNormalization()(conv5)
        #conv5 = Dropout(dropout_rate)(conv5)
        conv5 = keras.layers.GaussianDropout(gauss_dropout)(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        #ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = conv4 #Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(init_feats*8, (3, 3), activation='linear', use_bias=False, padding='same')(up6)
        conv6_orig = conv6
        conv6 = LeakyRELU(alpha=0.1)(conv6)
        conv6 = Conv2D(init_feats*8, (3, 3), activation='linear', use_bias=False, padding='same')(conv6)
        conv6 = LeakyRELU(alpha=0.1)(conv6)
        #conv6 = keras.layers.Add()([conv6, conv6_orig])
        #if batchnorm_layers[6]:
        #    conv6 = BatchNormalization()(conv6)
        #conv6 = Dropout(dropout_rate)(conv6)
        conv6 = keras.layers.GaussianNoise(noise_level)(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        #ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = conv3 #Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(init_feats*4, (3, 3), activation='linear', use_bias=False, padding='same')(up7)
        conv7_orig = conv7
        conv7 = LeakyRELU(alpha=0.1)(conv7)
        conv7 = Conv2D(init_feats*4, (3, 3), activation='linear', use_bias=False, padding='same')(conv7)
        conv7 = LeakyRELU(alpha=0.01)(conv7)
        #conv7 = keras.layers.Add()([conv7, conv7_orig])
        #if batchnorm_layers[7]:
        #    conv7 = BatchNormalization()(conv7)
        #conv7 = Dropout(dropout_rate)(conv7)
        conv7 = keras.layers.GaussianNoise(noise_level)(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv3)
        #ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = conv2 #Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(init_feats*2, (3, 3), activation='linear', use_bias=False, padding='same')(up8)
        conv8_orig = conv8
        conv8 = LeakyRELU(alpha=0.1)(conv8)
        conv8 = Conv2D(init_feats*2, (3, 3), activation='linear', use_bias=False, padding='same')(conv8)
        conv8 = LeakyRELU(alpha=0.1)(conv8)
        #conv8 = keras.layers.Add()([conv8, conv8_orig])
        if batchnorm_layers[8]:
            conv8 = BatchNormalization()(conv8)
        #conv8 = Dropout(dropout_rate)(conv8)
        conv8 = keras.layers.GaussianNoise(noise_level)(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        #ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = conv1 #Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(init_feats, (3, 3), activation='linear', use_bias=False, padding='same')(up9)
        conv9_orig = conv9
        conv9 = LeakyRELU(alpha=0.1)(conv9)
        conv9 = Conv2D(init_feats, (3, 3), activation='linear', use_bias=False, padding='same')(conv9)
        conv9 = LeakyRELU(alpha=0.1)(conv9)
        #conv9 = keras.layers.Add()([conv9, conv9_orig])
        #conv9 = Dropout(dropout_rate)(conv9)
        conv9 = keras.layers.GaussianNoise(noise_level)(conv9)

        #ch, cw = self.get_crop_shape(inputs, conv9)
        #conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(num_class, (1, 1), use_bias=False)(conv9)
        def adv_relu(x):
            return K.relu(x, max_value=1.0)
        conv10 = Activation(adv_relu)(conv10)
        #conv10 = K.minimum(conv10, 1.0)
        #conv10 = Activation(K.softmax)(conv10)

        model = Model(inputs=inputs, outputs=conv10)

        return model

def build_simple_cylinder(num_classes=6, ds=4, init_feats=32, feature_growth_rate=4, dilated_block_rates=[], ks=[(5,5),(3,3),(3,3),(3,3),(3,3)], use_transpose_conv=False, input_channels=3, model_save_path="model_checkpoint.h5"):
    x = inputs = Input(shape = (None, None, input_channels))
    x = Conv2D(init_feats, ks[0], activation='linear', padding='same', use_bias=False, name='conv1_1')(x)
    x = LeakyRELU(alpha=0.05)(x)
    nf = init_feats
    for layer in range(ds):
        x = Conv2D(nf, ks[1+layer], padding='same', use_bias=False)(x)
        x = LeakyRELU(alpha=0.05)(x)
        x = BatchNormalization()(x)
        nf += feature_growth_rate
    x = Conv2D(nf, ks[1+layer], padding='same', use_bias=False)(x)
    x = LeakyRELU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    for layer in range(ds):
        x = Conv2D(nf, (3,3), padding='same')(x)
        #x = LeakyRELU(alpha=0.05)(x)
        nf -= feature_growth_rate
    x = Conv2D(num_classes, (1,1), padding='same')(x)

    model = Model(inputs, x)
    model.compile(loss='mse', metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.002)) #'nadam' #'adadelta')

    try:
        if os.path.exists(model_save_path):
            print("Loading existing model weights...")
            model.load_weights(model_save_path, by_name=True)
    except:
        print("Could not load model weights. Initializing from scratch instead.")
    return model

# TODO: Abstract the learning rate, and optimizer, etc. away from the model architecture!!! This will be easy and fun!
def build_simple_hourglass(args):
    num_classes=args.num_classes
    ds=args.block_layers
    init_feats=args.initial_features_per_block #Original
    feature_growth_rate=args.feature_growth_rate #32#=8#2#4#Multiplicative now.
    loss=args.loss
    lr=args.lr
    dropout_rate=args.dropout_rate
    use_transpose_conv=False
    input_channels=args.input_channels
    model_save_path=args.model_save_path
    use_bias = args.use_bias
    conv=Conv2D #conv=SeparableConv2D
    
    # Input Layer
    x = inputs = Input(shape = (None, None, input_channels))
    x = Conv2D(init_feats, args.initial_kernel_size, activation='linear', padding='same', use_bias=False, name='conv1_1')(x)
    if args.batch_normalization:
        x = BatchNormalization()(x)
    x = LeakyRELU(alpha=args.lrelu_alpha)(x)
    
    nf = init_feats
    # Downsampling blocks.
    for layer in range(ds):
        for blocklayer in range(args.layers_per_block):
            x = conv(nf, args.kernel_size, padding='same', strides=(1,1), use_bias=use_bias)(x)
            x = LeakyRELU(alpha=args.lrelu_alpha)(x)
            x = Dropout(dropout_rate)(x)
            if args.batch_normalization:
                x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x) #AveragePooling2D
        if args.feature_growth_type == 'add':
            nf += feature_growth_rate#+= feature_growth_rate
        else:
            nf *= feature_growth_rate
    # Middle block.
    for blocklayer in range(args.layers_per_block):
        x = conv(nf, args.kernel_size, padding='same', use_bias=use_bias)(x)
        x = LeakyRELU(alpha=args.lrelu_alpha)(x)
        x = Dropout(dropout_rate)(x)
        if args.batch_normalization:
            x = BatchNormalization()(x)
            
    # Upsampling layers.
    for layer in range(ds):
        x = UpSampling2D(size=(2, 2))(x)
        x = conv(nf, (5,5), padding='same', use_bias=use_bias)(x)
        x = LeakyRELU(alpha=args.lrelu_alpha)(x)
        if args.batch_normalization:
            x = BatchNormalization()(x)
        #x = Conv2DTranspose(nf, (5,5), strides=(2,2), padding='same')(x)
        if args.feature_growth_type == 'add':
            nf -=args.upsampling_path_growth_rate
        else:
            nf /=args.upsampling_path_growth_rate
    
    # Final Classification Layer.
    x = Conv2D(num_classes, (1,1), padding='same', use_bias=False, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(loss=loss, metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=args.lr, clipvalue=0.5)) #'nadam' #'adadelta')

    try:
        if os.path.exists(model_save_path):
            print("Loading existing model weights...")
            model.load_weights(model_save_path, by_name=True)
    except:
        print("Could not load model weights. Initializing from scratch instead.")
    return model

def build_model_functional(args):
    num_classes=6
    num_feats=[[8, 16, 32, 32, 32, 32], [8,]]
    ks=[[(3,3),(3,3),(3,3),(5,5),(5,5),(5,5)],[(9,9)]]
    ds=[[2,2,2,-2,-2,-2],[(1,1)]]
    combine_modes='concat'
    output_strides=(1,1)
    input_channels=3
    model_save_path='model.h5'
    model_save_path = args.model_save_path
    num_classes = args.num_classes

    print("Building functional model", num_classes, model_save_path)

    model = UNet().create_model(img_shape=(None, None, input_channels), num_class=num_classes)
    #model = UNet().create_unet(img_shape=(None, None, input_channels), num_class=num_classes, init_feats=8, num_downsamplings=4)
    #model = UNet().create_linear(img_shape=(None, None, input_channels), num_class=num_classes)
    print("Model size:", model_size(model))

    #model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.00005)) #'nadam' #'adadelta')
    #model.compile(loss=pseudo_f_measure_loss, metrics=['accuracy',sensitivity,specificity,single_class_accuracy], optimizer=keras.optimizers.Nadam(lr=0.00002)) #'nadam' #'adadelta')
    model.compile(loss='mse', metrics=['accuracy',sensitivity,specificity,single_class_accuracy], optimizer=keras.optimizers.Nadam(lr=0.002)) #'nadam' #'adadelta')
    #model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.00005)) #'nadam' #'adadelta')
    #model.compile(loss=pseudo_f_measure_loss, metrics=['accuracy',sensitivity,specificity,single_class_accuracy], optimizer=keras.optimizers.Nadam(lr=0.00002)) #'nadam' #'adadelta')
    #model.compile(loss='poisson', metrics=['accuracy'], optimizer='nadam') #'nadam' #'adadelta')
    #flatmodel = Model(inputs=model_inputs, outputs=flatpredictions)
    #model.compile(loss="mse", metrics=['accuracy'], optimizer=keras.optimizers.Nadam(lr=0.00005))#optimizer='nadam') #'adadelta')

    if os.path.exists(model_save_path):
        print("Loading existing model weights...")
        model.load_weights(model_save_path, by_name=True)
    return model #, flatmodel

    #num_classes += 1 # TODO: Expand blank to its own class!

def build_model(args):
    print("")
    print("")
    print("")
    print("Building Keras model of type", args.model_type)
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    from regularizers import *
    from losses import *
    config = tf.ConfigProto()
    #try:
    #    print("Setting GPU memory usage to 90%")
    #    config.gpu_options.per_process_gpu_memory_fraction = 0.90
    #    set_session(tf.Session(config=config))
    #except:
    #    print("Setting GPU memory usage to 40%")
    #    config.gpu_options.per_process_gpu_memory_fraction = 0.40
    #    set_session(tf.Session(config=config))
    mem_fraction = 0.90
    while mem_fraction > 0.0:
        try:
            print("Setting GPU memory usage to %02f%", mem_fraction*100)
            config.gpu_options.per_process_gpu_memory_fraction = mem_fraction
            set_session(tf.Session(config=config))
            break
        except:
            mem_fraction *= 0.75
    
    model_type = args.model_type
    num_classes = args.num_classes

    import sys
    current_module = sys.modules[__name__]
    model = getattr(current_module, model_type)
    print("Importing model type named", model_type)
    
    #model = model(args)
    if os.path.exists(args.load_model_path):
        print("Loading existing model weights...", args.load_model_path, "With inferred number of classes", args.num_classes)
        # TODO: Get model loading without matching code working!
        try:
            model = keras.models.load_model(args.load_model_path)
        except Exception as ex:
            print(ex)
            print("BUILDING model from args...")
            model = model(args)
            model.load_weights(args.load_model_path, by_name=True)
    else:
        model = model(args)
        print("No model by this name exists; creating new model instead...", args.load_model_path)
            
    return model
