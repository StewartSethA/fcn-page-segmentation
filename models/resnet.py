from __future__ import print_function
import tensorflow as tf
import numpy as np
import math

from nn_utils import weight_variable, bias_variable, variable_summaries, conv_nonlin

###########################################################################################################
# A constructor for a single ResNet layer.
###########################################################################################################
def resnet_layer(input_tensor, num_inputs, num_outputs=64, kernel_size=(3,3), padding=(1,1), strides=(1,1), nonlinearity=tf.nn.relu, layer_name="resnet_layer"):
    print("Adding resnet layer: " + layer_name + " " + str(num_inputs) + " => " + str(num_outputs))
    with tf.variable_scope(layer_name):
        # ResNet1
        with tf.variable_scope("resnet1"):
            W_conv1 = weight_variable([kernel_size[0], kernel_size[1], num_inputs, num_outputs], 0.06)
            variable_summaries(W_conv1, layer_name + '/weights1')
            b_conv1 = bias_variable([num_outputs], 1)
            variable_summaries(b_conv1, layer_name + '/biases1')

            conv1 = tf.nn.conv2d(input_tensor, W_conv1, strides, padding='SAME', use_cudnn_on_gpu=True)
            variable_summaries(conv1, layer_name + '/conv1')
            h_conv1 = nonlinearity(conv1)
            variable_summaries(h_conv1, layer_name + '/h_conv1')

        # ResNet2
        with tf.variable_scope("resnet2"):
            W_conv2 = weight_variable([kernel_size[0], kernel_size[1], num_outputs, num_outputs], 0.06)
            variable_summaries(W_conv2, layer_name + '/weights2')
            b_conv2 = bias_variable([num_outputs], 1)
            variable_summaries(b_conv2, layer_name + '/biases2')

            conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides, padding='SAME', use_cudnn_on_gpu=True)
            variable_summaries(conv2, layer_name + '/conv2')
            h_conv2 = nonlinearity(conv2)
            variable_summaries(h_conv2, layer_name + '/h_conv2')

        # Add in an input downsampling if num_inputs != num_outputs
        if num_inputs != num_outputs:
            # TODO: For now, only allows downsampling by a factor of 2!
            print("Resampling input")
            input_sampled = tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            variable_summaries(input_sampled, layer_name + '/input_downsampled')
        else:
            input_sampled = input_tensor

        # RES PART: ADD identity to learned residual
        identity_plus_residual = input_sampled + h_conv2
        variable_summaries(identity_plus_residual, layer_name + '/identity_plus_residual')
        return identity_plus_residual, [W_conv1, W_conv2]

'''
# TODO Future rehash of flexible resnet definition.
###########################################################################################################
# Standard ResNet with the desired number of output features.
###########################################################################################################
def resnet(input_tensor, input_width, input_height, input_channels=1, start_feats=32, num_layers=27, target_width=5, target_height=5, nonlin=tf.nn.relu): # Alternative interface: Allow for input of set layer numbers and factors for downsampling.
    print("Building resnet with " + str(num_layers) + " (x2) layers.")
    init_featmaps=input_channels # 32
    current_featmaps = last_featmaps = init_featmaps
    ks = (3,3)
    ss = [1,1,1,1]
    ps = (1,1)
    current_output = input_tensor
    last_featmaps = init_featmaps
    current_featmaps = start_feats
    allweights = []
    # Compute the downsampling intervals based on the target feature dimensions.
    width_attenuation_ratio = float(input_width) / float(target_width)
    height_attenuation_ratio = float(input_height) / float(target_height)
    max_attenuation_ratio = max(width_attenuation_ratio, height_attenuation_ratio) # TODO: For now, assumes square images.
    num_downsamplings = math.log(max_attenuation_ratio) / math.log(2)
    layers_between_downsamplings = int(float(num_layers) / float(num_downsamplings))

    # Compute the intervals for downsampling to reach target width and height in time.
    for layer_num in range(0, num_layers+1):
        print("Adding layer " + str(layer_num))
        # RESNET layer
        if layer_num > 0:
            current_output, wts = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
            layer_num += 1
            last_featmaps = current_featmaps # Preserves dimensionality on ordinary layers
            allweights.extend(wts)
        # DOWNSAMPLE
        if layer_num == 0 or (layer_num > 1 and (layer_num - 1 + layers_between_downsamplings/2) % layers_between_downsamplings == 0):
            if layer_num > 1:
                # Add a pooling layer
                print("Downsampling with 2x2 max pooling")
                current_output = tf.nn.max_pool(current_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                current_featmaps *= 2 # Double feature maps on each downsample to preserve content
            print("Adding conv_nonlin layer" + "conv_nonlin" + str(layer_num))
            # UPSAMPLE feature maps only (not spatial dimensions) using a convolutional layer
            current_output, wts = conv_nonlin(current_output, last_featmaps, current_featmaps, ks, ss, ps, nonlin, "conv_nonlin" + str(layer_num), bias_init=1.0)
            allweights.extend(wts)
            layer_num += 1
    return current_output, allweights
'''

###########################################################################################################
# A flexible template for feed-forward residual networks with specifiable input/output dimensions and depth.
###########################################################################################################
# TODO: Currently downsampling is being done through max pooling layers, with a corresponding feature map doubling convolutional layer. An upgrade to allow strided resampling
# is desirable (all-convolutional, minimal pooling)
def resnet(input_tensor, input_width, input_height, input_channels=1, start_feats=32, num_layers=27, target_width=5, target_height=5, nonlin=tf.nn.relu): # Alternative interface: Allow for input of set layer numbers and factors for downsampling.
    print("Building resnet with " + str(num_layers) + " (x2) layers.")
    init_featmaps=input_channels # 32
    current_featmaps = last_featmaps = init_featmaps
    ks = (3,3)
    ss = [1,1,1,1]
    ps = (1,1)
    current_output = input_tensor
    last_featmaps = init_featmaps
    current_featmaps = start_feats
    allweights = []
    # Compute the downsampling intervals based on the target feature dimensions.
    width_attenuation_ratio = float(input_width) / float(target_width)
    height_attenuation_ratio = float(input_height) / float(target_height)
    max_attenuation_ratio = max(width_attenuation_ratio, height_attenuation_ratio) # TODO: For now, assumes square images.
    num_downsamplings = math.log(max_attenuation_ratio) / math.log(2)
    layers_between_downsamplings = int(float(num_layers) / float(num_downsamplings))

    # Compute the intervals for downsampling to reach target width and height in time.
    for layer_num in range(0, num_layers+1):
        print("Adding layer " + str(layer_num))
        # RESNET layer
        if layer_num > 0:
            current_output, wts = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
            layer_num += 1
            last_featmaps = current_featmaps # Preserves dimensionality on ordinary layers
            allweights.extend(wts)
        # DOWNSAMPLE
        if layer_num == 0 or (layer_num > 1 and (layer_num - 1 + layers_between_downsamplings/2) % layers_between_downsamplings == 0):
            if layer_num > 1:
                # Add a pooling layer
                print("Downsampling with 2x2 max pooling")
                current_output = tf.nn.max_pool(current_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                current_featmaps *= 2 # Double feature maps on each downsample to preserve content
            print("Adding conv_nonlin layer" + "conv_nonlin" + str(layer_num))
            # UPSAMPLE feature maps only (not spatial dimensions) using a convolutional layer
            current_output, wts = conv_nonlin(current_output, last_featmaps, current_featmaps, ks, ss, ps, nonlin, "conv_nonlin" + str(layer_num), bias_init=1.0)
            allweights.extend(wts)
            layer_num += 1
    return current_output, allweights

###########################################################################################################
# A ResNet for 28 x 28 x 1 input images (designed specifically for MNIST)
###########################################################################################################
def resnet28x28(input_tensor, init_channels):
    input_channels = 1
    nonlin = tf.nn.relu
    init_featmaps=init_channels # 32
    current_featmaps = last_featmaps = init_featmaps
    ks = (3,3)
    ss = [1,1,1,1]
    ps = (1,1)

    current_output = input_tensor
    layer_num = 1

    print("Group 1")
    current_output, _ = conv_nonlin(input_tensor, input_channels, init_featmaps, ks, ss, ps, nonlin, "conv1") # 256x256 x B x F
    layer_num += 1

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 28 x 28 x 32

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 28 x 28 x 32

    # Add a pooling layer
    current_output = tf.nn.max_pool(current_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    current_featmaps *= 2 # 14 x 14 x 64

    print("Group 2")
    current_output, _ = conv_nonlin(current_output, last_featmaps, current_featmaps, ks, ss, ps, nonlin, "conv2") # 256x256 x B x F
    layer_num += 1

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 14 x 14 x 64

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 14 x 14 x 64

    # Add a pooling layer
    current_output = tf.nn.max_pool(current_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    current_featmaps *= 2 # 7 x 7 x 128

    print("Group 3")
    current_output, _ = conv_nonlin(current_output, last_featmaps, current_featmaps, ks, ss, ps, nonlin, "conv3") # 256x256 x B x F
    layer_num += 1

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 7 x 7 x 128

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 7 x 7 x 128

    return current_output

###########################################################################################################
# A ResNet of arbitrary size and shape specified by iso-featuremap depth runs.
###########################################################################################################
def resnet_bydepths(input_tensor, depths=[3,5,3]):
    image_width = input_tensor.get_shape()[0]
    image_height = input_tensor.get_shape()[1]
    batch_size = input_tensor.get_shape()[2]

    input_channels = 1
    nonlin = tf.nn.relu
    init_featmaps=64
    ks1 = (7,7)
    ss1 = [1,2,2,1]
    ps1 = (3,3)
    conv_1, _ = conv_nonlin(input_tensor, input_channels, init_featmaps, ks1, ss1, ps1, nonlin, "conv1") # 256x256 x B x F

    current_output = conv_1

    layer_num = 1
    ks = (3,3)
    ss = [1,1,1,1]
    ps = (1,1)
    current_featmaps = init_featmaps
    last_featmaps = init_featmaps
    for depth_reducts in depths:
        print("Depth reduction:")
        for lnum in range(0, depth_reducts):
            print("Layer: " + str(layer_num))
            if last_featmaps != current_featmaps:
                sst = [1,2,2,1]
            else:
                sst = ss
            current_output = resnet_layer(current_output, last_featmaps, current_featmaps, ks, ps, sst, nonlin, "resnet_layer"+str(layer_num))
            layer_num += 1
            last_featmaps = current_featmaps

        # Add a pooling layer
        #current_output = tf.nn.max_pool(current_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #current_featmaps *= 2
    return current_output


###########################################################################################################
# A ResNet for 56 x 56 x 1 input images (designed specifically for MNIST)
###########################################################################################################
def resnet56(image_batch, init_channels=1, output_feats=10):
    # Random crop, translate, and rotate batch to get 224 x 224 images.

    h = 56
    w = 56
    start_feats = 32
    num_layers=3
    target_width=7
    target_height=7
    final_feats = (w / target_width) * start_feats
    fc_size = output_feats # 24 # 1024
    nonlin = tf.nn.relu
    ks = (3,3)
    ss = [1,1,1,1]
    ps = (1,1)
    current_output = image_batch
    current_featmaps = start_feats
    layer_num = 0

    current_output, _ = conv_nonlin(current_output, init_channels, current_featmaps, (7,7), [1,1,1,1], ps, nonlin, "conv" + str(layer_num)) # 224x224 x B x F => 56 x 56
    layer_num += 1

    current_output = tf.nn.max_pool(current_output, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 56 x 56 x 32
    # Add a pooling layer
    current_output = tf.nn.max_pool(current_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    current_featmaps *= 2 # 28 x 28 x 64
    current_output, _ = conv_nonlin(current_output, last_featmaps, current_featmaps, ks, ss, ps, nonlin, "conv" + str(layer_num)) # 256x256 x B x F
    layer_num += 1

    current_output = resnet_layer(current_output, current_featmaps, current_featmaps, ks, ps, ss, nonlin, "resnet_layer"+str(layer_num))
    layer_num += 1
    last_featmaps = current_featmaps # 14 x 14 x 64

    current_output = tf.nn.max_pool(current_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    current_featmaps *= 2

    current_output, _ = conv_nonlin(current_output, last_featmaps, current_featmaps, ks, ss, ps, nonlin, "conv" + str(layer_num)) # 256x256 x B x F
    layer_num += 1 # 7 x 7 x 128

    with tf.variable_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 128, fc_size])
        b_fc1 = bias_variable([fc_size])

        h_pool1_flat = tf.reshape(current_output, [-1, 7 * 7 * 128]) # Batch, EverythingElse
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    h_fc1_drop = h_fc1 #tf.nn.dropout(h_fc1, keep_prob)
    feature_embedding = h_fc1_drop

    print("Using resnet 56x56")
    return feature_embedding, [W_fc1,]
