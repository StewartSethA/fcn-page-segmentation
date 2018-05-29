from __future__ import print_function
import tensorflow as tf
import numpy as np
from nn_utils import weight_variable, bias_variable, variable_summaries, nn_layer, conv2d, maxpool, relu

def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0 ):
    shape = in_var.get_shape().as_list()

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b

def lrelu( x, leak=0.2, name="lrelu" ):
    return tf.maximum( x, leak*x )

def deconv2d( in_var, output_shape, name="deconv2d", stddev=0.02, bias_val=0.0 ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    # [ height, width, in_channels, number of filters ]
    var_shape = [ k_w, k_h, output_shape[-1], in_var.get_shape()[-1] ]

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", var_shape,
                             initializer=tf.truncated_normal_initializer( stddev=0.02 ) )
        b = tf.get_variable( "b", [output_shape[-1]],
                             initializer=tf.constant_initializer( bias_val ))

        dyn_input_shape = tf.shape(in_var)
        batch_size = dyn_input_shape[0]
        output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])

        deconv = tf.nn.conv2d_transpose( in_var, W, output_shape=output_shape, strides=[1, d_h, d_w, 1] )
        deconv = tf.reshape( tf.nn.bias_add( deconv, b), output_shape) #deconv.get_shape() )

        return deconv

def gen_model( z ):
    #print(z.get_shape())
    #z = tf.reshape()
    featmaps = 16
    H1 = linear(z, 7*7*featmaps , name="g_h1")
    #print(H1.get_shape())
    h1 = tf.nn.relu(H1)
    h1 = tf.reshape(h1, [batch_size, 7, 7, featmaps])

    D2 = deconv2d(h1, [batch_size, 14, 14, featmaps/4], name="g_d2")
    h2 = tf.nn.relu(D2)
    D3 = deconv2d(h2, [batch_size, 28, 28, 1], name="g_d3")
    D3 = tf.reshape(D3, [batch_size, 784]) # Map to 1D pixel array per image
    h3 = tf.sigmoid(D3)

    return h3,D3,h2,D2,h1,H1 # This is the generated image!

def cnn_layer(tensor, ks=3, infeats=16, outfeats=16, keep_prob=1.0, nonlin=relu, nameprefix="", layer_num=0, pool=False):
    W_conv = weight_variable([ks, ks, infeats, outfeats], name=nameprefix+"W"+str(layer_num)) # 28
    b_conv = bias_variable([outfeats], name=nameprefix+"b"+str(layer_num))
    h_conv = nonlin(conv2d(tensor, W_conv) + b_conv)
    if pool:
        h_pool = maxpool(h_conv)
    else:
        h_pool = h_conv
    h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
    return h_pool_drop

# Hourglass autoencoder
def cnn224x224_autoencoder_almostoptim(x_image, final_feats=10, keep_prob=0.5, batch_size=64, featmaps=32, ds=4, height=224, width=224, ks=3, dks=3, fc_layers=0, fc_feats=1024, class_splits=[], pred_splits=[], size=None):
    if size is not None:
        height = width = size
    if len(class_splits) > 0:
        final_feats = class_splits[0]
    mid_feats = x_image
    mid_featmaps = featmaps
    mf = mid_featmaps
    downsampling_layers = ds
    featscale_perlayer = 2
    w = tf.shape(x_image)[2] #width #int(width* 2**-downsampling_layers)
    h = tf.shape(x_image)[1]#height #int(height* 2**-downsampling_layers)
    cks = dks
    for i in range(0, downsampling_layers):
        #if i >= 2:
        #    cks = 5
        #if i >= 3:
        #    cks = 7
        mid_feats = cnn_layer(mid_feats, ks=cks, infeats=(1 if i == 0 else mf/featscale_perlayer), outfeats=mf, keep_prob=keep_prob, nameprefix="m", layer_num=i, pool=True)
        mf *= featscale_perlayer
        h /= 2
        w /= 2
    upsampling_layers = downsampling_layers
    mf /= featscale_perlayer
    h *= 2
    w *= 2

    # FC layers for global informativity
    #if fc_layers > 0:
    #    mid_feats = tf.reshape(mid_feats, [-1, h/2*w/2*mf])
    #    mid_feats = nn_layer(mid_feats, h/2*w/2*mf, fc_feats, "fc1", act=relu)
    #    mid_feats = nn_layer(mid_feats, fc_feats, h/2*w/2*mf, "fc2", act=relu)
    #    mid_feats = tf.reshape(mid_feats, [-1, h/2, w/2, mf])

    # UPSAMPLING layers.
    featscale_perlayer = 3
    for i in range(0, upsampling_layers):
        mid_feats = deconv2d(mid_feats, [batch_size, h, w, mf], name="d"+str(i)) # TODO: Interleave concatenations of upsampled features!!!
        mf /= featscale_perlayer # OR create multi-res convolutions!!! (Convs that work on some featmaps with half stride, etc.!)
        h *= 2
        w *= 2
    mf *= featscale_perlayer

    hi_feats = mid_feats
    capdepth = 3
    for i in range(0, capdepth):
        print(i, capdepth)
        #feats = cnn_layer(ks=3, infeats=(1 if i == 0 else featmaps)), outfeats=(featmaps if i != depth-1 else final_feats), keep_prob=keep_prob, layer_num=i)
        hi_feats = cnn_layer(hi_feats, ks=ks, infeats=mf, outfeats=(mf if i != capdepth-1 else final_feats), keep_prob=keep_prob, nameprefix="f", layer_num=i)
    hi_feats = tf.nn.relu(hi_feats)

    return hi_feats

def cnn224x224_autoencoder_regionpred(x_image, final_feats=10, keep_prob=0.5, batch_size=64, fc=False, width=224, height=224, featmaps=16, ds=4, ks=3, dks=5, fc_layers=0, fc_feats=1024):
    print("Using cnn224x224_autoencoder_regionpred model")
    depth = 1 #ds
    feats = x_image
    for i in range(0, depth):
        #feats = cnn_layer(ks=3, infeats=(1 if i == 0 else featmaps)), outfeats=(featmaps if i != depth-1 else final_feats), keep_prob=keep_prob, layer_num=i)
        feats = cnn_layer(feats, ks=ks, infeats=(1 if i == 0 else featmaps), outfeats=featmaps, keep_prob=keep_prob, nameprefix="", layer_num=i)

    mid_feats = x_image
    mid_featmaps = featmaps
    mf = mid_featmaps
    downsampling_layers = ds
    w = width #int(width* 2**-downsampling_layers)
    h = height #int(height* 2**-downsampling_layers)
    for i in range(0, downsampling_layers):
        mid_feats = cnn_layer(mid_feats, ks=dks, infeats=(1 if i == 0 else mf/2), outfeats=mf, keep_prob=keep_prob, nameprefix="m", layer_num=i, pool=True)
        mf *= 2
        h /= 2
        w /= 2
    upsampling_layers = downsampling_layers
    mf /= 2
    h *= 2
    w *= 2

    # FC layers for global informativity
    if fc_layers > 0:
        mid_feats = tf.reshape(mid_feats, [-1, h/2*w/2*mf])
        mid_feats = nn_layer(mid_feats, h/2*w/2*mf, fc_feats, "fc1", act=relu)
        mid_feats = nn_layer(mid_feats, fc_feats, h/2*w/2*mf, "fc2", act=relu)
        mid_feats = tf.reshape(mid_feats, [-1, h/2, w/2, mf])

    for i in range(0, upsampling_layers):
        mid_feats = deconv2d(mid_feats, [batch_size, h, w, mf], name="d"+str(i)) # TODO: Interleave concatenations of upsampled features!!!
        mf /= 2 # OR create multi-res convolutions!!! (Convs that work on some featmaps with half stride, etc.!)
        h *= 2
        w *= 2
    mf *= 2
    hi_feats = tf.concat([feats, mid_feats], axis=3)

    capdepth = 3
    for i in range(0, capdepth):
        print(i, capdepth)
        #feats = cnn_layer(ks=3, infeats=(1 if i == 0 else featmaps)), outfeats=(featmaps if i != depth-1 else final_feats), keep_prob=keep_prob, layer_num=i)
        hi_feats = cnn_layer(hi_feats, ks=ks, infeats=mf+featmaps, outfeats=(mf+featmaps if i != capdepth-1 else final_feats), keep_prob=keep_prob, nameprefix="f", layer_num=i)
    hi_feats = tf.nn.relu(hi_feats)

    return hi_feats

def cnn224x224_autoencoder_nopyramid(x_image, final_feats=10, keep_prob=0.5, batch_size=64, fc=False):
    featmaps = 16
    depth = 6
    feats = x_image
    for i in range(0, depth):
        W_conv = weight_variable([3, 3, (1 if i == 0 else featmaps), (featmaps if i != depth-1 else final_feats)], name="W"+str(i)) # 28
        b_conv = bias_variable([(featmaps if i != depth-1 else final_feats)], name="b"+str(i))
        h_conv = relu(conv2d(feats, W_conv) + b_conv)
        h_pool_drop = tf.nn.dropout(h_conv, keep_prob)
        feats = h_pool_drop
    return feats

def cnn224x224_autoencoder_nodownsampling(x_image, final_feats=10, keep_prob=0.5, batch_size=64, fc=False):
    featmaps = 32
    keep_prob = .9
    W_conv1 = weight_variable([3, 3, 1, featmaps/4], name="W1")  # 224
    b_conv1 = bias_variable([featmaps/4], name="b1")
    h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = maxpool(h_conv1)

    W_conv2 = weight_variable([3, 3, featmaps/4, featmaps/2], name="W2") # 112
    b_conv2 = bias_variable([featmaps/2], name="b2")
    h_conv2 = relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = maxpool(h_conv2)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

    W_conv3 = weight_variable([3, 3, featmaps/2, featmaps], name="W3") # 56
    b_conv3 = bias_variable([featmaps], name="b3")
    h_conv3 = relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
    h_pool3 = maxpool(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob)

    W_conv4 = weight_variable([3, 3, featmaps, featmaps*4], name="W4") # 28
    b_conv4 = bias_variable([featmaps*4], name="b4")
    h_conv4 = relu(conv2d(h_pool3_drop, W_conv4) + b_conv4)
    h_pool4 = maxpool(h_conv4)
    h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob)

    fc_size = 1024
    if fc:
        W_fc1 = weight_variable([7 * 7 * featmaps, fc_size], name="W3")
        b_fc1 = bias_variable([fc_size], name="b3")

        h_pool2_flat = tf.reshape(h_pool3_drop, [batch_size, 28*28*featmaps]) # Batch, EverythingElse
        h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        H1 = linear(h_fc1_drop, 28*28*featmaps , name="g_h1")
        h1 = tf.nn.relu(H1)
        h1 = tf.reshape(h1, [batch_size, 28, 28, featmaps])
        h1_drop = tf.nn.dropout(h1, keep_prob)
        inner = h1_drop
    else:
        inner = h_pool4_drop


    #h1 = inner
    D1 = deconv2d(inner, [batch_size, 28, 28, featmaps], name="g_d1")
    h1 = tf.nn.relu(D1)
    D2 = deconv2d(h1, [batch_size, 28*2, 28*2, featmaps/2], name="g_d2")
    h2 = tf.nn.relu(D2)
    D3 = deconv2d(h2, [batch_size, 28*4, 28*4, featmaps/4], name="g_d3")
    h3 = tf.nn.relu(D3)#)
    D4 = deconv2d(h3, [batch_size, 224, 224, final_feats], name="g_d4")
    #D3 = tf.reshape(D3, [-1, 784]) # Map to 1D pixel array per image
    #h3 = tf.sigmoid(D3)
    h4 = D4
    #h3 = tf.mul(h3, x_image)
    y_conv = h4

    return y_conv

def cnn224x224_autoencoder(x_image, final_feats=10, keep_prob=0.5, batch_size=64, featmaps = 32, fc=True, width=224, height=224):
    print("Creating standard autoencoder.")
    W_conv1 = weight_variable([3, 3, 1, featmaps/4], name="W1")  # 224
    b_conv1 = bias_variable([featmaps/4], name="b1")
    h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = maxpool(h_conv1)

    W_conv2 = weight_variable([3, 3, featmaps/4, featmaps/2], name="W2") # 112
    b_conv2 = bias_variable([featmaps/2], name="b2")
    h_conv2 = relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = maxpool(h_conv2)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

    W_conv3 = weight_variable([3, 3, featmaps/2, featmaps], name="W3") # 56
    b_conv3 = bias_variable([featmaps], name="b3")
    h_conv3 = relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
    h_pool3 = maxpool(h_conv3)
    h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob)

    W_conv4 = weight_variable([3, 3, featmaps, featmaps*4], name="W4") # 28
    b_conv4 = bias_variable([featmaps*4], name="b4")
    h_conv4 = relu(conv2d(h_pool3_drop, W_conv4) + b_conv4)
    h_pool4 = maxpool(h_conv4)
    h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob)

    fc_size = 1024
    if fc:
        W_fc1 = weight_variable([28 * 28 * featmaps, fc_size], name="W3")
        b_fc1 = bias_variable([fc_size], name="b3")

        pooled = tf.reshape(h_pool4_drop, [batch_size, 28*28*featmaps]) # Batch, EverythingElse
        h_fc1 = relu(tf.matmul(pooled, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        H1 = linear(h_fc1_drop, 28*28*featmaps , name="g_h1")
        h1 = tf.nn.relu(H1)
        h1 = tf.reshape(h1, [batch_size, 28, 28, featmaps])
        h1_drop = tf.nn.dropout(h1, keep_prob)
        inner = h1_drop
    else:
        inner = h_pool4_drop


    #h1 = inner
    D1 = deconv2d(inner, [batch_size, 28, 28, featmaps], name="g_d1")
    h1 = tf.nn.relu(D1)
    D2 = deconv2d(h1, [batch_size, 28*2, 28*2, featmaps/2], name="g_d2")
    h2 = tf.nn.relu(D2)
    D3 = deconv2d(h2, [batch_size, 28*4, 28*4, featmaps/4], name="g_d3")
    h3 = tf.nn.relu(D3)#)
    D4 = deconv2d(h3, [batch_size, 224, 224, final_feats], name="g_d4")
    #D3 = tf.reshape(D3, [-1, 784]) # Map to 1D pixel array per image
    #h3 = tf.sigmoid(D3)
    h4 = tf.relu(D4)
    #h3 = tf.mul(h3, x_image)
    y_conv = h4

    return y_conv

def cnn28x28_autoencoder_orig(x_image, final_feats=10, keep_prob=0.5, batch_size=64):
    W_conv1 = weight_variable([5, 5, 1, 32], name="W1")
    b_conv1 = bias_variable([32], name="b1")
    h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = maxpool(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], name="W2")
    b_conv2 = bias_variable([64], name="b2")
    h_conv2 = relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = maxpool(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W3")
    b_fc1 = bias_variable([1024], name="b3")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # Batch, EverythingElse
    h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #W_fc2 = weight_variable([1024, final_feats], name="W4")
    #b_fc2 = bias_variable([final_feats], name="b4")

    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    #wt_vars = [b_fc2, W_fc2, b_fc1, W_fc1, b_conv2, W_conv2, b_conv1, W_conv1]

    featmaps = 64
    H1 = linear(h_fc1_drop, 7*7*featmaps , name="g_h1")
    #print(H1.get_shape())
    h1 = tf.nn.relu(H1)
    h1 = tf.reshape(h1, [batch_size, 7, 7, featmaps])

    D2 = deconv2d(h1, [batch_size, 14, 14, featmaps/4], name="g_d2")
    h2 = tf.nn.relu(D2)
    D3 = deconv2d(h2, [batch_size, 28, 28, final_feats], name="g_d3")
    #D3 = tf.reshape(D3, [-1, 784]) # Map to 1D pixel array per image
    #h3 = tf.sigmoid(D3)
    h3 = D3

    y_conv = h3

    return y_conv

def cnn28x28(x_image, final_feats=10, keep_prob=0.5, batch_size=64, fc_feats=1024):
    feat_maps = 8
    W_conv1 = weight_variable([3, 3, 1, feat_maps], name="W1")
    b_conv1 = bias_variable([feat_maps], name="b1")
    h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = maxpool(h_conv1)
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    W_conv2 = weight_variable([3, 3, feat_maps, feat_maps*2], name="W2")
    b_conv2 = bias_variable([feat_maps*2], name="b2")
    h_conv2 = relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = maxpool(h_conv2)
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    W_fc1 = weight_variable([7 * 7 * feat_maps*2, fc_feats], name="W3")
    b_fc1 = bias_variable([fc_feats], name="b3")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*feat_maps*2]) # Batch, EverythingElse
    h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([fc_feats, final_feats], name="W4")
    b_fc2 = bias_variable([final_feats], name="b4")

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    wt_vars = [b_fc2, W_fc2, b_fc1, W_fc1, b_conv2, W_conv2, b_conv1, W_conv1]
    return y_conv

def cnn28x28_old(x_image, final_feats=10, keep_prob=0.5, batch_size=64):
    feat_maps = 8
    W_conv1 = weight_variable([5, 5, 1, 32], name="W1")
    b_conv1 = bias_variable([32], name="b1")
    h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = maxpool(h_conv1)
    h_pool1 = tf.nn.dropout(h_pool1)

    W_conv2 = weight_variable([5, 5, 32, 64], name="W2")
    b_conv2 = bias_variable([64], name="b2")
    h_conv2 = relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = maxpool(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W3")
    b_fc1 = bias_variable([1024], name="b3")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # Batch, EverythingElse
    h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, final_feats], name="W4")
    b_fc2 = bias_variable([final_feats], name="b4")

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    wt_vars = [b_fc2, W_fc2, b_fc1, W_fc1, b_conv2, W_conv2, b_conv1, W_conv1]
    return y_conv

def cnn224x224(x_image, final_feats=2, scalefeats=1.0, keep_prob=1.0):
    with tf.variable_scope("conv1"):
        W_conv1 = weight_variable([7, 7, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = maxpool(h_conv1) # 224/2 = 112

    with tf.variable_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = maxpool(h_conv2) # 112 / 2 = 56

    with tf.variable_scope("conv3"):
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = maxpool(h_conv3) # 56/2 = 28

    with tf.variable_scope("conv4"):
        W_conv4 = weight_variable([5, 5, 128, 256])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = maxpool(h_conv4) # 28/2 = 14

    with tf.variable_scope("conv5"):
        W_conv5 = weight_variable([5, 5, 256, 512])
        b_conv5 = bias_variable([512])
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = maxpool(h_conv5) # 14/2 = 7

    with tf.variable_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 512, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool5, [-1, 7*7*512]) # Batch, EverythingElse
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope("fc2"):
        W_fc2 = weight_variable([1024, final_feats])
        b_fc2 = bias_variable([final_feats])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.mul(scalefeats, y_conv)

    wt_vars = [W_fc2, W_fc1, W_conv5, W_conv4, W_conv3, W_conv2, W_conv1]
    return y_conv #, wt_vars

def cnn56x56(x_image, final_feats=2, scalefeats=0.0001, start_feats=64, keep_prob=1.0):
    with tf.variable_scope("conv3"):
        W_conv3 = weight_variable([3, 3, 1, start_feats*2])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(x_image, W_conv3) + b_conv3)
        h_pool3 = maxpool(h_conv3) # 56/2 = 28

    with tf.variable_scope("conv4"):
        W_conv4 = weight_variable([3, 3, start_feats*2, start_feats*4])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = maxpool(h_conv4) # 28/2 = 14

    with tf.variable_scope("conv5"):
        W_conv5 = weight_variable([3, 3, start_feats*4, start_feats*8])
        b_conv5 = bias_variable([512])
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = maxpool(h_conv5) # 14/2 = 7

    with tf.variable_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * start_feats*8, start_feats*16])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool5, [-1, 7*7*start_feats*8]) # Batch, EverythingElse
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope("fc2"):
        W_fc2 = weight_variable([start_feats*16, final_feats])
        b_fc2 = bias_variable([final_feats])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.mul(scalefeats, y_conv)

    wt_vars = [W_fc2, W_fc1, W_conv5, W_conv4, W_conv3]
    return y_conv, wt_vars
