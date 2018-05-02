import tensorflow as tf
import numpy as np

def model_size(model): # Compute number of params in a model (the actual number of floats)
    from keras import backend as K
    sizes = [np.prod(K.get_value(w).shape) for w in model.trainable_weights]
    return sum(sizes), sizes

#https://github.com/fchollet/keras/issues/2226
def get_gradients(model):
    import keras.backend as K
    # https://github.com/fchollet/keras/issues/5455
    '''
    from keras.objectives import mse
    m = get_your_model()
    y_true = K.placeholder(*your_models_output_shape)
    loss = K.mean(mse(y_true, m.output))
    get_grads = K.function([m.input, y_true], K.gradients(loss, m.input))

    grads = get_grads([np.random.rand(*your_models_input_shape), np.random.rand(*your_models_output_shape)])
    '''

    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    weights = model.trainable_weights #[tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    global last_input, last_gt
    gradient_tensors = optimizer.get_gradients(model.total_loss, weights)
    input_tensors = [model.inputs[0], model.sample_weights[0], model.targets[0], K.learning_phase()]
    gradients_function_keras = K.function(inputs=input_tensors, outputs=gradient_tensors)
    inputs = [last_input, np.ones((last_input.shape[0]), dtype=np.float32), last_gt, 0]
    np_gradients = gradients_function_keras(inputs)
    print("GRADIENTS:")
    print(np_gradients)
    return np_gradients # Instantiate and call the gradients function
    
        

#TODO ResNet redo!
#def identity_plus_delta_conv_variable(shape):
#    """Create a weight variable with appropriate initialization."""
#    initial = tf.truncated_normal(shape, stddev=0.1)
#
#    return tf.Variable(initial)


# This is a handy little function that takes as input a 2D batch tensor of 1D features,
# And partitions them for multiple distinct prediction tasks within the same network.
# For example, features 0-9 may be used for n-way class prediction, with feaures 10-13 used
# for another 4-way class prediction, plus two free predictors regressing real numbers at the end.
def multitask_head(tensor_in, class_splits=[], pred_splits=[]):
    current_ind = 0
    out = None
    if len(class_splits) > 0:
        pass
    elif len(pred_splits) > 0:
        pass
    else:
        raise Exception("Net predicts nothing! Add some number of output features to either class splits or predictor splits variables.")

    for i in range(len(class_splits)):
        extra_classes = tf.nn.softmax(tf.slice(tensor_in, [0,current_ind], [-1, class_splits[i]]))
        out = tf.concat(1, [out, extra_classes]) if out is not None else extra_classes
        current_ind += class_splits[i]

    for i in range(len(pred_splits)):
        extra_preds = tf.slice(tensor_in, [0,current_ind], [-1, pred_splits[i]])
        out = tf.concat(1, [out, extra_preds]) if out is not None else extra_preds
        current_ind += pred_splits[i]
    # Out is a concatenated tensor with softmaxes, etc. applied appropriately.
    return out


def save_model(sess, filepath):
    saver = tf.train.Saver()
    save_path = saver.save(sess, filepath)

def load_model(sess, filepath):
    saver = tf.train.Saver()
    saver.restore(sess, filepath)

# Leaky clipped relu
def relu(x, alpha=0.1, maxVal=5.0, name="relu"):
    return tf.maximum(alpha*x, tf.minimum(maxVal, x))

# Weight variable
def weight_variable(shape, stddev=0.1, allow_reuse=False, name="W"):
    """Create a weight variable with appropriate initialization."""
    if allow_reuse:
        W = tf.get_variable( name, shape, initializer=tf.truncated_normal_initializer( stddev=stddev ) )
    else:
        initial = tf.truncated_normal(shape, stddev=stddev)
        W = tf.Variable(initial, name=name)
    tf.add_to_collection('vars', W)
    return W

# Bias variable
def bias_variable(shape, bias_val=0.0, allow_reuse=False, name="b"):
    """Create a bias variable"""
    if allow_reuse:
        b = tf.get_variable( name, [shape[-1]], initializer=tf.constant_initializer( bias_val ))
    else:
        initial = tf.constant(bias_val, shape=shape)
        b = tf.Variable(initial, name=name)
    tf.add_to_collection('vars', b)
    return b

def variable_summaries(var, name):
    return
    """Attach a lot of summaries to a Tensor."""

    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
    '''

# Defines a 2D convolution layer
def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')

# Defines a max pooling layer
def maxpool(x,stride=2):
    return tf.nn.max_pool(x, ksize=[1,stride,stride,1], strides=[1,stride,stride,1], padding='SAME')

# Defines a Clipped ReLU function
def clipped_relu(clip_amount=1.0):
    return lambda x: tf.minimum(tf.nn.relu(x), clip_amount)

###########################################################################################################
# Standard convolutional (conv) layer with nonlinearity.
###########################################################################################################
def conv_nonlin(input_tensor, num_inputs, num_outputs, kernel_size=(3,3), strides=[1,1,1,1], padding=(1,1), nonlinearity=tf.nn.relu, name="conv_nonlin", bias_init=0.1):
    with tf.variable_scope(name):
        W_conv = weight_variable([kernel_size[0], kernel_size[1], num_inputs, num_outputs], 0.1)
        b_conv = bias_variable([num_outputs], bias_init)
        conv = tf.nn.conv2d(input_tensor, W_conv, strides, padding='SAME', use_cudnn_on_gpu=True, data_format=None, name="conv")
        h_conv = nonlinearity(conv)
    return h_conv, [W_conv,]

###########################################################################################################
# Standard fully connected (FC) layer with nonlinearity.
###########################################################################################################
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            #tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        #tf.histogram_summary(layer_name + '/activations', activations)
        return activations
