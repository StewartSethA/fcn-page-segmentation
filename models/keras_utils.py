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


if __name__ == '__main__':
    unittest.main(verbosity=2)
