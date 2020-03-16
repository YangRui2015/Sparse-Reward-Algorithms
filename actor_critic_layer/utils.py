import tensorflow as tf

def create_nn(input, input_num, output_num, init_val=0.01, relu=True, trainable=True, name=''):
    shape = [input_num, output_num]
    w_init = tf.random_uniform_initializer(minval=-init_val, maxval=init_val)
    b_init = tf.random_uniform_initializer(minval=-init_val, maxval=init_val)

    weights = tf.get_variable(name + "weights", shape, initializer=w_init, trainable=trainable)
    biases = tf.get_variable(name + "biases", [output_num], initializer=b_init, trainable=trainable)
    dot = tf.matmul(input, weights) + biases

    if not relu:
        return dot

    dot = tf.nn.relu(dot)
    return dot


def layer(input_layer, num_next_neurons, is_output=False):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]
    
    if is_output:
        weight_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        bias_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    else:
        # 1/sqrt(f)
        fan_in_init = 1 / num_prev_neurons ** 0.5
        weight_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init)
        bias_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init) 

    weights = tf.get_variable("weights", shape, initializer=weight_init)
    biases = tf.get_variable("biases", [num_next_neurons], initializer=bias_init)

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot

    relu = tf.nn.relu(dot)
    return relu







