import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse, is_square=False):
    with tf.variable_scope("G", reuse=reuse) as vs:
        print("in GenCNN z: " + str(z))
        # num_output = int(np.prod([8, 8, hidden_num]))
        if is_square:
            num_output = int(np.prod([8, 8, hidden_num])) # MEEEE
        else:
            num_output = int(np.prod([8, 11, hidden_num])) # MEEEE

        x = slim.fully_connected(z, num_output, activation_fn=None)
        print("MEEE x: " + str(x))
        # x = reshape(x, 8, 8, hidden_num, data_format)
        if is_square:
            x = reshape(x, 8, 8, hidden_num, data_format)
        else:
            x = reshape(x, 8, 11, hidden_num, data_format)

        print("MEEE reshape x: " + str(x))
        print("MEEE repeat_num: " + str(repeat_num))


        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            print("first conv: " + str(x))
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            print("second conv: " + str(x))
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)
                print("upscale: " + str(x))

        out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables
    # with tf.variable_scope("G", reuse=reuse) as vs:
    #     print("in GenCNN z: " + str(z))
    #     # num_output = int(np.prod([8, 8, hidden_num]))
    #     num_output = int(np.prod([8, 11, hidden_num])) # MEEEE
    #     x = slim.fully_connected(z, num_output, activation_fn=None)
    #     print("MEEE x: " + str(x))
    #     # x = reshape(x, 8, 8, hidden_num, data_format)
    #     x = reshape(x, 8, 11, hidden_num, data_format)
    #     print("MEEE reshape x: " + str(x))
    #     print("MEEE repeat_num: " + str(repeat_num))
    #
    #
    #     for idx in range(repeat_num):
    #         # x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    #         print("first conv: " + str(x))
    #         stage_name = str(idx) + "_1"
    #         weights_name = "GenCNN/Weights" + stage_name
    #         kernel_1 = tf.Variable(tf.truncated_normal([3, 3, hidden_num, 128], dtype=tf.float32, stddev=1e-1), name=weights_name)
    #         conv = tf.nn.conv2d(x, kernel_1, [1, 1, 1, 1], padding='SAME', data_format=data_format)
    #         biases_name = "GenCNN/Biases" + stage_name
    #         biases = tf.Variable(tf.constant(0.0, shape=[hidden_num], dtype=tf.float32), trainable=True, name=biases_name)
    #         bias = tf.nn.bias_add(conv, biases, data_format=data_format)
    #         conv_name = "GenCNN/Conv" + stage_name
    #         x = tf.nn.elu(bias, name=conv_name)
    #
    #         # x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    #         stage_name = str(idx) + "_2"
    #         weights_name = "GenCNN/Weights" + stage_name
    #         kernel_2 = tf.Variable(tf.truncated_normal([3, 3, hidden_num, 128], dtype=tf.float32, stddev=1e-1), name=weights_name)
    #         conv = tf.nn.conv2d(x, kernel_2, [1, 1, 1, 1], padding='SAME', data_format=data_format)
    #         biases_name = "GenCNN/Biases" + stage_name
    #         biases = tf.Variable(tf.constant(0.0, shape=[hidden_num], dtype=tf.float32), trainable=True, name=biases_name)
    #         bias = tf.nn.bias_add(conv, biases, data_format=data_format)
    #         conv_name = "GenCNN/Conv" + stage_name
    #         x = tf.nn.elu(bias, name=conv_name)
    #         print("second conv: " + str(x))
    #         if idx < repeat_num - 1:
    #             x = upscale(x, 2, data_format)
    #             print("upscale: " + str(x))
    #
    #
    #     stage_name = "3"
    #     weights_name = "GenCNN/Weights" + stage_name
    #     kernel_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 3], dtype=tf.float32, stddev=1e-1), name=weights_name)
    #     x = tf.nn.conv2d(x, kernel_3, [1, 1, 1, 1], padding='SAME', data_format=data_format)
    #     # biases_name = "GenCNN/Biases" + stage_name
    #     # biases = tf.Variable(tf.constant(0.0, shape=[channel_num], dtype=tf.float32), trainable=True, name=biases_name)
    #     # bias = tf.nn.bias_add(conv, biases)
    #     # conv_name = "GenCNN/Conv" + stage_name
    #     # x = tf.nn.elu(bias, name=conv_name)
    #     out = x
        # out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format, is_square=False):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        if is_square:
            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        else:
            x = tf.reshape(x, [-1, np.prod([8, 11, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        if is_square:
            num_output = int(np.prod([8, 8, hidden_num]))
        else:
            num_output = int(np.prod([8, 11, hidden_num]))

        x = slim.fully_connected(x, num_output, activation_fn=None)
        if is_square:
            x = reshape(x, 8, 8, hidden_num, data_format)
        else:
            x = reshape(x, 8, 11, hidden_num, data_format)


        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)
