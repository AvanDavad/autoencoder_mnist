import tensorflow as tf
from cons import IMG_H, IMG_W

def get_weight(shape, mean=0., stddev=0.5, name='weight'):
    w = tf.Variable(tf.truncated_normal(shape=shape, mean=0., stddev=0.1), name=name)
    tf.add_to_collection('weights', w)
    return w

def get_filter(shape, mean=0., stddev=0.5, name='filter'):
    fltr = tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev), name=name)
    tf.add_to_collection('weights', fltr)
    return fltr

def get_bias(shape, name='bias'):
    bs = tf.Variable(tf.zeros(shape=shape), name=name)
    tf.add_to_collection('biases', bs)
    return bs

def convolution(x, fltr, stride, padding='VALID', bs=None):
    x = tf.nn.conv2d(x, fltr, [1,stride,stride,1], padding=padding)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    if bs:
        x = tf.add(x,bs)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    return x

def upconvolution(x, fltr, output_shape, stride, padding='VALID', bs=None):
    x = tf.nn.conv2d_transpose(x, fltr, output_shape, [1,stride,stride,1], padding=padding)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    return x

def activation(x, act_func=tf.nn.relu):
    x = act_func(x)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    return x

def batchnorm(x, is_train, ema):
    gamma = tf.Variable(tf.ones(shape=[x.shape[-1]]), name='gamma')
    beta  = tf.Variable(tf.zeros(shape=[x.shape[-1]]), name='beta')
    
    batch_mean, batch_var = tf.nn.moments(x, axes=[0,1,2])
    update_op = ema.apply([batch_mean, batch_var])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
    
    moving_mean = ema.average(batch_mean)
    moving_var  = ema.average(batch_var)
    
    mean, var = tf.cond(is_train,
                        true_fn = lambda: [batch_mean, batch_var],
                        false_fn= lambda: [moving_mean, moving_var],
                        name = 'mean_var_condition')
    x = tf.divide((x-mean), tf.sqrt(var), name='normalized')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    
    x = x*gamma + beta
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    
    return x
    

def build_graph(model, batch_size, code_size):
    model.graph = tf.Graph()
    
    with model.graph.as_default():
        model.step = tf.Variable(0, dtype=tf.int32, trainable=False)

        model.code_ph = tf.placeholder(tf.float32, shape=[batch_size, code_size], name='code_ph')
        model.from_code_ph = tf.placeholder(tf.bool, shape=[], name='from_code_ph')

        model.input = tf.placeholder(tf.float32, shape=[batch_size,IMG_H,IMG_W,1], name='input')
        model.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        model.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        model.ema = tf.train.ExponentialMovingAverage(0.99)

        with tf.name_scope('Encoder'):

            with tf.name_scope('layer_1'):

                with tf.name_scope('convolution'):
                    model.w_e1 = get_filter([4,4,1,5], name='w_e1')
                    x = convolution(model.input, model.w_e1, 2)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)
                with tf.name_scope('activation'):
                    x = activation(x)

            with tf.name_scope('layer_2'):

                with tf.name_scope('convolution'):
                    model.w_e2 = get_filter([3,3,5,10], name='w_e2')
                    x = convolution(x, model.w_e2, 2)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)
                with tf.name_scope('activation'):
                    x = activation(x)

            with tf.name_scope('layer_3'):

                with tf.name_scope('convolution'):
                    model.w_e3 = get_filter([2,2,10,20], name='w_e3')
                    x = convolution(x, model.w_e3, 2)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)
                with tf.name_scope('activation'):
                    x = activation(x)

            with tf.name_scope('layer_4'):

                with tf.name_scope('convolution'):
                    model.w_e4 = get_filter([3,3,20,code_size], name='w_e4')
                    x = convolution(x, model.w_e4, 1)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)

            model.code_calc = tf.reshape(x, shape=[batch_size, code_size], name='code_calc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)


            model.code = tf.cond(model.from_code_ph, 
                                 true_fn=lambda: model.code_ph, 
                                 false_fn=lambda: model.code_calc, 
                                 name='code')

        with tf.name_scope('Decoder'):
            x = tf.reshape(model.code, shape=[batch_size,1,1,code_size])

            with tf.name_scope('layer_1'):

                with tf.name_scope('upconvolution'):
                    model.w_d1 = get_filter([3,3,20,code_size], name='w_d1')
                    x = upconvolution(x, model.w_d1, [batch_size,3,3,20], 1)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)
                with tf.name_scope('activation'):
                    x = activation(x)

            with tf.name_scope('layer_2'):

                with tf.name_scope('upconvolution'):
                    model.w_d2 = get_filter([2,2,10,20], name='w_d2')
                    x = upconvolution(x, model.w_d2, [batch_size,6,6,10], 2)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)
                with tf.name_scope('activation'):
                    x = activation(x)

            with tf.name_scope('layer_3'):

                with tf.name_scope('upconvolution'):
                    model.w_d3 = get_filter([3,3,5,10], name='w_d3')
                    x = upconvolution(x, model.w_d3, [batch_size,13,13,5], 2)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)
                with tf.name_scope('activation'):
                    x = activation(x)

            with tf.name_scope('layer_4'):

                with tf.name_scope('upconvolution'):
                    model.w_d4 = get_filter([4,4,1,5], name='w_d4')
                    x = upconvolution(x, model.w_d4, [batch_size,IMG_H,IMG_W,1], 2)
                with tf.name_scope('batchnorm'):
                    x = batchnorm(x, model.is_train, model.ema)

            model.generated = tf.nn.sigmoid(x, name='generated')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, model.generated)

        model.loss = tf.nn.l2_loss(model.generated - model.input, name='loss')

        model.opt = tf.train.AdamOptimizer(model.learning_rate)
        model.grads_and_vars = model.opt.compute_gradients(model.loss)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            model.train_op = model.opt.apply_gradients(model.grads_and_vars, model.step)

        # activation summaries
        act_summaries = []
        for x in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
            act_summaries.append(tf.summary.histogram(x.name.split(':')[0], x))
        model.act_summary_op = tf.summary.merge(act_summaries)

        # weight summaries
        w_summaries = []
        for w in tf.get_collection('weights'):
            w_summaries.append(tf.summary.histogram(w.name.split(':')[0], w))
        model.weights_summary_op = tf.summary.merge(w_summaries)

        model.saver = tf.train.Saver()

        model.init_op = tf.global_variables_initializer()

        model.graph.finalize()
