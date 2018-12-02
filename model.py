import tensorflow as tf
import numpy as np
from build_graph import build_graph
import utils
from cons import IMG_W, IMG_H

class Model(object):
    def __init__(self, batch_size=500, code_size=50):
        self.batch_size=batch_size
        self.code_size=code_size
        build_graph(self, batch_size, code_size)
    
    def graph_summary(self, logdir='logs/log1'):
        with self.graph.as_default():
            with tf.Session() as sess:
                with tf.summary.FileWriter(logdir, graph=sess.graph) as fw:
                    pass
    
    def activations_summary(self, feed_dict, model_path=None, logdir='logs/act1', sess=None):
        with self.graph.as_default():
            init = self._get_init(model_path)
            if sess is None:
                with tf.Session() as sess:
                    self.init_or_restore(sess, init, model_path)
                    self.activations_summary_helper(logdir, sess, feed_dict)
            else:
                self.activations_summary_helper(logdir, sess, feed_dict)
                
    def activations_summary_helper(self, logdir, sess, feed_dict):
        with tf.summary.FileWriter(logdir, graph=sess.graph) as fw:
            summ = sess.run(self.act_summary_op, feed_dict=feed_dict)
            fw.add_summary(summ, 1)
    
    def weights_summary(self, init=True, model_path=None, logdir='logs/wei1'):
        with self.graph.as_default():
            init = self._get_init(model_path)
            with tf.Session() as sess:
                self.init_or_restore(sess, init, model_path)
                with tf.summary.FileWriter(logdir, graph=sess.graph) as fw:
                    summ = sess.run(self.weights_summary_op)
                    fw.add_summary(summ, 1)
                
    def feedforward(self, input_batch, input_code=None, model_path='models/model_0'):
        with self.graph.as_default():
            if input_code is not None:
                feed_dict = self.get_feed_dict(code_ph = input_code, from_code_ph=True)
            else:
                feed_dict = self.get_feed_dict(input_batch=input_batch)
            with tf.Session() as sess:
                self.init_or_restore(sess, init=False, model_path=model_path)
                code, generated = sess.run([self.code, self.generated], feed_dict)
            return code, generated
    
    def train(self, learning_rate=0.001, niter=1000, load_model=None, 
              save_model='models/model_0', print_every=50, test_data=None):
        with self.graph.as_default():
            init = self._get_init(load_model)
            x_train, _ = utils.get_train_data()
            with tf.Session() as sess:
                self.init_or_restore(sess, init=init, model_path=load_model)
                for i in range(niter):
                    train_batch = utils.get_random_batch(x_train, self.batch_size)
                    feed_dict = self.get_feed_dict(input_batch=train_batch, 
                                                   is_train=True, learning_rate=learning_rate)
                    _, loss = sess.run([self.train_op, self.loss], feed_dict)
                    if i % print_every == 0:
                        msg = 'batch loss: {:.1f}'.format(loss)
                        if test_data is not None:
                            x_test_ = utils.get_random_batch(test_data, self.batch_size)
                            feed_dict = self.get_feed_dict(input_batch=x_test_, is_train=False)
                            loss = sess.run(self.loss, feed_dict)
                            msg += '\ttest loss: {:.1f}'.format(loss)
                        print(msg)
                path = self.saver.save(sess, save_model)
                print('model saved at {}'.format(path))

    def init_or_restore(self, sess, init=True, model_path='model_dir/model_0'):
        if init:
            sess.run(self.init_op)
            print('model initialized')
        else:
            self.saver.restore(sess, model_path)

    def get_feed_dict(self, input_batch=None, code_ph=None, 
                      from_code_ph=False, is_train=False, learning_rate=None):
        feed_dict = {}
        feed_dict[self.from_code_ph] = from_code_ph
        if from_code_ph:
            if code_ph is None:
                raise ValueError('if from_code_ph, you should specify code_ph')
            assert code_ph.shape == (self.batch_size, self.code_size)
            feed_dict[self.code_ph] = code_ph
            feed_dict[self.input] = np.zeros([self.batch_size, IMG_H, IMG_W, 1]).astype(np.float32)
        else:
            feed_dict[self.code_ph] = np.zeros([self.batch_size, self.code_size])
            assert input_batch is not None
            assert input_batch.shape == (self.batch_size, IMG_H, IMG_W, 1)
            feed_dict[self.input] = input_batch
        feed_dict[self.is_train] = is_train
        if is_train:
            assert learning_rate is not None, 'when training, you have to specify the learning rate'
            feed_dict[self.learning_rate] = learning_rate
        return feed_dict
    
    def transitions(self, nums, name='transitions.gif', model_path='models/model_0', delay=5, step_size=10):
        assert len(nums) > 0
        assert len(nums) <= self.batch_size
        x_train, y_train = utils.get_train_data()
        idxs = []
        for num in nums:
            idx = np.random.choice(np.where(y_train == num)[0])
            idxs.append(idx)
            
        batch = np.zeros([self.batch_size, IMG_H, IMG_W, 1])
        batch[:len(nums)] = x_train[idxs]
        feed_dict = self.get_feed_dict(input_batch=batch)
        code, _ = self.feedforward(batch, model_path=model_path)
        images = []
        for i in range(len(nums)-1):
            code_trans = np.zeros([self.batch_size, self.code_size])
            code_trans = (code[i:i+1]*np.linspace(1., 0., self.batch_size)[:,np.newaxis] +
                          code[i+1:i+2]*np.linspace(0., 1., self.batch_size)[:, np.newaxis])
            _, gen = self.feedforward(None, input_code=code_trans, model_path=model_path)
            images.extend(utils.get_images_from_generated(gen, step_size))

        utils.export_images(images, name=name, delay=delay)

    def _get_init(self, model_path):
        if model_path is None:
            return True
        return False