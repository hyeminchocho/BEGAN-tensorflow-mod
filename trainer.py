from __future__ import print_function

import datetime
import os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image, save_one_image
from math import sqrt

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
                "measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, measure, k_t))

            if step % (self.log_step * 10) == 0:
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                #cur_measure = np.mean(measure_history)
                #if cur_measure > prev_measure * 0.99:
                #prev_measure = cur_measure

    def build_model(self):
        self.x = self.data_loader
        x = norm_img(self.x)

        print("MEEE shapex0: " +  str(tf.shape(x)));
        print("MEEE znum: " + str(self.z_num))

        self.z = tf.random_uniform(
                (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')
        print("MEEE is square!: " + str(self.config.is_square))
        G, self.G_var = GeneratorCNN(
                self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=False, is_square=self.config.is_square)
        print("MEEE z is: " + str(self.z))
        print("MEEE G_var: " + str(self.G_var))

        print("build model MEEE G: " + str(G) + " x: " + str(x))

        d_out, self.D_z, self.D_var = DiscriminatorCNN(
                tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format, is_square=self.config.is_square)
        AE_G, AE_x = tf.split(d_out, 2)
        print("d_out: " + str(d_out))
        print("AE_G: " + str(AE_G) + " AE_x: " +  str(AE_x))

        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        # Kernels
        self.ConvWeights = None
        self.Conv1Weights = None
        for var in self.G_var:
            print("MEEE G_var name: " + var.name)
            if var.name == "G/Conv_10/weights:0":
                self.ConvWeights = var
            # if var.name == "G/Conv_1/weights:0":
            #     self.Conv1Weights = var
        print("MEEE conv weights: " + str(self.ConvWeights) + " conv1: " + str(self.Conv1Weights))

        self.DConv = None
        self.DConv_25 = None
        for var in self.D_var:
            print("MEEE D_var name: " + var.name + " shape: " + str(var.shape))
            if var.name == "D/Conv/weights:0":
                self.DConv = var
            if var.name == "D/Conv_25/weights:0":
                self.DConv_25 = var
            # if var.name == "G/Conv_1/weights:0":
            #     self.Conv1Weights = var


        print("MEE reduce passed")
        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_G - G))

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        summaries = [
            tf.summary.image("G", self.G),
            tf.summary.image("AE_G", self.AE_G),
            tf.summary.image("AE_x", self.AE_x),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ]
        #
        for var in self.G_var:
            print("MEEE G_var name: " + var.name)
            if "weights" in var.name and "Conv" in var.name:
                print("MEEE summary: " + str(var.shape))
                if var.shape == (3, 3, 3, 128):
                    summaries.append(tf.summary.image(var.name, put_kernels_on_grid(var), max_outputs=1))
                elif var.shape == (3, 3, 128 ,3):
                    summaries.append(tf.summary.image(var.name, put_kernels_on_grid(tf.transpose(var, perm=[0, 1, 3, 2])), max_outputs=1))
                else:
                    print("MEEE print var kernel: " + str(var))
                    summaries.append(tf.summary.image(var.name, put_kernels_on_grid(var[:, :, 0, :]), max_outputs=1))

                # tf.summary.image(var.name, put_kernels_on_grid(var), max_outputs=1)
            # if var.name == "G/Conv_1/weights:0":
            #     self.Conv1Weights = var

        self.summary_op = tf.summary.merge(summaries)
        # self.summary_op = tf.summary.merge([
        #     tf.summary.image("G", self.G),
        #     tf.summary.image("AE_G", self.AE_G),
        #     tf.summary.image("AE_x", self.AE_x),
        #
        #     # MEE Visualize Kernels
        #     tf.summary.image(self.ConvWeights.name, put_kernels_on_grid(tf.transpose(self.ConvWeights, perm=[0, 1, 3, 2])), max_outputs=1),
        #     tf.summary.image(self.DConv.name, put_kernels_on_grid(self.DConv), max_outputs=1),
        #     tf.summary.image(self.DConv_25.name, put_kernels_on_grid(tf.transpose(self.DConv_25, perm=[0, 1, 3, 2])), max_outputs=1),
        #     # tf.summary.image(self.Conv1Weights.name, put_kernels_on_grid(self.Conv1Weights), max_outputs=1)
        #
        #     tf.summary.scalar("loss/d_loss", self.d_loss),
        #     tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
        #     tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
        #     tf.summary.scalar("loss/g_loss", self.g_loss),
        #     tf.summary.scalar("misc/measure", self.measure),
        #     tf.summary.scalar("misc/k_t", self.k_t),
        #     tf.summary.scalar("misc/d_lr", self.d_lr),
        #     tf.summary.scalar("misc/g_lr", self.g_lr),
        #     tf.summary.scalar("misc/balance", self.balance),
        # ])

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r, _ = GeneratorCNN(
                self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True, is_square=self.config.is_square)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, root_path=None, path=None, idx=None, save=True, save_by_one=False):
        x = self.sess.run(self.G, {self.z: inputs})
        print("MEEE in generate print self.z shape: " + str(self.z.shape))
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            if save_by_one:
                now = datetime.datetime.now()
                for i in range(x.shape[0]):
                    path = os.path.join(root_path, '{}_single_G.png'.format(i))
                    save_one_image(x[i, :, :, :], path)
                print("MEEE x in generate: " + str(x.shape))
            else:
                save_image(x, path)
                print("[*] Samples saved: {}".format(path))

        return x


    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            print("MEEE image: " + str(img.shape) + " in " + key)
            x = self.sess.run(self.AE_x, {self.x: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size/2)
        print("MEEE in terp G real_batch shape: " + str(real_batch.shape))

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]
        print("MEEE in interp G z shape: " + str(z.shape))
        print("MEEE in interp G z1 shape: " + str(z1.shape) + " z2 shape: " + str(z2.shape))


        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            print("MEEE z after slerp shape: " + str(z.shape))
            z_decode = self.generate(z, save=False)
            bah = self.generate(z, root_path=self.model_dir, save=True, save_by_one=True) # MEEE
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_one_G(self):
        fps = 30
        z1 = np.random.uniform(-1, 1, size=(1, self.z_num))
        z2 = np.random.uniform(-1, 1, size=(1, self.z_num))
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.interpolate_one_G_helper(z1, z2, fps, date_str)

    def interpolate_one_G_helper(self, z1, z2, fps, timestamp, prev_counter=None):
        counter = 0
        for idx, ratio in enumerate(np.linspace(0, 1, fps)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            x = self.sess.run(self.G, {self.z: z})
            if prev_counter == None:
                counter = idx
            else:
                if idx == 0:
                    print("Continue in interp G helper")
                    continue
                counter = prev_counter + idx
            if counter == 0:
                # post_script = "_s"
                post_script = ""
                # post_script = "_s"
            else:
                post_script = ""
            filename = "./interps/interp_{}_{:03d}_G{}.jpg".format(timestamp, counter, post_script)
            print("In interp save filename: " + filename)
            save_one_image(x[0,:, :,:], filename)

        return counter
            # z_decode = self.generate(z, save=False)
            # generated.append(z_decode)

    def interpolate_many_G(self, num):
        fps = 30
        z1 = np.random.uniform(-1, 1, size=(1, self.z_num))
        z2 = np.random.uniform(-1, 1, size=(1, self.z_num))
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        counter = None
        for i in range(num):
            z1 = z2 # Update inter vars
            z2 = np.random.uniform(-1, 1, size=(1, self.z_num))
            counter = self.interpolate_one_G_helper(z1, z2, fps, date_str, prev_counter=counter)


    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test(self):
        root_path = "./" #self.model_dir # was "./"

        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(
                    real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
            self.autoencode(
                    real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch, step, root_path)
            #self.interpolate_D(real1_batch, real2_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))

            if all_G_z is None:
                all_G_z = G_z
            else:
                all_G_z = np.concatenate([all_G_z, G_z])
            save_image(all_G_z, '{}/G_z{}.png'.format(root_path, step))

        save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def generate_interpolation_G(self):
        root_path = "./"
        step = 333
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        print("MEE in gen interp G z_fixed shape: " + str(z_fixed.shape))
        G_z = self.generate(z_fixed, path=os.path.join(root_path, "interp_{}_G.png".format(step)))

        self.interpolate_G(G_z, step, root_path)

    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        print("MEEE Before transpose: " + str(x.shape))
        if self.data_format == 'NCHW':
            print("MEEE transposing")
            x = x.transpose([0, 2, 3, 1])
            print("MEEE after transpose: " + str(x.shape))
        return x

    from math import sqrt

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.

  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)

  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x
