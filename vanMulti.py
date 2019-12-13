import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
 
num_gpus = len(get_available_gpus())
print("Available GPU Number :"+str(num_gpus))


INPUT_SIZE = 784
NOISE_SIZE = 100
BATCH_SIZE = 128

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_Z(m, n):
    z = np.random.uniform(-1.0, 1.0, size=[m,n])
    return z.astype(np.float32)

def plot(samples: tf.Tensor):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)
    i = 0
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")
    return fig

def generator_loss(logits: tf.Tensor):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))

def descriminator_loss(real_logits: tf.Tensor, fake_logits: tf.Tensor):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    return D_loss_fake + D_loss_real

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device
 
    return _assign

class Generator:
    def __init__(self, hidden_size: int = 128):
        self.G_W1 = tf.Variable(xavier_init([NOISE_SIZE, hidden_size]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[hidden_size]))
        self.G_W2 = tf.Variable(xavier_init([hidden_size, INPUT_SIZE]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))
        self.theta_G = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]
    
    def forward(self, input_tensor):
        G_h1 = tf.nn.relu(tf.matmul(input_tensor, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

class Discriminator:
    def __init__(self, hidden_size: int = 128):
        self.D_W1 = tf.Variable(xavier_init([INPUT_SIZE, hidden_size]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[hidden_size]))
        self.D_W2 = tf.Variable(xavier_init([hidden_size, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

    def forward(self, input_tensor):
        D_h1 = tf.nn.relu(tf.matmul(input_tensor, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

def main():
    mnist = input_data.read_data_sets("data/mnist", one_hot=True)
    with tf.device("/cpu:0"):
        global_step=tf.train.get_or_create_global_step()
        tower_grads_g = []
        tower_grads_d= []
        X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name = 'X')
        Z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE], name = 'Z')
        od = tf.train.AdamOptimizer()
        og = tf.train.AdamOptimizer()

        generator = Generator()
        discriminator = Discriminator()

        samples = generator.forward(Z)

        out_path = "vanilla_multi_gan"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                    _x = X[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    _Z= Z[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    tf.get_variable_scope().reuse_variables()
                    D_real, D_logit_real = discriminator.forward(X)
                    D_fake, D_logit_fake = discriminator.forward(generator.forward(Z))

                    D_loss = descriminator_loss(D_logit_real, D_logit_fake)
                    G_loss = generator_loss(D_logit_fake)
                    g_grads = og.compute_gradients(G_loss)
                    d_grads = od.compute_gradients(D_loss)
                    tower_grads_d.append(d_grads)
                    tower_grads_g.append(g_grads)
                   
        D_grads = average_gradients(tower_grads_d)
        G_grads = average_gradients(tower_grads_g)
        train_op_d = od.apply_gradients(D_grads)
        train_op_g = og.apply_gradients(G_grads)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            ts = time.time()
            sess.run(tf.global_variables_initializer())
            for step in range(10000):
                if step % 1000 == 0:
                    sample_noise = sample_Z(16, NOISE_SIZE)
                    sp = sess.run(samples, feed_dict={Z: sample_noise})
                    fig = plot(sp)
                    file_path = os.path.join(out_path, "{}.png".format(str(i).zfill(3)))
                    plt.savefig(file_path, bbox_inches="tight")
                    i += 1
                    plt.close(fig)

                batch_x, _ = mnist.train.next_batch(BATCH_SIZE * num_gpus)
                sample_noise = sample_Z(BATCH_SIZE * num_gpus, NOISE_SIZE)
                sess.run(train_op_d, feed_dict={X: batch_x, Z: sample_noise})
                if step % 1000 == 0:
                    D_loss_curr = sess.run(D_loss, feed_dict={X: batch_x, Z: sample_noise})
                sample_noise = sample_Z(BATCH_SIZE * num_gpus, NOISE_SIZE)
                sess.run(train_op_g, feed_dict={Z: sample_noise})
                if step % 1000 == 0:
                    print('iter: {}'.format(step))
                    G_loss_curr = sess.run( G_loss, feed_dict={Z: sample_noise})
                    print('D loss {:4}'.format(D_loss_curr))
                    print('G loss {:4}'.format(G_loss_curr))
                    print()
            te = time.time() - ts
                
            print("Done with time {}".format(str(te)))


if __name__ == '__main__':
    main()


