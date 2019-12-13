import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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

    X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name = 'X')
    Z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE], name = 'Z')

    generator = Generator()
    discriminator = Discriminator()

    out_path = "vanilla_gan"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    D_real, D_logit_real = discriminator.forward(X)
    D_fake, D_logit_fake = discriminator.forward(generator.forward(Z))

    D_loss = descriminator_loss(D_logit_real, D_logit_fake)
    G_loss = generator_loss(D_logit_fake)

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=discriminator.theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator.theta_G)

    samples = generator.forward(Z)

    init = tf.initialize_all_variables()

    i = 0
    with tf.Session() as sess:
        sess.run(init)
        for it in range(10000):
            if it % 1000 == 0:
                sample_noise = sample_Z(16, NOISE_SIZE)
                sp = sess.run(samples, feed_dict={Z: sample_noise})
                fig = plot(sp)
                file_path = os.path.join(out_path, "{}.png".format(str(i).zfill(3)))
                plt.savefig(file_path, bbox_inches="tight")
                i += 1
                plt.close(fig)

            images, _ = mnist.train.next_batch(BATCH_SIZE)
            images = images.astype(np.float32)

            sample_noise = sample_Z(BATCH_SIZE, NOISE_SIZE)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: images, Z: sample_noise})

            sample_noise = sample_Z(BATCH_SIZE, NOISE_SIZE)
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_noise})

            if it % 1000 == 0:
                print('iter: {}'.format(it))
                print('D loss {:4}'.format(D_loss_curr))
                print('G loss {:4}'.format(G_loss_curr))
                print()

if __name__ == '__main__':
    main()


