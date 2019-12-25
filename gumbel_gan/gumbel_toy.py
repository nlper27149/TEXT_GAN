import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.set_random_seed(42)

BATCHS_IN_EPOCH = 100
BATCH_SIZE = 10
EPOCHS = 200
GENERATOR_TRAINING_FACTOR = 10
LEARNING_RATE = 0.0007
TEMPERATURE = 0.001

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# True_vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
#                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
True_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
def generate_text():
    while True:
        tmp = np.random.choice(True_vocab, 1, p=[float(1)/len(True_vocab)]*len(True_vocab))
        index = vocab.index(tmp)
        yield [index]


dataset = tf.data.Dataset.from_generator(generate_text,
                                         output_types=tf.int32,
                                         output_shapes=1).batch(BATCH_SIZE)
value = dataset.make_one_shot_iterator().get_next()
value = tf.one_hot(value, len(vocab))
value = tf.squeeze(value, axis=1)


def generator():
    with tf.variable_scope('generator'):
        z = tf.random_normal(shape=[1, 20], mean=0, stddev=1)
        logits = tf.layers.dense(z, len(vocab))
        gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(TEMPERATURE, logits=logits)
        probs = tf.nn.softmax(logits)
        generated = gumbel_dist.sample(BATCH_SIZE)
        return generated, probs

def discriminator(x):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        return tf.contrib.layers.fully_connected(x,
                                                 num_outputs=1,
                                                 activation_fn=None)

generated_outputs, generated_probs = generator()
discriminated_real = discriminator(value)
discriminated_generated = discriminator(generated_outputs)

d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_real,
                                            labels=tf.ones_like(discriminated_real)))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_generated,
                                            labels=tf.zeros_like(discriminated_generated)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_generated,
                                            labels=tf.ones_like(discriminated_generated)))

all_vars = tf.trainable_variables()
g_vars = [var for var in all_vars if var.name.startswith('generator')]
d_vars = [var for var in all_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    learned_probs = []
    for _ in range(EPOCHS):
        for _ in range(BATCHS_IN_EPOCH):
            sess.run(d_train_opt)
            t = sess.run(generated_outputs)
            print(t.tolist()[0][0].index(max(t.tolist()[0][0])))
            for _ in range(GENERATOR_TRAINING_FACTOR):
                sess.run(g_train_opt)
        learned_probs.append(sess.run(generated_probs))

