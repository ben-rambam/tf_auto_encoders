from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from layers import Dense
#from layers import Conv

tf.logging.set_verbosity(tf.logging.INFO)

def build_vae(sess):

    input_size = 784
    latent_size = 5
    nonlinearity = tf.nn.elu
    squashing = tf.nn.sigmoid
    learning_rate = 5e-4
    dropout = 1.
    lambda_l2_reg = 0.
    beta = 0.

    with tf.name_scope("encoder_decoder"):

        # create a placeholder for the unmodified input to the function by flattening the 28x28 image
        x_in = tf.placeholder(tf.float32, shape=(None,784), name="x_in")
        # create a dense layer object with 500 nodes
        il1 = Dense(scope="encoder_decoder", size=500, dropout=dropout, 
            nonlinearity=nonlinearity)
        # create a dense layer object with 500 nodes
        il2 = Dense(scope="encoder_decoder", size=500, dropout=dropout, 
            nonlinearity=nonlinearity)
        # create a dense layer object with 2 nodes and connect it backwards
        # to the input node
        z_mean = Dense(scope="encoder_decoder", size=latent_size, 
            dropout=dropout)(il2(il1(x_in)))
        # create a dense layer object with 2 nodes and connect it backwards
        # to the input node
        z_log_sigma = Dense(scope="encoder_decoder", size=latent_size, 
            dropout=dropout)(il2(il1(x_in)))
        #z_log_sigma = tf.fill(tf.shape(z_mean), 1., name="z_log_sigma")

        # sample from a unit gaussian
        #epsilon = tf.random_normal(tf.shape(z_log_sigma), name="epsilon")
        epsilon = tf.fill(tf.shape(z_log_sigma), 0., name="epsilon")

        # use the sample to regularize the latent space variable
        z = z_mean + tf.exp(z_log_sigma) * epsilon

        # create a dense layer object with 500 nodes for the decoder
        ol1 = Dense("encoder_decoder", size=500, dropout=dropout, 
            nonlinearity=nonlinearity)

        # create a dense layer object with 500 nodes for the decoder
        ol2 = Dense("encoder_decoder", size=500, dropout=dropout, 
            nonlinearity=nonlinearity)

        # create a dense layer object with 784 nodes to squash the decoder
        # back between zero and one
        ol3 = Dense("encoder_decoder", size=input_size, dropout=dropout,
            nonlinearity=squashing) 

        # create and output, connect it backwards backwards to the latent space
        x_reconstructed = ol3(ol2(ol1(z)))

    with tf.name_scope("decoder_from_latent"):
        # create a placeholder to allow access to the decoder without haveing to
        # go through the encoder first
        z_ = tf.placeholder_with_default(tf.random_normal([1, latent_size]), 
            shape=[None, latent_size], name="latent_in")
        z_ = tf.placeholder(tf.float32, shape=[None, latent_size], name="z")
        x_reconstructed_ = ol3(ol2(ol1(z_)))

    with tf.name_scope("loss"):
        #bound by clipping to avoid nan
        obs_ = tf.clip_by_value(x_reconstructed, 1e-7, 1-1e-7)
        rec_loss = -tf.reduce_sum(x_in * tf.log(obs_) 
            + (1-x_in) * tf.log(1-obs_),1)
        kl_loss = -0.5 * tf.reduce_sum(1+2 * z_log_sigma - z_mean**2 
            - tf.exp(2 * z_log_sigma), 1)
        tf.summary.scalar("KL-Divergence", tf.reduce_mean(kl_loss))
        tf.summary.scalar("Weighted_KL-Divergence", tf.reduce_mean(beta*kl_loss))
        tf.summary.scalar("Reconstruction", tf.reduce_mean(rec_loss))

    with tf.name_scope("l2_regularization"):
        regularizers = [tf.nn.l2_loss(var) for var in sess.graph.get_collection(
            "trainable_variables") if "weights" in var.name]
        l2_reg = lambda_l2_reg * tf.add_n(regularizers)

    with tf.name_scope("loss/"):
        cost = tf.reduce_mean(rec_loss + beta*kl_loss, name="vae_cost")
        cost += l2_reg
        tf.summary.scalar("Total", tf.reduce_mean(cost))

    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("Adam_optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads_and_vars = optimizer.compute_gradients(cost, tvars)
        #clipped = [(tf.clip_by_value(grad, -5, 5), tvar)
        #    for grad, tvar in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step,
            name="minimize_cost")
        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed, 
            z_, x_reconstructed_, cost, global_step, train_op)

def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    sess = tf.Session()

    (x_in, dropout_, z_mean, z_log_sigma, x_reconstructed, 
        z_, x_reconstructed_,cost, global_step, train_op) = build_vae(sess)

    saver = tf.train.Saver()


    train_writer = tf.summary.FileWriter('/tmp/jon_vae/train-5-direct')
    test_writer = tf.summary.FileWriter('/tmp/jon_vae/test-5-direct')
    summaries = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        # every run, get a new batch and train the autoencoder and record
        x, _ = mnist.train.next_batch(150)
        fetches = [summaries, train_op]
        feed_dict = {x_in: x}
        summary, _ = sess.run(fetches, feed_dict=feed_dict)
        train_writer.add_summary(summary,i)

        # every 10 runs, check how we're doing against the test data and record
        if i%10 == 0:
            x, _ = mnist.test.next_batch(150)
            fetches = [summaries, cost]
            feed_dict = {x_in: x}
            summary, cost_ = sess.run(fetches, feed_dict=feed_dict)
            test_writer.add_summary(summary,i)
        if i%100 == 0:
            save_path = saver.save(sess, "/tmp/jon_vae/model.ckpt")
            print("Model saved at step %d in %s" % (i, save_path))

if __name__ == "__main__":
    tf.app.run()
