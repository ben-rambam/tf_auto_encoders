from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from layers import Dense
import sys
import os
#from layers import Conv

tf.logging.set_verbosity(tf.logging.INFO)

def build_lae(sess,input_size,latent_size):

    nonlinearity = None
    squashing = tf.nn.sigmoid
    learning_rate = 5e-4
    dropout = 1.
    lambda_l2_reg = 0.
    beta = 0.

    with tf.name_scope("encoder_decoder"):

        # create a placeholder for the unmodified input to the function by flattening the 28x28 image
        x_in = tf.placeholder(tf.float32, shape=(None,input_size), name="x_in")
        encoder = Dense(scope="encoder_decoder", size=latent_size, dropout=dropout)
        z = encoder(x_in)
        decoder = Dense(scope="encoder_decoder", size=input_size, dropout=dropout)
        x_reconstructed = decoder(z)

    with tf.name_scope("decoder_from_latent"):
        # create a placeholder to allow access to the decoder without haveing to
        # go through the encoder first
        #z_ = tf.placeholder_with_default(tf.random_normal([1, latent_size]), 
        #    shape=[None, latent_size], name="latent_in")
        z_ = tf.placeholder(tf.float32, shape=[None, latent_size], name="z")
        x_reconstructed_ = decoder(z_)

    with tf.name_scope("loss"):
        #bound by clipping to avoid nan
        obs_ = tf.clip_by_value(x_reconstructed, 1e-7, 1-1e-7)
        rec_loss = -tf.reduce_sum(x_in * tf.log(obs_) 
            + (1-x_in) * tf.log(1-obs_),1)
        tf.summary.scalar("Reconstruction", tf.reduce_mean(rec_loss))

    with tf.name_scope("l2_regularization"):
        regularizers = [tf.nn.l2_loss(var) for var in sess.graph.get_collection(
            "trainable_variables") if "weights" in var.name]
        l2_reg = lambda_l2_reg * tf.add_n(regularizers)

    with tf.name_scope("loss/"):
        cost = tf.reduce_mean(rec_loss, name="vae_cost")
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
        return (x_in, dropout, z, x_reconstructed, 
            z_, x_reconstructed_, cost, global_step, train_op)

def plotSubset(input_size, x_in, n=10, cols=None, 
    outlines=True, save=True, name="subset", outdir="."):
    """Util to plot subset of inputs and reconstructed outputs"""
    #print(tf.shape(x_in))
    #print(tf.shape(x_reconstructed))
    n = min(n, x_in.shape[0])
    cols = (cols if cols else n)
    #rows = 2 * int(np.ceil(n / cols)) # doubled b/c input & reconstruction
    rows = 1

    plt.figure(figsize = (cols * 2, rows * 2))
    dim = int(input_size**0.5) # assume square images

    def drawSubplot(x_, ax_):
        plt.imshow(x_.reshape([dim, dim]), cmap="Greys")
        if outlines:
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)
        else:
            ax_.set_axis_off()

    for i, x in enumerate(x_in[:n], 1):
        # display original
        ax = plt.subplot(rows, cols, i) # rows, cols, subplot numbered from 1
        drawSubplot(x, ax)

    #for i, x in enumerate(x_reconstructed[:n], 1):
    #    # display reconstruction
    #    ax = plt.subplot(rows, cols, i + cols * (rows / 2))
    #    drawSubplot(x, ax)

    # plt.show()
    if save:
        #title = "{}_batch_{}_round_{}_{}.png".format(
        #    model.datetime, "_".join(map(str, model.architecture)), model.step, name)
        title = "{}.png".format(name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")

def main(to_reload=None):

    latent_size=5
    name = "linear-{}".format(latent_size)
    input_size=784
    steps=50000
    

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    sess = tf.Session()

    (x_in, dropout_, z, x_reconstructed, 
        z_, x_reconstructed_,cost, global_step, train_op) = build_lae(sess,input_size, latent_size)

    saver = tf.train.Saver()
            
    x_show = eval_data[100:109,:]
    tempName = "_".join((name, "truth"))
    plotSubset(input_size, x_show, n=10, cols=None, outlines=True,
                       save=True, name=tempName, outdir=".")


    train_writer = tf.summary.FileWriter("/tmp/jon_vae/{}-train".format(name))
    test_writer = tf.summary.FileWriter("/tmp/jon_vae/{}-test".format(name))
    summaries = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for i in range(steps):
        # every run, get a new batch and train the autoencoder and record
        x, _ = mnist.train.next_batch(100)
        fetches = [summaries, train_op]
        feed_dict = {x_in: x}
        summary, _ = sess.run(fetches, feed_dict=feed_dict)
        train_writer.add_summary(summary,i)

        # every 10 runs, check how we're doing against the test data and record
        if i%10 == 0:
            x, _ = mnist.test.next_batch(100)
            fetches = [summaries, cost]
            feed_dict = {x_in: x}
            summary, cost_ = sess.run(fetches, feed_dict=feed_dict)
            test_writer.add_summary(summary,i)
        if (i+1)%np.ceil(steps/5) == 0:
            save_path = saver.save(sess, "/tmp/jon_vae/model_{}.ckpt".format(name))
            print("Model saved at step %d in %s" % (i, save_path))
        
            fetches = [x_reconstructed]
            feed_dict = {x_in: x_show}
            x_reconstructed_ = sess.run(fetches, feed_dict=feed_dict)[0]
            #np.reshape(x_reconstructed, shape=(100,784))
            #print(np.shape(x))
            #print(np.shape(x_reconstructed_))

            tempName = "_".join((name, str(i)))
            print(tempName)
            plotSubset(input_size, x_reconstructed_, n=10, cols=None, 
                outlines=True, save=True, name=tempName, outdir=".")
    sess.close()

if __name__ == "__main__":
    tf.reset_default_graph()

    #for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR): try: os.mkdir(DIR)
    #    except(FileExistsError):
    #        pass

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()
