import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

######Parameters###################
init_eta = 1e-1
eta_decay = 0.9
eta_decays_every = 200
min_eta = 1e-2
gen_eta = 1e-2
nepochs = 20000 
eval_every = 100
eval_every_gen = 10 # how often to eval during generalization phase
termination_thresh = 0.01 # stop at this loss
nruns = 100 
num_inputs = 4 + 2 # four shapes, two colors, each feature is one-hot
num_outputs = 4 # four possible locations, one-hot
num_hidden = num_inputs
S = 4
nonlinearity_function = tf.nn.relu
###################################

dataset_1 = {"x": np.zeros([2, num_inputs]),
             "y": np.zeros([2, num_outputs])}
dataset_1["x"][0:, 0] = 1. # first shape
dataset_1["x"][0, 4] = 1. # red
dataset_1["y"][0, 0] = 1. # Q1

dataset_1["x"][1, 5] = 1. # blue
dataset_1["y"][1, 1] = 1. # Q2


dataset_2 = {"x": np.zeros([2, num_inputs]),
             "y": np.zeros([2, num_outputs])}
dataset_2["x"][0:, 1] = 1. # second shape
dataset_2["x"][0, 4] = 1. # red
dataset_2["y"][0, 2] = 1. # Q3

dataset_2["x"][1, 5] = 1. # blue
dataset_2["y"][1, 3] = 1. # Q4

dataset_1A = {"x": np.zeros([2, num_inputs]),
             "y": np.zeros([2, num_outputs])}
dataset_1A["x"][0:, 2] = 1. # third shape
dataset_1A["x"][0, 4] = 1. # red
dataset_1A["y"][0, 0] = 1. # Q1

dataset_1A["x"][1, 5] = 1. # blue
dataset_1A["y"][1, 1] = 1. # Q2

dataset_3 = {"x": np.zeros([2, num_inputs]),
             "y": np.zeros([2, num_outputs])}
dataset_3["x"][0:, 3] = 1. # fourth shape
dataset_3["x"][0, 4] = 1. # red
dataset_3["y"][0, 0] = 1. # Q1

dataset_3["x"][1, 5] = 1. # blue
dataset_3["y"][1, 3] = 1. # Q4

datasets = [dataset_1, dataset_2, dataset_1A, dataset_3]

def batch_datasets(dataset_list):
    batch_dataset = {
       "x": np.concatenate([d["x"] for d in dataset_list]),
       "y": np.concatenate([d["y"] for d in dataset_list])
    }
    return batch_dataset

batched_train_datasets = batch_datasets(datasets[:2])
batched_datasets = batch_datasets(datasets)

for rseed in xrange(nruns):
    for nlayer in [4, 2]:
        num_hidden = num_hidden
        print "nlayer %i run %i" % (nlayer, rseed)
        filename_prefix = "results/nlayer_%i_rseed_%i_" %(nlayer, rseed)

        np.random.seed(rseed)
        tf.set_random_seed(rseed)

        input_ph = tf.placeholder(tf.float32, shape=[None, num_inputs])
        target_ph = tf.placeholder(tf.float32, shape=[None, num_outputs])

        Win = tf.Variable(tf.random_uniform([num_inputs, num_hidden],0.,0.5/(num_hidden+num_inputs)))
        bi = tf.Variable(tf.ones([num_hidden,]))
        internal_rep = nonlinearity_function(tf.matmul(input_ph, Win) + bi)
        hidden_weights = []

        for layer_i in range(1, nlayer-1):
            W = tf.Variable(tf.random_normal([num_hidden, num_hidden],0.,0.5/num_hidden))
            b = tf.Variable(tf.ones([num_hidden,]))
            internal_rep = nonlinearity_function(tf.matmul(internal_rep, W) + b)

        bo = tf.Variable(tf.ones([num_outputs,]))
        Wout = tf.Variable(tf.random_uniform([num_hidden, num_outputs],0.,0.5/(num_hidden+num_outputs)))
        output_logits = tf.matmul(internal_rep, Wout) + bo
        output = tf.nn.softmax(output_logits) 

        item_losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_ph, logits=output_logits)
        loss = tf.reduce_mean(item_losses)
        eta_ph = tf.placeholder(tf.float32)
        optimizer = tf.train.GradientDescentOptimizer(eta_ph)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        def test_accuracy():
            losses = []
            for dataset in datasets:
                this_loss = sess.run(loss, feed_dict={input_ph: dataset["x"],
                                                      target_ph: dataset["y"]})
                losses.append(this_loss)
            return tuple(losses) # for format strings

        def print_outputs():
            for dataset in datasets:
                print(sess.run(output, feed_dict={eta_ph: curr_eta,
                                                  input_ph: dataset["x"],
                                                  target_ph: dataset["y"]}))

        def run_train_epoch(datasets):
            for dataset in datasets:
                sess.run(train, feed_dict={eta_ph: curr_eta,
                                           input_ph: dataset["x"],
                                           target_ph: dataset["y"]})


        def run_batched_train_epoch(batched_dataset):
            sess.run(train, feed_dict={eta_ph: curr_eta,
                                       input_ph: batched_dataset["x"],
                                       target_ph: batched_dataset["y"]})

        loss_p_format = "1: %.3f, 2: %.3f, 1a: %.3f, 3: %.3f"
        loss_f_format = ", ".join(["%f"]*4) + "\n" 
        print(tuple(test_accuracy()))
        print "Initial losses: " + loss_p_format % (test_accuracy())

        #loaded_output_logitss = np.loadtxt(output_logits_filename_to_load,delimiter=',')

        rep_track = []
        loss_filename = filename_prefix + "loss_track.csv"
        with open(loss_filename, 'w') as fout:
            fout.write("epoch, loss_1, loss_2, loss_1A, loss_3\n")
            curr_mses = test_accuracy()
            fout.write(("%i, " % 0) + loss_f_format % curr_mses)

            # initial training
            curr_eta = init_eta
            for epoch in xrange(1, nepochs+1):
#                run_train_epoch(datasets[:2])
                run_batched_train_epoch(batched_train_datasets)
                if epoch % eval_every == 0:
                    curr_mses = test_accuracy()
                    print ("epoch: %i, losses: " % epoch) + (loss_p_format % curr_mses) 
                    fout.write(("%i, " % epoch) + loss_f_format % curr_mses)
                    if np.all(np.array(curr_mses[:2]) < termination_thresh):
                        print("Early stop training!")
                        break

                if epoch % eta_decays_every == 0 and epoch > 0 and curr_eta > min_eta:
                    curr_eta *= eta_decay

            # generalization phase
            curr_eta = gen_eta
            for epoch in xrange(nepochs+1, 2*nepochs+1):
#                run_train_epoch(datasets)
                run_batched_train_epoch(batched_datasets)
                if epoch % eval_every_gen == 0:
                    curr_mses = test_accuracy()
                    print ("epoch: %i, losses: " % epoch) + (loss_p_format % curr_mses) 
                    fout.write(("%i, " % epoch) + loss_f_format % curr_mses)
                    if np.all(np.array(curr_mses) < termination_thresh):
                        print("Early stop generalization!")
                        break

        print "Final losses: " + loss_p_format % (test_accuracy())
