#!/usr/bin/python3

import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

import tensorflow.keras.backend as K 

from ntm import NTM
from seqgen import generate_patterns

parser = argparse.ArgumentParser(description="Neural Turing Machine")

# Train parameters
parser.add_argument('--train', action="store_true", default=False,
                    help="Train the NTM")
parser.add_argument('--test', action="store_true", default=False,
                    help="Test the NTM with a copy task input sequence")
parser.add_argument('--visualize', action="store_true", default=False,
                    help="Visualize the working of the NTM on the copy task")
parser.add_argument('--fixed_seq_len', action="store_true", default=True,
                    help="Whether a fixed or random sequence length should be used when visualizing")
parser.add_argument('--epochs', action="store", dest="epochs", default=100, type=int,
                    help="Epochs for training")
parser.add_argument('--batches', action="store", dest="batch_size", default=2, type=int,
                    help="Number of batches to be used in one training step")
parser.add_argument('--steps_per_epoch', action="store", dest="steps_per_epoch", default=2000, type=int,
                    help="The number of batches shown per epoch")
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=1e-4, type=float,
                    help="Learning rate to be used to be used by RMSprop for the NTM")
parser.add_argument('--momentum', action="store", dest="momentum", default=0.9, type=float,
                    help="Momentum used by RMSprop optimizer")
parser.add_argument('--clip_grad_min', action="store", dest="clip_grad_min", default=-10.0, type=float,
                    help="Minimum value to clip the gradients")
parser.add_argument('--clip_grad_max', action="store", dest="clip_grad_max", default=10.0, type=float,
                    help="Maximum value to clip the gradients")

# NTM parameters
parser.add_argument('--controller_size', action="store", dest="controller_size", default=100, type=int,
                    help="Controller size of the NTM")
parser.add_argument('--memory_locations', action="store", dest="memory_locations", default=128, type=int,
                    help="Number of memory locations")
parser.add_argument('--memory_vector_size', action="store", dest="memory_vector_size", default=20, type=int,
                    help="Number of memory vector size")
parser.add_argument('--maximum_shifts', action="store", dest="maximum_shifts", default=3, type=int,
                    help="The maximum number of shifts allowed over the memory")
parser.add_argument('--learn_r_bias', action="store_true", default=False,
                    help="Learn the read vector initialization")
parser.add_argument('--learn_w_bias', action="store_true", default=False,
                    help="Learn the weight vector initialization")
parser.add_argument('--learn_m_bias', action="store_true", default=False,
                    help="Learn the memory matrix initialization")

# Copy task sequence generator parameters
parser.add_argument('--max_sequence', action="store", dest="max_sequence", default=20, type=int,
                    help="The maximum allowed sequence")
parser.add_argument('--min_sequence', action="store", dest="min_sequence", default=1, type=int,
                    help="The minimum allowed sequence")
parser.add_argument('--in_bits', action="store", dest="in_bits", default=8, type=int,
                    help="The number of in bits")
parser.add_argument('--out_bits', action="store", dest="out_bits", default=8, type=int,
                    help="The number of out bits")

# Tensorflow checkpoints and tensorboard
parser.add_argument('--checkpoint_dir', action="store", dest="checkpoint_dir", default='./tf_ntm_ckpt/',
                    help="The location to save the checkpoint")
parser.add_argument('--max_to_keep', action="store", dest="max_to_keep", default=3, type=int,
                    help="Maximum number of checkpoint to keep")
parser.add_argument('--report_interval', action="store", dest="report_interval", default=10, type=int,
                    help="The report interval for the train information")
parser.add_argument('--train_log_dir', action="store", dest="train_log_dir", default='./tf_ntm_logs/gradient_tape/',
                    help="The location to save the training logs")

arg = parser.parse_args()

# Training
ntm_model = NTM(arg.controller_size, arg.memory_locations, arg.memory_vector_size, arg.maximum_shifts,
                arg.out_bits, arg.learn_r_bias, arg.learn_w_bias, arg.learn_m_bias)

optimizer = tf.optimizers.RMSprop(learning_rate=arg.learning_rate, momentum=arg.momentum)
bce_loss = tf.losses.BinaryCrossentropy()

# Training metrics
train_loss = tf.metrics.Mean(name="train_loss")
train_cost = tf.metrics.Mean(name="train_cost")

# Tensorboard
# tensorboard --logdir tf_ntm_logs/gradient_tape
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = arg.train_log_dir + current_time + '/train'

# Checkpoints
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=ntm_model)
manager = tf.train.CheckpointManager(ckpt, arg.checkpoint_dir, arg.max_to_keep)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restoring NTM model from {}".format(manager.latest_checkpoint))
else:
    print("Training NTM model from scratch")


def cost_per_sequence(y_true, y_pred):  # Calculates the bit errors per sequence
    cost = tf.reduce_sum(tf.abs(y_true - y_pred))
    cost = cost/tf.cast(y_true.shape[0], tf.float32)
    return cost


def train_one_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = ntm_model(x, training=arg.train)
        loss = bce_loss(y, y_pred)
        cost = cost_per_sequence(y, y_pred)

    gradients = tape.gradient(loss, ntm_model.trainable_variables)
    clipped_gradients = [tf.clip_by_value(grads, arg.clip_grad_min, arg.clip_grad_max) for grads in gradients]
    optimizer.apply_gradients(zip(clipped_gradients, ntm_model.trainable_variables))

    train_loss(loss)
    train_cost(cost)


# Training loop
if arg.train:
    print("Training NTM")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    try:
        for epoch in range(arg.epochs):
            for step in range(arg.steps_per_epoch):
                x, y = generate_patterns(arg.batch_size, arg.max_sequence, arg.min_sequence,
                                         arg.in_bits, arg.out_bits,
                                         pad=K.epsilon(), low_tol=K.epsilon())

                train_one_step(x, y)

                ckpt.step.assign_add(1)
                if int(ckpt.step) % arg.report_interval == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                template = "Epoch: {}, Train step: {}, Train loss: {}, Train cost: {}"
                print(template.format(
                    epoch + 1,
                    step + 1,
                    train_loss.result(),
                    train_cost.result()))

                with train_summary_writer.as_default():
                    # The loss and cost per sequence against the number of sequences shown to the model
                    tf.summary.scalar("loss", train_loss.result(), step=(int(ckpt.step)*arg.batch_size))
                    tf.summary.scalar("cost_per_sequence", train_cost.result(), step=(int(ckpt.step)*arg.batch_size))

            train_loss.reset_states()
            train_cost.reset_states()

    except KeyboardInterrupt:
        print("User interrupted")

# Visualize the prediction made by the model
if arg.test or arg.visualize:
    x, y = generate_patterns(arg.batch_size, arg.max_sequence, arg.min_sequence,
                             arg.in_bits, arg.out_bits, fixed_seq_len=arg.fixed_seq_len)

    y_pred = ntm_model(x)
    rt, r_wt, at, w_wt, Mt = ntm_model.debug_ntm()

    cmap_jet = plt.get_cmap('jet')
    cmap_gray = plt.get_cmap('gray')

    if arg.visualize:
        fig_ntm, (ax_at, ax_wwt, ax_mt, ax_rwt, ax_rt) = plt.subplots(5, 1)
        fig_ntm.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, hspace=0.3)

        ax_at.set_ylabel('Adds')
        ax_wwt.set_ylabel('Write Weights')
        ax_mt.set_ylabel("Memory")
        ax_rwt.set_ylabel("Read Weights")
        ax_rt.set_ylabel("Reads")

        ax_at.matshow(np.transpose(at), aspect='equal', cmap=cmap_jet)
        ax_wwt.matshow(w_wt,  aspect='auto', cmap=cmap_gray)
        ax_mt.matshow(np.transpose(Mt), aspect='auto', cmap=cmap_jet)
        ax_rwt.matshow(r_wt, aspect='auto', cmap=cmap_gray)
        ax_rt.matshow(np.transpose(rt), aspect='equal', cmap=cmap_jet)

    if arg.test:
        fig_ntm_out, (ax_t, ax_p) = plt.subplots(2, 1)
        fig_ntm_out.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, hspace=0.3)
        t = ax_t.matshow(np.transpose(x[0]), aspect='auto', cmap=cmap_jet)
        ax_t.set_ylabel("Target")
        p = ax_p.matshow(np.transpose(y_pred[0]), aspect='auto', cmap=cmap_jet)
        ax_p.set_ylabel("Prediction")

        fig_ntm_out.suptitle('NTM Copy Task (Sequence Length {})'.format(arg.max_sequence))
        fig_ntm_out.colorbar(t, ax=(ax_t, ax_p), orientation="vertical", fraction=0.1)

    plt.show()
