#!/usr/bin/python3

import datetime
import tensorflow as tf
import matplotlib.pyplot as plt 

import tensorflow.keras.backend as K 

from ntm import NTM
from seqgen import generate_patterns

# Train parameters
epochs = 100
batch_size = 1
steps_per_epoch = 2000
learning_rate = 1e-4
momentum = 0.9
clip_grad_min = -10.0
clip_grad_max = 10.0

# NTM parameters
controller_size = 100
memory_locations = 128
memory_vector_size = 20
maximum_shifts = 3

# Copy task sequence generator parameters
max_sequence = 20
min_sequence = 1
in_bits = 8
out_bits = 8

# Training
ntm_model = NTM(controller_size, memory_locations, memory_vector_size, maximum_shifts, out_bits)
optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum)
bce_loss = tf.losses.BinaryCrossentropy()

# Stateful metrics
train_loss = tf.metrics.Mean(name="train_loss")
train_cost = tf.metrics.Mean(name="train_cost")

# Checkpoints
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=ntm_model)
manager = tf.train.CheckpointManager(ckpt, './tf_ntm_ckpt', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

# Tensorboard
# tensorboard --logdir tf_ntm_logs/gradient_tape
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './tf_ntm_logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

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
        y_pred = ntm_model(x, training=True)
        loss = bce_loss(y, y_pred)
        cost = cost_per_sequence(y, y_pred)

    gradients = tape.gradient(loss, ntm_model.trainable_variables)
    clipped_gradients = [tf.clip_by_value(grads, clip_grad_min, clip_grad_max) for grads in gradients]
    optimizer.apply_gradients(zip(clipped_gradients, ntm_model.trainable_variables))

    train_loss(loss)
    train_cost(cost)


# Training loop
try:
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            x, y = generate_patterns(batch_size, max_sequence, min_sequence, in_bits, out_bits,
                                     pad=K.epsilon(), low_tol=K.epsilon())

            train_one_step(x, y)

            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
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
                tf.summary.scalar("loss", train_loss.result(), step=(int(ckpt.step)*batch_size))
                tf.summary.scalar("cost_per_sequence", train_cost.result(), step=(int(ckpt.step)*batch_size))

        train_loss.reset_states()
        train_cost.reset_states()

except KeyboardInterrupt:
    print("User interrupted")

# Visualize the prediction made by the model
x, y = generate_patterns(batch_size, max_sequence, min_sequence, in_bits, out_bits, fixed_seq_len=True)
y_pred = ntm_model(x)
rt, r_wt, at, w_wt, Mt = ntm_model.debug_ntm()

plt.matshow(rt, cmap=plt.get_cmap('jet'))
plt.matshow(at, cmap=plt.get_cmap('jet'))
plt.matshow(w_wt, cmap=plt.get_cmap('gray'))
plt.matshow(r_wt, cmap=plt.get_cmap('gray'))
plt.matshow(Mt, cmap=plt.get_cmap('jet'))

plt.matshow(x[0], cmap=plt.get_cmap('jet'))
plt.matshow(y_pred[0], cmap=plt.get_cmap('jet'))
plt.show()
