#!/usr/bin/python3

import tensorflow as tf
import matplotlib.pyplot as plt 

import tensorflow.keras.backend as K 

from ntm import NTM
from seqgen import generate_patterns

# Train parameters
epoch = 100
batch_size = 1
steps_per_epoch = 100*100
learning_rate = 1e-4

# NTM parameters
controller_size = 100
memory_locations = 128
memory_vector_size = 20
maximum_shifts = 3

# Copy task sequence generator parameters
max_sequence = 5
min_sequence = 1
in_bits = 8
out_bits = 8

# Training
ntm_model = NTM(controller_size, memory_locations, memory_vector_size, maximum_shifts, out_bits)
optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)
bce_loss = tf.losses.BinaryCrossentropy()

clip_grad_min = -10.0
clip_grad_max = 10.0

train_loss = tf.metrics.Mean(name="train_loss")
train_accuracy = tf.metrics.BinaryAccuracy(name="train_accuracy")

# Checkpoints
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=ntm_model)
manager = tf.train.CheckpointManager(ckpt, './tensorflow_ckpt', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    print("Restoring NTM model from {}".format(manager.latest_checkpoint))
else:
    print("Training NTM model from scratch")


def train_one_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = ntm_model(x)
        loss = bce_loss(y, y_pred)

    gradients = tape.gradient(loss, ntm_model.trainable_variables)
    clipped_gradients = [tf.clip_by_value(grads, clip_grad_min, clip_grad_max) for grads in gradients]
    optimizer.apply_gradients(zip(clipped_gradients, ntm_model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, y_pred)


# Training loop
try:
    for e in range(epoch):
        for step in range(steps_per_epoch):
            x, y = generate_patterns(batch_size, max_sequence, min_sequence, in_bits, out_bits,
                                     pad=K.epsilon(), low_tol=K.epsilon())

            train_one_step(x, y)

            template = "Epoch: {}, Train step: {}, Train loss: {}, Train accuracy: {}"
            print(template.format(
                                e+1,
                                step+1,
                                train_loss.result(),
                                train_accuracy.result()*100.0
                                ))

            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

except KeyboardInterrupt:
    print("User interrupted")

# Visualize the prediction made by the model
x, y = generate_patterns(batch_size, max_sequence, min_sequence, in_bits, out_bits, fixed_seq_len=False)
y_pred = ntm_model(x)

plt.matshow(x[0])
plt.matshow(y[0])
plt.matshow(y_pred[0])
plt.show()
