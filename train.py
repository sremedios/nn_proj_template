import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.forward import *
from utils.progbar import *

from models.tinynet import tinynet

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

if __name__ == '__main__':
    N_EPOCHS = 100
    BATCH_SIZE = 2**10
    BUFFER_SIZE = BATCH_SIZE * 2
    TARGET_SHAPE = (28, 28, 1)
    learning_rate = 1e-4
    train_color_code = "\033[0;32m"
    val_color_code = "\033[0;36m"
    test_color_code = "\033[0;37m"
    ds = 1
    CONVERGENCE_EPOCH_LIMIT = 10
    epsilon = 1e-3
    best_val_loss = 100000
    convergence_epoch_counter = 0
    progbar_length = 5

    ##### MODEL SETUP #####

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    MODEL_NAME = "small_cnn"
    WEIGHT_DIR = Path("models/weights") / MODEL_NAME
    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    RESULTS_DIR = Path("results") / MODEL_NAME
    TRAIN_CURVE_FILENAME = RESULTS_DIR / "training_curve.csv"
    TEST_PRED_FILENAME = RESULTS_DIR / "test_preds.csv"

    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not d.exists():
            d.mkdir(parents=True)

    model = tinynet(ds=8, num_outputs=10, shape=TARGET_SHAPE)
    with open(str(MODEL_PATH), 'w') as f:
        json.dump(model.to_json(), f)
    

    ##### LOAD DATA #####
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # add channel dimension
    x_train = np.array(x_train[..., np.newaxis], dtype=np.float32) / 255.
    y_train = np.array(tf.one_hot(y_train, depth=10), dtype=np.float32)
    x_test = np.array(x_test[..., np.newaxis], dtype=np.float32) / 255.
    y_test = np.array(tf.one_hot(y_test, depth=10), dtype=np.float32)


    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(BUFFER_SIZE)\
        .batch(BATCH_SIZE)\

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
        .batch(BATCH_SIZE)


    # Write header of training curve file
    with open(str(TRAIN_CURVE_FILENAME), 'w') as f:
        f.write(
                ("epoch,"
                 "train_loss,train_acc,"
                 "val_loss,val_acc,"
                 "\n")
        )


    print()

    best_epoch = 1
    for cur_epoch in range(N_EPOCHS):
        # metrics
        train_accuracy = tf.keras.metrics.Accuracy(name="train_acc")
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        val_accuracy = tf.keras.metrics.Accuracy(name="val_acc")
        val_loss = tf.keras.metrics.Mean(name="val_loss")


        cur_step = 1
        iterator = iter(train_dataset)
        elapsed_time = 0.0
        while True:
            try:
                batch_start_time = time.time()
                data = next(iterator)
            except StopIteration:
                break
            else:
                xs, ys = data
                grads, loss, logits = forward(
                    inputs=data,
                    model=model,
                    loss_fn=tf.nn.softmax_cross_entropy_with_logits,
                    training=True,
                )

                preds = tf.nn.softmax(logits)

                train_loss.update_state(loss)
                train_accuracy.update_state(
                        tf.argmax(ys, axis=1),
                        tf.argmax(preds, axis=1),
                    )

                opt.apply_gradients(zip(grads, model.trainable_variables))

                batch_end_time = time.time()
                elapsed_time = running_average(
                    elapsed_time,
                    batch_end_time - batch_start_time,
                    cur_step,
                )

                show_progbar(
                    cur_epoch + 1, 
                    N_EPOCHS, 
                    BATCH_SIZE,
                    cur_step,
                    len(x_train),
                    train_loss.result(),
                    train_accuracy.result(),
                    train_color_code,
                    elapsed_time,
                    progbar_length=progbar_length,
                )
                cur_step += 1
        print()

        # validation metrics
        cur_step = 1
        iterator = iter(val_dataset)
        elapsed_time = 0.0
        while True:
            try:
                batch_start_time = time.time()
                data = next(iterator)
            except StopIteration:
                break
            else:
                xs, ys = data
                loss, logits = forward(
                    inputs=data,
                    model=model,
                    loss_fn=tf.nn.softmax_cross_entropy_with_logits,
                    training=False,
                )

                preds = tf.nn.softmax(logits)

                val_loss.update_state(loss)
                val_accuracy.update_state(
                        tf.argmax(ys, axis=1),
                        tf.argmax(preds, axis=1),
                    )

                batch_end_time = time.time()
                elapsed_time = running_average(
                    elapsed_time,
                    batch_end_time - batch_start_time,
                    cur_step,
                )

                show_progbar(
                    cur_epoch + 1, 
                    N_EPOCHS, 
                    BATCH_SIZE,
                    cur_step,
                    len(x_test),
                    val_loss.result(),
                    val_accuracy.result(),
                    val_color_code,
                    elapsed_time,
                    progbar_length=progbar_length,
                )
                cur_step += 1
        print()


        with open(str(TRAIN_CURVE_FILENAME), 'a') as f:
            f.write(
                    ("{}," # epoch
                     "{:.4f},{:.4f}," # train_loss, train_acc 
                     "{:.4f},{:.4f}" # val_loss, val_acc 
                     "\n").format(
                cur_epoch + 1,
                train_loss.result(),
                train_accuracy.result(),
                val_loss.result(),
                val_accuracy.result(),
            ))

        if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
            print("\nNo improvement in {} epochs, model converged.\
                   \nModel achieved best val_loss at epoch {}.\
                   \nVal Loss: {:.4f}".format(
                CONVERGENCE_EPOCH_LIMIT,
                best_epoch,
                best_val_loss
            ))
            break
        
        if val_loss.result() > best_val_loss and\
                np.abs(val_loss.result() - best_val_loss) > epsilon:
            convergence_epoch_counter += 1
        else:
            convergence_epoch_counter = 0

        if val_loss.result() < best_val_loss:
            best_epoch = cur_epoch + 1
            best_val_loss = val_loss.result() 
            model.save_weights(
                str(WEIGHT_DIR / "best_weights.h5")
            )
