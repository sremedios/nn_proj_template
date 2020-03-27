import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import shutil
import time
import json

import tensorflow as tf
import numpy as np
from pathlib import Path

from utils.forward import *
from utils.progbar import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

if __name__ == '__main__':
    N_EPOCHS = 100000
    BATCH_SIZE = 2**6
    progbar_length = 5
    test_color_code = "\033[0;0m"

    MODEL_PATH = Path(sys.argv[1])
    WEIGHT_PATH = Path(sys.argv[2])
    RESULTS_DIR = Path(sys.argv[3])
    TEST_PRED_FILENAME = RESULTS_DIR / "test_preds.csv"

    if not RESULTS_DIR.exists():
        d.mkdir(parents=True)


    ##### LOAD MODEL #####
    with open(MODEL_PATH, 'r') as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))

    model.load_weights(str(WEIGHT_PATH))

    #### LOAD DATA #####
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 

    # add channel dimension
    x_test = np.array(x_test[..., np.newaxis], dtype=np.float32) / 255. 
    y_test = np.array(tf.one_hot(y_test, depth=10), dtype=np.float32) 

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
        .batch(BATCH_SIZE)
    

    ##### EVALUATE #####
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

    # test metrics

    y_pred = []
    y_true = []

    cur_step = 1
    iterator = iter(test_dataset)
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

            test_loss.update_state(loss)
            test_accuracy.update_state(
                    tf.argmax(ys, axis=1),
                    tf.argmax(preds, axis=1),
                )

            y_true.extend(tf.argmax(ys, axis=1))
            y_pred.extend(tf.argmax(preds, axis=1))

            batch_end_time = time.time()
            elapsed_time = running_average(
                elapsed_time,
                batch_end_time - batch_start_time,
                cur_step,
            )
            show_progbar(
                1,
                1,
                BATCH_SIZE,
                cur_step,
                len(x_test),
                test_loss.result(),
                test_accuracy.result(),
                test_color_code,
                elapsed_time,
                progbar_length=progbar_length,
            )

            cur_step += 1
    print()

    # Write header
    with open(str(TEST_PRED_FILENAME), 'w') as f:
        f.write("y_pred,y_true\n")

    # Write results 
    with open(str(TEST_PRED_FILENAME), 'a') as f:
        for y_p, y_t in zip(y_pred, y_true):
            f.write("{:.4f},{:.4f}\n".format(
                y_p.numpy(),
                y_t.numpy(),
            ))
