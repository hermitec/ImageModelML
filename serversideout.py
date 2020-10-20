import tensorflow as tf
tf.test.gpu_device_name()
from tensorflow.keras import layers, backend
import numpy as np
import os, sys, time
from tensorflow.keras.preprocessing import image

# h,w must be divisible by 4
h = 50
w = 50
channels = 1
with tf.device('/gpu:0'):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


    checkpoint = tf.train.Checkpoint(vertex_model=vertex_model,
                                    D3=D3,
                                    gan=gan)