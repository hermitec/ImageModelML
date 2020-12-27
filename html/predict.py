import tensorflow as tf
tf.test.gpu_device_name()
from tensorflow.keras import layers
import numpy as np
import os, sys, time
from tensorflow.keras.preprocessing import image
h = 32
w = 32
channels = 3
# == initialising == #

# TODO integrate better with main.py so i dont have to change stuff twice
# (will probably require changes to main.py more than here)


def model_vertex_creator():

    # Currently model is a DCGAN

    global h,w,channels
    a_input = layers.Input(shape=(6,w,h,channels))
    x = layers.Flatten()(a_input)
    x = layers.Dense(w*h*channels/4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((w/2,h/2,channels))(x)
    x = layers.Conv2D(32,(3,3),activation="relu")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(32,(3,3),activation="relu")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(32,(3,3),activation="relu")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(24)(x)
    x = layers.Reshape((8,3))(x)
    model = tf.keras.models.Model(a_input,x)
    return model

def model_vertex_discriminator():
    global h,w,channels
    b_input = layers.Input(shape=(8,3))
    x = layers.Conv1D(64,(1),activation="relu")(b_input)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)                  # Without dropout on D, D will always outtrain G early
    x = layers.Dense(1, activation="sigmoid")(x)# and G will never learn how to trick D.

    model = tf.keras.models.Model(b_input,x)
    return model

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

vertex_model = model_vertex_creator()
vertex_discriminator = model_vertex_discriminator()
gan_input = tf.keras.Input(shape=(6,w,h,channels))
gan_output = vertex_discriminator(vertex_model(gan_input))
gan = tf.keras.models.Model(gan_input,gan_output)

checkpoint = tf.train.Checkpoint(vertex_model=vertex_model,
                                vertex_discriminator=vertex_discriminator,
                                gan=gan)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# ====== #

raw_data = []
#THIS WILL NEED TO BE PASSED AS CMDLINE ARGS IN FINAL VERSION
for i in os.listdir(input("File path (folder with all 6 images):")):
        imgdirs = [data_folder+i+"/"+p for p in os.listdir(data_folder+i) if p.split(".")[1] == "png"]
        imgs = [load_preprocess(x) for x in imgdirs]
        raw_data.append(imgs)

print(raw_data)
