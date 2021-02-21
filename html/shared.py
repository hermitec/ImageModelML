

#File containing model setup

import tensorflow as tf
tf.test.gpu_device_name()
from tensorflow.keras import layers
import numpy as np
import os, sys, time
from tensorflow.keras.preprocessing import image

#test
# h,w must be divisible by 4
h = 32
w = 32
channels = 3

def load_preprocess(path):
    global h,w,channels
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize(image, [h, w,])
    image /= 255.0  # normalize to [0,1] range
    return image


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
    x = layers.Dropout(0.35)(x)                  # Without dropout on D, D will always outtrain G early
    x = layers.Dense(1, activation="sigmoid")(x)# and G will never learn how to trick D.

    model = tf.keras.models.Model(b_input,x)
    return model

def parse(obj_file):
    obj = open(obj_file,"r").readlines()
    parsed = []
    for i in obj:
        if i[0] == "v" and i[1] == " ":
            parsed.append([float(x.replace("\n","")) for x in i.split(" ")[1:]])
    return parsed


# Training parameters here (learning rate etc.) are an absolute nightmare
# and the slightest tweak can make or break the learning process
vertex_model = model_vertex_creator()
vertex_optimizer = tf.keras.optimizers.Adam(lr=0.00075, clipvalue=1.0, decay=1e-8,beta_1=0.5)
vertex_model.compile(optimizer=vertex_optimizer,loss="binary_crossentropy")

vertex_discriminator = model_vertex_discriminator()
d_optimizer = tf.keras.optimizers.Adam(lr=0.0005, clipvalue=1.0, decay=1e-8,beta_1=0.5)
vertex_discriminator.compile(optimizer=d_optimizer,loss="binary_crossentropy")

gan_input = tf.keras.Input(shape=(6,w,h,channels))
gan_output = vertex_discriminator(vertex_model(gan_input))
gan = tf.keras.models.Model(gan_input,gan_output)
gan_optimizer = tf.keras.optimizers.Adam(lr=0.0002, clipvalue=1.0, decay=1e-8,beta_1=0.5)
gan.compile(gan_optimizer,loss="binary_crossentropy")
