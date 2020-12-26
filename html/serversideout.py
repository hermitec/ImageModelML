import tensorflow as tf
tf.test.gpu_device_name()
from tensorflow.keras import layers, backend
import numpy as np
import os, sys, time
from tensorflow.keras.preprocessing import image

G = None
D = None
gan = None
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(G=G.x,
                                D=D.x,
                                gan=gan)
raw_data = []
raw_labels = []
data_folder = "../Dataset/"
def update_batch():
    # 8gb ram means manual batch loading
    # i hope you find humour in my suffering
    global raw_data, raw_labels, data_folder, BATCH_SIZE,current_index
    raw_data = []
    raw_labels = []
    for i in os.listdir(data_folder)[current_index:current_index+BATCH_SIZE]:

        if len(os.listdir(data_folder+i)) > 0:
            imgdirs = [data_folder+i+"/"+p for p in os.listdir(data_folder+i) if p.split(".")[1] == "png"]
            imgs = [load_preprocess(x) for x in imgdirs]
            raw_data.append(imgs)

            objdir = [data_folder+i+"/"+p for p in os.listdir(data_folder+i) if p.split(".")[1] == "obj"]
            obj = parse(objdir[0])
            raw_labels.append(obj)
    raw_data = np.reshape(raw_data, (BATCH_SIZE,6,h,w,channels))
    if(current_index + BATCH_SIZE >= len(os.listdir(data_folder))):
        current_index = 0
    else: current_index += BATCH_SIZE

update_batch()

out = []
for i in G.x.predict(np.array(raw_data[0]).reshape((1,6,w,h,channels))).tolist():
    out.append(i)

f = open("testfile.obj","w+")
f.write("o Cube\n")
f.close()
for i in out[0][0:]:
    f = open("testfile.obj","a+")
    print("v {0} {1} {2}\n".format(i[0],i[1],i[2]))
    f.write("v {0} {1} {2}\n".format(i[0],i[1],i[2]))
    f.close()
