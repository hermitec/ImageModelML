import tensorflow as tf
tf.test.gpu_device_name()
from tensorflow.keras import layers
import numpy as np
import os, sys, time
from tensorflow.keras.preprocessing import image

#test
# h,w must be divisible by 4
h = 72
w = 72
channels = 1
with tf.device('/cpu:0'):

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
        x = layers.Reshape((int(w/2),int(h/2),channels))(x)
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

    # -- #


    # larger batch size exponentially improves GAN training...
    # but I dont have a supercomputer so it has to be kept relatively small
    # so my 8GB of ram can handle it
    BATCH_SIZE = 4
    current_index = 0

    data_folder = input("Input folder: ")
    raw_data = []
    raw_labels = []

    def update_batch():
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
    print(raw_labels)

    # -- #

    gan_input = tf.keras.Input(shape=(6,w,h,channels))
    gan_output = vertex_discriminator(vertex_model(gan_input))
    gan = tf.keras.models.Model(gan_input,gan_output)
    gan_optimizer = tf.keras.optimizers.Adam(lr=0.0002, clipvalue=1.0, decay=1e-8,beta_1=0.5)
    gan.compile(gan_optimizer,loss="binary_crossentropy")


    # TRAINING PARAMS #

    EPOCHS = int(input("Epochs: ")) # Currently useless, need to implement because GANs become
                                # far easier to train the larger the batch size.

    history = []

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


    checkpoint = tf.train.Checkpoint(vertex_model=vertex_model,
                                    vertex_discriminator=vertex_discriminator,
                                    gan=gan)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    history = []
    # Actual training process:

    training = input("Perform training? y/n :")

    if training == "y":

        for i in range(EPOCHS):

            if i % 100 == 0:

                # Save progress
                checkpoint.save(file_prefix = checkpoint_prefix)

                out = []
                for x in vertex_model.predict(np.array(raw_data[0]).reshape((1,6,w,h,channels))).tolist():
                    out.append(x)
                out_str = "./Output/out_at_epoch_{}.obj".format(i)
                print(out)

                # Save current predictions of G
                f = open(out_str,"w+")
                f.write("o Cube\n")
                for x in out[0][0:]:
                    f = open(out_str,"a+")
                    print("v {0} {1} {2}\n".format(x[0],x[1],x[2]))
                    f.write("v {0} {1} {2}\n".format(x[0],x[1],x[2]))


            #... and train for another epoch

            update_batch()
            print(raw_data.shape)
            generated_objs = vertex_model.predict(raw_data, steps=1 )
            combined_obj = np.concatenate([generated_objs,raw_labels])

            misleading_targets = np.ones((len(generated_objs),1))
            misleading_targets += -1 * np.random.random(misleading_targets.shape)

            d_loss = vertex_discriminator.train_on_batch(combined_obj, np.concatenate([np.zeros((len(raw_labels))),np.ones((len(raw_labels)))]))
            a_loss = gan.train_on_batch(raw_data,misleading_targets)

            history.append([d_loss,a_loss])
            os.system("clear")
            print(vertex_model.predict(np.array(raw_data[0]).reshape((1,6,w,h,channels))))
            print(raw_labels[0])
            print("D LOSS: {0}".format(d_loss))
            print("GAN LOSS: {0}".format(a_loss))



        # -- Saving data after the fact -- #

        f = open("graph_raw.txt","w+")
        for l in [(str(history[x][0])+","+str(history[x][1])+";\n") for x,i in enumerate(history)]:
            f = open("graph_raw.txt","a+")
            f.write(l)

    out = []
    for i in vertex_model.predict(np.array(raw_data[0]).reshape((1,6,w,h,channels))).tolist():
        out.append(i)

    print(out)
    f = open("final_out.obj","w+")
    f.write("o Cube\n")
    for i in out[0][0:]:
        f = open("final_out.obj","a+")
        print("v {0} {1} {2}\n".format(i[0],i[1],i[2]))
        f.write("v {0} {1} {2}\n".format(i[0],i[1],i[2]))

    f = open("final_out.obj","a+")
    f.write("""f 1/1/1 5/2/1 7/3/1 3/4/1
    f 4/5/2 3/4/2 7/6/2 8/7/2
    f 8/8/3 7/9/3 5/10/3 6/11/3
    f 6/12/4 2/13/4 4/5/4 8/14/4
    f 2/13/5 1/1/5 3/4/5 4/5/5
    f 6/11/6 5/10/6 1/1/6 2/13/6""")
