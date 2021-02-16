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

    class Model:

        def __init__(self, input_shape, output_shape):
            self.input_dim = input_shape
            self.output_dim = output_shape
            self.x = None
            self.input_layer = None

        def initModel(self):
            self.input_layer = layers.Input(shape=self.input_dim)
            self.x = self.input_layer

        def finishModel(self):
            self.x = tf.keras.models.Model(self.input_layer,self.x)

        def addDense(self, x, res=32, chains=1):
            for i in range(chains):
                x = layers.Dense(res)(x)
            return x

        def addConv2D(self, x, chains=1, res=32, stride=2, activation="relu", alpha=0.1):
            for i in range(chains):
                x = layers.Conv2D(res,(stride,stride),activation=activation)(x)
                x = layers.LeakyReLU(alpha=alpha)(x)
            return x

        def addFlatten(self, x):
            x = layers.Flatten()(x)
            return x

        def addReshape(self, x, new_shape):
            x = layers.Reshape(new_shape)(x)
            return x

        def addDropout(self, x, alpha=0.1):
            x = layers.Dropout(alpha)(x)
            return x
            
        def addBatchNorm(self, x):
            x = layers.BatchNormalization()(x)
            return x


    class GAN(Model):

        def __init__(self, input_shape, output_shape, generator, discriminator):
            super().__init__(input_shape,output_shape)
            self.G = generator
            self.D = discriminator

        def runTrainLoop(self):
            return

        def predict(self, data):
            return G.x.predict(np.array(data).reshape((1,6,w,h,channels)))


    def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
        """
            input_tensor: tensor, with the shape (batch_size, time_steps, dims)
            filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
            kernel_size: int, size of the convolution kernel
            strides: int, convolution step size
            padding: 'same' | 'valid'
        """
        x = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=0))(input_tensor)
        x = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=1))(x)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
        x = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=0))(x)
        return x

    def load_preprocess(path):
        global h,w,channels
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.resize(image, [h, w,])
        image /= 255.0  # normalize to [0,1] range
        return image


# possible new architecture;

# D1 -> Input(images) -> operations -> Some info about image
# D2 -> Input(vertices) -> operations -> info about vertices in same form as output of D1
# D -> Input(Output of D1/2) -> operations -> Binary out

    G = Model((6,h,w,channels),(8,3))
    G.initModel()
    G.x = G.addFlatten(G.x)
    G.x = G.addDense(G.x,(w*h*channels)/4)
    G.x = G.addBatchNorm(G.x)
    G.x = G.addReshape(G.x, (int(w/2), int(h/2), channels))
    G.x = G.addConv2D(G.x,chains=3)
    G.x = G.addFlatten(G.x)
    G.x = G.addDense(G.x,24)
    G.x = G.addReshape(G.x, (8,3))
    G.finishModel()
    print(G.x.output_shape)
    print(G.x.summary())

    D = Model((8,3),(1))
    D.initModel()
    D.x = D.addFlatten(D.x)
    D.x = D.addDense(D.x,16)
    D.x = D.addDropout(D.x, alpha=0.15)
    D.x = D.addDense(D.x,16)
    D.x = D.addDense(D.x,1)
    D.finishModel()
    
    def new_d():
        input_layer = layers.Input(shape=(8,3))
        x = layers.Flatten()(input_layer)
        x = layers.Dense(128)(x)
        second_input = layers.Input(shape=(6,h,w,channels))
        y = layers.Flatten()(second_input)
        y = layers.Dense(128)(y)
        z = layers.concatenate(([x,y]))
        z = layers.Dense(1, activation = "sigmoid")(x)
        z = tf.keras.models.Model([input_layer,second_input],z)
        
        return z

    D.x = new_d()
    D.x.summary()
    input()
    GAN = GAN([(6,h,w,channels),(8,3)], (1), G, D)
    # Training parameters here (learning rate etc.) are an absolute nightmare
    # and the slightest tweak can make or break the learning process
    vertex_optimizer = tf.keras.optimizers.Adam(lr=0.00075, clipvalue=1.0, decay=1e-8,beta_1=0.5)
    G.x.compile(optimizer=vertex_optimizer,loss="binary_crossentropy")
    d_optimizer = tf.keras.optimizers.Adam(lr=0.0005, clipvalue=1.0, decay=1e-8,beta_1=0.5)
    D.x.compile(optimizer=d_optimizer,loss="binary_crossentropy")
    D1_input = tf.keras.Input(shape=(6,w,h,channels))
    D2_input = tf.keras.Input(shape=(8,3))
    # -- #

    gan_input = tf.keras.Input(shape=(6,w,h,channels))
    snd_input = tf.keras.Input(shape=(6,h,w,channels))
    gan_output = D.x([G.x(snd_input), snd_input])
    gan = tf.keras.models.Model([gan_input, snd_input],gan_output)
    gan_optimizer = tf.keras.optimizers.Adam(lr=0.0002, clipvalue=1.0, decay=1e-8,beta_1=0.5)
    gan.compile(gan_optimizer,loss="binary_crossentropy")
    gan.summary()
    input()
    def parse(obj_file):
        obj = open(obj_file,"r").readlines()
        parsed = []
        for i in obj:
            if i[0] == "v" and i[1] == " ":
                parsed.append([float(x.replace("\n","")) for x in i.split(" ")[1:]])
        return parsed

    # TRAINING PARAMS #

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


    checkpoint = tf.train.Checkpoint(G=G.x,
                                    D=D.x,
                                    gan=gan)



    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    history = []



    data_folder = "./Dataset/"
    raw_data = []
    raw_labels = []


    # larger batch size exponentially improves GAN training...
    # but I dont have a supercomputer so it has to be kept relatively small
    # so my 8GB of ram can handle it
    BATCH_SIZE = 15
    current_index = 0

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

    if "-s" in str(sys.argv):
        data_folder = "./user_input/"
        raw_data = []
        imgdirs = [data_folder+i for i in os.listdir(data_folder)]
        imgs = [load_preprocess(x) for x in imgdirs]
        raw_data.append(imgs)
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
        f.write("""f 1/1/1 5/2/1 7/3/1 3/4/1
        f 4/5/2 3/4/2 7/6/2 8/7/2
        f 8/8/3 7/9/3 5/10/3 6/11/3
        f 6/12/4 2/13/4 4/5/4 8/14/4
        f 2/13/5 1/1/5 3/4/5 4/5/5
        f 6/11/6 5/10/6 1/1/6 2/13/6""")
        f.close()
        sys.exit()

    update_batch()
    print(raw_labels)
    # Actual training process:
    training = input("Perform training? y/n :")

    EPOCHS = int(input("Epochs: "))

    if training == "y":

            for i in range(EPOCHS):

                if i % 100 == 0:

                    # Save progress
                    checkpoint.save(file_prefix = checkpoint_prefix)

                    out = []
                    for x in G.x.predict(np.array(raw_data[0]).reshape((1,6,w,h,channels))).tolist():
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

                # right now all data is trained every batch which is an absolutely
                # awful idea
                generated_objs = G.x.predict(raw_data, steps=1)
                print(generated_objs.shape)
                combined_obj = np.concatenate([generated_objs,raw_labels])
                x2 = np.concatenate([raw_data,raw_data])

                misleading_targets = np.ones((len(generated_objs),1))
                misleading_targets += -1 * np.random.random(misleading_targets.shape)

                d_loss = D.x.train_on_batch([combined_obj,x2], np.concatenate([np.zeros((len(raw_labels))),np.ones((len(raw_labels)))]))
                a_loss = gan.train_on_batch([raw_data, raw_data],misleading_targets)

                history.append([d_loss,a_loss])
                os.system("clear")
                print(G.x.predict(np.array(raw_data[0]).reshape((1,6,w,h,channels))))
                print(raw_labels[0])
                print("D LOSS: {0}".format(d_loss))
                print("GAN LOSS: {0}".format(a_loss))


            # #update_batch()
            # for no in range(len(raw_data)):
            #     #this is awful practice temporary terrible idea forces batchsize = 1

            #     # TODO
            #     # FIX
            #     generated_objs = G.predict(raw_data, steps=1)
            #     combined_obj = np.concatenate([generated_objs,raw_labels])
            #     #raw_data is raw images
            #     #raw_labels is raw vertices

            #     misleading_targets = np.ones((len(generated_objs),1))
            #     misleading_targets += -1 * np.random.random(misleading_targets.shape)
            #     d_loss = D.train_on_batch([raw_data[no].reshape(1,15000),combined_obj[no].reshape(1,24)], np.zeros(1))#np.concatenate([np.zeros(1),np.ones(1)]))
            #     a_loss = gan.train_on_batch([raw_data[no].reshape(1,15000),np.array(raw_labels[no]).reshape(1,24)],np.zeros(1))



            # -- Saving data after the fact -- #

            f = open("graph_raw.txt","w+")
            for l in [(str(history[x][0])+","+str(history[x][1])+";\n") for x,i in enumerate(history)]:
                f = open("graph_raw.txt","a+")
                f.write(l)

    f = open("out.obj")
    out = []
    for i in G.x.predict(np.array(raw_data[0]).reshape((1,6,w,h,channels))).tolist():
        out.append(i)

    print(out)
    f = open("out.obj","w+")
    f.write("o Cube\n")
    f.close()
    for i in out[0][0:]:
        f = open("final_out.obj","a+")
        print("v {0} {1} {2}\n".format(i[0],i[1],i[2]))
        f.write("v {0} {1} {2}\n".format(i[0],i[1],i[2]))
        f.close()

    print("HEllO")
    f = open("out.obj","a+")
    f.write("""f 1/1/1 5/2/1 7/3/1 3/4/1
f 4/5/2 3/4/2 7/6/2 8/7/2
f 8/8/3 7/9/3 5/10/3 6/11/3
f 6/12/4 2/13/4 4/5/4 8/14/4
f 2/13/5 1/1/5 3/4/5 4/5/5
f 6/11/6 5/10/6 1/1/6 2/13/6""")
    f.close()
