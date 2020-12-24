f = open("testfile.txt","a+")
f.write("Use your imagination and pretend this is a model!")
f.close()
G = None
D = None
gan = None
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(G=G.x,
                                D=D.x,
                                gan=gan)

G.predict()
