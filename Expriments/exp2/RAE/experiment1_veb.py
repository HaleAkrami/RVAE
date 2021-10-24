import PIL.Image as Image
import numpy as np
import tensorflow as tf
import os
import time

import sys
sys.path.append("/big_disk/akrami/git_repos_old/lesion-detector/src/AutoEncoder/L12/original_code/RobustAutoencoder-master/model")
import l21RobustDeepAutoencoderOnST as l21RDA

sys.path.append("/big_disk/akrami/git_repos_old/lesion-detector/src/AutoEncoder/L12/original_code/RobustAutoencoder-master/data")
import ImShow as I

def l21RDAE(X, layers, lamda, folder, learning_rate = 0.15, inner = 100, outer = 10, 
            batch_size = 133,re_init=False,inputsize = (28,28)):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            rael21 = l21RDA.RobustL21Autoencoder(sess = sess, lambda_= lamda*X.shape[0], 
                                                 layers_sizes=layers)
            l21L, l21S = rael21.fit(X = X, sess = sess, inner_iteration = inner, iteration = outer, 
                                    batch_size = batch_size, learning_rate = learning_rate,  
                                    re_init=re_init,verbose = False)
            l21R = rael21.getRecon(X = X, sess = sess)
            l21H = rael21.transform(X, sess)
            Image.fromarray(I.tile_raster_images(X=l21S,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21S.png")
            Image.fromarray(I.tile_raster_images(X=l21R,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21R.png")
            Image.fromarray(I.tile_raster_images(X=l21L,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21L.png")
            l21S.dump("l21S.npk")
    os.chdir("../")

def compare_frame():

    X = np.load(r"/big_disk/akrami/git_repos_old/lesion-detector/src/AutoEncoder/L12/original_code/RobustAutoencoder-master/data/data.npk",allow_pickle=True)

    inner = 50
    outer = 20



    lambda_list = [0.0001, 0.0003, 0.0008, 0.001, 0.0015, 0.00035, 0.00045, 
         0.00055, 0.00065, 0.00075, 0.00085, 0.00095, 0.00105, 0.00115, 0.00125]
#     lambda_list = [0.00015,0.00018,0.0002,0.00025,0.00028,0.0003]
    print(lambda_list)
    
    layers = [784, 400, 200] ## S trans
    print("start")
    start_time = time.time()
    image_X = Image.fromarray(I.tile_raster_images(X = X, img_shape = (28,28), tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    for lam in lambda_list:
        folder = "lam" + str(lam)
        l21RDAE(X = X, layers=layers, lamda = lam, folder = folder, learning_rate = 0.001, 
                inner = inner, outer = outer, batch_size = 133,re_init=True,inputsize = (28,28))
        print("done: lam", str(lam))
    print ("Runing time:" + str(time.time() - start_time) + " s")
if __name__ == "__main__":
    compare_frame()