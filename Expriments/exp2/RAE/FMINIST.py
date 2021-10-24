import PIL.Image as Image
import numpy as np
import tensorflow as tf
import os
import time
import scipy.io as spio
import sys
sys.path.append("/ImagePTE1/akrami/git_repos_old/lesion-detector/src/AutoEncoder/L12/original_code/RobustAutoencoder-master/model")
import l21RobustDeepAutoencoderOnST as l21RDA
sys.path.append("/ImagePTE1/akrami/git_repos_old/lesion-detector/src/AutoEncoder/L12/original_code/RobustAutoencoder-master/data")
import ImShow as I
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist



seed = 10004

def create_data(frac_anom):

    np.random.seed(seed)

    (X, X_lab), (test, _test_lab) = fashion_mnist.load_data()
    X_lab = np.array(X_lab)

    # find bags
    ind = np.isin(X_lab, (0, 1, 2, 3, 4, 5, 6, 8))  #(1, 5, 7, 9)
    X_lab_outliers = X_lab[ind]
    X_outliers = X[ind]

    # find sneaker and ankle boots
    ind = np.isin(X_lab, (7, 9))  # (0, 2, 3, 4, 6))  #
    X_lab = X_lab[ind]
    X = X[ind]

    # X = ((X / 255.0) > 0.01).astype(np.float)
    X = X / 255.0
    #    X = X[:10000, ]
    #    X_lab = X_lab[:10000, ]

    # test_images = test_images / 255

    Nsamp = np.int(np.rint(len(X) * frac_anom)) + 1

    X_outliers = X_outliers / 255.0

    #N=np.ones((10000,28,28))

    #X=np.concatenate((X,N),axis=0)
    X[:Nsamp, :, :] = X_outliers[:Nsamp, :, :]
    X_lab[:Nsamp] = 10

    X = np.clip(X, 0, 1)

    # find bags
    ind = np.isin(_test_lab, (0, 1, 2, 3, 4, 5, 6, 8))  #(1, 5, 7, 9)
    _test_lab_outliers = _test_lab[ind]
    test_outliers = test[ind]

    # find sneaker and ankle boots
    ind = np.isin(_test_lab, (7, 9))  # (0, 2, 3, 4, 6))  #
    _test_lab = _test_lab[ind]
    test = test[ind]

    # X = ((X / 255.0) > 0.01).astype(np.float)
    test = test / 255.0
    Nsamp = np.int(np.rint(len(X) * frac_anom)) + 1
    test_outliers = test_outliers / 255.0

    test[:Nsamp, :, :] =test_outliers[:Nsamp, :, :]
    _test_lab[:Nsamp] = 10

    test = np.clip(test, 0, 1)

    X_train, X_valid, X_lab_train, X_lab_valid = train_test_split(
        X, X_lab, test_size=0.33, random_state=10003)
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))
    test = test.reshape((len(test), np.prod(test.shape[1:])))

                                   

    return X_train, X_valid, X_lab_train, X_lab_valid



 

def l21RDAE(X, X_valid, X_lab_valid,layers, lamda, folder, learning_rate = 0.15, inner = 100, outer = 10, 
            batch_size = 133,re_init=False,inputsize = (28,28)):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    #tf.compat.v1.disable_eager_execution()
    
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            rael21 = l21RDA.RobustL21Autoencoder(sess = sess, lambda_= lamda*X.shape[0], 
                                                 layers_sizes=layers)
            l21L, l21S = rael21.fit(X = X,X_valid=X_valid,X_lab_valid=X_lab_valid, sess = sess, inner_iteration = inner, iteration = outer, 
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

    #X = np.load(r"/big_disk/akrami/git_repos_old/lesion-detector/src/AutoEncoder/L12/original_code/RobustAutoencoder-master/data/data.npk",allow_pickle=True)

    X, X_valid, X_lab_train, X_lab_valid=create_data(0.1)
    inner = 50
    outer = 20



    #lambda_list = [0.0001, 0.0003, 0.0008, 0.001, 0.0015, 0.00035, 0.00045, 
         #0.00055, 0.00065, 0.00075, 0.00085, 0.00095, 0.00105, 0.00115, 0.00125]
#     lambda_list = [0.00015,0.00018,0.0002,0.00025,0.00028,0.0003]
    lambda_list = [2, 0.9,0.3,0.2,0.1,0.02,0.01,0.005,0.002,0.001,0.0002,0.00002]
    print(lambda_list)
    
    layers = [784, 400, 20] ## S trans
    print("start")
    start_time = time.time()
    image_X = Image.fromarray(I.tile_raster_images(X = X, img_shape = (28,28), tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    for lam in lambda_list:
        folder = "lam" + str(lam)
        l21RDAE(X = X, X_valid=X_valid ,X_lab_valid=X_lab_valid,layers=layers, lamda = lam, folder = folder, learning_rate = 0.001, 
                inner = inner, outer = outer, batch_size = 133,re_init=True,inputsize = (28,28))
        print("done: lam", str(lam))
    print ("Runing time:" + str(time.time() - start_time) + " s")
if __name__ == "__main__":
    compare_frame()