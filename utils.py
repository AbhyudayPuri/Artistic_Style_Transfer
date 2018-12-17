import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf 


def gram_matrix(x):

    # Getting the diemnsions of x
    dim = x.get_shape().as_list()

    # Computing the dimensions of the feature map
    M = dim[1] * dim[2]
    N = dim[3]

    # Reshaping the feature map
    F = tf.reshape(x, (N,M))

    # Computing the Gram matrix of the feature map
    G = (1 / (2 * M * N)) * tf.matmul(F, tf.transpose(F))

    return G
    
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0

    # We crop the image from the center
    short_edge = min(img.shape[:2])
    y = int((img.shape[0] - short_edge) / 2)
    x = int((img.shape[1] - short_edge) / 2)
    crop_img = img[y: y + short_edge, x: x + short_edge]
    
    # Resizing the image to [224,224,3]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

