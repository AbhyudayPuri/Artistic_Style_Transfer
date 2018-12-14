import Load_VGG as VGG
import tensorflow as tf 
import numpy as np 
import cv2 as cv
from utils import rgb_resize

# rgb = cv.imread('/Users/abhyudaypuri/Downloads/temp.jpeg')
rgb = cv.imread('temp.jpeg')
batch = rgb_resize(rgb)


with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: batch}

        model = VGG.VGG19()
        with tf.name_scope("content_vgg"):
            model.build(images)

        prob = sess.run(model.prob, feed_dict=feed_dict)
        print(prob.argsort()[::-1][-5:])
        print("Done")
