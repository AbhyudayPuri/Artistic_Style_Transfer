import tensorflow as tf 
import numpy as np 

VGG_MEAN = [103.939, 116.779, 123.68]

class VGG19: 
	def __init__(self):
		# Loading the pre-trained weights into a dict
		self.weights_dict = np.load('vgg19.npy', encoding = 'latin1').item()
		print("Weights loaded")

	def build(self, rgb):
		#####################################################
		# Using the pre-trained weight to build the network #
		# rgb: rgb image [batch, height, width, 3]
		#####################################################

		print("Build model started")

		# Convert RGB to BGR
		red, green, blue = tf.split(axis = 3, num_or_size_splits = 3, value = rgb)
		assert red.get_shape().as_list()[1:] == [224,224,1]
		assert green.get_shape().as_list()[1:] == [224,224,1]
		assert blue.get_shape().as_list()[1:] == [224,224,1]

		bgr = tf.concat(axis = 3, values = [red - VGG_MEAN[2],
											green - VGG_MEAN[1],
											blue - VGG_MEAN[0]])

		assert bgr.get_shape().as_list()[1:] == [224,224,3]

		# Creating the layers of the network
		self.conv1_1 = self.conv_layer(bgr, "conv1_1")
		self.relu1_1 = tf.nn.relu(self.conv1_1, "relu1_1")
		self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
		self.relu1_2 = tf.nn.relu(self.conv1_2, "relu1_2")
		self.pool1 = self.max_pool(self.relu1_2, "pool1")

		self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
		self.relu2_1 = tf.nn.relu(self.conv2_1, "relu2_1")
		self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
		self.relu2_2 = tf.nn.relu(self.conv2_2, "relu2_2")
		self.pool2 = self.max_pool(self.relu2_2, "pool2")

		self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
		self.relu3_1 = tf.nn.relu(self.conv3_1, "relu3_1")
		self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
		self.relu3_2 = tf.nn.relu(self.conv3_2, "relu3_2")		
		self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
		self.relu3_3 = tf.nn.relu(self.conv3_3, "relu3_3")
		self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
		self.relu3_4 = tf.nn.relu(self.conv3_4, "relu3_4")
		self.pool3 = self.max_pool(self.relu3_4, "pool3")

		self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
		self.relu4_1 = tf.nn.relu(self.conv4_1, "relu4_1")
		self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
		self.relu4_2 = tf.nn.relu(self.conv4_2, "relu4_2")		
		self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
		self.relu4_3 = tf.nn.relu(self.conv4_3, "relu4_3")
		self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
		self.relu4_4 = tf.nn.relu(self.conv4_4, "relu4_4")
		self.pool4 = self.max_pool(self.relu4_4, "pool4")

		self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
		self.relu5_1 = tf.nn.relu(self.conv5_1, "relu5_1")
		self.conv5_2 = self.conv_layer(self.relu5_1, "conv5_2")
		self.relu5_2 = tf.nn.relu(self.conv5_2, "relu5_2")		
		self.conv5_3 = self.conv_layer(self.relu5_2, "conv5_3")
		self.relu5_3 = tf.nn.relu(self.conv5_3, "relu5_3")
		self.conv5_4 = self.conv_layer(self.relu5_3, "conv5_4")
		self.relu5_4 = tf.nn.relu(self.conv5_4, "relu5_4")
		self.pool5 = self.max_pool(self.relu5_4, "pool5")

		self.fc6 = self.fc_layer(self.pool5, "fc6")
		assert self.fc6.get_shape().as_list()[1:] == [4096]
		self.relu6 = tf.nn.relu(self.fc6)

		self.fc7 = self.fc_layer(self.fc6, "fc7")
		self.relu7 = tf.nn.relu(self.fc7)

		self.fc8 = self.fc_layer(self.relu7, "fc8")

		self.prob = tf.nn.softmax(self.fc8, name="prob")

		self.weights_dict = None
		print("Build model complete!")



	def max_pool(self, bottom, name):
		return tf.nn.max_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
	
	def avg_pool(self, bottom, name):
		return tf.nn.avg_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

	def conv_layer(self, bottom, name):
		with tf.variable_scope(name):
			# Reading the wieghts and biases from the dict
			filt = tf.constant(self.weights_dict[name][0], name = "filter")
			bias = tf.constant(self.weights_dict[name][1], name = "biases")

			conv = tf.nn.conv2d(bottom, filt, [1,1,1,1], padding='SAME')
			output = tf.nn.bias_add(conv, bias)

			# relu = tf.nn.relu(output)
			return output

	def fc_layer(self, bottom, name):
		with tf.variable_scope(name):
			shape = bottom.get_shape().as_list()
			dim = 1
			for d in shape[1:]:
				dim *= d
			x = tf.reshape(bottom, [-1, dim])

			weights = tf.constant(self.weights_dict[name][0], name = "weights")
			biases = tf.constant(self.weights_dict[name][1], name = "biases")

			fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
			return fc