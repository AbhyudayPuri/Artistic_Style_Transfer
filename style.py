import numpy as np
import tensorflow as tf
import vgg19_style
import utils
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pickle


# Reading the style and the content image
style_image = utils.load_image("./test_data/van.jpg")
content_image = utils.load_image("./test_data/lake.jpg")

# Adding an extra dimension to the images
image_c = np.float32(content_image.reshape((1, 224, 224, 3)))
image_s = np.float32(style_image.reshape((1, 224, 224, 3)))


#################################################################
# Setting the hyper-parameters for the program                  #
#################################################################

# Number of epochs to run the optimization for
num_epoch = 1200
# The penalty for content reconstruction
alpha = 0.001
# The penalty for style reconstruction
beta = 1

#################################################################
# Setting up the model for our program                          #
#################################################################

# Creating the randomly initialized image that acts as our generated image
x = tf.Variable(tf.random_uniform([1,224,224,3], minval=0, maxval=None, dtype=tf.float32, seed=None, name=None), name="x")

#################################################################
# Creating the feature maps for the 3 images                    #
#################################################################

# Creating the feature map for the content image
model_c = vgg19_style.Vgg19()
model_c.build(image_c)

# Creating the feature map for the style image
model_s = vgg19_style.Vgg19()
model_s.build(image_s)

# Creating the feature map for the generated image
model_x = vgg19_style.Vgg19()
model_x.build(x)


#################################################################
# Setting up the loss functions                                 #
#################################################################

# Content reconstruction loss
loss_content = (1/2)*tf.reduce_sum(tf.square(model_c.conv4_2 -  model_x.conv4_2))

# Style reconstruction loss
loss_style = 0

loss_style += 0.2*tf.reduce_sum(tf.square(utils.gram_matrix(model_s.conv1_1) -  utils.gram_matrix(model_x.conv1_1)))
loss_style += 0.2*tf.reduce_sum(tf.square(utils.gram_matrix(model_s.conv2_1) -  utils.gram_matrix(model_x.conv2_1)))
loss_style += 0.2*tf.reduce_sum(tf.square(utils.gram_matrix(model_s.conv3_1) -  utils.gram_matrix(model_x.conv3_1)))
loss_style += 0.2*tf.reduce_sum(tf.square(utils.gram_matrix(model_s.conv4_1) -  utils.gram_matrix(model_x.conv4_1)))
loss_style += 0.2*tf.reduce_sum(tf.square(utils.gram_matrix(model_s.conv5_1) -  utils.gram_matrix(model_x.conv5_1)))

# The total loss for the optimization problem at hand
total_loss = alpha*loss_content + beta*loss_style

################################################################
# Setting up the optimizer for the problem                     #
################################################################
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(total_loss)

# Intitializing the variables
init = tf.initialize_all_variables()

################################################################
# The main function with the tensorflow session                #
################################################################
def optimize():
	
	# Stores the loss for each iteration
	l = []

	# Startign the TF session
	with tf.Session() as session:
		session.run(init)
		for step in range(num_epoch):  
			# Running the optimizer over our loss functiojn
			session.run(train)
			l.append(session.run(total_loss))
			print('iter',step)
			print('loss:', l[step])
			# Plots the generated image after every 300 iterations
			if (step % 300 == 0):
				a = session.run(x)
				a = a.reshape(224,224,3)
				plt.imshow(a)
				plt.show()

		a = session.run(x)

	return l,a

###############################################################
# Calling the main fuction in order to run the TF session.    #
###############################################################
l,x = optimize()

a = x.reshape(224, 224, 3)

# Plotting the final output
plt.imshow(a)
plt.figure()
plt.show() 