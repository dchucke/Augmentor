#!/usr/bin/env python
# coding: utf-8

# # Training a Neural Network using Augmentor and Keras
# 
# In this notebook, we will train a simple convolutional neural network on the MNIST dataset using Augmentor to augment images on the fly using a generator.
# 
# ## Import Required Libraries
# 
# We start by making a number of imports:

# In[1]:


import Augmentor

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Define a Convolutional Neural Network
# 
# Once the libraries have been imported, we define a small convolutional neural network. See the Keras documentation for details of this network: <https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py> 
# 
# It is a three layer deep neural network, consisting of 2 convolutional layers and a fully connected layer:

# In[2]:


num_classes = 10
input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Once a network has been defined, you can compile it so that the model is ready to be trained with data:

# In[3]:


model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# You can view a summary of the network using the `summary()` function:

# In[4]:


model.summary()


# ## Use Augmentor to Scan Directory for Data
# 
# Now we will use Augmentor to scan a directory containing our data that we will eventually feed into the neural network in order to train it. 
# 
# When you point a pipeline to a directory, it will scan each subdirectory and treat each subdirectory as a class for your machine learning problem. 
# 
# For example, within the directory `mnist`, there are subdirectories for each digit:
# 
# ```
# mnist/
# ├── 0/
# │   ├── 0001.png
# │   ├── 0002.png
# │   ├── ...
# │   └── 5985.png
# ├── 1/
# │   ├── 0001.png
# │   ├── 0002.png
# │   ├── ...
# │   └── 6101.png
# ├── 2/
# │   ├── 0000.png
# │   ├── 0001.png
# │   ├── ...
# │   └── 5801.png
# │ ...
# ├── 9/
# │   ├── 0001.png
# │   ├── 0002.png
# │   ├── ...
# │   └── 6001.png
# └
# ```
# 
# The directory `0` contains all the images corresponding to the 0 class.
# 
# To get the data, we can use `wget` (this may not work under Windows):

# In[5]:


get_ipython().system('wget https://github.com/unifyid-labs/DeepGenStruct-Notebooks/raw/master/seed/seed_images_Kannada.npy')
get_ipython().system('mkdir -p images')
from PIL import Image
from IPython.core.debugger import set_trace
import logging, sys
img = np.load("./seed_images_Kannada.npy")
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.debug(img)
get_ipython().run_line_magic('debug', '')
for i in range(10):
	for j in range(28):
		Image.fromarray(img[i][j].astype(np.uint8)).save("./images/"+str(i)+"/"+str(j)+".png")
		get_ipython().run_line_magic('debug', '')


# After the MNIST data has downloaded, we can instantiate a `Pipeline` object in the `training` directory to add the images to the current pipeline:

# In[6]:


p = Augmentor.Pipeline("images/")


# ## Add Operations to the Pipeline
# 
# Now that a pipeline object `p` has been created, we can add operations to the pipeline. Below we add several simple  operations:

# In[7]:


p.flip_top_bottom(probability=0.1)
p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)


# You can view the status of pipeline using the `status()` function, which shows information regarding the number of classes in the pipeline, the number of images, and what operations have been added to the pipeline:

# In[8]:


p.status()


# ## Creating a Generator
# 
# A generator will create images indefinitely, and we can use this generator as input into the model created above. The generator is created with a user-defined batch size, which we define here in a variable named `batch_size`. This is used later to define number of steps per epoch, so it is best to keep it stored as a variable.

# In[9]:


batch_size = 128
g = p.keras_generator(batch_size=batch_size)


# The generator can now be used to created augmented data. In Python, generators are invoked using the `next()` function - the Augmentor generators will return images indefinitely, and so `next()` can be called as often as required. 
# 
# You can view the output of generator manually:

# In[10]:


images, labels = next(g)


# Images, and their labels, are returned in batches of the size defined above by `batch_size`. The `image_batch` variable is a tuple, containing the augmentented images and their corresponding labels.
# 
# To see the label of the first image returned by the generator you can use the array's index:

# In[11]:


print(labels[0])


# Or preview the images using Matplotlib (the image should be a 5, according to the label information above):

# In[12]:


plt.imshow(images[0].reshape(28, 28), cmap="Greys");


# ## Train the Network
# 
# We train the network by passing the generator, `g`, to the model's fit function. In Keras, if a generator is used we used the `fit_generator()` function as opposed to the standard `fit()` function. Also, the steps per epoch should roughly equal the total number of images in your dataset divided by the `batch_size`.
# 
# Training the network over 5 epochs, we get the following output:

# In[13]:


h = model.fit_generator(g, steps_per_epoch=len(p.augmentor_images)/batch_size, epochs=5, verbose=1)


# ## Summary
# 
# Using Augmentor with Keras means only that you need to create a generator when you are finished creating your pipeline. This has the advantage that no images need to be saved to disk and are augmented on the fly.
