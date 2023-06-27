import tensorflow as tf
import tensorflow.keras.layers as tfl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import imageio

import model

# hyperparameters

# model params
kernel_size = 3
pool_size = (2,2)
n_class = 32
n_filter = 32
img_height = 96
img_width = 128
n_channel = 3

# training params
epochs = 5
batch_size = 32
buffer_size = 500
val_subsplits = 5

# select images and masks
path = '/content/drive/MyDrive/image_seg_dataset/seg_dataset/Files/'
image_path = os.path.join(path, './data/CameraRGB/')
mask_path = os.path.join(path, './data/CameraMask/')
image_path_orig = os.listdir(image_path)
image_list = [image_path+i for i in image_path_orig]
mask_list = [mask_path+i for i in image_path_orig]

# convert to tensor then dataset
image_filenames = tf.constant(image_list)
mask_filenames = tf.constant(mask_list)

dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))

# define the preprocessing functions
def process_path(image_path, mask_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_png(mask, channels=3)
  mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)

  return img, mask

def preprocess(img, mask):
  img = tf.image.resize(img, (96,128), method='nearest')
  mask = tf.image.resize(mask, (96,128), method='nearest')
  return img, mask

# implement preprocessing
image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)

# instantiate the model
model = unet_model((img_height, img_width, n_channel), n_filter, n_class)

# compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# load the data
train_dataset = processed_image_ds.cache().shuffle(buffer_size).batch(batch_size)

# train the model
model_history = model.fit(train_dataset, epochs=epochs)

# display images
def display(display_list):
  plt.figure(figsize=(15,15))
  titles = ['Image', 'True mask', 'Predicted mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(titles[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# define a function to see predicted mask
def create_mask(pred_mask): #(h, w, n_class)
  pred_mask = tf.math.argmax(pred_mask, axis=-1) #(h, w)
  pred_mask = pred_mask[..., tf.newaxis] # (h, w, 1)
  return pred_mask[0] # (w, 1)

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])

show_predictions(train_dataset, 6)