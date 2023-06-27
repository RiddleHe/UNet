import tensorflow as tf
import tensorflow.keras.layers as tfl

# define a traditional downsampling conv block
def conv_block(input=None, n_filter=32, dropout_prob=0, max_pooling=True):
  conv = tfl.Conv2D(n_filter, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(input)
  conv = tfl.Conv2D(n_filter, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
  if dropout_prob>0:
    conv = tfl.Dropout(dropout_prob)(conv)
  if max_pooling:
    next_layer = tfl.MaxPooling2D(pool_size=pool_size)(conv)
  else:
    next_layer = conv
  skip_connection = conv
  return next_layer, skip_connection

# define a upsampling block
def upsampling_block(expansive_input, contractive_input, n_filter=32):
  up = tfl.Conv2DTranspose(n_filter, kernel_size, strides=(2,2), padding='same')(expansive_input)
  merge = tfl.concatenate([up, contractive_input], axis=-1)
  conv = tfl.Conv2D(n_filter, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
  conv = tfl.Conv2D(n_filter, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
  return conv

# define the whole U-Net
def unet_model(input_shape=(96,128,3), n_filter=32, n_class=n_class):
  inputs = tfl.Input(input_shape)
  cblock1 = conv_block(inputs, n_filter)
  cblock2 = conv_block(cblock1[0], 2*n_filter)
  cblock3 = conv_block(cblock2[0], 4*n_filter)
  cblock4 = conv_block(cblock3[0], 8*n_filter, dropout_prob=0.3)
  cblock5 = conv_block(cblock4[0], 16*n_filter, dropout_prob=0.3, max_pooling=False)

  ublock6 = upsampling_block(cblock5[0], cblock4[1], 8*n_filter)
  ublock7 = upsampling_block(ublock6, cblock3[1], 4*n_filter)
  ublock8 = upsampling_block(ublock7, cblock2[1], 2*n_filter)
  ublock9 = upsampling_block(ublock8, cblock1[1], n_filter)

  conv9 = tfl.Conv2D(n_filter, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
  conv10 = tfl.Conv2D(n_class, 1, padding='same')(conv9)
  model = tf.keras.Model(inputs=inputs, outputs=conv10)
  return model