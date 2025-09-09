import os
import sys

if len(sys.argv) != 3:
    print("Usage: python start_training.py <normal_images_folder> <ground_truth_folder>")
    sys.exit(1)

import numpy as np
import tensorflow as tf
from src.dataGenerator import DataGenerator
from tensorflow.keras.layers import Input, Conv2D, concatenate, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Data preparation
folder_to_search = sys.argv[1]
DATA_NAMES = [f for f in os.listdir(folder_to_search) if f.endswith(".png")]
print(f"Found {len(DATA_NAMES)} images for training.")

# Training parameters
NB_EPOCH = 50
BATCH_SIZE = 16
EPOCH_SIZE = 40
PATCH_SIZE = 512
IMG_FOLDER = folder_to_search
GROUNDTRUTH_FOLDER = sys.argv[2]
PATCH_RATIO = 0.5
ROTATION_STEP = 10
NOISE_SCALE = 64
NOISE_MAX_ANGLE = 5
RESCALE = 1.6

gen = DataGenerator(
	DATA_NAMES,
	BATCH_SIZE,
	EPOCH_SIZE,
	PATCH_SIZE,
	IMG_FOLDER,
	GROUNDTRUTH_FOLDER,
	PATCH_RATIO,
	ROTATION_STEP,
	NOISE_SCALE,
	NOISE_MAX_ANGLE,
	RESCALE,
	flip=True,
)

# Model definition (U-Net - Receptive Field ~ 400x400px)
def build_unet_big(input_shape):
	inputs = Input(input_shape)
	c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
	p1 = MaxPooling2D((2, 2))(c1)

	c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
	c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
	p2 = MaxPooling2D((2, 2))(c2)

	c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
	c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
	p3 = MaxPooling2D((2, 2))(c3)

	c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
	c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
	p4 = MaxPooling2D((2, 2))(c4)

	c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p4)
	c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
	p5 = MaxPooling2D((2, 2))(c5)

	b = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p5)
	b = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b)

	u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(b)
	u5 = concatenate([u5, c5])
	c8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u5)
	c8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c8)

	u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c8)
	u4 = concatenate([u4, c4])
	c9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u4)
	c9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c9)

	u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c9)
	u3 = concatenate([u3, c3])
	c10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u3)
	c10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c10)

	u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c10)
	u2 = concatenate([u2, c2])
	c11 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
	c11 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c11)

	u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c11)
	u1 = concatenate([u1, c1])
	c12 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
	c12 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c12)

	outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(c12)
	return Model(inputs, outputs)

IMG_HEIGHT = PATCH_SIZE
IMG_WIDTH = PATCH_SIZE
IMG_CHANNELS = 3
model = build_unet_big((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Losses and metrics
def dice_loss(y_true, y_pred, smooth=1e-8):
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	intersection = tf.reduce_sum(y_true*y_pred)
	den1 = tf.reduce_sum(y_true * y_true)
	den2 = tf.reduce_sum(y_pred * y_pred)
	dice = (2. * intersection + smooth) / (den1 + den2 + smooth)
	return 1 - tf.reduce_mean(dice)

def bce_dice_loss(y_true, y_pred):
	bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
	dice = dice_loss(y_true, y_pred)
	return bce + dice

def focal_dice_loss(y_true, y_pred, lambda_bfce=1.0):
	bfce = tf.keras.losses.BinaryFocalCrossentropy()(y_true, y_pred)
	dice = dice_loss(y_true, y_pred)
	return lambda_bfce * bfce + dice

def f1_score_metric(y_true, y_pred):
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	tp = tf.reduce_sum(y_true * y_pred)
	fp = tf.reduce_sum((1 - y_true) * y_pred)
	fn = tf.reduce_sum(y_true * (1 - y_pred))
	precision = tp / (tp + fp + 1e-6)
	recall = tp / (tp + fn + 1e-6)
	f1 = 2 * precision * recall / (precision + recall + 1e-6)
	return f1

# Callbacks
checkpoint = ModelCheckpoint(filepath='checkpoints/best.weights.h5', save_weights_only=True, save_best_only=True, monitor='loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

# Compile and fit
adam2 = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5, beta_2=0.5, epsilon=1e-07, amsgrad=False)
# np.seterr(all='raise')
# tf.debugging.enable_check_numerics() # this slows code
model.compile(
	optimizer=adam2,
	loss=focal_dice_loss,
	metrics=[f1_score_metric, dice_loss]
)

history = model.fit(
	gen,
	epochs=NB_EPOCH,
	verbose=2,
	#callbacks=[checkpoint, early_stopping]
)

model.save('unet_model_from_normals.h5')
